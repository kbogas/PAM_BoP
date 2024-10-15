import os
import time

import catboost
import numpy as np
import pandas as pd
from scipy.sparse import hstack as sp_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import (
    create_pam_matrices_with_start_and_end,
    get_sparsity,
    load_data_nc,
    set_all_seeds,
)

# Set random state
random_state = 42
set_all_seeds(random_state)

############################## USER SELECTION #######################################


# Select the datasets for which to run the script.
dataset_list = [
    "AIFB",
    # "MUTAG",
    # "BGS",
    # "AM"
]

# Number of runs
num_runs = 5

# Device to use (by default CPU, but can use GPU for CatBoost training)
device = "GPU"  # feel free to change to "GPU" if available

# Maximum number of hops to do for PAMs
k = 4

# Alpha for self BoP feature vector weighting when aggregating neigbhorhood
alpha = 2

############################## END USER SELECTION ####################################


print(f"Will run in {device}")


res = []


root_path = "./"

path_out = os.path.join(
    root_path, f"./results/{'_'.join(dataset_list)}_{num_runs}_runs.txt"
)


for dataset in dataset_list:
    print(f"###### {dataset}  ######")

    # Load data
    path_to_files = os.path.join(root_path, f"./data/{dataset}/")

    df_train, df_nodes = load_data_nc(path_to_files)

    unique_rels = sorted(list(df_train["rel"].unique()))
    unique_nodes = sorted(
        set(df_train["head"].values.tolist() + df_train["tail"].values.tolist())
    )
    print(
        f"# of unique rels: {len(unique_rels)} \t | # of unique nodes: {len(unique_nodes)}"
    )
    print()

    # Generate features only for the needed nodes (those with label)
    wanted_label_nodes = df_nodes["id"].values.tolist()

    # Keep track of nodes in the 1-hop neighborhood of the wanted nodes as well
    for _ in range(1):
        wanted_label_nodes += df_train[df_train["head"].isin(wanted_label_nodes)][
            "tail"
        ].values.tolist()
    wanted_label_nodes = np.array(list(set(wanted_label_nodes)))
    print(
        f"For efficiency will focus on {len(wanted_label_nodes)} nodes. (The labeled nodes + their 1-hop neighbors)\n"
    )

    # Pring generate pams
    time_s = time.time()
    pams, node2id, rel2id, _ = create_pam_matrices_with_start_and_end(
        df_train,
        max_order=k,
        wanted_nodes_start=wanted_label_nodes,
        wanted_nodes_end=wanted_label_nodes,
    )

    print(
        f"Generated PAMS in {time.time() - time_s:.3f} seconds.. Will aggregate BoPs from PAMs.\n"
    )

    # Aggregate BoP features using PAMs
    wanted_label_indices = np.array(list(map(node2id.get, wanted_label_nodes)))
    node_feats = []
    for hop_i, cp in enumerate(pams):
        print(
            f"Hop: {hop_i + 1}\t Shape: {cp.shape}\t Sparsity: {get_sparsity(cp):.2f}\t Nnz: {cp.nnz}"
        )
        if cp.shape[0] > len(wanted_label_indices):
            cur_cp = cp[wanted_label_indices, :]
        else:
            cur_cp = cp
        if cur_cp.shape[1] > len(wanted_label_indices):
            cur_cp = cur_cp[:, wanted_label_indices]

        cur_cp.eliminate_zeros()
        # These are the outgoing paths of each node
        outgoing = cur_cp
        # Thease are the incoming paths of each node
        incoming = cur_cp.T

        cur_feats = sp_hstack((outgoing, incoming))
        node_feats.append(cur_feats)
    node_feats = sp_hstack(node_feats)

    print(f"\nVectorizing with sklearn.\n")
    vect = TfidfVectorizer(
        tokenizer=lambda x: x,
        token_pattern=None,
        lowercase=False,
        preprocessor=lambda x: x,
        min_df=5,
        max_df=0.95,
        max_features=10000,
    )

    node_feats = vect.fit_transform(node_feats.tolil().data).toarray()
    print(f"Node Features shape: {node_feats.shape}.\n")

    # Do the 1-hop aggregation of features
    adj = pams[0]
    if adj.shape[0] > len(wanted_label_indices):
        adj = adj[wanted_label_indices, :][:, wanted_label_indices]
    if adj.shape[1] > len(wanted_label_indices):
        adj = adj[:, wanted_label_indices]
    adj.eliminate_zeros()
    adj.data = np.ones(len(adj.data))
    adj.setdiag(alpha)

    node_feats_final = adj @ node_feats

    # No longer needed
    del node_feats
    del adj
    del pams

    # Create X_train (Num_Train_Nodes X Feats)
    wanted_label_indices_list = wanted_label_indices.tolist()
    X_train = node_feats_final[
        np.array(
            [
                wanted_label_indices_list.index(node2id[train_id])
                for train_id in df_nodes[df_nodes["split"] == "train"]["id"]
            ]
        )
    ]

    # Create X_test (Num_Test_Nodes X Feats)
    X_test = node_feats_final[
        np.array(
            [
                wanted_label_indices_list.index(node2id[test_id])
                for test_id in df_nodes[df_nodes["split"] == "test"]["id"]
            ]
        )
    ]

    # Standard Scaling of features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Keep track of the labels
    y_train = df_nodes[df_nodes["split"] == "train"]["label"].astype(int).values
    y_test = df_nodes[df_nodes["split"] == "test"]["label"].astype(int).values

    time_pam_and_feats = time.time() - time_s

    time_s = time.time()

    # Iterate over different runs
    for num_run in range(num_runs):

        X_train_run, X_val_run, y_train_run, y_val_run = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=num_run,
            stratify=y_train,
        )

        print(f"(Run {num_run}) Training Classifier..\n")

        clf = catboost.CatBoostClassifier(
            iterations=1000,
            early_stopping_rounds=20,
            task_type=device,
            verbose=False,
            devices="0",
            random_seed=num_run,
            use_best_model=True,
            auto_class_weights="Balanced",
        )

        clf.fit(
            X_train_run,
            y_train_run,
            eval_set=(X_val_run, y_val_run),
            use_best_model=True,
        )
        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))
        print("\n\n")
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        micro_f1 = f1_score(y_test, y_pred, average="micro")

        time_clf = time.time() - time_s
        time_total = time_clf + time_pam_and_feats

        res.append(
            (
                dataset,
                len(df_train),
                len(unique_nodes),
                len(unique_rels),
                len(X_train_run),
                len(X_test),
                k,
                X_train.shape[1],
                len(wanted_label_nodes),
                num_run,
                time_total,
                acc,
                macro_f1,
                micro_f1,
            )
        )
        res_df = pd.DataFrame(
            res,
            columns=[
                "dataset",
                "num_edges",
                "num_nodes",
                "num_rels",
                "num_train",
                "num_test",
                "max_order",
                "num_feats",
                "num_focus_nodes",
                "num_run",
                "time_total",
                "acc",
                "macro_f1",
                "micro_f1",
            ],
        )
        res_df.to_csv(path_out, index=False)

res_df = pd.DataFrame(
    res,
    columns=[
        "dataset",
        "num_edges",
        "num_nodes",
        "num_rels",
        "num_train",
        "num_test",
        "max_order",
        "num_feats",
        "num_focus_nodes",
        "num_run",
        "time_total",
        "acc",
        "macro_f1",
        "micro_f1",
    ],
)
means = res_df.groupby(["dataset", "max_order"]).aggregate("mean")[["acc"]]
std = res_df.groupby(["dataset", "max_order"]).aggregate("std")[["acc"]]
res = means.astype(str) + " ± " + std.astype(str)
print(f"Acc ± 1*std")
print(res.iloc[0].to_string())
