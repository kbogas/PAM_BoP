import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sympy.ntheory import factorint
from utils import (
    create_lossless_khops,
    create_pam_matrices,
    get_filtering_cache,
    load_data,
    set_all_seeds,
)

# For reproducibility
random_state = 42
set_all_seeds(random_state)


############################## USER SELECTION #######################################

root_path_to_data = "./data"
path_to_save_res = "./results/lossy_and_lossless_WN18RR_dataset_runs.txt"

runs_to_do = {
    "WN18RR_lossy": {
        "dataset": "WN18RR",
        "method": "lossy",
        "max_order": 5,
        "sim_pairs": 100,
    },
    "WN18RR_lossless": {
        "dataset": "WN18RR",
        "method": "lossless",
        "max_order": 2,
        "sim_pairs": 100,
    },
    # "DDB14_lossy": {
    #     "dataset": "DDB14",
    #     "method": "lossy",
    #     "max_order": 3,
    #     "sim_pairs": 20,
    # },
    # "DDB14_lossless": {
    #     "dataset": "DDB14",
    #     "method": "lossless",
    #     "max_order": 1,
    #     "sim_pairs": 20,
    # },
    # "NELL995_lossy": {
    #     "dataset": "NELL995",
    #     "method": "lossy",
    #     "max_order": 4,
    #     "sim_pairs": 20,
    # },
    # "NELL995_lossless": {
    #     "dataset": "NELL995",
    #     "method": "lossless",
    #     "max_order": 1,
    #     "sim_pairs": 100,
    # },
}


######################## END USER SELECTION #######################################

total_results = []
for run_name, run_settings in runs_to_do.items():
    dataset_name = run_settings["dataset"]
    path_to_files = os.path.join(root_path_to_data, dataset_name)

    method = run_settings["method"]
    max_order = run_settings["max_order"]
    sim_pairs = run_settings["sim_pairs"]

    # Loading the data
    time_s = time.time()
    df_train_orig, df_train, df_eval, df_test, already_seen_triples = load_data(
        path_to_files, dataset_name, add_inverse_edges="NO"
    )

    # Info on the data
    unique_rels = sorted(list(df_train["rel"].unique()))
    unique_nodes = sorted(
        set(df_train["head"].values.tolist() + df_train["tail"].values.tolist())
    )
    print(
        f"# of unique rels: {len(unique_rels)} \t | # of unique nodes: {len(unique_nodes)}"
    )

    time_prev = time.time()
    time_needed = time_prev - time_s
    print(
        f"Total time for reading: {time_needed:.5f} secs ({time_needed/60:.2f} mins)\n"
    )

    time_s = time.time()

    if method == "lossy":
        (
            pam_1hop_lossless,
            power_A,
            node2id,
            rel2id,
            broke_cause_of_sparsity,
        ) = create_pam_matrices(
            df_train_orig,
            max_order=max_order,
            break_with_sparsity_threshold=0.8,
        )
        max_order = len(power_A)

        id2rel = {v: k for k, v in rel2id.items()}
    elif method == "lossless":
        (
            pam_1hop_lossless,
            power_A,
            node2id,
            rel2id,
            broke_cause_of_sparsity,
            mappings,
        ) = create_lossless_khops(df_train_orig, max_hop=max_order, print_=False)
        max_order = len(power_A)

        id2rel = {v: k for k, v in rel2id.items()}

    else:
        raise NotImplementedError(f"{method} not implemented")

    time_prev = time.time()
    time_needed = time_prev - time_s
    print(
        f"Total time for P^{max_order}: {time_needed:.5f} secs ({time_needed/60:.2f} mins)\n"
    )

    # Map the initial data to node indices and relation primes
    df_train_mapped = df_train.copy()
    df_train_mapped["rel"] = df_train["rel"].map(rel2id)
    df_train_mapped["head"] = df_train["head"].map(node2id).astype(int)
    df_train_mapped["tail"] = df_train["tail"].map(node2id).astype(int)

    df_train_mapped.dropna(inplace=True)

    # Create the features of the paths for the original train pairs
    features_ij, labels_ij = {}, {}

    if method == "lossless":
        value2indices = {}

        total_feats = 0
        unq_path_total = 0
        for hop, hop_dict in mappings.items():
            if hop == 0:
                num_feats_per_hop = {0: 0}
                continue
            unq_paths = []
            prime2indices = {}
            for prime, paths in hop_dict["prime2path"].items():
                cur_indices = []
                for path in paths:
                    if path not in unq_paths:
                        unq_paths += [path]
                    cur_indices.append(unq_paths.index(path))
                prime2indices[prime] = cur_indices
            value2indices[hop] = prime2indices
            num_feats_per_hop[hop] = len(unq_paths)
            unq_path_total += len(unq_paths)
            print(f"Hop {hop}, UNQ: {len(unq_paths)}")

        print(f"Total Feats: {sum(num_feats_per_hop.values())}")
        print(f"Total UNQ: {unq_path_total}")

    if method == "lossless":
        node_feats = np.empty((len(unique_nodes), 2 * unq_path_total))
    else:
        node_feats = np.zeros((len(unique_nodes), 2 * max_order))

    for i, row in df_train_mapped.iterrows():
        cur_row, cur_col = int(row["head"]), int(row["tail"])
        if (cur_row, cur_col) not in features_ij:
            features_ij[(cur_row, cur_col)] = np.zeros((max_order,))
            if method == "lossless":
                total_feats = unq_path_total  # sum(num_feats_per_hop.values())
                features_ij[(cur_row, cur_col)] = np.zeros((total_feats,))

            labels_ij[(cur_row, cur_col)] = set()
        for cur_hop in range(0, max_order):
            cur_power = power_A[cur_hop]
            cur_value = cur_power[(cur_row, cur_col)]
            if cur_value > 0:
                if method == "lossless":

                    if cur_hop == 0:
                        factors = factorint(cur_value, multiple=True)
                    else:
                        factors = [cur_value]
                    offset = num_feats_per_hop[cur_hop]
                    for value in factors:
                        for index in value2indices[cur_hop + 1][value]:
                            features_ij[(cur_row, cur_col)][offset + index] += 1
                            node_feats[cur_row, offset + index] += 1
                            node_feats[cur_col, unq_path_total + offset + index] += 1
                else:
                    features_ij[(cur_row, cur_col)][cur_hop] = cur_value
                    node_feats[cur_row, cur_hop] += cur_value
                    node_feats[cur_col, max_order + cur_hop] += cur_value

        labels_ij[(cur_row, cur_col)] = labels_ij[(cur_row, cur_col)].union(
            [row["rel"]]
        )

    print(f"Feats, shapes: {node_feats.shape}")

    X_train = []
    y_train = []
    train_pairs = []

    add_inverse_pair_features = True

    for ij, ij_feature_dict in features_ij.items():
        cur_features_pair = ij_feature_dict
        if add_inverse_pair_features:
            try:
                cur_features_inverse = features_ij[(ij[1], features_ij[0])]
            except KeyError:
                if method == "lossless":
                    cur_features_inverse = np.zeros((total_feats,))
                else:
                    cur_features_inverse = np.zeros((max_order,))

            cur_features_pair = np.hstack(
                (cur_features_pair, cur_features_inverse)
            ).reshape(1, -1)
        cur_features_nodes = np.hstack(
            (
                node_feats[ij[0], :].reshape(1, -1),
                node_feats[ij[1], :].reshape(1, -1),
            )
        )
        cur_features = np.hstack((cur_features_pair, cur_features_nodes))
        X_train.append(cur_features.reshape(-1))
        y_train.append(list(labels_ij[ij]))
        train_pairs.append(ij)

    assert 0 == (np.array(X_train).sum(axis=1) == 0).sum()
    assert len(X_train) == len(features_ij)

    X_train = np.array(X_train)
    if method == "lossless":
        X_train_ohe = X_train
        X_train_ohe = csr_array(X_train_ohe)
    else:
        # One-hot encode the data
        ohe = OneHotEncoder(handle_unknown="ignore")
        X_train_ohe = ohe.fit_transform(X_train)  # np.array(X_train)  #

    print(f"Extracted features for the train pairs {X_train_ohe.shape} ...")

    # Repeat the same feature extraction procedure for the test data
    df_test_mapped = df_test.copy()
    df_test_mapped["rel"] = df_test["rel"].map(rel2id)
    df_test_mapped["head"] = df_test["head"].map(node2id)
    df_test_mapped["tail"] = df_test["tail"].map(node2id)
    df_test_mapped.dropna(inplace=True)

    df_test_mapped["ij"] = tuple(
        zip(
            df_test_mapped["head"].astype(int).values,
            df_test_mapped["tail"].astype(int).values,
        )
    )
    X_test = []
    y_test = []
    count_no_features = 0
    for i, row in df_test_mapped[["rel", "ij"]].iterrows():

        if method == "lossless":
            cur_dict = np.zeros((2 * total_feats))
        else:
            cur_dict = np.zeros((2 * max_order))
        for cur_hop in range(0, max_order):
            cur_power = power_A[cur_hop]
            cur_value = cur_power[row["ij"]]
            if cur_value > 0:
                if method == "lossless":

                    if cur_hop == 0:
                        factors = factorint(cur_value, multiple=True)
                    else:
                        factors = [cur_value]
                    offset = num_feats_per_hop[cur_hop]
                    for value in factors:
                        for index in value2indices[cur_hop + 1][value]:
                            cur_dict[offset + index] += 1
                else:
                    cur_dict[cur_hop] = cur_value
            if add_inverse_pair_features:
                cur_value = cur_power[row["ij"][1], row["ij"][0]]
                if cur_value > 0:
                    if method == "lossless":

                        if cur_hop == 0:
                            factors = factorint(cur_value, multiple=True)
                        else:
                            factors = [cur_value]
                        offset = num_feats_per_hop[cur_hop]
                        for value in factors:
                            for index in value2indices[cur_hop + 1][value]:
                                cur_dict[total_feats + offset + index] += 1
                    else:
                        cur_dict[cur_hop + max_order] = cur_value

            cur_features_nodes = np.hstack(
                (
                    node_feats[row["ij"][0], :].reshape(1, -1),
                    node_feats[row["ij"][1], :].reshape(1, -1),
                )
            )
            cur_features = np.hstack(
                (cur_dict.reshape(1, -1), cur_features_nodes.reshape(1, -1))
            )
        X_test.append(cur_features.reshape(-1))
        y_test.append([row["rel"]])
    X_test = np.array(X_test)

    if method == "lossless":
        X_test_ohe = X_test
        # X_test_ohe = X_test[:, indices_to_keep]
        X_test_ohe = csr_array(X_test_ohe)
    else:
        X_test_ohe = ohe.transform(X_test)  # X_test  #

    print(f"Extracted features for the test pairs {X_test_ohe.shape}...\n")

    # from sklearn.neighbors import KNeighborsClassifier

    # id2rel = {v: k for k, v in rel2id.items()}
    # y_train_one = [id2rel[y[0]] for y in y_train]

    # y_test_one = [id2rel[y[0]] for y in y_test]

    # clf = KNeighborsClassifier(n_neighbors=sim_pairs, n_jobs=30, metric="manhattan")
    # clf.fit(X_train_ohe, y_train_one)
    # probas = clf.predict_proba(X_test_ohe)
    # res = []
    # for i_test, cur_row in df_test_mapped.reset_index().iterrows():
    #     cur_probas = probas[i_test]
    #     sorted_indices = np.argsort(cur_probas)[::-1]
    #     pred_labels = clf.classes_[sorted_indices].tolist()
    #     proba_labels = cur_probas[sorted_indices].tolist()
    #     rank = pred_labels.index(id2rel[cur_row["rel"]]) + 1
    #     cur_res = {
    #         "predicted": pred_labels,
    #         "probas": proba_labels,
    #         "rank": rank,
    #         **cur_row,
    #     }
    #     res.append(cur_res)

    print(f"Distances between {X_test_ohe.shape} and {X_train_ohe.shape}")
    distances = pairwise_distances(X_test_ohe, X_train_ohe, metric="manhattan")
    sorted_ids = np.argsort(distances, axis=1)

    # Keep track of already seen relations between pairs, to exclude them from predictions
    cache_triples = get_filtering_cache(df_train, df_eval)
    cache_mapped = {}
    for (h, t), rels in cache_triples.items():
        try:
            cache_mapped[(node2id[h], node2id[t])] = [rel2id[rel] for rel in rels]
        except KeyError:
            continue

    # Iterate over the test pairs
    list_of_pairs = set(list(features_ij.keys()))

    res = []
    train_pairs = np.array(train_pairs)

    # For each test pair keep their most similar pair
    print(f"Iterating over tests")
    for i_test, similar_pairs in enumerate(sorted_ids[:, :sim_pairs]):
        cur_row = df_test_mapped.iloc[i_test]

        current_similar_pairs = train_pairs[similar_pairs, :]
        # remove the test pair from its list of most similar pairs, if it exists
        if (cur_row["head"], cur_row["tail"]) in list_of_pairs:
            similar_pairs = similar_pairs[1 : sim_pairs + 1]
        else:
            similar_pairs = similar_pairs[:sim_pairs]

        # Find the possible labels, according to the labels of the most similar pairs
        poss_labels = [label for id_ in similar_pairs for label in y_train[id_]]
        pred_labels = list(dict.fromkeys(poss_labels))

        predictions = [
            (label, count)
            for (label, count) in (Counter(poss_labels).most_common())[:sim_pairs]
        ]
        pred_labels = [pred[0] for pred in predictions]
        proba_labels = [pred[1] for pred in predictions]

        try:
            rank = pred_labels.index(cur_row["rel"]) + 1
        except ValueError:
            rank = len(unique_rels) + 1

        cur_res = {
            "predicted": pred_labels,
            "probas": proba_labels,
            "rank": rank,
            "similar_ij": current_similar_pairs,
            **cur_row,
        }
        res.append(cur_res)

    # break

    df_res = pd.DataFrame(res)
    pr_results = {}
    pr_results["MRR"] = (1 / df_res["rank"]).mean()
    time_taken = time.time() - time_s
    print(f"\n #### RESULTS FOR {run_name} #######")
    print(f"MRR:{pr_results['MRR']:.4f}")

    for k in [1, 3, 10]:
        pr_results[f"h@{k}"] = (df_res["rank"] <= k).sum() / df_res.shape[0]
        print(f"Hits@{k}: {pr_results[f'h@{k}']:.4f}")
    total_results.append(
        [method, dataset_name] + list(pr_results.values()) + [time_taken]
    )
    print(f"\n")


df = pd.DataFrame(
    total_results,
    columns=["method", "dataset", "mrr", "h@1", "h@3", "h@10", "time_sec"],
)
df = df.sort_values(["dataset", "mrr"], ascending=False)
df.to_csv(path_to_save_res, index=False)
print(df.to_string())
