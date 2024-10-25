import multiprocessing
import time
from functools import partial

import numpy as np
import scipy
import torch
import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from utils import BoP_peptides, process_data_zinc, set_all_seeds

from dgl.data import ZINCDataset

max_order = 5
use_node_feats = False

random_state = 42
num_features_to_keep = 10000

set_all_seeds(random_state)

all_pams_full = []
all_labels_full = []
indices = {}
prev_index = 0
for split in ["train", "valid", "test"]:

    dataset = ZINCDataset(mode=split)
    print(
        f"Start mapping for {len(dataset)}-{split} graphs + PAM generation @ {max_order} hops"
    )

    time_s = time.time()
    pool = multiprocessing.Pool()

    # Process the dataset in parallel
    results = list(
        tqdm.tqdm(
            pool.imap(
                partial(
                    process_data_zinc,
                    max_order=max_order,
                ),
                dataset,
            ),
            total=len(dataset),
        )
    )
    print(
        f"PAM time took for {split} : {time.time()-time_s:.2f} seconds ({(time.time()-time_s)/60:.2f} mins)"
    )

    # Close the pool to release resources
    pool.close()
    pool.join()

    # Extract results
    all_pams, all_labels = zip(*results)

    all_pams = np.array(all_pams, dtype=object)
    all_labels = np.vstack(all_labels)

    if split == "train":
        tr = BoP_peptides(
            min_df=2, max_df=len(all_pams) - 1, max_features=num_features_to_keep
        )
        all_pams_tr = tr.fit_transform(all_pams)  # .toarray()
    else:
        all_pams_tr = tr.transform(all_pams)

    if use_node_feats:
        # node_feats = []
        # edge_feats = []
        # for g, _ in dataset:
        #     node_feats.append(g.ndata["feat"].cpu().numpy().sum(axis=0))
        #     edge_feats.append(g.edata["feat"].cpu().numpy().sum(axis=0))
        # node_feats = np.vstack(node_feats)
        # edge_feats = np.vstack(edge_feats)
        # node_feats = np.hstack((node_feats, edge_feats))

        # if split == "train":
        #     sc = MinMaxScaler()
        #     sc.fit_transform(node_feats)
        # node_feats = sc.transform(node_feats)

        # node_feats = csr_matrix(node_feats)
        # all_pams_tr = sparse_hstack((all_pams_tr, node_feats)).tocsc()
        # print(f"BoP + Nodes (Samples X Feats): {all_pams_tr.shape}")

        node_feats = []
        for g, _ in dataset:
            node_counts = g.ndata["feat"].bincount()
            node_hist = (
                torch.nn.functional.pad(
                    node_counts, (0, dataset.num_atom_types - node_counts.shape[0])
                )
                / g.num_nodes()
            )
            edge_counts = g.edata["feat"].bincount()
            edge_hist = (
                torch.nn.functional.pad(
                    edge_counts, (0, dataset.num_atom_types - edge_counts.shape[0])
                )
                / g.num_edges()
            )
            node_feats.append(torch.cat((node_hist, edge_hist)).cpu().numpy())

        node_feats = np.vstack(node_feats)
        node_feats = csr_matrix(node_feats)
        print("NF", node_feats.shape)
        all_pams_tr = sparse_hstack((all_pams_tr, node_feats)).tocsc()
        if split == "train":
            sc = MaxAbsScaler()
            all_pams_tr = sc.fit_transform(all_pams_tr)  # .toarray()
        else:
            all_pams_tr = sc.transform(all_pams_tr)
        print("BoP + Nodes", all_pams_tr.shape)

    all_pams_full.append(all_pams_tr)
    all_labels_full.append(all_labels)
    indices[split] = np.arange(prev_index, prev_index + all_pams_tr.shape[0])
    prev_index = prev_index + all_pams_tr.shape[0]


all_pams_tr = scipy.sparse.vstack(all_pams_full)
all_labels = np.concatenate(all_labels_full)


print("LABELS", all_labels.shape)
print("BoP", all_pams_tr.shape)


train_graphs = all_pams_tr[indices["train"]]
val_graphs = all_pams_tr[indices["valid"]]
test_graphs = all_pams_tr[indices["test"]]


y_train = all_labels[indices["train"]]
y_valid = all_labels[indices["valid"]]
y_test = all_labels[indices["test"]]


from catboost import CatBoostRegressor, Pool
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression

# from sklearn.linear_model import LinearRegression, MultiTaskLasso
# from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

# time_s = time.time()
# print("Feature selection...")

# # # Keep only 10000 paths
# print("val", val_graphs.shape)
# fs = SelectFromModel(
#     DecisionTreeRegressor(random_state=random_state), max_features=num_features_to_keep
# )
# # fs = SelectKBest(f_regression, k=num_features_to_keep)
# train_graphs = fs.fit_transform(train_graphs, y_train.ravel())
# val_graphs = fs.transform(val_graphs)
# # val_graphs = fs.fit_transform(val_graphs, y_valid.ravel())
# # train_graphs = fs.transform(train_graphs)

# test_graphs = fs.transform(test_graphs)

# # # print("val", val_graphs.shape)
# print(train_graphs.shape)


# print(
#     f"Feature selection took : {time.time()-time_s:.2f} seconds ({(time.time()-time_s)/60:.2f} mins)"
# )
# exit()


#


train_pool = Pool(train_graphs, y_train)
valid_pool = Pool(val_graphs, y_valid)
test_pool = Pool(test_graphs, y_test)


max_num_iter = 10000
num_early_stopping = 100  # int(max_num_iter*0.05)
model1 = CatBoostRegressor(
    iterations=max_num_iter,
    learning_rate=0.2,
    loss_function="MAE",
    random_seed=0,
    early_stopping_rounds=num_early_stopping,
    verbose=True,
    # thread_count=-1,
    task_type="GPU",
    devices="0",
)
model1.fit(X=train_graphs, y=y_train, eval_set=valid_pool, use_best_model=True)

y_train_pred = model1.predict(train_pool)
y_pred = model1.predict(test_pool)

# model1 = LinearSVR(C=1, max_iter=5000, random_state=42, dual="auto")
# model1.fit(train_graphs, y_train)

# y_train_pred = model1.predict(train_graphs)
# y_pred = model1.predict(test_graphs)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_pred)


print(f"Features: {train_graphs.shape[1]}")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print()
