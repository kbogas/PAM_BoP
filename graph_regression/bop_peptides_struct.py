import multiprocessing
import time
from functools import partial

import numpy as np
import tqdm
from catboost import CatBoostRegressor, Pool
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from utils import BoP_peptides, process_data_peptides, set_all_seeds

from dgl.data.lrgb import PeptidesStructuralDataset

############################## USER SELECTION #######################################

# k-hop order for PAMs
max_order = 6
# number of BoP features to keep
num_features_to_keep = 10000


device = "GPU"  # use "GPU" if available, else "CPU"

random_state = 42
max_num_iter = 10000
num_early_stopping = 100


########################### END USER SELECTION #######################################

# set random states
set_all_seeds(random_state)

# load dataset from dgl
dataset = PeptidesStructuralDataset()

print(
    f"Start mapping for {len(dataset)} graphs + PAM generation @ {max_order} hops..\n"
)


time_s = time.time()
pool = multiprocessing.Pool()

# Generate PAMs in parallel
results = list(
    tqdm.tqdm(
        pool.imap(partial(process_data_peptides, max_order=max_order), dataset),
        total=len(dataset),
    )
)
print(
    f"PAM time taken : {time.time()-time_s:.2f} seconds ({(time.time()-time_s)/60:.2f} mins)"
)

# Close the pool to release resources
pool.close()
pool.join()


time_s = time.time()
print(f"Generating BoP features..\n")
# Get pams and labels
all_pams, all_labels = zip(*results)

all_pams = np.array(all_pams, dtype=object)
all_labels = np.vstack(all_labels)

# Generate BoP features
tr = BoP_peptides(min_df=2, max_df=1.0)
all_pams_tr = tr.fit_transform(all_pams)  # .toarray()


print(
    f"BoP time took : {time.time()-time_s:.2f} seconds ({(time.time()-time_s)/60:.2f} mins)"
)


node_feats = []
edge_feats = []
for g, _ in dataset:
    node_feats.append(g.ndata["feat"].cpu().numpy().sum(axis=0))
    edge_feats.append(g.edata["feat"].cpu().numpy().sum(axis=0))
node_feats = np.vstack(node_feats)
edge_feats = np.vstack(edge_feats)
node_feats = np.hstack((node_feats, edge_feats))

node_feats = MinMaxScaler().fit_transform(node_feats)

node_feats = csr_matrix(node_feats)
all_pams_tr = sparse_hstack((all_pams_tr, node_feats)).tocsc()
print(f"BoP + Nodes (Samples X Feats): {all_pams_tr.shape}")


# Split the data
train_graphs = all_pams_tr[dataset.get_idx_split()["train"].cpu().numpy()]
val_graphs = all_pams_tr[dataset.get_idx_split()["val"].cpu().numpy()]
test_graphs = all_pams_tr[dataset.get_idx_split()["test"].cpu().numpy()]


y_train = all_labels[dataset.get_idx_split()["train"].cpu().numpy()]
y_valid = all_labels[dataset.get_idx_split()["val"].cpu().numpy()]
y_test = all_labels[dataset.get_idx_split()["test"].cpu().numpy()]


time_s = time.time()
print("Feature selection...")

# Keep only 10000 paths
fs = SelectFromModel(
    DecisionTreeRegressor(random_state=random_state), max_features=num_features_to_keep
)
val_graphs = fs.fit_transform(val_graphs, y_valid)
train_graphs = fs.transform(train_graphs)
test_graphs = fs.transform(test_graphs)


print(
    f"Feature selection took : {time.time()-time_s:.2f} seconds ({(time.time()-time_s)/60:.2f} mins)"
)


time_s = time.time()
print("Training classifier...")

train_pool = Pool(train_graphs, y_train)
valid_pool = Pool(val_graphs, y_valid)
test_pool = Pool(test_graphs, y_test)


model = CatBoostRegressor(
    iterations=max_num_iter,
    loss_function="MultiRMSE",
    eval_metric="MultiRMSE",
    random_seed=random_state,
    early_stopping_rounds=num_early_stopping,
    verbose=1,
    task_type=device,
    devices="0",
    boosting_type="Plain",
    # thread_count=-1,
)
model.fit(X=train_graphs, y=y_train, eval_set=valid_pool, use_best_model=True)


print(
    f"Training took : {time.time()-time_s:.2f} seconds ({(time.time()-time_s)/60:.2f} mins)"
)

y_train_pred = model.predict(train_pool)
y_pred = model.predict(test_pool)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_pred)


print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print()
