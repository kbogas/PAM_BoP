import time
from collections import Counter

import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import csr_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from utils import (
    create_lossless_khops,
    create_pam_matrices,
    get_filtering_cache,
    load_data,
    set_all_seeds,
)

# For reproducibility
set_all_seeds(42)


projects = {
    # "DDB14_plus_times": {
    #     "project_name": "DDB14",
    #     "path_to_files": "./data/DDB14/",
    #     "add_inverse_edges": "NO",
    #     "method": "plus_times",
    #     "spacing_strategy": "step_100",
    #     "ranking_clf": "knn",
    #     "eliminate_zeros": True,
    #     "max_order": 5,
    #     "sim_pairs": 20,
    #     "selection_strategy": "most_common",
    # },
    # "WN18RR_plus_times": {
    #     "project_name": "WN18RR",
    #     "path_to_files": "./data/WN18RR/",
    #     "add_inverse_edges": "YES",
    #     "method": "plus_times",
    #     "spacing_strategy": "step_100",
    #     "ranking_clf": "knn",
    #     "eliminate_zeros": True,
    #     "add_node_features": True,
    #     "max_order": 5,
    #     "sim_pairs": 100,
    #     "selection_strategy": "most_common",
    # },
    "NELL995_plus_times": {
        "project_name": "NELL995",
        "path_to_files": "./data/NELL995/",
        "add_inverse_edges": "NO",
        "method": "plus_times",
        "max_order": 4,
        "sim_pairs": 20,
        "spacing_strategy": "step_100",
        "ranking_clf": "knn",
        "selection_strategy": "most_common",
    },
    # "DDB14_lossless": {
    #     "project_name": "DDB14",
    #     "path_to_files": "../../GIT/Prime_Adj/data/DDB14/",
    #     "add_inverse_edges": "NO",
    #     "method": "lossless",
    #     "spacing_strategy": "step_100",
    #     "ranking_clf": "knn",
    #     "eliminate_zeros": True,
    #     "max_order": 4,
    #     "sim_pairs": 50,
    #     "selection_strategy": "most_common",
    # },
    # "WN18RR_legacy": {
    #     "project_name": "WN18RR",
    #     "path_to_files": "../../GIT/Prime_Adj/data/WN18RR",
    #     "method": "plus_times",
    #     "add_inverse_edges": "YES",
    #     "max_order": 6,
    #     "sim_pairs": 40,
    #     "selection_strategy": "most_common",
    # },
    # "WN18RR_lossless": {
    #     "project_name": "DDB14",
    #     "path_to_files": "/home/kbougatiotis/GIT/Prime_Adj/data/WN18RR/",
    #     "add_inverse_edges": "YES",
    #     "method": "lossless",
    #     "spacing_strategy": "step_100",
    #     "ranking_clf": "knn",
    #     "eliminate_zeros": True,
    #     "add_node_features": True,
    #     "max_order": 2,
    #     "sim_pairs": 100,
    #     "selection_strategy": "most_common",
    # },
    # "WN18RR_plus_times_nozeros_factor10": {
    #     "project_name": "WN18RR",
    #     "path_to_files": "../../GIT/Prime_Adj/data/WN18RR/",
    #     "add_inverse_edges": "YES",
    #     "method": "plus_times",
    #     "spacing_strategy": "factor_10",
    #     "eliminate_zeros": True,
    #     "max_order": 5,
    #     "sim_pairs": 20,
    #     "selection_strategy": "most_common",
    # },
    # "NELL995_legacy": {
    #     "project_name": "NELL995",
    #     "path_to_files": "../../GIT/Prime_Adj/data/NELL995/",
    #     "add_inverse_edges": "NO",
    #     "method": "plus_times",
    #     "max_order": 4,
    #     "sim_pairs": 100,
    #     "selection_strategy": "tf-idf",
    # },
    # "NELL995_lossless": {
    #     "project_name": "NELL995",
    #     "path_to_files": "../../GIT/Prime_Adj/data/NELL995/",
    #     "add_inverse_edges": "NO",
    #     "method": "lossless",
    #     "max_order": 3,
    #     "sim_pairs": 50,
    #     "spacing_strategy": "step_100",
    #     "ranking_clf": "knn",
    #     "sim_pairs": 100,
    #     "selection_strategy": "most_common",
    # },
    # "NELL995_plus_plus": {
    #     "project_name": "NELL995",
    #     "path_to_files": "../../GIT/Prime_Adj/data/NELL995/",
    #     "add_inverse_edges": "NO",
    #     "method": "plus_plus",
    #     "max_order": 3,
    #     "sim_pairs": 50,
    #     "selection_strategy": "most_common",
    # },
}


total_results = []
for run_name, project_settings in projects.items():
    print(run_name)

    project_name = project_settings["project_name"]
    path_to_files = project_settings["path_to_files"]
    method = project_settings["method"]
    add_inverse_edges = project_settings["add_inverse_edges"]
    max_order = project_settings["max_order"]
    sim_pairs = project_settings["sim_pairs"]
    selection_strategy = project_settings["selection_strategy"]
    try:
        spacing_strategy = project_settings["spacing_strategy"]
    except KeyError:
        spacing_strategy = "step_1"
    try:
        use_log = project_settings["use_log"]
    except KeyError:
        use_log = True

    try:
        eliminate_zeros = project_settings["eliminate_zeros"]
    except KeyError:
        eliminate_zeros = False

    try:
        ranking_clf = project_settings["ranking_clf"]
    except KeyError:
        ranking_clf = "knn"

    try:
        add_node_features = project_settings["add_node_features"]
    except KeyError:
        add_node_features = True

    # Loading the data
    time_s = time.time()
    df_train_orig, df_train, df_eval, df_test, already_seen_triples = load_data(
        path_to_files, project_name, add_inverse_edges=add_inverse_edges
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

    if method == "plus_plus" or method == "plus_times" or method == "plus_mapping":
        (
            pam_1hop_lossless,
            power_A,
            node2id,
            rel2id,
            broke_cause_of_sparsity,
        ) = create_pam_matrices(
            df_train_orig,
            max_order=max_order,
            method=method,
            use_log=use_log,
            eliminate_zeros=eliminate_zeros,
            spacing_strategy=spacing_strategy,
            break_with_sparsity_threshold=0.8,
        )
        max_order = len(power_A)

        id2rel = {v: k for k, v in rel2id.items()}
    elif method == "preloaded":
        (
            pam_1hop_lossless,
            power_A,
            node2id,
            rel2id,
            broke_cause_of_sparsity,
        ) = create_pam_matrices(
            df_train_orig,
            max_order=1,
            method="legacy",
            use_log=use_log,
            eliminate_zeros=eliminate_zeros,
            spacing_strategy=spacing_strategy,
            break_with_sparsity_threshold=0.8,
        )
        max_order = len(power_A)

        id2rel = {v: k for k, v in rel2id.items()}
        with np.load("./wn18rr_pam_lossless_2.npz", allow_pickle=True) as data:
            power_A = data["arr_0"]
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
    print(f" Length of hops: {len(power_A)}")

    # Map the initial data to node indices and relation primes
    df_train_mapped = df_train.copy()
    df_train_mapped["rel"] = df_train["rel"].map(rel2id)
    df_train_mapped["head"] = df_train["head"].map(node2id).astype(int)
    df_train_mapped["tail"] = df_train["tail"].map(node2id).astype(int)
    for i, row in df_train_mapped.iterrows():
        print(row)
        break
    print(f"Length before dropping nan", len(df_train_mapped))
    df_train_mapped.dropna(inplace=True)
    print(f"Length after dropping nan", len(df_train_mapped))
    df_train_mapped = df_train_mapped

    # Create the features of the paths for the original train pairs
    features_ij, labels_ij = {}, {}
    if add_inverse_edges == "YES":
        true_train = df_train_mapped.iloc[: df_train_mapped.shape[0] // 2]
    elif add_inverse_edges == "NO":
        true_train = df_train_mapped
    elif add_inverse_edges == "YES__INV":
        true_train = df_train_mapped[
            df_train_mapped["rel"].isin(list(rel2id.values())[: int(len(rel2id) / 2)])
        ]

    else:
        raise KeyError(f"{add_inverse_edges} not understood..")

    true_train["head"] = true_train["head"].astype(int)
    true_train["tail"] = true_train["tail"].astype(int)

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
        # exit()

    if method == "lossless":
        node_feats = np.empty((len(unique_nodes), 2 * unq_path_total))
    else:
        node_feats = np.zeros((len(unique_nodes), 2 * max_order))
    for i, row in true_train.iterrows():
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
                    from sympy.ntheory import factorint

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

    # Check unique (i,j) pairs are as expected
    assert len(features_ij) == true_train.groupby(["head", "tail"]).nunique().shape[0]

    # Create the feature vectors for the nodes
    # node_feats = []
    # for cur in power_A:
    #     cp = cur.copy()
    #     outgoing = cp.sum(axis=1).reshape(
    #         -1,
    #     )
    #     incoming = cp.sum(axis=0).reshape(
    #         -1,
    #     )
    #     node_feats.append(outgoing)
    #     node_feats.append(incoming)
    # node_feats = np.array(node_feats).T

    # node_feats = csr_array(node_feats)
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
        if add_node_features:
            cur_features_nodes = np.hstack(
                (
                    node_feats[ij[0], :].reshape(1, -1),
                    node_feats[ij[1], :].reshape(1, -1),
                )
            )
            cur_features = np.hstack((cur_features_pair, cur_features_nodes))
        else:
            cur_features = cur_features_pair
        X_train.append(cur_features.reshape(-1))
        y_train.append(list(labels_ij[ij]))
        train_pairs.append(ij)

    assert 0 == (np.array(X_train).sum(axis=1) == 0).sum()
    assert len(X_train) == len(features_ij)

    X_train = np.array(X_train)
    if method == "lossless":
        X_train_ohe = X_train
        # from sklearn.feature_selection import SelectKBest, chi2
        # from sklearn.preprocessing import MultiLabelBinarizer
        # y_train_mb = MultiLabelBinarizer().fit_transform(y_train)
        # feature_scores = []
        # num_to_keep = 500
        # for label_index in range(y_train_mb.shape[1]):
        #     cur_y = y_train_mb[:, label_index]
        #     ohe = SelectKBest(chi2, k='all')
        #     ohe.fit_transform(X_train, cur_y)
        #     feature_scores.append(list(ohe.scores_))
        # # Labels X Feats
        # feature_scores = np.array(feature_scores)
        # print(feature_scores.shape)
        # # Feats
        # feature_scores_collapsed = np.max(feature_scores, axis=0)
        # print('collapsed', feature_scores_collapsed.shape)
        # indices_to_keep = np.argsort(feature_scores_collapsed)[::-1][:num_to_keep]
        # #
        # indices_to_keep = set()
        # label_freq = y_train_mb.sum(axis=0) / y_train_mb.sum()
        # print('label freq', label_freq.shape)
        # nums_per_label = [int(np.round(freq*num_to_keep)) for freq in label_freq]
        # print(f'Will keep: {sum(nums_per_label)} broken in {[nums_per_label]}')
        # for label_index in np.argsort(nums_per_label)[::-1]:
        #     sorted_indices = np.argsort(feature_scores[label_index, :].flatten())[::-1]
        #     leftover_indices = [index for index in sorted_indices if index not in indices_to_keep]
        #     cur_indices = leftover_indices[:nums_per_label[label_index]]
        #     indices_to_keep = indices_to_keep.union(cur_indices)
        # indices_to_keep = np.array(list(indices_to_keep))
        # print(f'Will keep {indices_to_keep}')
        # X_train_ohe = X_train[:, indices_to_keep]
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
    for i, row in df_test_mapped.iterrows():
        print(row)
        break
    print(f"Length before dropping nan", len(df_test_mapped))
    df_test_mapped.dropna(inplace=True)
    print(f"Length after dropping nan", len(df_test_mapped))

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
                    from sympy.ntheory import factorint

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
                        from sympy.ntheory import factorint

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
            if add_node_features:
                cur_features_nodes = np.hstack(
                    (
                        node_feats[row["ij"][0], :].reshape(1, -1),
                        node_feats[row["ij"][1], :].reshape(1, -1),
                    )
                )
                cur_features = np.hstack(
                    (cur_dict.reshape(1, -1), cur_features_nodes.reshape(1, -1))
                )
            else:
                cur_features = cur_dict
        X_test.append(cur_features.reshape(-1))
        y_test.append([row["rel"]])
    X_test = np.array(X_test)

    if method == "lossless":
        X_test_ohe = X_test
        # X_test_ohe = X_test[:, indices_to_keep]
        X_test_ohe = csr_array(X_test_ohe)
    else:
        X_test_ohe = ohe.transform(X_test)  # X_test  #

    # if method == "lossless":
    #     from sklearn.linear_model import Lasso
    #     from sklearn.multioutput import MultiOutputClassifier
    #     from sklearn.preprocessing import MultiLabelBinarizer
    #     from sklearn.tree import DecisionTreeClassifier

    #     mlb = MultiLabelBinarizer()
    #     y_train_mlb = mlb.fit_transform(y_train)
    #     clf = DecisionTreeClassifier(random_state=42)
    #     clf.fit(X_train_ohe, y_train_mlb)
    #     columns_to_keep = clf.feature_importances_.argsort()[::-1][:200]

    #     # clf = MultiOutputClassifier(Lasso(alpha=0.1))
    #     # clf.fit(X_train_ohe, y_train_mlb)
    #     # feature_importances = [est.coef_ for est in clf.estimators_]
    #     # columns_to_keep = (
    #     #     np.vstack(feature_importances).mean(axis=1).argsort()[::-1][:200]
    #     # )

    #     X_train_ohe = X_train_ohe[:, columns_to_keep]
    #     X_test_ohe = X_test_ohe[:, columns_to_keep]

    print(f"Extracted features for the test pairs {X_test_ohe.shape}...\n")

    print(f"Calculating distances..")

    # Calculate distances between train and test
    if "svm" in ranking_clf:
        import scipy
        from sklearn.svm import LinearSVC

        distances = []
        stacked = scipy.sparse.vstack((X_test_ohe, X_train_ohe))
        for i_test in tqdm.tqdm(range(X_test_ohe.shape[0]), total=X_test_ohe.shape[0]):
            # create the "Dataset"
            mask = np.zeros(stacked.shape[0])
            mask[i_test] = 1
            mask[X_test_ohe.shape[0] :] = 1
            cur_data = stacked[mask.astype(bool)]

            if "subsample" in ranking_clf:
                rng = np.random.default_rng()
                indices_to_keep = [0] + (
                    1 + rng.choice(cur_data.shape[0] - 1, size=100, replace=False)
                ).tolist()
                cur_data_train = cur_data[indices_to_keep, :]
            else:
                cur_data_train = cur_data

            y = np.zeros(cur_data_train.shape[0])
            y[0] = 1  # we have a single positive example, mark it as such
            # train our (Exemplar) SVM
            # docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

            clf = LinearSVC(
                class_weight="balanced",
                penalty="l1",
                verbose=False,
                max_iter=10000,
                tol=1e-6,
                C=0.5,
            )
            clf.fit(cur_data_train, y)  # train

            # infer on whatever data you wish, e.g. the original data
            cur_distances = -clf.decision_function(cur_data)
            distances.append(cur_distances)
        distances = np.array(distances)
    else:
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
    test_index = 0
    train_pairs = np.array(train_pairs)

    # Calculate the IDF of the relations on the train set
    idf = np.log(
        (len(df_train) / df_train["rel"].value_counts(ascending=True))
    ).to_dict()

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

        # Rank them
        # Either by calculating their tf-idf weights (largest tf-idf weights on top)
        if selection_strategy == "tf-idf":
            tf = dict(Counter(poss_labels))
            for key, val in tf.items():
                tf[key] = tf[key] / len(poss_labels)
            tfidf = {}
            for rel_id, rel_freq in tf.items():
                tfidf[rel_id] = rel_freq * idf[id2rel[rel_id]]
            sorted_tfidf = dict(sorted(tfidf.items(), key=lambda item: item[1])[::-1])
            pred_labels = list(sorted_tfidf.keys())
            proba_labels = list(sorted_tfidf.values())
        # Or rank them by their term frequency (most frequent on top)
        elif selection_strategy == "most_common":
            predictions = [
                (label, count)
                for (label, count) in (Counter(poss_labels).most_common())[:sim_pairs]
            ]
            pred_labels = [pred[0] for pred in predictions]
            proba_labels = [pred[1] for pred in predictions]
        else:
            raise KeyError(f"{selection_strategy} not understood")

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
        [run_name, project_name] + list(pr_results.values()) + [time_taken]
    )
    print(f"\n")


df = pd.DataFrame(
    total_results, columns=["run", "dataset", "mrr", "h@1", "h@3", "h@10", "time_sec"]
)
print(df.sort_values(["dataset", "mrr"], ascending=False).to_string())


# Using 2 hops in plus plus vs 3 hops in legacy. All other the same
#                run dataset       mrr       h@1       h@3      h@10
# 0  DDB14_plus_plus   DDB14  0.939602  0.900309  0.978877  0.983514
# 1     DDB14_legacy   DDB14  0.914965  0.868109  0.961360  0.962648
