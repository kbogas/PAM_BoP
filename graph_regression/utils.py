import random
import time

import datasets
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import csr_array, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sympy.ntheory import nextprime

import graphblas as gb


def generate_pams_peptides(g, max_order: int = 6) -> list[np.ndarray]:
    """Per graph PAM generator

    Args:
        g (dgl.graph): DGL Graph.
        max_order (int): The max value k of PAMs to calculate.


    Returns:
        list[np.ndarray]: List of arrays corresponding to the non-zero values of each
        PAM matrix
    """
    source, dest = g.edges()
    source = source.cpu().numpy()
    dest = dest.cpu().numpy()
    # Map each edge of (head_atom_type, bond_type, tail_atom_type) to a specific value
    # that will act as the relation of this edge, encoding both the atom labels
    # and the bond between them
    mapped_triples = np.log(
        (
            torch.cat(
                (
                    g.ndata["feat"][g.edges()[0]],
                    g.edata["feat"],
                    g.ndata["feat"][g.edges()[1]],
                ),
                dim=1,
            )
        )
        .sum(-1)
        .cpu()
        .numpy()
    )
    df = pd.DataFrame({"head": source, "rel": mapped_triples, "tail": dest})
    pam_1hop_lossless, pam_powers, node2id, rel2id, broke_cause_of_sparsity = (
        create_pam_matrices(
            df,
            max_order=max_order,
        )
    )
    return [pam_power.data for pam_power in pam_powers]


def process_data_peptides(data, max_order):
    "Wrapper for partial function"

    (g, label) = data
    pam_powers = generate_pams_peptides(g, max_order)
    return pam_powers, label.cpu().numpy()


def generate_pams_zinc(g, max_order):
    # rels = torch.vstack(
    #     (g.ndata["feat"][g.edges()[0]], g.edata["feat"], g.ndata["feat"][g.edges()[1]])
    # ).T @ torch.tensor([1000, 100, 1])
    rels = torch.vstack(
        (g.ndata["feat"][g.edges()[0]], g.edata["feat"], g.ndata["feat"][g.edges()[1]])
    ).T
    rels = np.apply_along_axis(lambda x: "_".join(x), 1, rels.numpy().astype(str))
    source, dest = g.edges()
    source = source.cpu().numpy()
    dest = dest.cpu().numpy()
    df = pd.DataFrame({"head": source, "rel": rels, "tail": dest})
    pam_1hop_lossless, pam_powers, node2id, rel2id, broke_cause_of_sparsity = (
        create_pam_matrices(
            df,
            max_order=max_order,
            method="plus_plus",
            spacing_strategy="step_1",
            use_log=False,
        )
    )
    return [pam_power.data for pam_power in pam_powers]


def process_data_zinc(data, max_order):

    (g, label) = data
    pam_powers = generate_pams_zinc(g, max_order)
    return pam_powers, label.cpu().numpy()


class BoP_peptides(BaseEstimator, TransformerMixin):
    "Simple sklearn-based Tf-idf Vectorizer for the nnz BoP Values"

    def __init__(
        self,
        min_df=1,
        max_df=1.0,
        max_features=None,
    ):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vector = TfidfVectorizer(
            tokenizer=lambda x: x,
            token_pattern=None,
            lowercase=False,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )

        self.vectorizers = []
        self.unused_powers = []

    def fit(self, X, y=None):
        self.max_p = len(X[0])
        self.feature_names = []
        self.feature_names_level = []
        self.features_per_hop = []
        all_features = []
        for power in range(self.max_p):
            paths = X[:, power]
            cur_vector = clone(self.vector)

            tr_paths = cur_vector.fit_transform(paths)
            all_features.append(tr_paths)
            self.vectorizers.append(cur_vector)
            self.feature_names.extend(
                [
                    f"{feat_name}_{power+1}"
                    for feat_name in cur_vector.get_feature_names_out()
                ]
            )
            self.feature_names_level.extend(
                [power + 1] * len(cur_vector.get_feature_names_out())
            )
            self.features_per_hop.append(tr_paths.shape[1])
        return self

    def transform(self, X):
        all_features = []
        index_to_use = 0
        for power in range(self.max_p):
            if power in self.unused_powers:
                continue
            paths = X[:, power]
            cur_vector = self.vectorizers[index_to_use]
            tr_paths = cur_vector.transform(paths)
            all_features.append(tr_paths)
            index_to_use += 1
        return sp.hstack(all_features)

    def get_feature_names_out(self):
        return self.feature_names


def get_prime_map_from_rel(
    list_of_rels: list,
    starting_value: int = 1,
    spacing_strategy: str = "step_10",
    add_inverse_edges: bool = False,
) -> tuple[dict, dict]:
    """
    Helper function that given a list of relations returns the mappings to and from the
    prime numbers used.

    Different strategies to map the numbers are available.
    "step_X", increases the step between two prime numbers by adding X to the current prime
    "factor_X", increases the step between two prime numbers by multiplying the current prime with X
    "natural",increases by 1 starting from 2

    Args:
        list_of_rels (list): iterable, contains a list of the relations that need to be mapped.
        starting_value (int, optional): Starting value of the primes. Defaults to 1.
        spacing_strategy (str, optional):  Spacing strategy for the primes. Defaults to "step_1".
        add_inverse_edges (bool, optional):  Whether to create mapping for inverse edges. Defaults to False.

    Returns:
        rel2prime: dict, relation to prime dictionary e.g. {"rel1":2}.
        prime2rel: dict, prime to relation dictionary e.g. {2:"rel1"}.
    """
    # add inverse edges if needed
    if add_inverse_edges:
        list_of_rels = [str(relid) for relid in list_of_rels] + [
            str(relid) + "__INV" for relid in list_of_rels
        ]
    else:
        list_of_rels = [str(relid) for relid in list_of_rels]

    # Initialize dicts
    rel2prime = {}
    prime2rel = {}
    # Starting value for finding the next prime
    current_int = starting_value
    # Map each relation id to the next available prime according to the strategy used
    if spacing_strategy == "natural":
        c = 1
        for relid in list_of_rels:
            rel2prime[relid] = c
            prime2rel[c] = relid
            c += 1
    else:
        if spacing_strategy == "constant":
            spacing_strategy = "step_10"
        for relid in list_of_rels:
            cur_prime = int(nextprime(current_int))  # type: ignore
            rel2prime[relid] = cur_prime
            prime2rel[cur_prime] = relid
            if "step" in spacing_strategy:
                step = float(spacing_strategy.split("_")[1])
                current_int = cur_prime + step
            elif "factor" in spacing_strategy:
                factor = float(spacing_strategy.split("_")[1])
                current_int = cur_prime * factor
            else:
                raise NotImplementedError(
                    f"Spacing strategy : {spacing_strategy}  not understood!"
                )
    return rel2prime, prime2rel


def create_pam_matrices(
    df_train: pd.DataFrame,
    max_order: int = 5,
    use_log: bool = True,
    method: str = "plus_times",
    spacing_strategy: str = "step_100",
    break_with_sparsity_threshold: float = -1,
    print_: bool = False,
) -> tuple[csr_array, list[csr_array], dict, dict, bool]:
    """Helper function that creates the pam matrices.

    Args:
        df_train (pd.DataFrame): The triples in the form of a pd.DataFrame with columns
        (head, rel, tail).
        max_order (int, optional): The maximum order for the PAMs (i.e. the k-hops).
        Defaults to 5.
        use_log (bool, optional): Whether to use log of primes for numerical stability.
        Defaults to True.
        spacing_strategy (str, optional): he spacing strategy as mentioned in get_prime_map_from_rel.
        Defaults to "step_100".
        eliminate_zeros (bool, optional): Whether to eliminate zeros between matrix hops.
        Defaults to True.
        break_with_sparsity_threshold (int, optional): The percentage of sparsity that is not accepted.
        If one of the k-hop PAMs has lower sparsity we break the calculations and do not include it
        in the returned matrices list.

    Returns:
        tuple[csr_matrix, list[csr_matrix], dict, dict]: The first argument is the lossless 1-hop PAM with products.
        The second is a list of the lossy PAMs powers, the third argument is the node2id dictionary and
        the fourth argument is the relation to id dictionary.
    """

    unique_rels = sorted(list(df_train["rel"].unique()))
    unique_nodes = sorted(
        set(df_train["head"].values.tolist() + df_train["tail"].values.tolist())  # type: ignore
    )

    if print_:
        print(
            f"# of unique rels: {len(unique_rels)} \t | # of unique nodes: {len(unique_nodes)}"
        )

    node2id = {}
    id2node = {}
    for i, node in enumerate(unique_nodes):
        node2id[node] = i
        id2node[i] = node

    time_s = time.time()

    # Map the relations to primes
    rel2id, id2rel = get_prime_map_from_rel(
        unique_rels,
        starting_value=2,
        spacing_strategy=spacing_strategy,
    )

    # Create the adjacency matrix
    df_train["rel_mapped"] = df_train["rel"].astype(str).map(rel2id)
    df_train["head_mapped"] = df_train["head"].map(node2id)
    df_train["tail_mapped"] = df_train["tail"].map(node2id)

    aggregated_df_lossless = (
        df_train.groupby(["head_mapped", "tail_mapped"])["rel_mapped"]
        .aggregate("prod")
        .reset_index()
    )
    # if any(aggregated_df_lossless["rel_mapped"].values > np.iinfo(np.int64).max):
    #     raise OverflowError(
    #         f"You have overflowing due to large prime number products in 1-hop PAM. Please lower the spacing strategy (lowest is 'step_1'), current is {spacing_strategy}"
    #     )

    pam_1hop_lossless = csr_array(
        (
            aggregated_df_lossless["rel_mapped"],
            (
                aggregated_df_lossless["head_mapped"],
                aggregated_df_lossless["tail_mapped"],
            ),
        ),
        shape=(len(unique_nodes), len(unique_nodes)),
    )

    if use_log:
        id2rel = {}
        for k, v in rel2id.items():
            rel2id[k] = np.log(v)
            id2rel[np.log(v)] = k

    df_train["rel_mapped"] = df_train["rel"].astype(str).map(rel2id)

    aggregated_df = (
        df_train.groupby(["head_mapped", "tail_mapped"])["rel_mapped"]
        .aggregate("sum")
        .reset_index()
    )
    pam_1hop_lossy = csr_array(
        (
            aggregated_df["rel_mapped"],
            (aggregated_df["head_mapped"], aggregated_df["tail_mapped"]),
        ),
        shape=(len(unique_nodes), len(unique_nodes)),
        dtype=np.float64,
    )

    A_gb = gb.io.from_scipy_sparse(pam_1hop_lossy)
    # Generate the PAM^k matrices
    pam_powers = [pam_1hop_lossy]
    pam_power_gb = [A_gb]
    broke_cause_of_sparsity = False
    for ii in range(1, max_order):
        updated_power_gb = pam_power_gb[-1].mxm(A_gb, method).new()
        # updated_power_gb.setdiag(0)
        updated_power = gb.io.to_scipy_sparse(updated_power_gb)
        updated_power.sort_indices()
        updated_power.eliminate_zeros()

        sparsity = get_sparsity(updated_power)
        if print_:
            print(f"Sparsity {ii + 1}-hop: {sparsity:.2f} %")
        if sparsity < 100 * break_with_sparsity_threshold and ii > 1:
            if print_:
                print(
                    f"Stopped at {ii + 1} hops due to non-sparse matrix.. Current sparsity {sparsity:.2f} % < {break_with_sparsity_threshold}"
                )
            broke_cause_of_sparsity = True
            break
        pam_powers.append(updated_power)
        pam_power_gb.append(updated_power_gb)

    return pam_1hop_lossless, pam_powers, node2id, rel2id, broke_cause_of_sparsity


def set_all_seeds(seed: int = 0):
    """Fix random seeds
    Args:
        seed (int): Random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    return 1


def get_sparsity(A: csr_array) -> float:
    """Calculate sparsity % of scipy sparse matrix.
    Args:
        A (scipy.sparse): Scipy sparse matrix
    Returns:
        (float)): Sparsity as a float
    """

    return 100 * (1 - A.nnz / (A.shape[0] ** 2))


if __name__ == "__main__":
    pass
