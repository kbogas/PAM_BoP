import os
import random
import time
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sympy import nextprime

import graphblas as gb


def set_all_seeds(seed: int = 42):
    """Random seeding utility

    Args:
        seed (int, optional): The random seed number. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_data_nc(path_to_files: str):
    """
    Helper function to load data
    """

    project_name = path_to_files.split("/")[-2].split("_")[0]
    PROJECT_DETAILS = {
        "MUTAG": {"skiprows": 0, "sep": "\t"},
        "BGS": {"skiprows": 0, "sep": "\t"},
        "AM": {"skiprows": 0, "sep": "\t"},
        "AIFB": {"skiprows": 0, "sep": "\t"},
    }

    df_train = pd.read_csv(
        os.path.join(path_to_files, "graph.csv"),
        sep=PROJECT_DETAILS[project_name]["sep"],
        header=None,
        dtype=str,
        skiprows=PROJECT_DETAILS[project_name]["skiprows"],
    )
    df_train.columns = ["head", "rel", "tail"]

    df_nodes = pd.read_csv(
        os.path.join(path_to_files, "labels.csv"),
        sep=PROJECT_DETAILS[project_name]["sep"],
        header=None,
        dtype=str,
        skiprows=PROJECT_DETAILS[project_name]["skiprows"],
    )
    df_nodes.columns = ["id", "label", "split"]
    df_nodes["id"] = df_nodes["id"].astype(str)
    df_nodes["label"] = df_nodes["label"].astype(str)

    print(f"In train: {len(df_train)}")
    print(
        f"# labels in train: {len(df_nodes[df_nodes['split']=='train'])}/{len(df_nodes)}"
    )
    print(
        f"# labels in test: {len(df_nodes[df_nodes['split']=='test'])}/{len(df_nodes)}"
    )
    return df_train, df_nodes


def sum_of_logs(x: np.ndarray):
    """
    Helper function to calculate the sum of logs
    instead of the product of an array.

    Args:
        x (np.array): Array to aggregate

    Returns:
        float: The sum(log(x))
    """

    return np.sum(np.log(x))


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


def create_pam_matrices_with_start_and_end(
    df_train: pd.DataFrame,
    max_order: int = 5,
    method: Literal["plus_times", "plus_plus", "constant"] = "plus_times",
    spacing_strategy: str = "step_1000",
    eliminate_diagonal: bool = False,
    break_with_sparsity_threshold: float = 0.01,
    wanted_nodes_start: np.ndarray = np.array([]),
    wanted_nodes_end: np.ndarray = np.array([]),
) -> tuple[list[csr_array], dict, dict, bool]:
    """Helper function that creates the pam matrices.

    Args:
        df_train (pd.DataFrame): The triples in the form of a pd.DataFrame with columns
        (head, rel, tail).

        max_order (int, optional): The maximum order for the PAMs (i.e. the k-hops).
        Defaults to 5.


        method (Literal["plus_times", "plus_plus"], optional, "constant"):  Method of multiplication.
        - "plus_times": Generic matrix multiplication GrapBLAS multiplication.
        - "plus_plus": Matrix multiplication using a plus_plus semiring using GraphBLAS. Use with use_log=True recommended.
        - "constant": Matrix
        Defaults to "plus_times".

        spacing_strategy (str, optional): The spacing strategy as mentioned in get_prime_map_from_rel.
        Defaults to "step_10".

        eliminate_eliminate_diagonalzeros (bool, optional): Whether to zero-out the diagonal in each k-hop.
        (This essentially removes cyclic paths from being propagated).
        Defaults to False.

        break_with_sparsity_threshold (int, optional): The percentage of sparsity that is not accepted.
        If one of the k-hop PAMs has lower sparsity we break the calculations and do not include it
        in the returned matrices list.
        Defaults to -1.

        eliminate_diagonal (bool, optional): Whether to zero-out the diagonal in each k-hop.
        (This essentially removes cyclic paths from being propagated).
        Defaults to False.


        wanted_nodes_start (list, optional): Whether to generate matrix with size
        len(wanted_nodes_start) X len(nodes), keeping only the start nodes of choice. Defaults to [].

        wanted_nodes_end (list, optional): Whether to generate matrix with size
        len(nodes) X len(wanted_nodes_end), keeping only the final nodes of choice. Defaults to [].

    Returns:
        tuple[list[csr_array], dict, dict, bool]: The first argument is a list of the lossy PAMs powers.
        The second argument is the node2id dictionary.
        The third argument is the relation to id dictionary.
        The final argument is weather the matrix creation process was broken due to sparsity.

    """

    # Number of unique rels and nodes

    time_s = time.time()
    unique_rels = sorted(list(df_train["rel"].unique()))
    unique_nodes = sorted(
        set(df_train["head"].values.tolist() + df_train["tail"].values.tolist())  # type: ignore
    )

    node2id = {}
    id2node = {}
    for i, node in enumerate(unique_nodes):
        node2id[node] = int(i)
        id2node[int(i)] = node

    # Map the relations to primes
    rel2id, id2rel = get_prime_map_from_rel(
        unique_rels,
        starting_value=2,
        spacing_strategy=spacing_strategy,
    )

    # Map all node and rel values to the corresponding numerical ones.
    df_train["rel_mapped"] = df_train["rel"].astype(str).map(rel2id)
    df_train["head_mapped"] = df_train["head"].map(node2id)
    df_train["tail_mapped"] = df_train["tail"].map(node2id)

    id2rel = {}
    for k, v in rel2id.items():
        rel2id[k] = np.log(v)
        id2rel[np.log(v)] = k

    df_train["rel_mapped"] = df_train["rel"].astype(str).map(rel2id)

    # Create the lossy 1-hop with log-sum-sum
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

    # Map everything to 1. This is only for testing scenarios.
    if spacing_strategy == "constant":
        pam_1hop_lossy.data = np.ones_like(pam_1hop_lossy.data)

    # Create the GraphBLAS equivalent
    A_gb = gb.io.from_scipy_sparse(pam_1hop_lossy)
    A_start = A_gb
    if len(wanted_nodes_start) > 0:
        wanted_ids = np.array([node2id[node] for node in wanted_nodes_start])
        A_start = A_gb[wanted_ids, :].new()

    A_end = A_gb
    if len(wanted_nodes_end) > 0:
        wanted_ids = np.array([node2id[node] for node in wanted_nodes_end])
        A_end = A_gb[:, wanted_ids].new()

    pam_powers = [gb.io.to_scipy_sparse(A_start)]
    pam_power_gb = [A_start]
    broke_cause_of_sparsity = False
    for ii in range(1, max_order):
        time_hop_s = time.time()
        print(f"Hop {ii + 1}")
        if ii == max_order - 1:
            updated_power_gb = pam_power_gb[-1].mxm(A_end, method).new()
        else:
            updated_power_gb = pam_power_gb[-1].mxm(A_gb, method).new()
            if eliminate_diagonal:
                updated_power_gb.setdiag(0)

        updated_power = gb.io.to_scipy_sparse(updated_power_gb)
        updated_power.eliminate_zeros()

        sparsity = get_sparsity(updated_power)
        print(
            f"Sparsity {ii + 1}-hop: {sparsity:.2f} % (Time needed for this hop: {(time.time() - time_hop_s)/60:.2f} mins)"
        )
        if sparsity < 100 * break_with_sparsity_threshold and ii > 1:
            print(
                f"Stopped at {ii + 1} hops due to non-sparse matrix.. Current sparsity {sparsity:.2f} % < {break_with_sparsity_threshold}"
            )
            broke_cause_of_sparsity = True
            break
        pam_powers.append(updated_power)
        pam_power_gb.append(updated_power_gb)

    print(f"Total time taken for PAMs creation: {(time.time() - time_s)/60:.2f} mins..")

    del pam_power_gb

    return pam_powers, node2id, rel2id, broke_cause_of_sparsity


def get_sparsity(A: csr_array) -> float:
    """Calculate sparsity % of scipy sparse matrix.
    Args:
        A (scipy.sparse): Scipy sparse matrix
    Returns:
        (float)): Sparsity as a float
    """

    return 100 * (1 - A.nnz / (A.shape[0] * A.shape[1]))


if __name__ == "__main__":
    pass
