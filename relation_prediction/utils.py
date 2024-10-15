import os
import random
import time
from collections import defaultdict
from itertools import product
from typing import List, Union

import numba as nb
import numpy as np
import pandas as pd
import tqdm
from numba import types
from numba.typed import Dict
from scipy.sparse import csr_array
from sympy.ntheory import factorint, nextprime

import graphblas as gb


def load_data(
    path_to_folder: str, project_name: str, add_inverse_edges: str = "NO"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """
    Helper function that loads the data in pd.DataFrames and returns them.
    Args:
        path_to_folder (str): path to folder with train.txt, valid.txt, test.txt
        project_name (str): name of the project
        add_inverse_edges (str, optional):  Whether to add the inverse edges.
        Possible values "YES", "YES__INV", "NO". Defaults to "NO".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]: [description]
    """
    PROJECT_DETAILS = {
        "lc-neo4j": {"skiprows": 1, "sep": "\t"},
        "codex-s": {"skiprows": 0, "sep": "\t"},
        "WN18RR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10-DR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10": {"skiprows": 0, "sep": "\t"},
        "FB15k-237": {"skiprows": 0, "sep": "\t"},
        "NELL995": {"skiprows": 0, "sep": "\t"},
        "DDB14": {"skiprows": 0, "sep": "\t"},
    }

    df_train = pd.read_csv(
        os.path.join(path_to_folder, "train.txt"),
        sep=PROJECT_DETAILS[project_name]["sep"],
        header=None,
        dtype="str",
        skiprows=PROJECT_DETAILS[project_name]["skiprows"],
    )
    df_train.columns = ["head", "rel", "tail"]
    df_train_orig = df_train.copy()
    if "YES" in add_inverse_edges:
        print(f"Will add the inverse train edges as well..")
        df_train["rel"] = df_train["rel"].astype(str)
        df_train_inv = df_train.copy()
        df_train_inv["head"] = df_train["tail"]
        df_train_inv["tail"] = df_train["head"]
        if add_inverse_edges == "YES__INV":
            df_train_inv["rel"] = df_train["rel"] + "__INV"
        df_train = pd.concat([df_train, df_train_inv], ignore_index=True)
    if project_name in ["lc-neo4j"]:
        df_eval = None
        df_test = None
        already_seen_triples = set(df_train.to_records(index=False).tolist())
    else:
        try:
            df_eval = pd.read_csv(
                os.path.join(path_to_folder, "valid.txt"),
                sep=PROJECT_DETAILS[project_name]["sep"],
                header=None,
                dtype="str",
                skiprows=PROJECT_DETAILS[project_name]["skiprows"],
            )
            df_eval.columns = ["head", "rel", "tail"]
        except FileNotFoundError:
            print(
                f"No valid.txt found in {path_to_folder}... df_eval will contain the train data.."
            )
            df_eval = df_train.copy()
        df_test = pd.read_csv(
            os.path.join(path_to_folder, "test.txt"),
            sep=PROJECT_DETAILS[project_name]["sep"],
            header=None,
            dtype="str",
            skiprows=PROJECT_DETAILS[project_name]["skiprows"],
        )
        df_test.columns = ["head", "rel", "tail"]
        if "YAGO" in project_name:
            for cur_df in [df_train, df_eval, df_test]:
                for col in cur_df.columns:
                    cur_df[col] = cur_df[col]  # + "_YAGO"

        already_seen_triples = set(
            df_train.to_records(index=False).tolist()
            + df_eval.to_records(index=False).tolist()
        )
    print(f"Total: {len(already_seen_triples)} triples in train + eval!)")
    print(f"In train: {len(df_train)}")
    print(f"In valid: {len(df_eval)}")
    print(f"In test: {len(df_test)}")
    return df_train_orig, df_train, df_eval, df_test, already_seen_triples


def get_filtering_cache(df_train, df_eval):
    """
    Helper function that create a dict of the form:
    [(h,t)]-> [r1, ...] keeping an index of which (h,t) are
    connected through which relations from the train and eval
    triples.
    """
    cache_triples = defaultdict(list)
    all_triples = pd.concat((df_train, df_eval))
    for triple in tqdm.tqdm(all_triples.to_records()):
        # Adding h,t -> r
        cache_triples[(triple[1], triple[3])].append(triple[2])
    return cache_triples


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
    spacing_strategy: str = "step_10",
    eliminate_zeros: bool = False,
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
        Defaults to "step_10".
        break_with_sparsity_threshold (int, optional): The percentage of sparsity that is not accepted.
        If one of the k-hop PAMs has lower sparsity we break the calculations and do not include it
        in the returned matrices list.
        Defaults to "step_10"

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
    df_train["rel_mapped"] = df_train["rel"].map(rel2id)
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
        if print_:
            print(f"Will map to logs!")
        id2rel = {}
        for k, v in rel2id.items():
            rel2id[k] = np.log(v)
            id2rel[np.log(v)] = k

    df_train["rel_mapped"] = df_train["rel"].map(rel2id)

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

    if spacing_strategy == "constant":
        pam_1hop_lossy.data = np.ones_like(pam_1hop_lossy.data)

    if method == "plus_mapping":
        pam_1hop_lossy.data = np.array([dat * np.log(3) for dat in pam_1hop_lossy.data])

    A_gb = gb.io.from_scipy_sparse(pam_1hop_lossy)
    # Generate the PAM^k matrices
    pam_powers = [pam_1hop_lossy]
    pam_power_gb = [A_gb]
    broke_cause_of_sparsity = False
    base_logs = [3, 5, 7, 11, 13, 17, 19, 23, 27, 31, 37]
    for ii in range(1, max_order):
        if print_:
            print(f"Hop {ii + 1}")
        updated_power_gb = pam_power_gb[-1].mxm(A_gb, method).new()
        if eliminate_zeros:
            updated_power_gb.setdiag(0)
        updated_power = gb.io.to_scipy_sparse(updated_power_gb)
        # updated_power.sort_indices()
        # updated_power.eliminate_zeros()

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


def extend_paths(
    hop_k_paths: list[Union[tuple, int]], hop_1_vals: list[int]
) -> list[tuple]:
    """Helper function that extend a relational chain by appending each possible
    relation to each existing path. E.g.
    hop_k_paths = [(3, 3), (3, 5)]
    hop_1_vals = [3,5]
    extended_paths -> [(3,3,3), (3,5,3)]

    Args:
        hop_k_paths (list[Union[tuple, int]]): The existing relational chains.
        hop_1_vals (list[int]): The values of the 1-hop relation to append to each path.

    Returns:
        list[tuple]: The extended paths as a list (all possible) of tuples (paths)
    """
    extended_paths = []
    for comb in product(hop_k_paths, hop_1_vals):
        if isinstance(comb[0], tuple):
            res = tuple(list(comb[0]) + [comb[1]])
        else:
            res = tuple(comb)
        extended_paths.append(res)
    return extended_paths


def create_lattice_mappings(
    rel2id: dict[str, int], num_hops: int, use_log: bool, print_: bool = False
) -> dict[int, dict[str, dict]]:
    """Helper function that generates the possible |R|^num_hops lattice of mappings of paths to primes.
    E.g. {r1, r2, r3} -> {2,3,5}, {(r1,r1), (r1,r2), (r1,r3), ...} -> {2, 3, 5, ...}


    Args:
        rel2id (dict[str, int]): The original mapping from the relation name to a prime
        num_hops (int): The maximum number of hops.
        use_log (bool): Whether to map to logs. This is done for efficiency when aggregating a lot of values
        so we do not have integer overflowing after multiplying a lot of primes together
        print_ (bool, optional): Whether to print with tqdm. Defaults to False.

    Returns:
        dict[int, dict[str, dict]]: _description_
    """
    mappings = {
        0: {
            "path2prime": rel2id,
            "prime2path": dict(zip(rel2id.values(), rel2id.keys())),
        }
    }
    mappings[1] = {
        "path2prime": dict(
            zip([tuple([val]) for val in rel2id.values()], rel2id.values())
        ),
        "prime2path": dict(
            zip(rel2id.values(), [tuple([val]) for val in rel2id.values()])
        ),
    }
    hop_1_values = list(rel2id.values())
    if print_:
        iterator = tqdm.tqdm(range(2, num_hops + 1), total=num_hops - 1)
    else:
        iterator = range(2, num_hops + 1)
    for k in iterator:
        next_mapping = {}
        cur_prime = 3
        current_path_chains = (
            list(mappings[k - 1]["path2prime"].keys()) if k > 2 else hop_1_values
        )
        extended_paths = extend_paths(current_path_chains, hop_1_values)
        for ext_path in extended_paths:
            if use_log:
                value_to_use = np.log(cur_prime)
            else:
                value_to_use = cur_prime
            next_mapping[ext_path] = value_to_use
            cur_prime = nextprime(cur_prime)
        mappings[k] = {
            "path2prime": next_mapping,
            "prime2path": dict(zip(next_mapping.values(), next_mapping.keys())),
        }
    return mappings


@nb.jit(nopython=True)
def inner_direct(
    value_k_hop,
    value_1_hop,
    integer2path_k_hop,
    integer2paths_nb_1hop,
    path2prime_k_plus_1_hop,
):
    """This facilitates the lossless algorithm for a specific cell.


    Args:
        value_k_hop (int): The value of the cell at k-hop
        value_1_hop (int): The value of the final hop, coming from the 1-hop matrix
        integer2path_k_hop (dict[int, list]): The mapping of ints to k-hop paths
        e.g. for 2 hop {33->[(11,r2), (r1,r3)]}, with (r1,r2)->11 and (r1,r3)->3
        integer2paths_nb_1hop (dict[int, list]): The mapping of ints to 1-hop paths
        e.g. {35->[3, 5]} where r1 <-> 3, r2 <->5
        path2prime_k_plus_1_hop (dict[int, list]): The mapping for the resulting k+1 matrix.


    Returns:
        (product, extended_paths_k_plus_1): The resulting value and the extended paths
    """

    path_chains_at_k = integer2path_k_hop[value_k_hop]
    factor_last_hop = [prime for prime in integer2paths_nb_1hop[value_1_hop].flatten()]
    extended_paths_k_plus_1 = np.zeros(
        (len(path_chains_at_k) * len(factor_last_hop), len(path_chains_at_k[0]) + 1),
        dtype=np.int64,
    )
    product = 1
    path_k_i_counter = 0
    for path_chain in path_chains_at_k:
        for last_hop in factor_last_hop:
            res = np.append(path_chain, np.asarray(last_hop, dtype=np.int64))
            rest_str = "".join([str(item) for item in res])
            extended_paths_k_plus_1[path_k_i_counter, :] = res
            product = product * path2prime_k_plus_1_hop[rest_str]
            path_k_i_counter += 1

    return product, extended_paths_k_plus_1


@nb.jit(nopython=True)
def inner_log_direct(
    value_k_hop,
    value_1_hop,
    float2path_k_hop,
    float2paths_nb_1hop,
    path2prime_k_plus_1_hop,
):
    """The same as before but now we have logs of values instead of ints.

    Args:
        value_k_hop (int): The value of the cell at k-hop
        value_1_hop (int): The value of the final hop, coming from the 1-hop matrix
        float2path_k_hop (dict[float, list]): The mapping of floats to k-hop paths
        e.g. for 2 hop {33->[(11,r2), (r1,r3)]}, with (r1,r2)->11 and (r1,r3)->3
        float2paths_nb_1hop (dict[float, list]): The mapping of floats to 1-hop paths
        e.g. {35->[3, 5]} where r1 <-> 3, r2 <->5
        path2prime_k_plus_1_hop (dict[float, list]): The mapping for the resulting k+1 matrix.


    Returns:
        (product, extended_paths_k_plus_1): The resulting value and the extended paths
    """

    path_chains_at_k = float2path_k_hop[value_k_hop]
    factor_last_hop = [prime for prime in float2paths_nb_1hop[value_1_hop].flatten()]
    extended_paths_k_plus_1 = np.zeros(
        (len(path_chains_at_k) * len(factor_last_hop), len(path_chains_at_k[0]) + 1),
        dtype=np.int64,
    )
    product = 0
    path_k_i_counter = 0
    for path_chain in path_chains_at_k:
        for last_hop in factor_last_hop:
            res = np.append(path_chain, np.asarray(last_hop, dtype=np.int64))
            rest_str = "".join([str(item) for item in res])
            extended_paths_k_plus_1[path_k_i_counter, :] = res
            product += np.log(path2prime_k_plus_1_hop[rest_str])
            path_k_i_counter += 1
    return product, extended_paths_k_plus_1


@nb.jit(nopython=True)
def sparse_dot2(
    matrix_left_data,
    matrix_left_indices,
    matrix_left_indptr,
    left_n_rows,
    matrix_right_data,
    matrix_right_indices,
    matrix_right_indptr,
    number2paths_nb,
    number2paths_nb_1hop,
    path2prime_k_plus_1_hop_nb,
    semiring,
):
    """Sparse matrix multiplication matrix_left x matrix_right

    Both matrices must be in the CSR sparse format.
    Data, indices and indptr are the standard CSR representation where the column indices for row i
    are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in
    data[indptr[i]:indptr[i+1]].

    Args:
        matrix_left_data (numpy.array): Non-zero value of the sparse matrix.
        matrix_left_indices (numpy.array): Column positions of non-zero values.
        matrix_left_indptr (numpy.array): Array with the count of non-zero values per row.
        matrix_right_data (numpy.array): Non-zero value of the sparse matrix.
        matrix_right_indices (numpy.array): Column positions of non-zero values.
        matrix_right_indptr (numpy.array): Array with the count of non-zero values per row.

    Returns:
        numpy.array: 2D array with the result of the matrix multiplication.
    """

    rows, cols, values = [], [], []
    k_plus_1_value_to_map = Dict.empty(
        key_type=types.float64,
        value_type=types.int64[:, :],
    )
    value = 0
    for row_left in range(left_n_rows):
        for left_i in range(
            matrix_left_indptr[row_left], matrix_left_indptr[row_left + 1]
        ):
            col_left = matrix_left_indices[left_i]
            value_left = matrix_left_data[left_i]
            for right_i in range(
                matrix_right_indptr[col_left], matrix_right_indptr[col_left + 1]
            ):
                col_right = matrix_right_indices[right_i]
                value_right = matrix_right_data[right_i]
                if "lossless" in semiring:
                    if "log" in semiring:
                        value, extended_paths_k_plus_1 = inner_log_direct(
                            value_left,
                            value_right,
                            number2paths_nb,
                            number2paths_nb_1hop,
                            path2prime_k_plus_1_hop_nb,
                        )
                    else:
                        value, extended_paths_k_plus_1 = inner_direct(
                            value_left,
                            value_right,
                            number2paths_nb,
                            number2paths_nb_1hop,
                            path2prime_k_plus_1_hop_nb,
                        )
                k_plus_1_value_to_map[value] = extended_paths_k_plus_1

                rows.append(row_left)
                cols.append(col_right)
                values.append(value)
    return rows, cols, values, k_plus_1_value_to_map


def spmm(
    matrix_left: csr_array,
    matrix_right: csr_array,
    number2paths_nb,
    number2paths_nb_1hop,
    path2prime_k_plus_1_hop_nb,
    semiring,
) -> tuple[csr_array, dict]:
    """A wrapper function for sparse matrix multpiplication in numba.

    Args:
        matrix_left (csr_array): The left array.
        matrix_right (csr_array): The right array.
        number2paths_nb (_type_): The number (ints or floats) to paths mapping for k-hops paths.
        number2paths_nb_1hop (_type_): The number (ints or floats) to paths mapping for 1-hops paths.
        path2prime_k_plus_1_hop_nb (_type_): The mappting of paths to number for the resulting k+1 matrix
        semiring (_type_): The semiring under which to run this.

    Raises:
        NotImplementedError: If semiring is ["lossless", "plus_times", "lossless_log_plus"], we don't support it.

    Returns:
        _type_: The resulting k+1 sparse csr_array
    """

    if semiring not in ["lossless", "plus_times", "lossless_log_plus"]:
        raise NotImplementedError(f"{semiring} not implemented...")

    rows, cols, values, k_plus_1_value_to_map = sparse_dot2(
        matrix_left.data,
        matrix_left.indices,
        matrix_left.indptr,
        matrix_left.shape[0],
        matrix_right.data,
        matrix_right.indices,
        matrix_right.indptr,
        number2paths_nb,
        number2paths_nb_1hop,
        path2prime_k_plus_1_hop_nb,
        semiring=semiring,
    )

    dictionary_of_next_hop = {}
    k_plus_1_value_to_map_aggregated = {}
    for index, (row, col) in enumerate(zip(rows, cols)):
        value = values[index]
        # if "log" in semiring:
        #    value = np.log(value)
        if (row, col) not in dictionary_of_next_hop:
            dictionary_of_next_hop[(row, col)] = value
            k_plus_1_value_to_map_aggregated[value] = np.asarray(
                k_plus_1_value_to_map[value], np.int64
            )
        else:
            cur_value = dictionary_of_next_hop[(row, col)]
            if semiring == "plus_times" or semiring == "lossless_log_plus":
                dictionary_of_next_hop[(row, col)] = cur_value + value
            else:
                dictionary_of_next_hop[(row, col)] = cur_value * value
            k_plus_1_value_to_map_aggregated[dictionary_of_next_hop[(row, col)]] = (
                np.vstack(
                    (
                        k_plus_1_value_to_map_aggregated[cur_value],
                        np.asarray(k_plus_1_value_to_map[value], np.int64),
                    )
                )
            )
    indices = list(zip(*list(dictionary_of_next_hop.keys())))
    if "log" in semiring:
        type_ = np.float64
    else:
        type_ = np.int64

    return (
        csr_array(
            (list(dictionary_of_next_hop.values()), (indices[0], indices[1])),
            shape=(matrix_left.shape[0], matrix_right.shape[1]),
            dtype=type_,
        ),
        k_plus_1_value_to_map_aggregated,
    )


def create_lossless_khops(
    df_train_orig: pd.DataFrame, max_hop: int, print_: bool = False
) -> tuple[csr_array, list[csr_array], dict, dict, bool, dict]:
    """Wrapper facilitating the whole process.

    Args:
        df_train_orig (pd.DataFrame): The original graph in triples form.
        max_hop (int): The number of hops to calculate.
        print_ (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        tuple[csr_array, list[csr_array], dict, dict, bool, dict]: Exactly the same returns as in the lossy case +
        the final dict with the mappings prime2paths (and the inverse) over each layer.
    """
    (
        A_sparse,
        _,
        node2id,
        rel2id,
        broke_cause_of_sparsity,
    ) = create_pam_matrices(
        df_train_orig,
        max_order=2,
        method="plus_times",
        use_log=False,  ############################## HARDCODED FALSE  #########
        spacing_strategy="step_1",
    )

    if print_:
        print(f"Creating Mappings..")
    mappings = create_lattice_mappings(rel2id, max_hop, False)

    number2paths = {}
    for value, paths in mappings[1]["prime2path"].items():
        number2paths[value] = paths

    number2paths_nb = Dict.empty(
        key_type=types.float64,
        value_type=types.int64[:, :],
    )
    number2paths_nb_1hop = Dict.empty(
        key_type=types.float64,
        value_type=types.int64[:, :],
    )
    for k, v in number2paths.items():
        number2paths_nb[k] = np.asarray(v, dtype=np.int64).reshape(-1, 1)
        number2paths_nb_1hop[k] = number2paths_nb[k]

    for value in A_sparse.data:
        if value not in number2paths_nb_1hop:
            number2paths_nb[value] = np.asarray(
                factorint(value, multiple=True), dtype=np.int64
            ).reshape(-1, 1)
            number2paths_nb_1hop[value] = number2paths_nb[value]

    power_A = [A_sparse]

    type_ = types.float64
    for cur_hop_index in range(max_hop - 1):
        # Prepare mappings
        if print_:
            print(f"K is {cur_hop_index + 2}")
        path2prime_k_plus_1_hop = mappings[cur_hop_index + 2]["path2prime"]

        path2prime_k_plus_1_hop_nb = Dict.empty(
            key_type=types.string,
            value_type=type_,
        )

        for k, v in path2prime_k_plus_1_hop.items():
            path2prime_k_plus_1_hop_nb["".join([str(k_) for k_ in k])] = v

        updated_A, new_paths = spmm(
            power_A[-1],
            power_A[0],
            number2paths_nb,
            number2paths_nb_1hop,
            path2prime_k_plus_1_hop_nb,
            "lossless_log_plus",
        )

        mappings[cur_hop_index + 2] = {"path2prime": {}, "prime2path": {}}
        for float_, paths_arr in new_paths.items():
            number2paths_nb[float_] = np.asarray(paths_arr, dtype=np.int64)
            path_list = paths_arr.tolist()

            tupled_paths = tuple([tuple(sub) for sub in path_list])
            mappings[cur_hop_index + 2]["path2prime"][tupled_paths] = float_
            mappings[cur_hop_index + 2]["prime2path"][float_] = tupled_paths

        num_overflow = (updated_A.data < 0).sum()
        num_inf = np.isinf(updated_A.data).sum()
        updated_A.data[(updated_A.data < 0)] = 0
        updated_A.data[np.isinf(updated_A.data)] = 0
        updated_A.eliminate_zeros()
        if num_overflow > 0 or num_inf > 0:
            # pass
            # raise ArithmeticError(f"Created {num_overflow} overflows and {num_inf} infinities during calculations")

            print(
                f"Hop {cur_hop_index + 2}: Created {num_overflow} overflows and {num_inf} infinities during calculations"
            )

        sparsity = get_sparsity(updated_A)
        if print_:
            print(f"Sparsity {cur_hop_index + 2}-hop: {sparsity:.2f} %")

        power_A.append(updated_A)

    return A_sparse, power_A, node2id, rel2id, broke_cause_of_sparsity, mappings


if __name__ == "__main__":
    pass
