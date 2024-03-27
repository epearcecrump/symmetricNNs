import more_itertools

from .setpartitions import *

def set_partitions(order_k: int, order_l: int, max_blocks: int) -> list:
    """
    Calculates a list of all set partitions having at most max_blocks blocks.
    Here, we assume that the set partition corresponds to a set partition diagram
    having order_k nodes at the bottom and order_l nodes at the top of the diagram.

    Returns
    -------
    list
    """

    # We first make sure that the maximum number of blocks is 
    # at most the sum of the orders of the tensor powers.
    total_order = order_k + order_l
    if max_blocks > total_order:
        max_blocks = total_order

    collection = tuple(i for i in range(1, order_k + order_l + 1))

    set_part = []
    for k in range(1, max_blocks + 1):
        set_part_k_blocks = more_itertools.set_partitions(collection,k)
        for part in set_part_k_blocks:
            set_part.append(part)

    return set_part

def set_partition_weight_matrices_by_indices(
        dim: int, order_k: int, order_l: int, diag=True
    ) -> list:
    """
    Returns a list consisting of lists of indices, where each list of indices 
    corresponds to where a partition spanning matrix is non-zero,
    and the number of set partitions that appeared in the calculation.

    To use the diagram basis, set diag = True.
    Otherwise, to use the orbit basis, set diag = False. 
    """
    lst = []
    set_parts = set_partitions(order_k, order_l, dim)
    for set_part in set_parts:
        part_indices = set_part_diag_basis_indices_list(set_part, dim, diag)
        mat_indices = mat_indices_list(part_indices, dim, order_k, order_l)
        lst.append(mat_indices)
    return lst
