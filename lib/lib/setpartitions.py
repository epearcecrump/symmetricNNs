def convert_tuple_to_matrix_index(indices: tuple, dim: int, order_k: int) -> int:
    """
    Helper function that converts a tuple with indices
    (i_1, ..., i_k) where i_j is an element of {1, ..., dim}
    that indexes a tensor to the equivalent index that indexes
    the same tensor represented in matrix form.

    Returns
    -------
    int
    """
    assert(len(indices) == order_k)
    total = 0
    for index in indices:
        total += pow(dim, order_k - 1)*(index - 1)
        order_k -=1
    return total 

def convert_int_to_tuple(num: int, dim: int, order_k: int) -> list:
    """
    The reverse function of convert_tuple_to_matrix_index().
    Converts an integer back into a tuple of length order_k
    according to the base = dim

    Returns
    -------
    list
    """
    lst = []
    for i in range(order_k):
        if num < dim:
            lst.append(int(num+1))
            num = 0
        else:
            val = num % dim
            lst.append(int(val+1))
            sub = pow(dim, order_k - (order_k + i))*val
            num -= sub 
            num //= dim
    lst.reverse()
    return lst

def set_part_diag_basis_indices_list(lst: list, dim: int, diag=True) -> list:
    """
    Takes a pattern corresponding to a set partition
    and calculates all indices that match it 
    in the diagram basis if diag = True, 
    else in the orbit basis if diag = False,
    based on the value of dim (== n).
    
    Note that this returns a list of elements of the form [I,J], i.e both the row
    and the column tuples, that can be divided appropriately
    by the function mat_indices_list according to the orders k, l.

    Returns
    -------
    list
    """
    num_blocks = len(lst)

    # Total sum of the tensor power orders
    total_tensor_orders = sum(len(sublst) for sublst in lst)

    # Enumerate the blocks given in the input lst
    blocks = {i: list(lst[i-1]) for i in range(1, num_blocks+1)}
    
    indices_lst = []    
    for i in range(pow(dim, num_blocks)):
        block_labels = convert_int_to_tuple(i,dim,num_blocks)  

        # Labels must all be different if we're using the orbit basis 
        if not diag and len(block_labels) != len(set(block_labels)): 
            continue

        # vertex_labels consists of pairs of the form
        # diagram vertex label : value that appears in the [I,J] list for that vertex
        vertex_labels = {}
        for j in range(num_blocks):
            I_J_val = block_labels[j]
            block = blocks[j+1]
            for k in range(len(block)):
                vertex_labels[block[k]] = I_J_val
        list_tup = [vertex_labels[i] for i in range(1, total_tensor_orders+1)] 
        indices_lst.append(list_tup)
    
    return indices_lst

def mat_indices_list(indices_lst: list, dim: int, order_k: int, order_l: int) -> list:
    """
    Converts a list of tuple indices corresponding to a set partition, 
    for a given dim and orders k,l into their equivalent matrix index form.

    Returns
    -------
    list
    """

    assert(len(indices_lst[0]) == order_k + order_l)
   
    lst = []
    for indices in indices_lst:
        row_indices = indices[:order_l]
        col_indices = indices[order_l:]
        row_index = convert_tuple_to_matrix_index(row_indices, dim, order_l)
        col_index = convert_tuple_to_matrix_index(col_indices, dim, order_k)
        lst.append([row_index, col_index])
    return lst

