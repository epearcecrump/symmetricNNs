# Example of weight matrices that are generated.

from nn.symmequiv import SymmetricGrpEquivLinear

symm_model_4_1_1_diag = SymmetricGrpEquivLinear(dim_n = 4, order_k = 1, order_l = 1)
print("Number of weights: ", len(symm_model_4_1_1_diag.basis_set_matrices))
for basis_matrix in symm_model_4_1_1_diag.basis_set_matrices:
    print(basis_matrix)

symm_model_4_1_1_orbit = SymmetricGrpEquivLinear(dim_n = 4, order_k = 1, order_l = 1, diag=False)
print("Number of weights: ", len(symm_model_4_1_1_orbit.basis_set_matrices))
for basis_matrix in symm_model_4_1_1_orbit.basis_set_matrices:
    print(basis_matrix)

symm_model_2_2_2 = SymmetricGrpEquivLinear(dim_n = 2, order_k = 2, order_l = 2)
print("Number of weights: ", len(symm_model_2_2_2.basis_set_matrices))
for basis_matrix in symm_model_2_2_2.basis_set_matrices:
    print(basis_matrix)


from nn.lib.setpartitions import set_part_diag_basis_indices_list

print(set_part_diag_basis_indices_list([[1], [2]], 2))
print(set_part_diag_basis_indices_list([[1, 2]], 2))
print(set_part_diag_basis_indices_list([[1, 2], [3]], 2))
print(set_part_diag_basis_indices_list([[1, 2, 3]], 3))
