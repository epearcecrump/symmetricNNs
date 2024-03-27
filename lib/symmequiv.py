import torch
import torch.nn as nn
import torch.optim as optim

from .lib import symmpartitions

class SymmetricGrpEquivLinear(nn.Module):
    """
    Creates a trainable S_n-equivariant linear layer 
    (R^n)^{otimes k} rightarrow (R^n)^{otimes l} using
    the diagram basis for the Partition vector space P_k^l(n)
    to create the weight matrices.
    
    dim_n: the dimension of the space, i.e the n in S_n
    order_k: the tensor power k
    order_l: the tensor power l
    diag: chooses diagram basis if True, else orbit basis
    """

    def __init__(self, dim_n: int, order_k: int, order_l: int, diag = True):
        super().__init__()
        self.n = dim_n
        self.k = order_k
        self.l = order_l
        self.diag = diag    
        
        #Trick used below to get everything on same device!
        self.dummy_param = nn.Parameter(torch.empty(0))
        
        self.basis_set_matrices, self.num_weights = self.__basis_set_matrices_generation()

        self.weights = nn.ParameterList([])
        for i in range(self.num_weights): 
            self.weights.append(nn.Parameter(torch.randn(())))

    def print_weights(self) -> None:
        for i in range(len(self.weights)):
            print(self.weights[i])

    def __basis_set_matrices_generation(self):
        part_lst_by_indices = symmpartitions.set_partition_weight_matrices_by_indices(
                                dim=self.n, order_k=self.k, order_l=self.l, diag=self.diag
                              )
        matrices = []
        for val in part_lst_by_indices:
            mat = torch.zeros(pow(self.n, self.l),pow(self.n, self.k))
            for ind in val:
                mat[ind[0]][ind[1]] = 1
            matrices.append(mat)
        return matrices, len(matrices)

    def forward(self, X):
        #Trick to get everything on the same device for training etc.
        device = self.dummy_param.device

        #Move all weight matrices onto device first before performing calculations.
        for i in range(len(self.basis_set_matrices)):
            self.basis_set_matrices[i] = self.basis_set_matrices[i].to(device)

        #Move weight_matrix onto device before calculating its value.
        weight_matrix = torch.zeros(pow(self.n, self.l),pow(self.n, self.k)).to(device)
    
        for weight_index, mat in enumerate(self.basis_set_matrices):
            weight_matrix += mat * self.weights[weight_index]
        
        linear = torch.einsum('ij,kj->ki', weight_matrix, X)    # allows for batch processing
        return linear

