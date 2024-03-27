import unittest

from nn.lib import setpartitions

class setpartitionsTest(unittest.TestCase):
    def test_convert_tup1(self):
        indices = (1,3,1)
        dim = 3
        order_k = 3
        self.assertEqual(setpartitions.convert_tuple_to_matrix_index(indices, dim, order_k), 6, f"Wrong Tuple to Matrix conversion for indices = {indices}, dim = {dim}, and order_k = {order_k}.")

if __name__ == "__main__":
    unittest.main()

