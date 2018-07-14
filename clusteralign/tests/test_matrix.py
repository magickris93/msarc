from unittest import TestCase
from scipy.sparse import csr_matrix
from clusteralign.matrix import prune
from clusteralign.tests.testing import assert_array_equal


class PruneTest(TestCase):
    def test_prune(self):
        input = csr_matrix([[0, 1, 2, 3],
                            [3, 2, 1, 0],
                            [2, 0, 3, 1]])
        output0 = input
        output1 = [[0, 0, 2, 3],
                   [3, 2, 0, 0],
                   [2, 0, 3, 0]]
        output2 = [[0, 0, 0, 3],
                   [3, 0, 0, 0],
                   [0, 0, 3, 0]]
        output3 = [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]
        input = input

        prune(input, 1)
        assert_array_equal(input, output0)
        prune(input, 2)
        assert_array_equal(input, output1)
        prune(input, 3)
        assert_array_equal(input, output2)
        prune(input, 4)
        assert_array_equal(input, output3)
