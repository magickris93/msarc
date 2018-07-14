from scipy.sparse import isspmatrix_csr


def prune(array, cut):
    assert isspmatrix_csr(array)
    array.data[array.data < cut] = 0.
    array.eliminate_zeros()
