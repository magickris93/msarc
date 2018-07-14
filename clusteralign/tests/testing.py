from numpy import array
from numpy.testing import (assert_array_equal as np_assert_array_equal,
                           assert_array_almost_equal as np_assert_array_almost_equal)
from scipy.sparse import issparse
from scipy.sparse.sputils import isdense


def to_dense(a):
    if not (isdense(a) or issparse(a)):
        a = array(a)
    if a.ndim == 1:
        a = a.reshape(1, a.shape[0])
    return a.toarray() if issparse(a) else a


def assert_array_equal(x, y, *args, **kwargs):
    np_assert_array_equal(to_dense(x), to_dense(y), *args, **kwargs)


def assert_array_almost_equal(x, y, *args, **kwargs):
    np_assert_array_almost_equal(to_dense(x), to_dense(y), *args, **kwargs)
