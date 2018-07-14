from libc.stdlib cimport free
from libc.string cimport strncpy
from cpython.ref cimport Py_INCREF
from cpython.object cimport PyObject
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "pfgoto.h":
    ctypedef struct sequ:
        char *name
        char *seq

    ctypedef struct aligm:
        pass

    ctypedef struct u_sc:
        char *name
        char *monomers
        double **mat
        double endgaps

    u_sc scmat

    aligm c_align "align" (sequ *seq_array)
    void free_align(aligm alig)
    double **partition_f(aligm alig)
    void free_partition_f(double **m, sequ *seq_array)

# define global variables used by probA
#   usually these would be defined in gotoh_input.c, mt.c, option.c and psplot.c
#   but the functionality of those is not required and the _probA module is not
#   linked with them.
cdef public:
    # from gotoh_input.c
    float BETA = 1
    float ENDGAP = 0
    float OPENGAP = 0
    float EXTENDGAP = 0
    char MAT_SER[20]
    float DISTANCE = -1
    long Nr = 0
    float CUT = 0

    void usage(int bla):
        pass

    # from mt.c
    void sgenrand(unsigned long seed):
        pass

    double genrand():
        return 0.

    # from options.c
    int verbose_flag = -1   # disable output
    int typ_flag = -1
    int Egap_flag = 1
    int Ogap_flag = 0
    int Xgap_flag = 0

    # from psplot.c
    int PS_plot(sequ *so, double **P, char *wastlfile):
        return 0

np.import_array()


cdef inline void set_array_base(np.ndarray arr, object base):
     Py_INCREF(base)
     arr.base = <PyObject*>base


# wrap c allocated data for garbage collection when it is no longer referenced
# in Python
#   Based on: https://gist.github.com/2924976
cdef class dealloc_array_2d:
    cdef void *data_ptr
    cdef int size_x
    cdef int size_y
    cdef int typenum

    cdef set_data(self, int size_x, int size_y, void* data_ptr, int typenum):
        """ Set the data of the array

        This cannot be done in the constructor as it must recieve C-level
        arguments.

        Parameters:
        -----------
        size_x, size_y: int
            Lengths of the array.
        data_ptr: void*
            Pointer to the data.
        typenum: int
            Data type of the array.
        """
        self.data_ptr = data_ptr
        self.size_x = size_x
        self.size_y = size_y
        self.typenum = typenum

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp>self.size_x
        shape[1] = <np.npy_intp>self.size_y
        # Create a 2D array, of size 'size_x' x 'size_y'
        ndarray = np.PyArray_SimpleNewFromData(2, shape, self.typenum,
                                               self.data_ptr)
        set_array_base(ndarray, self)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(<void *>self.data_ptr)


cdef dealloc_array_2d_double(int size_x, int size_y, double *data):
    cdef dealloc_array_2d obj = dealloc_array_2d()
    obj.set_data(size_x, size_y, <void *>data, np.NPY_DOUBLE)
    return np.array(obj, copy=False)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def align(record1, record2, double temperature,
          opengap, extendgap, endgap,
          int type, char *matrix, float set, double cut):
    global typ_flag, Egap_flag, BETA, ENDGAP, MAT_SER, DISTANCE, CUT
    global Ogap_flag, Xgap_flag, OPENGAP, EXTENDGAP

    # set run parameters
    BETA = 1. / temperature
    if opengap is not None:
        OPENGAP = <float>opengap
        Ogap_flag = 1
    else:
        Ogap_flag = 0
    if extendgap is not None:
        EXTENDGAP = <float>extendgap
        Xgap_flag = 1
    else:
        Xgap_flag = 0
    if endgap is not None:
        ENDGAP = <float>endgap
        Egap_flag = 1
    else:
        Egap_flag = 0
    typ_flag = type
    strncpy(MAT_SER, matrix, 19);
    MAT_SER[19] = 0
    DISTANCE = set
    CUT = cut

    cdef sequ s[2]
    cdef aligm a
    cdef double **m
    cdef int x = len(record1.seq) + 1
    cdef int y = len(record2.seq) + 1
    cdef int l

    # align
    s[0].name = <char *>record1.id
    seq1 = str(record1.seq)
    s[0].seq = <char *>seq1
    s[1].name = <char *>record2.id
    seq2 = str(record2.seq)
    s[1].seq = <char *>seq2
    a = c_align(s)
    l = len(scmat.monomers)
    params = (typ_flag, str(scmat.name), scmat.mat[0][l - 2], scmat.mat[0][l - 1], scmat.endgaps)
    m = partition_f(a)

    # create numpy array
    pp = dealloc_array_2d_double(x, y, m[0])

    # free memory
    #   m[0] memory will be freed by numpy
    #   modified probA will ignore NULL
    #   m memory must be freed manually
    free(<void *>m)
    free_partition_f(NULL, s)
    free_align(a)

    return pp[1:, 1:], params
