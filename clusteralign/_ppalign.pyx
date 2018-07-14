from libc.stdlib cimport malloc, realloc, free
import numpy as np
cimport numpy as np
cimport cython

DTYPE_INT = np.int
DTYPE_BOOL = np.bool
DTYPE_UINT8 = np.uint8
DTYPE_FLOAT = np.float
ctypedef np.int_t DTYPE_INT_t
ctypedef np.uint8_t DTYPE_UINT8_t
ctypedef np.float_t DTYPE_FLOAT_t
cdef DTYPE_INT_t UP = 1, LEFT = 2, DIAGONAL = 3

cdef DTYPE_FLOAT_t *row0 = NULL, *row1 = NULL
cdef DTYPE_INT_t row0_size = 0, row1_size = 0
cdef DTYPE_INT_t *matrix = NULL
cdef DTYPE_INT_t matrix_size = 0


cdef inline DTYPE_FLOAT_t *alloc_row0(int size):
    global row0, row0_size
    cdef DTYPE_FLOAT_t *tmp

    if row0 != NULL:
        if size > row0_size:
            tmp = <DTYPE_FLOAT_t *>realloc(<void *>row0, size * sizeof(DTYPE_FLOAT_t))
            if tmp == NULL:
                free(<void *>row0)
                row0 = NULL
                row0_size = 0
            else:
                row0 = tmp
                row0_size = size
    if row0 == NULL:
        row0 = <DTYPE_FLOAT_t *>malloc(size * sizeof(DTYPE_FLOAT_t))
        row0_size = size


cdef inline DTYPE_FLOAT_t *alloc_row1(int size):
    global row1, row1_size
    cdef DTYPE_FLOAT_t *tmp

    if row1 != NULL:
        if size > row1_size:
            tmp = <DTYPE_FLOAT_t *>realloc(<void *>row1, size * sizeof(DTYPE_FLOAT_t))
            if tmp == NULL:
                free(<void *>row1)
                row1 = NULL
                row1_size = 0
            else:
                row1 = tmp
                row1_size = size
    if row1 == NULL:
        row1 = <DTYPE_FLOAT_t *>malloc(size * sizeof(DTYPE_FLOAT_t))
        row1_size = size


cdef inline DTYPE_INT_t *alloc_matrix(int size):
    global matrix, matrix_size
    cdef DTYPE_INT_t *tmp

    if matrix != NULL:
        if size > matrix_size:
            tmp = <DTYPE_INT_t *>realloc(matrix, size * sizeof(DTYPE_INT_t))
            if tmp == NULL:
                free(<void *>matrix)
                matrix = NULL
                matrix_size = 0
            else:
                matrix = tmp
                matrix_size = size
    if matrix == NULL:
        matrix = <DTYPE_INT_t *>malloc(size * sizeof(DTYPE_INT_t))
        matrix_size = size


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def align_probabilities(np.ndarray[DTYPE_FLOAT_t, ndim=2] pp not None):
    global row0, row1, matrix

    cdef unsigned int imax = pp.shape[0]
    cdef unsigned int jmax = pp.shape[1]
    cdef unsigned int imax_plus_1 = imax + 1, jmax_plus_1 = jmax + 1
    cdef np.ndarray[DTYPE_INT_t, ndim=1] map1 = np.empty(imax, dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_INT_t, ndim=1] map2 = np.empty(jmax, dtype=DTYPE_INT)
    cdef unsigned int i, j, j_plus_1, k, d
    cdef DTYPE_INT_t *matrix_ptr
    cdef DTYPE_FLOAT_t v_UP, v_LEFT, v_DIAGONAL

    alloc_matrix(imax * jmax)
    alloc_row0(jmax_plus_1)
    alloc_row1(jmax_plus_1)

    for j in range(jmax_plus_1):
        row0[j] = 0.
    row1[0] = 0.

    matrix_ptr = matrix
    for i in range(imax):
        for j in range(jmax):
            j_plus_1 = j + 1

            v_DIAGONAL = row0[j] + pp[i, j]
            v_UP = row0[j_plus_1]
            v_LEFT = row1[j]

            if v_DIAGONAL > v_UP:
                if v_DIAGONAL >= v_LEFT:
                    row1[j_plus_1] = v_DIAGONAL
                    matrix_ptr[j] = DIAGONAL
                else:
                    row1[j_plus_1] = v_LEFT
                    matrix_ptr[j] = LEFT
            elif v_DIAGONAL < v_UP:
                if v_UP > v_LEFT:
                    row1[j_plus_1] = v_UP
                    matrix_ptr[j] = UP
                elif v_UP < v_LEFT:
                    row1[j_plus_1] = v_LEFT
                    matrix_ptr[j] = LEFT
                elif i > j:
                    row1[j_plus_1] = v_UP
                    matrix_ptr[j] = UP
                elif i < j:
                    row1[j_plus_1] = v_LEFT
                    matrix_ptr[j] = LEFT
                elif imax >= jmax:
                    row1[j_plus_1] = v_UP
                    matrix_ptr[j] = UP
                elif imax < jmax:
                    row1[j_plus_1] = v_LEFT
                    matrix_ptr[j] = LEFT
            elif v_DIAGONAL >= v_LEFT:
                row1[j_plus_1] = v_DIAGONAL
                matrix_ptr[j] = DIAGONAL
            else:
                row1[j_plus_1] = v_LEFT
                matrix_ptr[j] = LEFT

        row0, row1 = row1, row0
        matrix_ptr += jmax

    i, j, k = imax, jmax, imax + jmax
    matrix_ptr = matrix + imax * jmax - 1
    while i or j:
        if not i:
            d = LEFT
        elif not j:
            d = UP
        else:
            d = matrix_ptr[0]
        k -= 1
        if d == UP:
            i -= 1
            map1[i] = k
            matrix_ptr -= jmax
        elif d == LEFT:
            j -= 1
            map2[j] = k
            matrix_ptr -= 1
        elif d == DIAGONAL:
            i -= 1
            j -= 1
            map1[i] = map2[j] = k
            matrix_ptr -= jmax_plus_1

    for i in range(imax):
        map1[i] -= k
    for j in range(jmax):
        map2[j] -= k

    return map1, map2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def remap(np.ndarray[DTYPE_UINT8_t, ndim=2] map not None,
          np.ndarray[DTYPE_INT_t, ndim=1] group1 not None,
          np.ndarray[DTYPE_INT_t, ndim=1] group2 not None,
          np.ndarray[DTYPE_INT_t, ndim=1] mapping1 not None,
          np.ndarray[DTYPE_INT_t, ndim=1] mapping2 not None,
          np.ndarray[DTYPE_INT_t, ndim=1] new_mapping1 not None,
          np.ndarray[DTYPE_INT_t, ndim=1] new_mapping2 not None):

    cdef unsigned int i, j1, j2, g
    cdef DTYPE_INT_t k, c1, c2

    cdef DTYPE_UINT8_t *map_data, *map_data_row, *new_map_data, *new_map_data_row
    cdef DTYPE_INT_t *mapping1_data, *mapping2_data
    cdef DTYPE_INT_t *new_mapping1_data, *new_mapping2_data
    cdef unsigned int length = max(new_mapping1[new_mapping1.shape[0] - 1],
                                   new_mapping2[new_mapping2.shape[0] - 1]) + 1

    cdef np.ndarray[DTYPE_UINT8_t, ndim=2] new_map = np.zeros((map.shape[0], length), dtype=DTYPE_UINT8)

    map_data = &map[0, 0]
    new_map_data = &new_map[0, 0]
    mapping1_data = &mapping1[0]
    mapping2_data = &mapping2[0]
    new_mapping1_data = &new_mapping1[0]
    new_mapping2_data = &new_mapping2[0]

    for k in group1:
#        map_data_row = map_data + k * l_o
#        new_map_data_row = new_map_data + k * l_n

        for i in range(len(new_mapping1)):
            #new_map_data_row[new_mapping1[i]] = map_data_row[mapping1[i]]
            new_map[k, new_mapping1[i]] = map[k, mapping1[i]]

    for k in group2:
#        map_data_row = map_data + k * l_o
#        new_map_data_row = new_map_data + k * l_n

        for i in range(len(new_mapping2)):
            #new_map_data_row[new_mapping2[i]] = map_data_row[mapping2[i]]
            new_map[k, new_mapping2[i]] = map[k, mapping2[i]]

    new_map.dtype = DTYPE_BOOL
    return new_map
