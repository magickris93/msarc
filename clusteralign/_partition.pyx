import numpy as np
cimport numpy as np
cimport cython

DTYPE_FLOAT = np.float

ctypedef np.int_t DTYPE_INT_t
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.uint8_t DTYPE_UINT8_t
ctypedef np.float_t DTYPE_FLOAT_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def bestmove(np.ndarray[DTYPE_INT_t, ndim=1] parts not None,
             np.ndarray gains not None, np.ndarray allowed not None, int direction):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] gains_item = None
    cdef np.ndarray[DTYPE_UINT8_t, ndim=1] allowed_item = None
    cdef unsigned int num_groups = parts.shape[0]
    cdef unsigned int i, j, start, end, l
    cdef unsigned int max_group = 0
    cdef int max_partition = -1, group_max_partition
    cdef DTYPE_FLOAT_t gain, curgain, max_gain = 0., group_max_gain
    cdef DTYPE_INT_t curpart

    for i in range(num_groups):
        allowed_item = <np.ndarray[DTYPE_UINT8_t, ndim=1]>allowed[i]
        l = allowed_item.shape[0]
        if not l:
            continue
        gains_item = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>gains[i]
        curpart = parts[i]
        if direction < 0:
            start, end = 0, curpart
        elif direction > 0:
            start, end = curpart, l
        else:
            start, end = 0, l
        group_max_partition = -1
        group_max_gain = 0.
        gain = 0.
        if curpart < start:
            curgain = 0.
            for j in range(curpart + 1, start):
                gain += gains_item[j]
        if start <= curpart < end:
            for j in range(start, curpart + 1):
                gain += gains_item[j]
                if not allowed_item[j]:
                    continue
                if group_max_partition == -1 or gain > group_max_gain:
                    group_max_partition = j
                    group_max_gain = gain
            curgain = gain
            for j in range(curpart + 1, end):
                gain += gains_item[j]
                if not allowed_item[j]:
                    continue
                if group_max_partition == -1 or gain > group_max_gain:
                    group_max_partition = j
                    group_max_gain = gain
        else:
            for j in range(start, end):
                gain += gains_item[j]
                if not allowed_item[j]:
                    continue
                if group_max_partition == -1 or gain > group_max_gain:
                    group_max_partition = j
                    group_max_gain = gain
        if curpart >= end:
            for j in range(end, curpart + 1):
                gain += gains_item[j]
            curgain = gain
        if group_max_partition >= 0:
            group_max_gain -= curgain
            if max_partition == -1 or group_max_gain > max_gain:
                max_group = i
                max_partition = group_max_partition
                max_gain = group_max_gain

    if max_partition == -1:
        return None

    return max_group, max_partition, max_gain

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def calcgains_group(object self, unsigned int group):
    cdef np.ndarray[DTYPE_INT_t, ndim=1] lengths = self._lengths
    cdef unsigned int l_group = <unsigned int>lengths[group]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] gains = np.zeros(l_group + 1,
                                                            dtype=DTYPE_FLOAT)

    if l_group == 0:
        return gains

    cdef unsigned int num_groups = self._numgroups
    cdef np.ndarray[DTYPE_INT_t, ndim=1] parts = self._parts
    cdef unsigned int i, j, p_i, l_i
    cdef np.ndarray[object, ndim=2] matrices = self._edges
    cdef object edges = None
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] edges_data
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] edges_rows
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] edges_indices

    cdef DTYPE_FLOAT_t *edges_data_data
    cdef DTYPE_INT32_t *edges_rows_data
    cdef DTYPE_INT32_t *edges_indices_data
    cdef DTYPE_FLOAT_t *gains_data = &gains[1]

    cdef DTYPE_INT32_t k, row_start, row_end, index

    for i in range(num_groups):
        if i == group:
            continue

        l_i = lengths[i]
        if l_i == 0:
            continue

        p_i = parts[i]          # the current point of partition

        if group < i:           # edge matrix
            edges = matrices[group, i]
            edges_rows = <np.ndarray[DTYPE_INT32_t, ndim=1]>edges.indptr
            edges_rows_data = &edges_rows[0]

            if edges_rows_data[edges_rows.shape[0] - 1] == 0:
                continue

            edges_data = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>edges.data
            edges_indices = <np.ndarray[DTYPE_INT32_t, ndim=1]>edges.indices
            edges_data_data = &edges_data[0]
            edges_indices_data = &edges_indices[0]

            for j in range(l_group):
                row_start = edges_rows_data[j]
                row_end = edges_rows_data[j + 1]

                for k in range(row_start, row_end):
                    if edges_indices_data[k] >= p_i:
                        for k in range(k, row_end):
                            gains_data[j] -= edges_data_data[k]
                        break
                    gains_data[j] += edges_data_data[k]

        else:                   # transposed edge matrix
            edges = matrices[i, group]
            edges_rows = <np.ndarray[DTYPE_INT32_t, ndim=1]>edges.indptr
            edges_rows_data = &edges_rows[0]

            if edges_rows_data[edges_rows.shape[0] - 1] == 0:
                continue

            edges_data = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>edges.data
            edges_indices = <np.ndarray[DTYPE_INT32_t, ndim=1]>edges.indices
            edges_data_data = &edges_data[0]
            edges_indices_data = &edges_indices[0]

            for j in range(p_i):
                row_start = edges_rows_data[j]
                row_end = edges_rows_data[j + 1]

                for k in range(row_start, row_end):
                    gains_data[edges_indices_data[k]] += edges_data_data[k]

            for j in range(p_i, l_i):
                row_start = edges_rows_data[j]
                row_end = edges_rows_data[j + 1]

                for k in range(row_start, row_end):
                    gains_data[edges_indices_data[k]] -= edges_data_data[k]

    # divide by 2 here so we don't have to double the difference in
    # _fixgain_all, which is called more frequently
    for i in range(l_group):
        gains_data[i] /= 2.

    return gains


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def fixgains_group(object self, unsigned int group,
                   unsigned int move_group, unsigned int move_partition,
                   np.ndarray[DTYPE_FLOAT_t, ndim=1] gains):
    if move_group == group:
        raise ValueError

    cdef np.ndarray[DTYPE_INT_t, ndim=1] lengths = self._lengths
    cdef unsigned int l_group = <unsigned int>lengths[group]
    cdef unsigned int l_move_group = <unsigned int>lengths[move_group]

    if l_group == 0 or l_move_group == 0:
        return gains

    cdef np.ndarray[DTYPE_INT_t, ndim=1] parts = self._parts
    cdef unsigned int p_move_group = parts[move_group]

    if move_partition == p_move_group:
        return ValueError

    cdef unsigned int j
    cdef np.ndarray[object, ndim=2] matrices = self._edges
    cdef object edges = None
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] edges_data
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] edges_rows
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] edges_indices

    cdef DTYPE_FLOAT_t *edges_data_data
    cdef DTYPE_INT32_t *edges_rows_data
    cdef DTYPE_INT32_t *edges_indices_data
    cdef DTYPE_FLOAT_t *gains_data = &gains[1]

    cdef DTYPE_INT32_t k, row_start, row_end, index

    if group < move_group:      # edge matrix
        edges = matrices[group, move_group]
        edges_rows = <np.ndarray[DTYPE_INT32_t, ndim=1]>edges.indptr
        edges_rows_data = &edges_rows[0]

        if edges_rows_data[edges_rows.shape[0] - 1] == 0:
            return

        edges_data = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>edges.data
        edges_indices = <np.ndarray[DTYPE_INT32_t, ndim=1]>edges.indices
        edges_data_data = &edges_data[0]
        edges_indices_data = &edges_indices[0]

        if move_partition > p_move_group:
            for j in range(l_group):
                row_start = edges_rows_data[j]
                row_end = edges_rows_data[j + 1]

                if row_start == row_end:
                    continue
                if edges_indices_data[row_start] >= move_partition:
                    continue
                if edges_indices_data[row_end - 1] < p_move_group:
                    continue
                for k in range(row_start, row_end):
                    if edges_indices_data[k] >= p_move_group:
                        row_start = k
                        break
                if edges_indices_data[row_start] >= move_partition:
                    continue
                for k in range(row_end - 1, row_start - 1, -1):
                    if edges_indices_data[k] < move_partition:
                        row_end = k + 1
                        break
                else:
                    continue

                for k in range(row_start, row_end):
                    gains_data[j] += edges_data_data[k]

        else:   # less
            for j in range(l_group):
                row_start = edges_rows_data[j]
                row_end = edges_rows_data[j + 1]

                if row_start == row_end:
                    continue
                if edges_indices_data[row_start] >= p_move_group:
                    continue
                if edges_indices_data[row_end - 1] < move_partition:
                    continue
                for k in range(row_start, row_end):
                    if edges_indices_data[k] >= move_partition:
                        row_start = k
                        break
                if edges_indices_data[row_start] >= p_move_group:
                    continue
                for k in range(row_end - 1, row_start - 1, -1):
                    if edges_indices_data[k] < p_move_group:
                        row_end = k + 1
                        break
                else:
                    continue

                for k in range(row_start, row_end):
                    gains_data[j] -= edges_data_data[k]

    else:                       # transposed edge matrix
        edges = matrices[move_group, group]
        edges_rows = <np.ndarray[DTYPE_INT32_t, ndim=1]>edges.indptr
        edges_rows_data = &edges_rows[0]

        if edges_rows_data[edges_rows.shape[0] - 1] == 0:
            return

        edges_data = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>edges.data
        edges_indices = <np.ndarray[DTYPE_INT32_t, ndim=1]>edges.indices
        edges_data_data = &edges_data[0]
        edges_indices_data = &edges_indices[0]

        if move_partition > p_move_group:
            for j in range(p_move_group, move_partition):
                row_start = edges_rows_data[j]
                row_end = edges_rows_data[j + 1]

                for k in range(row_start, row_end):
                    gains_data[edges_indices_data[k]] += edges_data_data[k]
        else:   # less
            for j in range(move_partition, p_move_group):
                row_start = edges_rows_data[j]
                row_end = edges_rows_data[j + 1]

                for k in range(row_start, row_end):
                    gains_data[edges_indices_data[k]] -= edges_data_data[k]
