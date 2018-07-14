import numpy as np
from scipy.sparse import csr_matrix
cimport numpy as np
cimport cython

DTYPE_FLOAT = np.float

ctypedef np.int_t DTYPE_INT_t
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.float_t DTYPE_FLOAT_t


# Based on Relax from ProbCons in Main.cc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def relax_XZ_ZY(object matXZ not None, object matZY not None, object matXY not None, double weight):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] dataXZ, dataZY, dataXY
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] rowsXZ, rowsZY, rowsXY
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] indicesXZ, indicesZY, indicesXY

    dataXZ = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>matXZ.data
    rowsXZ = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXZ.indptr
    indicesXZ = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXZ.indices
    dataZY = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>matZY.data
    rowsZY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matZY.indptr
    indicesZY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matZY.indices
    dataXY = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>matXY.data
    rowsXY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXY.indptr
    indicesXY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXY.indices

    cdef DTYPE_FLOAT_t *dataXZ_data, *dataZY_data, *dataXY_data
    cdef DTYPE_INT32_t *rowsXZ_data, *rowsZY_data, *rowsXY_data
    cdef DTYPE_INT32_t *indicesXZ_data, *indicesZY_data, *indicesXY_data

    cdef unsigned int lengthX = matXY.shape[0]
    cdef unsigned int x
    cdef DTYPE_INT32_t z, indexXZ, indexZY, indexXY
    cdef DTYPE_INT32_t startXZ, endXZ, startZY, endZY, startXY, endXY, iterXY
    cdef DTYPE_FLOAT_t valXZ

    dataXZ_data = &dataXZ[0]
    dataZY_data = &dataZY[0]
    dataXY_data = &dataXY[0]
    rowsXZ_data = &rowsXZ[0]
    rowsZY_data = &rowsZY[0]
    rowsXY_data = &rowsXY[0]
    indicesXZ_data = &indicesXZ[0]
    indicesZY_data = &indicesZY[0]
    indicesXY_data = &indicesXY[0]

    for x in range(lengthX):
        startXZ = rowsXZ_data[x]
        endXZ = rowsXZ_data[x + 1]
        startXY = rowsXY_data[x]
        endXY = rowsXY_data[x + 1]

        for z in range(startXZ, endXZ):
            indexXZ = indicesXZ_data[z]
            startZY = rowsZY_data[indexXZ]
            endZY = rowsZY_data[indexXZ + 1]
            valXZ = dataXZ_data[z]
            iterXY = startXY

            indexZY = indicesZY_data[startZY]
            indexXY = indicesXY_data[iterXY]
            while startZY < endZY and iterXY < endXY:
                if indexZY < indexXY:
                    startZY += 1
                    indexZY = indicesZY_data[startZY]
                elif indexZY > indexXY:
                    iterXY += 1
                    indexXY = indicesXY_data[iterXY]
                else:   # equal
                    dataXY_data[iterXY] += valXZ * dataZY_data[startZY] * weight
                    startZY += 1
                    iterXY += 1
                    indexZY = indicesZY_data[startZY]
                    indexXY = indicesXY_data[iterXY]


# Based on Relax1 from ProbCons in Main.cc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def relax_ZX_ZY(object matZX not None, object matZY not None, object matXY not None, double weight):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] dataZX, dataZY, dataXY
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] rowsZX, rowsZY, rowsXY
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] indicesZX, indicesZY, indicesXY

    dataZX = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>matZX.data
    rowsZX = <np.ndarray[DTYPE_INT32_t, ndim=1]>matZX.indptr
    indicesZX = <np.ndarray[DTYPE_INT32_t, ndim=1]>matZX.indices
    dataZY = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>matZY.data
    rowsZY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matZY.indptr
    indicesZY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matZY.indices
    dataXY = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>matXY.data
    rowsXY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXY.indptr
    indicesXY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXY.indices

    cdef DTYPE_FLOAT_t *dataZX_data, *dataZY_data, *dataXY_data
    cdef DTYPE_INT32_t *rowsZX_data, *rowsZY_data, *rowsXY_data
    cdef DTYPE_INT32_t *indicesZX_data, *indicesZY_data, *indicesXY_data

    cdef unsigned int lengthZ = matZX.shape[0]
    cdef unsigned int z
    cdef DTYPE_INT32_t x, indexZX, indexZY, indexXY
    cdef DTYPE_INT32_t startZX, endZX, startZY, endZY, startXY, endXY, iterZY
    cdef DTYPE_FLOAT_t valZX

    dataZX_data = &dataZX[0]
    dataZY_data = &dataZY[0]
    dataXY_data = &dataXY[0]
    rowsZX_data = &rowsZX[0]
    rowsZY_data = &rowsZY[0]
    rowsXY_data = &rowsXY[0]
    indicesZX_data = &indicesZX[0]
    indicesZY_data = &indicesZY[0]
    indicesXY_data = &indicesXY[0]

    for z in range(lengthZ):
        startZX = rowsZX_data[z]
        endZX = rowsZX_data[z + 1]
        startZY = rowsZY_data[z]
        endZY = rowsZY_data[z + 1]

        for x in range(startZX, endZX):
            indexZX = indicesZX_data[x]
            startXY = rowsXY_data[indexZX]
            endXY = rowsXY_data[indexZX + 1]
            valZX = dataZX_data[x]
            iterZY = startZY

            indexZY = indicesZY_data[iterZY]
            indexXY = indicesXY_data[startXY]
            while iterZY < endZY and startXY < endXY:
                if indexZY < indexXY:
                    iterZY += 1
                    indexZY = indicesZY_data[iterZY]
                elif indexZY > indexXY:
                    startXY += 1
                    indexXY = indicesXY_data[startXY]
                else:   # equal
                    dataXY_data[startXY] += valZX * dataZY_data[iterZY] * weight
                    iterZY += 1
                    startXY += 1
                    indexZY = indicesZY_data[iterZY]
                    indexXY = indicesXY_data[startXY]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def relax_XZ_YZ(object matXZ not None, object matYZ not None, object matXY not None, double weight):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] dataXZ, dataYZ, dataXY
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] rowsXZ, rowsYZ, rowsXY
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] indicesXZ, indicesYZ, indicesXY

    dataXZ = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>matXZ.data
    rowsXZ = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXZ.indptr
    indicesXZ = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXZ.indices
    dataYZ = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>matYZ.data
    rowsYZ = <np.ndarray[DTYPE_INT32_t, ndim=1]>matYZ.indptr
    indicesYZ = <np.ndarray[DTYPE_INT32_t, ndim=1]>matYZ.indices
    dataXY = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>matXY.data
    rowsXY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXY.indptr
    indicesXY = <np.ndarray[DTYPE_INT32_t, ndim=1]>matXY.indices

    cdef DTYPE_FLOAT_t *dataXZ_data, *dataYZ_data, *dataXY_data
    cdef DTYPE_INT32_t *rowsXZ_data, *rowsYZ_data, *rowsXY_data
    cdef DTYPE_INT32_t *indicesXZ_data, *indicesYZ_data, *indicesXY_data

    cdef unsigned int lengthX = matXY.shape[0]
    cdef unsigned int x
    cdef DTYPE_INT32_t y, indexXZ, indexYZ, indexXY
    cdef DTYPE_INT32_t startXZ, endXZ, startYZ, endYZ, startXY, endXY, iterXZ

    dataXZ_data = &dataXZ[0]
    dataYZ_data = &dataYZ[0]
    dataXY_data = &dataXY[0]
    rowsXZ_data = &rowsXZ[0]
    rowsYZ_data = &rowsYZ[0]
    rowsXY_data = &rowsXY[0]
    indicesXZ_data = &indicesXZ[0]
    indicesYZ_data = &indicesYZ[0]
    indicesXY_data = &indicesXY[0]

    for x in range(lengthX):
        startXY = rowsXY_data[x]
        endXY = rowsXY_data[x + 1]
        startXZ = rowsXZ_data[x]
        endXZ = rowsXZ_data[x + 1]

        for y in range(startXY, endXY):
            indexXY = indicesXY_data[y]
            startYZ = rowsYZ_data[indexXY]
            endYZ = rowsYZ_data[indexXY + 1]
            iterXZ = startXZ

            indexYZ = indicesYZ_data[startYZ]
            indexXZ = indicesXZ_data[iterXZ]
            while startYZ < endYZ and iterXZ < endXZ:
                if indexYZ < indexXZ:
                    startYZ += 1
                    indexYZ = indicesYZ_data[startYZ]
                elif indexYZ > indexXZ:
                    iterXZ += 1
                    indexXZ = indicesXZ_data[iterXZ]
                else:   # equal
                    dataXY_data[y] += dataXZ_data[iterXZ] * dataYZ_data[startYZ] * weight
                    startYZ += 1
                    iterXZ += 1
                    indexYZ = indicesYZ_data[startYZ]
                    indexXZ = indicesXZ_data[iterXZ]


# Based on BuildPosterior from ProbCons in ComputeAlignment.cc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def project(np.ndarray[DTYPE_INT_t, ndim=1] group1 not None,
            np.ndarray[DTYPE_INT_t, ndim=1] group2 not None,
            np.ndarray[object, ndim=1] mappings not None,
            np.ndarray pair_pp_all not None):
    cdef object pair_pp = None
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] pair_pp_data
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] pair_pp_indptr
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] pair_pp_indices
    cdef DTYPE_FLOAT_t *pp_data, *pp_data_row
    cdef DTYPE_FLOAT_t *pair_pp_data_data
    cdef DTYPE_INT32_t *pair_pp_indptr_data, *pair_pp_indices_data
    cdef DTYPE_INT_t g1, g2
    cdef unsigned int x, y, i, j, mi, mj, pp_x = 0, pp_y = 0
    cdef DTYPE_INT32_t row_start, row_end

    cdef np.ndarray[DTYPE_INT_t, ndim=1] mapping1 = None
    cdef np.ndarray[DTYPE_INT_t, ndim=1] mapping2 = None

    for g1 in group1:
        mapping1 = <np.ndarray[DTYPE_INT_t, ndim=1]>mappings[<unsigned int>g1]
        pp_x = max(pp_x, mapping1[<unsigned int>(mapping1.shape[0] - 1)] + 1)
    for g2 in group2:
        mapping2 = <np.ndarray[DTYPE_INT_t, ndim=1]>mappings[<unsigned int>g2]
        pp_y = max(pp_y, mapping2[<unsigned int>(mapping2.shape[0] - 1)] + 1)

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] pp = np.zeros((pp_x, pp_y), dtype=DTYPE_FLOAT)
    pp_data = &pp[0, 0]

    for g1 in group1:
        mapping1 = <np.ndarray[DTYPE_INT_t, ndim=1]>mappings[<unsigned int>g1]

        for g2 in group2:
            mapping2 = <np.ndarray[DTYPE_INT_t, ndim=1]>mappings[<unsigned int>g2]

            if g1 < g2:
                pair_pp = pair_pp_all[<unsigned int>g1, <unsigned int>g2]
                x = pair_pp.shape[0]
                y = pair_pp.shape[1]
                pair_pp_data = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>pair_pp.data
                pair_pp_indptr = <np.ndarray[DTYPE_INT32_t, ndim=1]>pair_pp.indptr
                pair_pp_indices = <np.ndarray[DTYPE_INT32_t, ndim=1]>pair_pp.indices

                pair_pp_data_data = &pair_pp_data[0]
                pair_pp_indptr_data = &pair_pp_indptr[0]
                pair_pp_indices_data = &pair_pp_indices[0]


                for i in range(x):
                    row_start = pair_pp_indptr[i]
                    row_end = pair_pp_indptr[i + 1]
                    mi = mapping1[i]
                    pp_data_row = pp_data + mi * pp_y

                    for j in range(row_start, row_end):
                        mj = mapping2[pair_pp_indices_data[j]]
                        pp_data_row[mj] += pair_pp_data_data[j]

            else:
                pair_pp = pair_pp_all[<unsigned int>g2, <unsigned int>g1]
                x = pair_pp.shape[0]
                y = pair_pp.shape[1]
                pair_pp_data = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>pair_pp.data
                pair_pp_indptr = <np.ndarray[DTYPE_INT32_t, ndim=1]>pair_pp.indptr
                pair_pp_indices = <np.ndarray[DTYPE_INT32_t, ndim=1]>pair_pp.indices

                pair_pp_data_data = &pair_pp_data[0]
                pair_pp_indptr_data = &pair_pp_indptr[0]
                pair_pp_indices_data = &pair_pp_indices[0]

                for j in range(x):
                    row_start = pair_pp_indptr[j]
                    row_end = pair_pp_indptr[j + 1]
                    mj = mapping2[j]
                    pp_data_row = pp_data + mj

                    for i in range(row_start, row_end):
                        mi = mapping1[pair_pp_indices_data[i]]
                        pp_data_row[mi * pp_y] += pair_pp_data_data[i]

    return pp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def coarsen_weights(np.ndarray[DTYPE_INT_t, ndim=1] weights not None):
    cdef unsigned int L = weights.shape[0]

    if L <= 2:
        return weights

    cdef unsigned int newL = (L + 1) / 2
    if L <= 2:
        newL = L

    cdef np.ndarray[DTYPE_INT_t, ndim=1] new_weights = np.empty(newL, dtype=weights.dtype)
    cdef DTYPE_INT_t *weights_data, *new_weights_data
    cdef int i, j = 0

    weights_data = &weights[0]
    new_weights_data = &new_weights[0]

    for i in range(L / 2):
        new_weights_data[i] = weights_data[j] + weights_data[j + 1]
        j += 2
    if i < newL - 1:
        new_weights_data[newL - 1] = weights_data[L - 1]

    return new_weights


cdef inline int mapping(int i, unsigned int L):
    if L <= 2:
        return i
    return i / 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def coarsen_pair_pp(object pair_pp not None):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] pair_pp_data
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] pair_pp_rows
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] pair_pp_indices
    cdef DTYPE_FLOAT_t *pair_pp_data_data
    cdef DTYPE_INT32_t *pair_pp_rows_data, *pair_pp_indices_data

    cdef unsigned int X = pair_pp.shape[0]
    cdef unsigned int Y = pair_pp.shape[1]

    if X <= 2 and Y <= 2:
        return pair_pp

    cdef unsigned int newX = (X + 1) / 2
    cdef unsigned int newY = (Y + 1) / 2
    if X <= 2:
        newX = X
    if Y <= 2:
        newY = Y

    if X == 0 or Y == 0:
        return np.empty((newX, newY), dtype=pair_pp.dtype)

    pair_pp_data = <np.ndarray[DTYPE_FLOAT_t, ndim=1]>pair_pp.data
    pair_pp_rows = <np.ndarray[DTYPE_INT32_t, ndim=1]>pair_pp.indptr
    pair_pp_indices = <np.ndarray[DTYPE_INT32_t, ndim=1]>pair_pp.indices

    pair_pp_data_data = &pair_pp_data[0]
    pair_pp_rows_data = &pair_pp_rows[0]
    pair_pp_indices_data = &pair_pp_indices[0]

    cdef unsigned int nnz = pair_pp_rows_data[pair_pp_rows.shape[0] - 1]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] new_pair_pp_data = np.empty(nnz, dtype=pair_pp_data.dtype)
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] new_pair_pp_rows = np.empty(newX + 1, dtype=pair_pp_rows.dtype)
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] new_pair_pp_indices = np.empty(nnz, dtype=pair_pp_indices.dtype)
    cdef DTYPE_FLOAT_t *new_pair_pp_data_data
    cdef DTYPE_INT32_t *new_pair_pp_rows_data, *new_pair_pp_indices_data

    new_pair_pp_data_data = &new_pair_pp_data[0]
    new_pair_pp_rows_data = &new_pair_pp_rows[0]
    new_pair_pp_indices_data = &new_pair_pp_indices[0]

    cdef int row1_start, row1_end, row2_start, row2_end
    cdef int row1_y, row2_y, n_index = -1
    cdef int nx, ny, my, x = 0

    for nx in range(newX):
        new_pair_pp_rows_data[nx] = n_index + 1
        ny = -1

        # linked rows
        if x < X - 1 and X > 2:
            row1_start = pair_pp_rows_data[x]
            x += 1
            row2_start = row1_end = pair_pp_rows_data[x]
            x += 1
            row2_end = pair_pp_rows_data[x]

            row1_y = pair_pp_indices_data[row1_start]
            row2_y = pair_pp_indices_data[row2_start]
            while row1_start < row1_end and row2_start < row2_end:
                if row1_y <= row2_y:
                    my = mapping(row1_y, Y)
                    if my > ny:
                        n_index += 1
                        new_pair_pp_indices_data[n_index] = ny = my
                        new_pair_pp_data_data[n_index] = pair_pp_data_data[row1_start]
                    else:
                        new_pair_pp_data_data[n_index] += pair_pp_data_data[row1_start]
                    row1_start += 1
                    row1_y = pair_pp_indices_data[row1_start]
                else:
                    my = mapping(row2_y, Y)
                    if my > ny:
                        n_index += 1
                        new_pair_pp_indices_data[n_index] = ny = my
                        new_pair_pp_data_data[n_index] = pair_pp_data_data[row2_start]
                    else:
                        new_pair_pp_data_data[n_index] += pair_pp_data_data[row2_start]
                    row2_start += 1
                    row2_y = pair_pp_indices_data[row2_start]

            if row1_start >= row1_end:
                row1_start = row2_start
                row1_end = row2_end
            for row1_start in range(row1_start, row1_end):
                row1_y = pair_pp_indices_data[row1_start]

                my = mapping(row1_y, Y)
                if my > ny:
                    n_index += 1
                    new_pair_pp_indices_data[n_index] = ny = my
                    new_pair_pp_data_data[n_index] = pair_pp_data_data[row1_start]
                else:
                    new_pair_pp_data_data[n_index] += pair_pp_data_data[row1_start]

        # single row
        else:
            row1_start = pair_pp_rows_data[x]
            x += 1
            row1_end = pair_pp_rows_data[x]

            for row1_start in range(row1_start, row1_end):
                row1_y = pair_pp_indices_data[row1_start]

                my = mapping(row1_y, Y)
                if my > ny:
                    n_index += 1
                    new_pair_pp_indices_data[n_index] = ny = my
                    new_pair_pp_data_data[n_index] = pair_pp_data_data[row1_start]
                else:
                    new_pair_pp_data_data[n_index] += pair_pp_data_data[row1_start]

    n_index += 1
    new_pair_pp_rows_data[newX] = n_index
    return csr_matrix((new_pair_pp_data[:n_index],
                       new_pair_pp_indices[:n_index],
                       new_pair_pp_rows),
                      shape=(newX, newY))
