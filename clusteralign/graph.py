from Bio.SeqRecord import SeqRecord
from numpy import ndarray, array, empty, empty_like, ones, zeros, fromiter
from scipy.sparse import csr_matrix
try:
    from scipy.sparse.sparsetools._csr import get_csr_submatrix
except ImportError:
    try:
        from scipy.sparse.sparsetools import get_csr_submatrix
    except ImportError:
        def get_csr_submatrix(M, N, indptr, indices, data, i0, i1, j0, j1):
            matrix = csr_matrix((data,indices,indptr), shape=(M, N))
            submatrix = matrix[i0:i1, j0:j1]
            return submatrix.indptr, submatrix.indices, submatrix.data
from .sequence import SPACE, EMPTY_SEQ
from .matrix import prune
from .probA import pairwise
from .partition import GraphPartitioner
from ._graph import (relax_XZ_ZY, relax_ZX_ZY, relax_XZ_YZ, project,
                     coarsen_weights, coarsen_pair_pp)
import math


class Graph(object):
    """A graph of interactions between symbols in multiple sequences."""

    def __init__(self, records_or_parent, verbose=False, **kwargs):
        if records_or_parent is None:
            raise ValueError("First parameter is None, should be a list of SeqRecord objects or a Graph instance.")
        records = None
        parent = None
        if isinstance(records_or_parent, Graph):
            parent = records_or_parent
        else:
            for r in records_or_parent:
                if not isinstance(r, SeqRecord):
                    raise ValueError("Records parameter is not a list of SeqRecord objects.")
            records = records_or_parent

        self._verbose = verbose

        self._parent = parent

        self._records = records
        self._size = None
        self._lengths = None
        self._max_length = None
        self._value = None
        self._starts = None
        self._ends = None
        self._parts = None
        self._rel_parts = None
        self._coarse = []

        self.__weights = None
        self.__pair_pp = None

        self.processes = kwargs.get('processes', 1)

        self._params = {}

        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)

    def __len__(self):
        return self.size

    def __str__(self):
        if self.is_column:
            return "".join((r.seq[0] if len(r.seq) > 0 else SPACE for r in self.records))
        return " ".join((str(r.seq) if len(r.seq) > 0 else SPACE for r in self.records))

    @property
    def size(self):
        """Total number of symbols contained in the graph."""
        if self._size is None:
            self._size = self.lengths.sum()
        return self._size

    @property
    def num_sequences(self):
        """Number of sequences contained in the graph."""
        return len(self.records)

    @property
    def lengths(self):
        """Lengths of the sequences contained in the graph."""
        if self._lengths is None:
            if self._ends is not None:
                self._lengths = self._ends - self.starts
            else:
                self._lengths = fromiter((len(r.seq) for r in self.records),
                                         dtype=int, count=self.num_sequences)
        return self._lengths

    @property
    def max_length(self):
        """Length of the longest sequence contained in the graph."""
        if self._max_length is None:
            self._max_length = self.lengths.max()
        return self._max_length

    @property
    def value(self):
        """
        Total value of the graph, ie. sum of all edge weights.

        Only inter-sequence edges are counted, because only those can remain
        uncut after alignment.
        """

        if self._value is None:
            self._value = sum(m.sum() for m in self._pair_pp.flat if m is not None)
        return self._value

    @property
    def starts(self):
        """
        An array of start offsets, where the graph sequences start in the whole
        sequences.
        """
        if self._starts is None:
            self._starts = zeros(self.num_sequences, dtype=int)
        return self._starts

    @property
    def ends(self):
        """
        An array of end offsets, where the graph sequences end in the whole
        sequences.
        """
        if self._ends is None:
            self._ends = self.lengths
        return self._ends

    @property
    def is_column(self):
        """Describes whether the graph contains a single alignment column."""
        return self.max_length <= 1

    @property
    def nonzero(self):
        """Number of non-zero length sequences."""
        return (self.lengths > 0).sum()

    @property
    def records(self):
        if self._records is None and self._parent is not None:
            self.__slice_self()
        return self._records

    @property
    def partition(self):
        """An array of indices that bisect the graph."""
        if self._parts is None:
            if self._rel_parts is None:
                self._partition()
            self._parts = self.starts + self._rel_parts
        return self._parts

    def relax(self, iterations_left=0, weighted_transform=False):
        """Do a consistency transformation on the graph."""
        if iterations_left < 0:
            raise ValueError

        pair_pp = self._pair_pp
        self.__pair_pp = new_pair_pp = empty_like(pair_pp)
        numprocs = self._params['processes']

        cut = self._params['cut']
        # for every pair of sequences
        n = self.num_sequences
        weights=[[1]*n for i in xrange(n)]
        if weighted_transform:
            for x in xrange(n):
                for y in xrange(x + 1, n):
                    weights[x][y]=pair_pp[x,y].sum()/min(pair_pp[x,y].shape)

        if numprocs > 1:
            import multiprocessing

            xylist=sum([[(x, y) for y in xrange(x + 1, n)] for x in xrange(n)], [])
            chunksize = int(math.ceil(len(xylist)/float(numprocs)))
            procs=[]
            outq=multiprocessing.Queue()

            def _mprelax(x, y):
                sum_w = weights[x][y]*2
                new_pp_xy = pair_pp[x, y] * sum_w

                for z in xrange(n):
                    if z is x or z is y:
                        continue
                    if z < x:
                        relax_ZX_ZY(pair_pp[z, x], pair_pp[z, y], new_pp_xy, weights[z][x]*weights[z][y])
                        sum_w += weights[z][x]*weights[z][y]
                    elif x < z < y:
                        relax_XZ_ZY(pair_pp[x, z], pair_pp[z, y], new_pp_xy, weights[x][z]*weights[z][y])
                        sum_w += weights[x][z]*weights[z][y]
                    else:
                        relax_XZ_YZ(pair_pp[x, z], pair_pp[y, z], new_pp_xy, weights[x][z]*weights[y][z])
                        sum_w += weights[x][z]*weights[y][z]

                # normalize
                new_pp_xy.data /= sum_w
                # cut-off
                prune(new_pp_xy, cut)
                # x i y potrzebne, do prawidlowego przyporzadkowania pozniej
                return new_pp_xy, x, y

            def worker(xylist, outq):
                outl=[]
                for x, y in xylist:
                    outl.append(_mprelax(x, y))
                outq.put(outl)

            for i in range(numprocs):
                p=multiprocessing.Process(target=worker, args=(xylist[chunksize*i:chunksize*(i+1)], outq))
                procs.append(p)
                p.start()

            results=[]
            for i in range(numprocs):
                results += outq.get()
            for p in procs:
                p.join()
            for new_pp_xy, x, y in results:
                new_pair_pp[x, y] = new_pp_xy
        else:
            for x in xrange(n):
                for y in xrange(x + 1, n):
                    sum_w = weights[x][y]*2
                    new_pp_xy = pair_pp[x, y] * sum_w

                    for z in xrange(n):
                        if z is x or z is y:
                            continue
                        if z < x:
                            relax_ZX_ZY(pair_pp[z, x], pair_pp[z, y], new_pp_xy, weights[z][x]*weights[z][y])
                            sum_w += weights[z][x]*weights[z][y]
                        elif x < z < y:
                            relax_XZ_ZY(pair_pp[x, z], pair_pp[z, y], new_pp_xy, weights[x][z]*weights[z][y])
                            sum_w += weights[x][z]*weights[z][y]
                        else:
                            relax_XZ_YZ(pair_pp[x, z], pair_pp[y, z], new_pp_xy, weights[x][z]*weights[y][z])
                            sum_w += weights[x][z]*weights[y][z]

                    # normalize
                    new_pp_xy.data /= sum_w
                    # cut-off
                    prune(new_pp_xy, cut)

                    new_pair_pp[x, y] = new_pp_xy



    def slice(self, starts, ends):
        """
        Return a slice of the graph from starts to ends.

        args:
            starts      an array of indices in the sequences where they are to
                        be cut from the left
            ends        an array of indices in the sequences where they are to
                        be cut from the right
        """
        slice = type(self)(self,
                           _params=self._params,
                           _starts=starts,
                           _ends=ends)

        return slice

    def split(self, parts=None):
        """
        Split the graph into two along parts.

        args:
            parts       an array of indices in the sequences where they are to
                        be cut
        """

        if parts is None:
            parts = self.partition
        else:
            parts = array(parts, dtype=int)

        return (self.slice(self.starts, parts),
                self.slice(parts, self.ends))

    def _slice_records(self, starts, ends):
        records = []
        for i, r in enumerate(self.records):
            seq = r.seq[starts[i]:ends[i]] if ends[i] - starts[i] > 0 else EMPTY_SEQ
            records.append(SeqRecord(seq, id=r.id, name=r.name, description=r.description))
        return tuple(records)

    def _slice_weights(self, starts, ends):
        weights = empty(self.num_sequences, dtype=ndarray)
        for i, w in enumerate(self._weights):
            weights[i] = w[starts[i]:ends[i]]
        return weights

    def _slice_pair_pp(self, starts, ends):
        pair_pp = empty((self.num_sequences, self.num_sequences), dtype=object)
        lengths = ends - starts
        for i in xrange(self.num_sequences):
            len_i = lengths[i]
            for j in xrange(i + 1, self.num_sequences):
                pair_pp_ij = self._pair_pp[i, j]
                assert (len(self.records[i].seq), len(self.records[j].seq)) == pair_pp_ij.shape
                len_j = lengths[j]
                if len_i > 0 < len_j:
                    M, N = pair_pp_ij.shape
                    i0, i1 = starts[i], ends[i]
                    j0, j1 = starts[j], ends[j]
                    indptr, indices, data\
                        = get_csr_submatrix(M, N,
                                            pair_pp_ij.indptr,
                                            pair_pp_ij.indices,
                                            pair_pp_ij.data,
                                            int(i0), int(i1), int(j0), int(j1))
                    pair_pp[i, j] = csr_matrix((data,indices,indptr),
                                               shape=(i1 - i0, j1 - j0))
                else:
                    pair_pp[i, j] = empty((len_i, len_j), pair_pp_ij.dtype)
                assert pair_pp[i, j].shape == (len_i, len_j)
        return pair_pp

    def __slice_self(self):
        starts = self.starts - self._parent.starts
        ends = self.ends - self._parent.starts
        self._records = self._parent._slice_records(starts, ends)
        self.__weights = self._parent._slice_weights(starts, ends)
        self.__pair_pp = self._parent._slice_pair_pp(starts, ends)
        self._parent = None

    @property
    def _weights(self):
        """Weights of graph nodes (symbols)."""
        if self.__weights is None:
            if self._parent is not None:
                self.__slice_self()
            else:
                w = empty(self.num_sequences, dtype=ndarray)
                for i, r in enumerate(self.records):
                    w[i] = ones(len(r.seq), dtype=int)
                self.__weights = w
        return self.__weights

    @property
    def _pair_pp(self):
        """Posterior probabilities that symbols in sequences align."""
        if self.__pair_pp is None:
            if self._parent is not None:
                self.__slice_self()
            else:
                self.__calc_pp()
        return self.__pair_pp

    def __calc_pp(self):
        """Calculates the posterior probabilities for the input sequences."""
        if self.records is None:
            raise ValueError("No sequences in graph")

        params = {'verbose': self._verbose, 'processes': self._params['processes']}
        if 'temperature' in self._params:
            params['temperature'] = self._params['temperature']
        if 'opengap' in self._params:
            params['opengap'] = self._params['opengap']
        if 'extendgap' in self._params:
            params['extendgap'] = self._params['extendgap']
        if 'endgap' in self._params:
            params['endgap'] = self._params['endgap']
        if 'type' in self._params:
            params['type'] = self._params['type']
        if 'matrix' in self._params:
            params['matrix'] = self._params['matrix']
        if 'set' in self._params:
            params['set'] = self._params['set']
        if 'cut' in self._params:
            params['cut'] = self._params['cut']
        self.__pair_pp, sequence_type = pairwise(self.records, **params)
        self._params['type'] = sequence_type

    def _coarsen(self):
        lengths = self.lengths
        coarse_lengths = (lengths + 1) / 2

        for i, l in enumerate(coarse_lengths):
            if lengths[i] == 2:
                coarse_lengths[i] = 2

        weights = self._weights
        coarse_weights = empty_like(weights)
        for i, w in enumerate(weights):
            coarse_weights[i] = coarsen_weights(w)

        pair_pp = self._pair_pp
        coarse_pair_pp = empty_like(pair_pp)
        n = self.num_sequences
        for x in xrange(n):
            for y in xrange(x + 1, n):
                coarse_pair_pp[x, y] = coarsen_pair_pp(pair_pp[x, y])

        uncoarse = type(self)(self.records,
                            _lengths=self._lengths,
                            _starts=self._starts,
                            _ends=self._ends,
                            _Graph__weights=self.__weights,
                            _Graph__pair_pp=self.__pair_pp)
        self._coarse.append(uncoarse)

        self._lengths = coarse_lengths
        self.__weights = coarse_weights
        self.__pair_pp = coarse_pair_pp

        self._max_length = None

    def _loop_coarsen(self):
        while self.max_length > 2:
            self._coarsen()

    def _uncoarsen(self):
        uncoarse = self._coarse.pop()

        coarse_lengths = self._lengths
        lengths = self._lengths = uncoarse.lengths
        self.__weights = uncoarse._weights
        self.__pair_pp = uncoarse._pair_pp

        if self._rel_parts is not None:
            parts = empty_like(self._rel_parts)
            for i, p in enumerate(self._rel_parts):
                if coarse_lengths[i] == lengths[i]:
                    parts[i] = p
                elif p == coarse_lengths[i]:
                    parts[i] = lengths[i]
                else:
                    parts[i] = p + p
            self._rel_parts = parts

        self._max_length = None

    def _loop_uncoarsen(self):
        while self._coarse:
            self._uncoarsen()
            self._refine()

    def _refine(self):
        gp = GraphPartitioner(self)
        if self._rel_parts is None:
            raise ValueError("Initial partition is None")
        self._rel_parts = gp.partition(self._rel_parts)

    def _partition(self):
        if self.size == self.max_length:
            self.__partition_equal()
        elif ('multilevel' in self._params) and self._params['multilevel']:
            self.__partition_multilevel()
        else:
            self.__partition_singlelevel()

    def __partition_equal(self):
        self._rel_parts = self.lengths / 2

    def __partition_initial(self):
        #from numpy.random import rand

        #self._rel_parts = (self.lengths * rand(self.num_sequences)).astype(int)
        #self._rel_parts = zeros_like(self.lengths)
        #self._rel_parts = self.lengths
        self.__partition_equal()

    def __partition_singlelevel(self):
        self.__partition_initial()
        self._refine()

    def __partition_multilevel(self):
        self._loop_coarsen()
        self.__partition_singlelevel()
        self._loop_uncoarsen()

    def project(self, group1, group2, column_idxs1, column_idxs2, map):
        mappings = empty(self.num_sequences, dtype=object)
        for i in group1:
            mappings[i] = map[i, column_idxs1].nonzero()[0]
        for i in group2:
            mappings[i] = map[i, column_idxs2].nonzero()[0]

        pp = project(group1, group2, mappings, self._pair_pp)
        return pp
