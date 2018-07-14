from __future__ import print_function
from sys import stdout
from copy import deepcopy
from collections import defaultdict

from numpy import empty, array, logical_not, uint8
from numpy.random import randint
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO import parse as seq_parse
from Bio.AlignIO import parse as align_parse, write as align_write

from .sequence import SPACE, SPACES, GAPPED_ALPHABET
from .msf import file_read_msf, file_write_msf
from .graph import Graph
from .probA import TYPE_DNA
from ._ppalign import align_probabilities, remap

CONSISTENCY = 2
REFINE = 100

try:
    from Bio.Align import MultipleSeqAlignment
except ImportError:
    from Bio.Align.Generic import Alignment as GenericAlignment

    class MultipleSeqAlignment(GenericAlignment):
        def __init__(self, records, alphabet):
            super(MultipleSeqAlignment, self).__init__(alphabet)
            for r in records:
                self.add_sequence(r.id, str(r.seq))


def tomsa(records):
    """
    Creates a BioPython MultipleSeqAlignment object from a list of SeqRecords.
    Falls back to Generic.Alignment object in older versions of BioPython.
    """
    for r in records:
        if not isinstance(r, SeqRecord):
            raise ValueError("Records parameter is not a list of SeqRecord objects")

    return MultipleSeqAlignment(records, GAPPED_ALPHABET)

def work(gp):
    p, g = gp
    l, r = g.split()
    return (p+'0', l), (p+'1', r)

class FakeLock(object):
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass

class Alignment(object):
    """Creates and can output an alignment from an input file."""
    def __init__(self, file=None, format="fasta", consistency=CONSISTENCY, refine=REFINE, verbose=False, _lock=FakeLock(), **kwargs):
        if file is None:
            raise ValueError("File parameter is None")

        self._verbose = verbose
        self._lock = _lock

        if verbose:
            print("reading sequences ...", end=' ')

        self.file = file
        with open(file, "rU") as f:
            with _lock:
                self._records = tuple(seq_parse(f, format))

        if verbose:
            print("done")
            for r in self._records:
                print("  %s" % r.id)
                for j in xrange(0, len(r.seq), 76):
                    print("    %s" % r.seq[j:j + 76])
            print()

        self._params = kwargs
        self.__graph = Graph(self._records, verbose=verbose, _params=kwargs)
        self.__relaxed = 0
        self._consistency = consistency
        self._refinements = refine
        self._columns = None
        self._map = None
        self.processes = kwargs.get('processes', 1)

        self._alignment = None

        self._cuts = []
        self._cut_levels = defaultdict(int)

    def __str__(self):
        return self.alignment

    @property
    def _graph(self):
        if self._verbose:
            # precompute posterior probabilities for verbose output order
            self.__graph._pair_pp
        if self.__relaxed < self._consistency:
            self._relax()
        return self.__graph

    @property
    def alignment(self):
        """
        An BioPython MultipleSeqAlignment object containing the alignment of
        the sequences contained in this object.
        """
        if self._alignment is None:
            if self._map is None:
                if self._columns is not None:
                    self.__map_columns()
                else:
                    self._map = self._align(self._graph)
            self._refine_each()
            if self._refinements:
                self._refine()
            assert self._map.shape[1] > 0, "Alignment has no columns"
            records = deepcopy(self._records)
            for i, record in enumerate(records):
                seq = record.seq
                aligned_seq = []
                map = self._map[i]
                index = 0
                for symbol in map:
                    if symbol:
                        aligned_seq.append(seq[index])
                        index += 1
                    else:
                        aligned_seq.append(SPACE)
                record.seq = Seq("".join(aligned_seq), GAPPED_ALPHABET)
            self._alignment = tomsa(records)
        return self._alignment

    @alignment.setter
    def alignment(self, alignment):
        columns = []
        starts = self._graph.starts
        for i in xrange(alignment.get_alignment_length()):
            ends = starts.copy()
            for j, r in enumerate(alignment):
                if r.seq[i] not in SPACES:
                    ends[j] += 1
            column = self._graph.slice(starts, ends)
            columns.append(column)
            starts = ends

        self._columns = columns
        self._alignment = alignment

    def _write(self, f, format="msf"):
        # trigger calculations before acquiring lock
        self.alignment

        if format == "msf":
            with self._lock:
                file_write_msf(self.alignment, f,
                               infile=self.file, outfile=f.name,
                               prot=self._params['type'] is not TYPE_DNA,
                               endgap=self._params['endgap'])
        else:
            with self._lock:
                align_write(self.alignment, f, format)

    def _read(self, f, format="msf"):
        if format == "msf":
            with self._lock:
                self.alignment = file_read_msf(f)
        else:
            with self._lock:
                self.alignment = align_parse(f, format).next()

    def output(self, *args, **kwargs):
        self._write(stdout, *args, **kwargs)

    def write(self, file, *args, **kwargs):
        with open(file, "w") as f:
            self._write(f, *args, **kwargs)

    def read(self, file, *args, **kwargs):
        with open(file, "r") as f:
            self._read(f, *args, **kwargs)

    @property
    def value(self):
        if self._columns is None:
            if self._map is None:
                self.alignment
            self.__unmap_columns()
        # self._columns is a list of Graph instances
        return sum(c.value for c in self._columns)

    def _relax(self):
        """Run the consistency transformation."""
        if self._consistency:
            if self._verbose:
                print("performing consistency transformation ...")

            for count, i in enumerate(reversed(xrange(self._consistency - self.__relaxed))):
                if self._verbose:
                    print("  iteration", count + 1, "...", end=' ')
                self.__graph.relax(i, self._params['weightrans'])
                self.__relaxed += 1
                if self._verbose:
                    print("done")

            if self._verbose:
                print()

    def _align(self, graph_or_starts, ends=None):
        """Align the sequences."""
        if isinstance(graph_or_starts, Graph):
            graph = graph_or_starts
        else:
            graph = self._graph.slice(graph_or_starts, ends)

        if self.processes > 1:
            from multiprocessing import Pool
            from operator import itemgetter
            pool = Pool(self.processes)
            graphs = [('', graph)]
            map_columns = []

            if self._verbose:
                print("partitioning graph ...", end="")
                line_position = 22

            while graphs:
                if self._verbose and len(graphs) == 1:
                    print(".", end="")
                    line_position += 1
                    if line_position >= 80:
                        print(end="\n  ")
                        line_position = 2

                resultslist=pool.map(work, graphs)

                graphs=[]
                for l, r in resultslist:
                    if r[1].is_column:
                        map_columns.append((r[0], r[1].lengths))
                    else:
                        graphs.append(r)
                    if l[1].is_column:
                        map_columns.append((l[0], l[1].lengths))
                    else:
                        graphs.append(l)

            pool.close()
            map_columns.sort(key=itemgetter(0))
            map_columns=[g[1] for g in map_columns]
        else:
            graphs = [graph]
            map_columns = []

            if self._verbose:
                print("partitioning graph ...", end="")
                line_position = 22

            while graphs:
                g = graphs.pop()

                if self._verbose and len(graphs) == 1:
                    print(".", end="")
                    line_position += 1
                    if line_position >= 80:
                        print(end="\n  ")
                        line_position = 2

                # Graph depicts a single column, add to columns
                if g.is_column:
                    map_columns.append(g.lengths)
                    continue

                # Iterativly partition the graph
                l, r = g.split()
                if graph is self._graph:
                    self._cuts.append((g.partition, g))
                    if l not in self._cut_levels:
                        self._cut_levels[l] = self._cut_levels[g] + 1
                    if r not in self._cut_levels:
                        self._cut_levels[r] = self._cut_levels[g] + 1
                graphs.append(r)
                graphs.append(l)

        if self._verbose:
            if line_position < 75:
                print(" done")
            else:
                print("\n  done")
            print()

        map = empty((len(self._records), len(map_columns)), dtype=bool)
        for i, c in enumerate(map_columns):
            map[:, i] = c
        return map

    def _refine_each(self):
        """Refine the alignment."""
        self._alignment = None

        if self._verbose:
            print("realigning sequences ...")

        partitions = defaultdict(int)
        num_records = len(self._records)
        for c in self._map.transpose():
            c_sum = c.sum()
            if c_sum == 1 or c_sum == num_records - 1:
                if c[0]:
                    c = logical_not(c)
                partitions[tuple(c)] += 1

        for count, partition in sorted(zip(partitions.values(), partitions.keys()), reverse=True):
            partition = array(partition)
            group1 = partition.nonzero()[0]
            group2 = logical_not(partition).nonzero()[0]
            self.__realign(group1, group2)

        if self._verbose:
            print()

    def _refine(self):
        self._alignment = None

        if self._verbose:
            print("refining alignment ...")

        for i in xrange(self._refinements):
            self.__realign(*self.__group())

        if self._verbose:
            print()

    def __map_columns(self):
        columns = self._columns
        if columns is None:
            self._map = None
        else:
            map = empty((len(self._records), len(columns)), dtype=bool)
            for i, c in enumerate(columns):
                map[:, i] = c.lengths
            self._map = map
            self._columns = None

    def __unmap_columns(self):
        new_columns = []

        starts = self._graph.starts.copy()
        ends = starts.copy()
        for c in self._map.transpose():
            ends += c
            new_columns.append(self._graph.slice(starts, ends))
            starts = ends
            ends = starts.copy()

        self._columns = new_columns
        self._map = None

    def __group(self):
        while True:
            random = randint(0, 2, len(self._records))
            random = random.astype(bool)
            g1, g2 = random.nonzero()[0], logical_not(random).nonzero()[0]
            if len(g1) and len(g2):
                return g1, g2

    def __realign(self, group1, group2):
        if self._verbose:
            print("  %s with %s ..." % (", ".join(self._records[i].id for i in group1),
                                        ", ".join(self._records[i].id for i in group2)),
                  end=" ")

        # project groups
        def project(group):
            group_map = self._map[group]
            column_idxs = group_map.any(axis=0)
            return column_idxs.nonzero()[0]

        column_idxs1 = project(group1)
        column_idxs2 = project(group2)
        matrix = self._graph.project(group1, group2,
                                     column_idxs1, column_idxs2,
                                     self._map)

        # align matrix
        new_column_idxs1, new_column_idxs2 = align_probabilities(matrix)

        if self._verbose:
            print("done")

        # un-project alignment
        if ((column_idxs1 == new_column_idxs1).all()
            and (column_idxs2 == new_column_idxs2).all()):
            return False

        new_map = remap(self._map.view(uint8),
                        group1, group2,
                        column_idxs1, column_idxs2,
                        new_column_idxs1, new_column_idxs2)

        self._map = new_map
        return True
