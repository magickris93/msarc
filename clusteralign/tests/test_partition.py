from unittest import TestCase
from copy import deepcopy
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from numpy import array, arange, empty, zeros_like
from clusteralign.sequence import GAPPED_ALPHABET
from clusteralign.graph import Graph
from clusteralign.partition import GraphPartitioner
from clusteralign._partition import calcgains_group, fixgains_group
from clusteralign.tests.testing import assert_array_equal, assert_array_almost_equal


class GraphPartitionerTest(TestCase):
    records = (
        SeqRecord(Seq("TLRP", GAPPED_ALPHABET), id="seq 1"),
        SeqRecord(Seq("YCIKP", GAPPED_ALPHABET), id="seq 2"),
        SeqRecord(Seq("YLRP", GAPPED_ALPHABET), id="seq 3"),
        SeqRecord(Seq("ALRP", GAPPED_ALPHABET), id="seq 4")
    )
    records_same = (
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKG", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKG", GAPPED_ALPHABET), id="1aab_"),
    )
    records_offset = (
        SeqRecord(Seq("SSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYI", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("RGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREM", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("PKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARY", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKAD", GAPPED_ALPHABET), id="1aab_"),
    )
    records_diff = (
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK", GAPPED_ALPHABET), id="1j46_A"),
        SeqRecord(Seq("MKKLKKHPDFPKKPLTPYFRFFMEKRAKYAKLHPEMSNLDLTKILSKKYKELPEKKKMKYIQDFQREKQEFERNLARFREDHPDLIQNAKK", GAPPED_ALPHABET), id="1k99_A"),
        SeqRecord(Seq("MHIKKPLNAFMLYMKEMRANVVAESTLKESAAINQILGRRWHALSREEQAKYYELARKERQLHMQLYPGWSARDNYGKKKKRKREK", GAPPED_ALPHABET), id="2lef_A"),
    )
    records_synthetic = (
        SeqRecord(Seq("ABCD", GAPPED_ALPHABET), id="seq 1"),
        SeqRecord(Seq("EFGH", GAPPED_ALPHABET), id="seq 2"),
    )
    params = {
        "temperature": 0.02,
        "set": 160
    }

    def setUp(self):
        self.graph = Graph(self.records)
        self.graph_same = Graph(self.records_same, _params=self.params)
        self.graph_offset = Graph(self.records_offset, _params=self.params)
        self.graph_diff = Graph(self.records_diff, _params=dict(self.params, temperature=10., cut=0))

        self.graph_synthetic = Graph(self.records_synthetic)
        self.graph_synthetic._pair_pp[0, 1] = arange(41, 57).reshape(4, 4)

        self.partitioner = GraphPartitioner(self.graph)
        self.partitioner_same = GraphPartitioner(self.graph_same)
        self.partitioner_offset = GraphPartitioner(self.graph_offset)
        self.partitioner_diff = GraphPartitioner(self.graph_diff)
        self.partitioner_synthetic = GraphPartitioner(self.graph_synthetic)

    def test_numgroups(self):
        self.assertEqual(self.partitioner._numgroups, 4)
        self.assertEqual(self.partitioner_same._numgroups, 2)
        self.assertEqual(self.partitioner_offset._numgroups, 4)
        self.assertEqual(self.partitioner_diff._numgroups, 4)

    def test_lengths(self):
        assert_array_equal(self.partitioner._lengths, [4, 5, 4, 4])
        assert_array_equal(self.partitioner_same._lengths, [82, 82])
        assert_array_equal(self.partitioner_offset._lengths, [66, 66, 66, 66])
        assert_array_equal(self.partitioner_diff._lengths, [83, 85, 91, 86])

    def test_move(self):
        self.partitioner._parts = array([0, 0, 0, 0])
        self.partitioner._move([(0, 1),
                                (0, 0),
                                (1, 1),
                                (0, 1),
                                (1, 2),
                                (0, 0)])
        assert_array_equal(self.partitioner._parts, [0, 2, 0, 0])

    def test_movepartition(self):
        self.partitioner._parts = array([4, 0, 0, 4])
        initial_parts = self.partitioner._parts.copy()
        initial_balance = deepcopy(self.partitioner._balance_weight)
        parts = self.partitioner._parts = array([0, 5, 2, 2])
        gains = self._calcgains(self.partitioner)
        self.partitioner._GraphPartitioner__balance_weight = None
        balance = self.partitioner._balance_weight
        self.partitioner._parts = initial_parts
        self.partitioner._GraphPartitioner__balance_weight = initial_balance
        for step in [(0, 0),
                     (1, 5),
                     (2, 2),
                     (3, 2)]:
            self.partitioner._movepartition(*step)
        parts_many = self.partitioner._parts
        gains_many = self.partitioner._gains
        balance_many = self.partitioner._balance_weight
        assert_array_equal(parts, parts_many)
        for i in xrange(self.partitioner._numgroups):
            assert_array_almost_equal(gains[i] / 2, gains_many[i])
        assert_array_equal(balance, balance_many)

    def test_balance_weight(self):
        self.partitioner._parts = array([0, 0, 0, 0])
        assert_array_equal(self.partitioner._balance_weight, [0, self.partitioner._lengths.sum()])

        self.partitioner._parts = array([1, 4, 3, 2])
        # reset property
        self.partitioner._GraphPartitioner__balance_weight = None
        assert_array_equal(self.partitioner._balance_weight, [10, self.partitioner._lengths.sum() - 10])

        self.partitioner._movepartition(0, 0)
        assert_array_equal(self.partitioner._balance_weight, [9, self.partitioner._lengths.sum() - 9])
        self.partitioner._movepartition(3, 3)
        assert_array_equal(self.partitioner._balance_weight, [10, self.partitioner._lengths.sum() - 10])

        self.partitioner._parts = array([1, 4, 3, 2])
        self.partitioner._graph._Graph__weights = array([array([1, 2, 3, 4]),
                                                         array([5, 6, 7, 8, 9]),
                                                         array([10, 11, 12, 13]),
                                                         array([14, 15, 16, 17])])
        # reset property
        self.partitioner._GraphPartitioner__balance_weight = None
        assert_array_equal(self.partitioner._balance_weight, [1 + (5 + 6 + 7 + 8) + (10 + 11 + 12) + (14 + 15),
                                                              (2 + 3 + 4) + 9 + 13 + (16 + 17)])

    def test_calcgain(self):
        cut__0_0 = 0
        cut__0_1 = - (41 + 45 + 49 + 53)
        cut__1_0 = - (41 + 42 + 43 + 44)
        cut__1_1 = - (42 + 43 + 44 + 45 + 49 + 53)
        cut__1_2 = - (43 + 44 + 45 + 46 + 49 + 50 + 53 + 54)
        cut__2_1 = - (42 + 43 + 44 + 46 + 47 + 48 + 49 + 53)
        cut__2_2 = - (43 + 44 + 47 + 48 + 49 + 50 + 53 + 54)

        self.partitioner_synthetic._parts = array([0, 0])
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(1, 1), cut__0_1 - cut__0_0)
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(0, 1), cut__1_0 - cut__0_0)

        self.partitioner_synthetic._parts = array([0, 1])
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(1, 1), cut__0_1 - cut__0_0)
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(0, 1), cut__1_1 - cut__0_1)

        self.partitioner_synthetic._parts = array([1, 0])
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(1, 1), cut__1_1 - cut__1_0)
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(0, 1), cut__1_0 - cut__0_0)

        self.partitioner_synthetic._parts = array([1, 1])
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(1, 1), cut__1_1 - cut__1_0)
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(0, 1), cut__1_1 - cut__0_1)
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(1, 2), cut__1_2 - cut__1_1)
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(0, 2), cut__2_1 - cut__1_1)

        self.partitioner_synthetic._parts = array([1, 2])
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(1, 2), cut__1_2 - cut__1_1)
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(0, 2), cut__2_2 - cut__1_2)

        self.partitioner_synthetic._parts = array([2, 1])
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(1, 2), cut__2_2 - cut__2_1)
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(0, 2), cut__2_1 - cut__1_1)

        self.partitioner_synthetic._parts = array([2, 2])
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(1, 2), cut__2_2 - cut__2_1)
        self.assertAlmostEqual(self.partitioner_synthetic._calcgain(0, 2), cut__2_2 - cut__1_2)

    def _calcgain_all(self, partitioner, i):
        length = partitioner._lengths[i] + 1
        return array([partitioner._calcgain(i, j) for j in xrange(length)])

    def test_calcgain_all(self):
        def _test_calcgain_all(parts):
            self.partitioner_diff._parts = parts

            for i in xrange(self.partitioner_diff._numgroups):
                a1 = self._calcgain_all(self.partitioner_diff, i)
                a2 = self.partitioner_diff._calcgain_all(i)
                assert_array_almost_equal(a1 / 2, a2)

        _test_calcgain_all(array([0, 0, 0, 0]))
        _test_calcgain_all(array([20, 20, 1, 1]))

    def test_calcgain_cython(self):
        def _test_calcgain_all(parts):
            self.partitioner_diff._parts = parts

            for i in xrange(self.partitioner_diff._numgroups):
                a1 = calcgains_group(self.partitioner_diff, i)
                a2 = self.partitioner_diff._calcgain_all(i)
                assert_array_almost_equal(a1, a2)

        _test_calcgain_all(array([0, 0, 0, 0]))
        _test_calcgain_all(array([20, 20, 1, 1]))

    def _fixgain_all(self, partitioner, i, group, partition):
        length = partitioner._lengths[i] + 1
        return array([partitioner._fixgain(i, j, group, partition) for j in xrange(length)])

    def test_fixgain(self):
        self.partitioner_diff._parts = array([20, 20, 1, 1])
        # trigger initial gains calculation
        self.partitioner_diff._GraphPartitioner__gains = None
        self.partitioner_diff._gains

        def _test_fixgains(g, n):
            for i in xrange(self.partitioner_diff._numgroups):
                if i == g:
                    continue
                a1 = self.partitioner_diff._fixgain_all(i, g, n)
                a2 = self._fixgain_all(self.partitioner_diff, i, g, n)
                assert_array_almost_equal(a1, a2 / 2)

        # forwards ...
        for i in range(len(self.records_diff)):
            _test_fixgains(i, self.partitioner_diff._parts[i])

        # ... and backwards
        for i in range(len(self.records_diff)):
            _test_fixgains(i, self.partitioner_diff._parts[i] - 1)

    def test_fixgain_cython(self):
        self.partitioner_diff._parts = array([20, 20, 1, 1])
        # trigger initial gains calculation
        self.partitioner_diff._GraphPartitioner__gains = None
        self.partitioner_diff._gains

        def _test_fixgains(g, n):
            for i in xrange(self.partitioner_diff._numgroups):
                if i == g:
                    continue
                a1 = self.partitioner_diff._fixgain_all(i, g, n)
                a2 = zeros_like(a1)
                fixgains_group(self.partitioner_diff, i, g, n, a2)
                assert_array_almost_equal(a1, a2)

        # forwards ...
        for i in range(len(self.records_diff)):
            _test_fixgains(i, self.partitioner_diff._parts[i])

        # ... and backwards
        for i in range(len(self.records_diff)):
            _test_fixgains(i, self.partitioner_diff._parts[i] - 1)

    def test_fixgains_single(self):
        self.partitioner_diff._parts = array([0, 0, 0, 0])

        # trigger initial gains calculation
        self.partitioner_diff._GraphPartitioner__gains = None
        self.partitioner_diff._gains

        def _test_fixgains_single(g, n):
            initial_parts = self.partitioner_diff._parts.copy()
            self.partitioner_diff._parts[g] = n
            gains = self._calcgains(self.partitioner_diff)
            parts = self.partitioner_diff._parts

            self.partitioner_diff._parts = initial_parts
            self.partitioner_diff._movepartition(g, n)
            gains_many = self.partitioner_diff._gains
            parts_many = self.partitioner_diff._parts

            for i in xrange(self.partitioner_diff._numgroups):
                assert_array_almost_equal(gains[i] / 2, gains_many[i])
            assert_array_equal(parts, parts_many)

        # forwards ...
        for i in range(1, 11):
            _test_fixgains_single(0, i)
            _test_fixgains_single(1, i)

        # ... and backwards
        for i in reversed(range(10)):
            _test_fixgains_single(0, i)
            _test_fixgains_single(1, i)

    def _calcgains(self, partitioner):
        g = empty(partitioner._numgroups, dtype=object)

        for i in xrange(partitioner._numgroups):
            g[i] = self._calcgain_all(partitioner, i)

        return g

    def test_fixgains_multiple(self):
        self.partitioner_diff._parts = array([0, 0, 0, 0])

        # trigger initial gains calculation
        self.partitioner_diff._GraphPartitioner__gains = None
        initial_parts = self.partitioner_diff._parts.copy()
        initial_gains = deepcopy(self.partitioner_diff._gains)

        def _test_fixgains_multiple(g, n):
            self.partitioner_diff._parts[g] = n
            gains = self._calcgains(self.partitioner_diff)
            parts = self.partitioner_diff._parts

            self.partitioner_diff._parts = initial_parts.copy()
            self.partitioner_diff._GraphPartitioner__gains = deepcopy(initial_gains)
            self.partitioner_diff._movepartition(g, n)
            gains_many = self.partitioner_diff._gains
            parts_many = self.partitioner_diff._parts

            for i in xrange(self.partitioner_diff._numgroups):
                assert_array_almost_equal(gains[i] / 2, gains_many[i])
            assert_array_equal(parts, parts_many)

        # forwards ...
        for i in range(1, 11):
            _test_fixgains_multiple(0, i)

        initial_parts = self.partitioner_diff._parts.copy()
        initial_gains = deepcopy(self.partitioner_diff._gains)

        # ... and backwards
        for i in reversed(range(10)):
            _test_fixgains_multiple(0, i)

    def test_fixgains(self):
        self.partitioner_diff._parts = array([0, 0, 0, 0])

        # trigger initial gains calculation
        self.partitioner_diff._GraphPartitioner__gains = None
        self.partitioner_diff._gains

        def _test_fixgains(g, n):
            self.partitioner_diff._movepartition(g, n)

            gains = self.partitioner_diff._gains.copy()
            gains_proper = self.partitioner_diff._calcgains()
            for i in xrange(self.partitioner_diff._numgroups):
                assert_array_almost_equal(gains[i], gains_proper[i])

        # forwards ...
        for i in range(1, len(self.records_diff[0].seq) + 1):
            _test_fixgains(0, i)
            _test_fixgains(1, i)

        # ... and backwards
        for i in reversed(range(len(self.records_diff[0].seq))):
            _test_fixgains(0, i)
            _test_fixgains(1, i)

    def test_move_gains(self):
        self.partitioner_diff._parts = array([10, 10, 10, 10])

        def assert_moves(moves):
            parts = self.partitioner_diff._parts.copy()
            gains = deepcopy(self.partitioner_diff._gains)
            gain = 0.

            for group, node in moves[:-1]:
                if node > self.partitioner_diff._parts[group]:
                    gain += self.partitioner_diff._gains[group][node]
                else:
                    gain -= self.partitioner_diff._gains[group][self.partitioner_diff._parts[group]]
                self.partitioner_diff._movepartition(group, node)
                self.assertRaises(AssertionError, assert_array_equal, parts, self.partitioner_diff._parts)
                self.assertRaises(AssertionError, self.assertAlmostEqual, gain, 0)

            group, node = moves[-1]
            if node > self.partitioner_diff._parts[group]:
                gain += self.partitioner_diff._gains[group][node]
            else:
                gain -= self.partitioner_diff._gains[group][self.partitioner_diff._parts[group]]
            self.partitioner_diff._movepartition(group, node)
            self.assertNotEqual(id(parts), id(self.partitioner_diff._parts))
            assert_array_equal(parts, self.partitioner_diff._parts)
            for i in range(len(self.records_diff)):
                self.assertNotEqual(id(gains[i]), id(self.partitioner_diff._gains[i]))
                assert_array_almost_equal(gains[i], self.partitioner_diff._gains[i])
            self.assertAlmostEqual(gain, 0)

        assert_moves([(0, 11),
                      (0, 10)])
        assert_moves([(0, 11), (1, 11),
                      (1, 10), (0, 10)])
        assert_moves([(0, 11), (1, 11),
                      (0, 10), (1, 10)])
        assert_moves([(0, 11), (1, 11), (2, 11), (3, 11),
                      (3, 10), (2, 10), (1, 10), (0, 10)])
        assert_moves([(0, 11), (1, 11), (2, 11), (3, 11),
                      (0, 10), (1, 10), (2, 10), (3, 10)])
        assert_moves([(0, 11), (1, 11), (0, 12), (1, 12),
                      (1, 11), (0, 11), (1, 10), (0, 10)])
        assert_moves([(0, 11), (1, 11), (0, 12), (1, 12),
                      (0, 11), (1, 11), (0, 10), (1, 10)])
        assert_moves([(0, 11), (1, 11), (2, 11), (3, 11), (0, 12), (1, 12), (2, 12), (3, 12),
                      (3, 11), (2, 11), (1, 11), (0, 11), (3, 10), (2, 10), (1, 10), (0, 10)])
        assert_moves([(0, 11), (1, 11), (2, 11), (3, 11), (0, 12), (1, 12), (2, 12), (3, 12),
                      (0, 11), (1, 11), (2, 11), (3, 11), (0, 10), (1, 10), (2, 10), (3, 10)])

    def test_partition(self):
        assert_array_equal(self.partitioner_same.partition([0, 0]), [41, 41])
        assert_array_equal(self.partitioner_offset.partition([0, 0, 0, 0]), [27, 31, 35, 39])
