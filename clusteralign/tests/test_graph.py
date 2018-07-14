from unittest import TestCase
from numpy import identity, hstack, vstack
from scipy.sparse import issparse
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from clusteralign.graph import Graph
from clusteralign.sequence import GAPPED_ALPHABET
from clusteralign.tests.testing import assert_array_equal, assert_array_almost_equal


class GraphTest(TestCase):
    records = (
        SeqRecord(Seq("TLRP", GAPPED_ALPHABET), id="seq 1"),
        SeqRecord(Seq("YCIKP", GAPPED_ALPHABET), id="seq 2"),
        SeqRecord(Seq("YLRP", GAPPED_ALPHABET), id="seq 3"),
        SeqRecord(Seq("ALRP", GAPPED_ALPHABET), id="seq 4")
    )
    records_l = (
        SeqRecord(Seq("TLR", GAPPED_ALPHABET), id="seq 1"),
        SeqRecord(Seq("YCIK", GAPPED_ALPHABET), id="seq 2"),
        SeqRecord(Seq("YLR", GAPPED_ALPHABET), id="seq 3"),
        SeqRecord(Seq("ALR", GAPPED_ALPHABET), id="seq 4")
    )
    records_r = (
        SeqRecord(Seq("P", GAPPED_ALPHABET), id="seq 1"),
        SeqRecord(Seq("P", GAPPED_ALPHABET), id="seq 2"),
        SeqRecord(Seq("P", GAPPED_ALPHABET), id="seq 3"),
        SeqRecord(Seq("P", GAPPED_ALPHABET), id="seq 4")
    )
    records_4 = (
        SeqRecord(Seq("P", GAPPED_ALPHABET), id="seq 1"),
        SeqRecord(Seq("", GAPPED_ALPHABET), id="seq 2"),
        SeqRecord(Seq("", GAPPED_ALPHABET), id="seq 3"),
        SeqRecord(Seq("P", GAPPED_ALPHABET), id="seq 4")
    )
    records_same = (
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKG", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKG", GAPPED_ALPHABET), id="1aab_"),
    )
    records_offset = (
        SeqRecord(Seq("KPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYI", GAPPED_ALPHABET), id="1aab_"),
    )
    records_partition = (
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
    params = {
        "temperature": 0.02,
        "set": 160,
    }

    def setUp(self):
        self.graph = Graph(self.records)
        self.graph_l = Graph(self.records_l)
        self.graph_r = Graph(self.records_r)
        self.graph_4 = Graph(self.records_4)
        self.graph_same = Graph(self.records_same, _params=self.params)
        self.graph_offset = Graph(self.records_offset, _params=self.params)
        self.graph_partition = Graph(self.records_partition, _params=self.params)
        self.graph_diff = Graph(self.records_diff, _params=dict(self.params,
                                                                cut=0))

    def test_len(self):
        self.assertEquals(len(self.graph), 17)
        self.assertEquals(len(self.graph_l), 13)
        self.assertEquals(len(self.graph_r), 4)
        self.assertEquals(len(self.graph_4), 2)

    def test_str(self):
        self.assertEquals(str(self.graph), 'TLRP YCIKP YLRP ALRP')
        self.assertEquals(str(self.graph_l), 'TLR YCIK YLR ALR')
        self.assertEquals(str(self.graph_r), 'PPPP')
        self.assertEquals(str(self.graph_4), 'P--P')

    def test_size(self):
        self.assertEquals(self.graph.size, 17)
        self.assertEquals(self.graph_l.size, 13)
        self.assertEquals(self.graph_r.size, 4)
        self.assertEquals(self.graph_4.size, 2)

    def test_num_sequences(self):
        self.assertEqual(self.graph.num_sequences, 4)
        self.assertEqual(self.graph_l.num_sequences, 4)
        self.assertEqual(self.graph_r.num_sequences, 4)
        self.assertEqual(self.graph_4.num_sequences, 4)

    def test_lengths(self):
        assert_array_equal(self.graph.lengths, [4, 5, 4, 4])
        assert_array_equal(self.graph_l.lengths, [3, 4, 3, 3])
        assert_array_equal(self.graph_r.lengths, [1, 1, 1, 1])
        assert_array_equal(self.graph_4.lengths, [1, 0, 0, 1])

    def test_max_length(self):
        self.assertEqual(self.graph.max_length, 5)
        self.assertEqual(self.graph_l.max_length, 4)
        self.assertEqual(self.graph_r.max_length, 1)
        self.assertEqual(self.graph_4.max_length, 1)

    def test_value(self):
        self.assertAlmostEqual(self.graph_same.value, 82)
        self.assertAlmostEqual(self.graph_offset.value, 72)

    def test_starts(self):
        assert_array_equal(self.graph.starts, [0, 0, 0, 0])
        assert_array_equal(self.graph_l.starts, [0, 0, 0, 0])
        assert_array_equal(self.graph_r.starts, [0, 0, 0, 0])
        assert_array_equal(self.graph_4.starts, [0, 0, 0, 0])

    def test_ends(self):
        assert_array_equal(self.graph.ends, [4, 5, 4, 4])
        assert_array_equal(self.graph_l.ends, [3, 4, 3, 3])
        assert_array_equal(self.graph_r.ends, [1, 1, 1, 1])
        assert_array_equal(self.graph_4.ends, [1, 0, 0, 1])

    def test_is_column(self):
        self.assertFalse(self.graph.is_column)
        self.assertFalse(self.graph_l.is_column)
        self.assertTrue(self.graph_r.is_column)
        self.assertTrue(self.graph_4.is_column)

    def test_records(self):
        self.assertEqual(self.graph.records, self.records)
        self.assertEqual(self.graph_l.records, self.records_l)
        self.assertEqual(self.graph_r.records, self.records_r)
        self.assertEqual(self.graph_4.records, self.records_4)

    def test_partition(self):
        assert_array_equal(self.graph_same.partition, [41, 41])
        assert_array_equal(self.graph_partition.partition, [27, 31, 35, 39])

    def test_partition_multilevel(self):
        self.graph_partition._params['multilevel'] = False
        self.graph_partition._partition()
        p_s = self.graph_partition._rel_parts
        self.graph_partition._rel_parts = None
        self.graph_partition._params['multilevel'] = True
        self.graph_partition._partition()
        p_m = self.graph_partition._rel_parts
        assert_array_equal(p_s, p_m)

    def test_relax(self):
        graph = self.graph_diff
        records = graph.records
        pair_pp = graph._pair_pp
        graph.relax()
        new_pair_pp = graph._pair_pp

        def get_pair_pp(x, y, pair_pp):
            if x == y:
                return identity(len(records[x]), float)
            elif y > x:
                pp = pair_pp[x, y]
            else:
                pp = pair_pp[y, x].transpose()
            if issparse(pp):
                return pp.todense()
            return pp

        def to_full(pair_pp):
            return vstack(tuple(hstack(tuple(get_pair_pp(x, y, pair_pp) for y in xrange(len(records)))) for x in xrange(len(records))))

        def simple_relax(pair_pp):
            relaxed = pair_pp.dot(pair_pp) / len(records)
            #sum = 0
            #for r in records:
            #    relaxed[sum:sum + len(r), sum:sum + len(r)] = 0.
            #    sum += len(r)
            relaxed[pair_pp == 0.] = 0.
            relaxed[range(relaxed.shape[0]), range(relaxed.shape[1])] = 1.
            #relaxed[relaxed < cut] = 0  # cut = 0
            return relaxed

        assert_array_almost_equal(to_full(new_pair_pp),
                                  simple_relax(to_full(pair_pp)))

    def test_split_left(self):
        g_l, g_r = self.graph.split(self.graph.starts)
        self.assertEqual('----', str(g_l))
        self.assertEqual(str(self.graph), str(g_r))

    def test_split_right(self):
        g_l, g_r = self.graph.split(self.graph.ends)
        self.assertEqual(str(self.graph), str(g_l))
        self.assertEqual('----', str(g_r))

    def test_split(self):
        g_l, g_r = self.graph.split(self.graph_l.ends)

        self.assertEqual(str(self.graph_l), str(g_l))
        self.assertEqual(str(self.graph_r), str(g_r))

        assert_array_equal(g_l.starts, [0, 0, 0, 0])
        assert_array_equal(g_l.ends, self.graph_l.ends)
        assert_array_equal(g_r.starts, self.graph_l.ends)
        assert_array_equal(g_r.ends, self.graph.ends)

    def test_split_pairs(self):
        g_l, g_r = self.graph_diff.split([25, 25, 25, 25])

        p = self.graph_diff._pair_pp
        p_l = g_l._pair_pp
        p_r = g_r._pair_pp
        l = p.shape[0]
        for i in range(l):
            for j in range(i + 1, l):
                assert_array_equal(p_l[i, j], p[i, j][:25, :25])
                assert_array_equal(p_r[i, j], p[i, j][25:, 25:])
            for j in range(i + 1):
                self.assertEqual(p[i, j], None)
                self.assertEqual(p_l[i, j], None)


class GraphTestSparse(GraphTest):
    params = dict(GraphTest.params, sparse=True)
