from unittest import TestCase
from numpy import zeros
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from clusteralign.sequence import GAPPED_ALPHABET
from clusteralign.probA import align, pairwise
from clusteralign.tests.testing import to_dense, assert_array_equal, assert_array_almost_equal


class AlignTest(TestCase):
    records_same = (
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
    )
    records_offset = (
        SeqRecord(Seq("KPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYI", GAPPED_ALPHABET), id="1aab_"),
    )
    records_repeat = (
        SeqRecord(Seq("KPRG", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("KPRGKPRGKPRG", GAPPED_ALPHABET), id="1aab_ repeat"),
    )
    records_diff = (
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK", GAPPED_ALPHABET), id="1j46_A"),
    )
    defaults = {
        "temperature": 0.01,
        "set": 160
    }

    def _align(self, records, **kwargs):
        pairs, params = align(records[0], records[1], **dict(self.defaults, **kwargs))
        return pairs

    def test_align_same(self):
        pairs = self._align(self.records_same)
        expected = zeros((83, 83), dtype=float)
        expected[range(83), range(83)] = 1
        assert_array_almost_equal(pairs, expected)

    def test_align_offset(self, **kwargs):
        pairs = self._align(self.records_offset, **kwargs)
        expected_pairs = zeros((77, 78), dtype=float)
        expected_pairs[range(77 - 5),
                       range(6, 78)] = 1
        assert_array_almost_equal(pairs, expected_pairs)

    def test_align_repeat(self):
        pairs = self._align(self.records_repeat)
        expected_pairs = zeros((4, 12), dtype=float)
        expected_pairs[range(4) * 3, range(12)] = 1. / 3
        assert_array_almost_equal(pairs, expected_pairs)

    def test_align_reverse(self):
        pairs = self._align(self.records_diff)
        pairs_r = self._align(tuple(reversed(self.records_diff)))
        assert_array_almost_equal(pairs, pairs_r.transpose())

    def test_align_params(self):
        self.assertRaises(AssertionError, self.test_align_offset, temperature=100)
        self.assertRaises(AssertionError, self.test_align_offset, set=40)
        self.assertRaises(AssertionError, self.test_align_offset, matrix="blosum")

    def test_align_cut(self):
        MIN = 0
        MAX = 0.1

        def is_cut(ar):
            return ((ar > MIN) & (ar < MAX)).any()

        pairs0 = self._align(self.records_diff, cut=MIN)
        pairs0 = to_dense(pairs0)
        self.assertTrue(is_cut(pairs0))

        pairs1 = self._align(self.records_diff, cut=MAX)
        pairs1 = to_dense(pairs1)
        self.assertFalse(is_cut(pairs1))

        def uncut(ar):
            return (ar <= MIN) | (ar >= MAX)

        assert_array_equal(pairs0[uncut(pairs0)], pairs1[uncut(pairs0)])

        pairs0[pairs0 < MAX] = 0
        assert_array_equal(pairs0, pairs1)


class PairwiseTest(TestCase):
    records_same = (
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
    )
    records_diff = (
        SeqRecord(Seq("GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTYIPPKGE", GAPPED_ALPHABET), id="1aab_"),
        SeqRecord(Seq("MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK", GAPPED_ALPHABET), id="1j46_A"),
        SeqRecord(Seq("MKKLKKHPDFPKKPLTPYFRFFMEKRAKYAKLHPEMSNLDLTKILSKKYKELPEKKKMKYIQDFQREKQEFERNLARFREDHPDLIQNAKK", GAPPED_ALPHABET), id="1k99_A"),
        SeqRecord(Seq("MHIKKPLNAFMLYMKEMRANVVAESTLKESAAINQILGRRWHALSREEQAKYYELARKERQLHMQLYPGWSARDNYGKKKKRKREK", GAPPED_ALPHABET), id="2lef_A"),
    )
    defaults = {
        "temperature": 0.02,
        "set": 160,
        "cut": 0
    }

    def _pairwise(self, records):
        alignments, sequence_type = pairwise(records, **self.defaults)
        return alignments

    def test_pairwise_same(self):
        pairs = self._pairwise(self.records_same)
        assert_array_equal(pairs[0, 1], pairs[0, 2])
        assert_array_equal(pairs[0, 1], pairs[1, 2])

    def test_pairwise_shape(self):
        pairs = self._pairwise(self.records_diff)
        for i in range(len(self.records_diff)):
            for j in range(i + 1, len(self.records_diff)):
                self.assertEqual(pairs[i, j].shape, (len(self.records_diff[i]), len(self.records_diff[j])))
            for j in range(i + 1):
                self.assertEqual(pairs[i, j], None)
