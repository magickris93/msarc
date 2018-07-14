from Bio.Alphabet import Gapped, SingleLetterAlphabet
from Bio.Seq import Seq

SPACES = ["-", ".", " ", "~"]
SPACE = SPACES[0]
MSF_SPACE = SPACES[1]
MSF_TERMINAL_SPACE = SPACES[3]

GAPPED_ALPHABET = Gapped(SingleLetterAlphabet(), SPACE)


class EmptySeq(Seq):
    def __init__(self):
        super(EmptySeq, self).__init__(SPACE, GAPPED_ALPHABET)

    def __len__(self):
        return 0

    def __asseq(self):
        return Seq(str(self), self.alphabet)

    def __add__(self, other):
        return self.__asseq().__add__(other)

    def __radd__(self, other):
        return self.__asseq().__radd__(other)

EMPTY_SEQ = EmptySeq()
