import random


class Weave:
    def __init__(self, sequences):
        self.sequences = sequences

    @property
    def shuffle_sequences(self):
        concat_sequences = list("".join(self.sequences))
        random.shuffle(concat_sequences)
        return "".join(concat_sequences)
