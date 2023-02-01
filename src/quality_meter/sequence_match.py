class SequenceMatch:
    def __init__(self, test_sequence, indices):
        self.test_sequence = test_sequence
        indices.sort(reverse=True)
        self.indices = indices

    def get_test_sequence(self):
        return self.test_sequence

    def get_indices(self):
        return self.indices

    def get_max_index(self):
        return self.indices[0]