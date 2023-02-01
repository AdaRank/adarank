from abc import abstractmethod

from src.quality_meter.sequence_match import SequenceMatch


class QualityMeasure:

    def __init__(self, y=None):
        self.qualities = []
        self.mean_quality = None
        self.y = y

    def filter_test_sequences(self, sequence):
        matches = []
        for s in self.y:
            contains_all = True
            indices = []
            for item in sequence:
                try:
                    found_index = s.index(item)
                    if found_index not in indices:
                        indices.append(found_index)
                    else:
                        raise ValueError
                except ValueError as ve:
                    contains_all = False
            if contains_all:
                matches.append(SequenceMatch(s, indices))
        return matches


    def reset_values(self):
        self.qualities = []
        self.mean_quality = None

    @abstractmethod
    def add_sequence(self, sequence, recommendations):
        ...

    @abstractmethod
    def compute_quality(self):
        ...


