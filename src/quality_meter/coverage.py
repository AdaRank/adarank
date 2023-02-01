import collections
import itertools

from src.quality_meter.quality_measure import QualityMeasure

is_equal_list = lambda x, y: collections.Counter(x) == collections.Counter(y)

class Coverage(QualityMeasure):

    def __init__(self, y, y_hat, max_val):
        super().__init__(y, y_hat, max_val)

    def compute_quality(self):
        """
        Computes the ratio of relevant items which were retrieved. Duplicates get deleted first.
        :return:
        """
        nr_relevant = len(self.y_hat)

        # remove duplicate sequences from list of sequences
        self.y.sort()
        self.y = list(self.y for self.y, _ in itertools.groupby(self.y))

        matches = 0
        for l1 in self.y:
            for l2 in self.y_hat:
                if is_equal_list(l1, l2):
                    matches += 1
                    break
        return matches/nr_relevant