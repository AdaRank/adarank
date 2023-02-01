from src.quality_meter.quality_measure import QualityMeasure
import collections

is_equal_list = lambda x, y: collections.Counter(x) == collections.Counter(y)

class PrecisionNovelty(QualityMeasure):

    def __init__(self, y, y_hat, max_val):
        super().__init__(y, y_hat, max_val)


    def compute_quality(self):
        """
        Computes the ratio of retrieved and relevant items (precision) and the ratio of unknown and retrieved items (novelty)
        :return:
        """
        nr_retrieved = len(self.y)
        matches = 0
        nr_novel = 0
        for l1 in self.y:
            found = False
            for l2 in self.y_hat:
                if is_equal_list(l1, l2):
                    matches += 1
                    found = True
                    break
            if not found:
                nr_novel += 1
        return matches/nr_retrieved, nr_novel/nr_retrieved