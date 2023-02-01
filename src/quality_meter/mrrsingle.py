from src.quality_meter.quality_measure import QualityMeasure
from src.quality_meter.sequence_match import SequenceMatch
import utils

class MrrSingle(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)
        self.number_sequences = 0

    def add_sequence(self, sequence, recommendations):
        self.number_sequences += 1

        for rank, rec in enumerate(recommendations):
            if all(x in sequence[-len(rec):] for x in rec):
                self.qualities.append(1/(rank+1))
                break
            break

    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / self.number_sequences

    def reset_values(self):
        self.number_sequences = 0
        self.qualities = []
        self.mean_quality = None

