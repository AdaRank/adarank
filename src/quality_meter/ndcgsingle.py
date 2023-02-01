import math

from src.quality_meter.quality_measure import QualityMeasure
from src.quality_meter.sequence_match import SequenceMatch

class NdcgSingle(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)

    def add_sequence(self, sequence, recommendations):
        rels = []
        irels = []
        for rank, rec in enumerate(recommendations):
            if all(x in sequence[-len(rec):] for x in rec):
                rels.append(1 / math.log2((rank + 1) + 1))
            irels.append(1 / math.log2((rank + 1) + 1))
        # Divide sum of discounted cumulative gains by ideal discounted cumulative gains
        # (IDCG = every item in recommendation list is relevant)
        self.qualities.append(sum(rels)/sum(irels))

    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)



