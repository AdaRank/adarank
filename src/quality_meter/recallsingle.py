from src.quality_meter.quality_measure import QualityMeasure


class RecallSingle(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)

    def add_sequence(self, sequence, recommendations):
        hit = 0
        for rec in recommendations:
            if all(x in sequence[-len(rec):] for x in rec):
                hit = 1
                break
        self.qualities.append(hit)


    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)
