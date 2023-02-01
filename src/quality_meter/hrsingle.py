from src.quality_meter.quality_measure import QualityMeasure


class HRSingle(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)


    def add_sequence(self, sequence, recommendations):
        hits = 0
        for rec in recommendations:
            if all(x in sequence[-len(rec):] for x in rec):
                hits += 1
        self.qualities.append(hits/len(recommendations))


    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)
