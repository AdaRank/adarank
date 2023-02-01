from src.quality_meter.quality_measure import QualityMeasure


class Hr(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)

    def add_sequence(self, sequence, recommendations):
        filtered_gt = self.filter_test_sequences(sequence)  # filter for matching prefixes
        temp_hrs = []
        for s in filtered_gt:
            for rec in recommendations:
                hits = 0
                if len(s.get_test_sequence()[s.get_max_index() + 1:]) > 0:
                    for x in rec:
                        if x in s.get_test_sequence()[s.get_max_index() + 1:]:
                            hits += 1
                temp_hrs.append(hits/len(recommendations))

        if len(temp_hrs) > 0:
            self.qualities.append(max(temp_hrs))
        else:
            self.qualities.append(0)


    def add_sequence_succ(self, new_sequence, gt_sequence, recommendations):
        hits = 0
        for rec in recommendations:
            if all(x in gt_sequence[len(new_sequence):] for x in rec):
                hits += 1
        self.qualities.append(hits / len(recommendations))

    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)
