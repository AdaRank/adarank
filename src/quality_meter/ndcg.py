import math
from statistics import mean

from src.quality_meter.quality_measure import QualityMeasure
from src.quality_meter.sequence_match import SequenceMatch

class Ndcg(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)


    def add_sequence(self, sequence, recommendations):
        #print(f"Checking a possible next item for this sequence: {sequence}")
        # Search for matching sequences in test data (antecedents has to match)
        filtered_gt = self.filter_test_sequences(sequence)  # filter for matching prefixes
        #if len(filtered_gt) == 0:
            #print("No match")
        temp_ndcgs = []
        for s in filtered_gt:
            rels = []
            #print(s.get_test_sequence(), s.get_indices())
            #print(f"All recommendations:\n{recommendations}")
            for rank, rec in enumerate(recommendations):

                #if len(s.get_test_sequence()[s.get_max_index() + 1:]) > 0 and all(
                #        x in s.get_test_sequence()[s.get_max_index() + 1:] for x in
                #        rec): # rec can be a batch of items that's why "all" is used

                # Following code should be used if one wants to check only the next few elements
                # (number of "next elements" is defined by number of elements in recommendation
                # Recommendation can be a batch of items that's why "all" is used
                if len(s.get_test_sequence()[s.get_max_index() + 1:]) > 0 and all(
                       x in s.get_test_sequence()[s.get_max_index() + 1:(s.get_max_index() + 1+len(rec))] for x in rec):
                    rels.append(1/math.log2((rank+1)+1))
                else:
                    rels.append(0)
            if len(rels) > 0:
                temp_ndcgs.append(sum(rels)/len(rels))
            else:
                temp_ndcgs.append(0)
        if len(temp_ndcgs) > 0:
            self.qualities.append(max(temp_ndcgs))
        else:
            self.qualities.append(0)

    def add_sequence_succ(self, new_sequence, gt_sequence, recommendations):
        rels = []
        irels = []
        for rank, rec in enumerate(recommendations):
            if all(x in gt_sequence[len(new_sequence):] for x in rec):
                rels.append(1 / math.log2((rank + 1) + 1))
            irels.append(1 / math.log2((rank + 1) + 1))
        # Divide sum of discounted cumulative gains by ideal discounted cumulative gains
        # (every item in recommendation list is relevant)
        self.qualities.append(sum(rels) / sum(irels))

    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)



