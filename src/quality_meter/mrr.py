from src.quality_meter.quality_measure import QualityMeasure
from src.quality_meter.sequence_match import SequenceMatch
import utils

class Mrr(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)
        self.number_sequences = 0

    def add_sequence(self, sequence, recommendations):
        self.number_sequences += 1
        #print(f"Checking a possible next item for this sequence: {sequence}")
        filtered_gt = self.filter_test_sequences(sequence) # filter for matching prefixes
        #if len(filtered_gt) == 0:
        #    print("No match")
        for s in filtered_gt:
            #print(s.get_test_sequence(), s.get_indices())
            #print(f"All recommendations: {recommendations}")
            for rank, rec in enumerate(recommendations):
                #print(rank, rec)
                #if len(s.get_test_sequence()[s.get_max_index()+1:]) > 0 and all(x in s.get_test_sequence()[s.get_max_index()+1:] for x in rec): # rec can be a batch of items that's why "all" is used
                # Following code should be used if one wants to check only the next few elements
                # (number of "next elements" is defined by number of elements in recommendation
                # Recommendation can be a batch of items that's why "all" is used
                if len(s.get_test_sequence()[s.get_max_index()+1:]) > 0 and all(x in s.get_test_sequence()[s.get_max_index()+1:(s.get_max_index()+1+len(rec))] for x in rec):  # rec can be a batch of items that's why "all" is used
                    #print(f"Adding new RRS {1/rank+1}")
                    self.qualities.append(1/(rank+1))
                    #print(self.rrs)
                    #print("Is in suffix and can be counted as relevant recommendation")
                    break
            break
        #print(self.rrs)

    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / self.number_sequences

    def reset_values(self):
        self.number_sequences = 0
        self.qualities = []
        self.mean_quality = None

