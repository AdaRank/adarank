import utils
from src.quality_meter.quality_measure import QualityMeasure
from src.quality_meter.sequence_match import SequenceMatch

class Map(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)

    def add_sequence(self, sequence, recommendations):
        #print(f"Checking a possible next item for this sequence: {sequence}")
        # Search for matching sequences in test data (antecedents has to match)
        filtered_gt = self.filter_test_sequences(sequence)  # filter for matching prefixes
        temp_aps = []
        for s in filtered_gt:
            ps = []
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
                    ps.append((len(ps)+1)/(rank+1))
                    #print(f"Adding new ps {(len(ps)+1)/(rank+1)}")
                    #print(ps)
                    #print("Is in suffix and can be counted as relevant recommendation")
            if len(ps) > 0:
                temp_aps.append(sum(ps)/len(ps))
            else:
                temp_aps.append(0)
        if len(temp_aps) > 0:
            self.qualities.append(max(temp_aps))
        else:
            self.qualities.append(0)


    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)
