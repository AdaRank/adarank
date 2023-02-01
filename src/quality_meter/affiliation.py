
from src.quality_meter.quality_measure import QualityMeasure


class Affiliation(QualityMeasure):

    def __init__(self, y, y_hat, max_val):
        super().__init__(y, y_hat, max_val)


    def compute_quality(self):
        """
        y consists of all sequences which were created by replacing a random event
        y_hat consists of all sequences which should be created
        :return:
        """
        return len([x for x,y in zip(self.y,self.y_hat) if x == y])/len(self.y_hat)

