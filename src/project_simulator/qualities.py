import logging

from src.enums.selection_methods import SelectionMethods
from src.quality_meter.div import Div
from src.quality_meter.hr import Hr
from src.quality_meter.hrsingle import HRSingle
from src.quality_meter.map import Map
from src.quality_meter.mrr import Mrr
from src.quality_meter.mrrsingle import MrrSingle
from src.quality_meter.ndcg import Ndcg
from src.quality_meter.ndcgsingle import NdcgSingle
from src.quality_meter.recallsingle import RecallSingle

logger = logging.getLogger(__name__)


class Qualities:

    def __init__(self, y):
        self.mrr = Mrr(y)
        self.map = Map(y)
        self.ndcg = Ndcg(y)
        self.div = Div(y=None)
        self.hr = Hr(y)
        self.hr_single = HRSingle(y)
        self.recall_single = RecallSingle(y)
        self.mrr_single = MrrSingle(y)
        self.ndcg_single = NdcgSingle(y)
        self.cancelled_recommendations = 0
        self.successful_recommendations = 0

    def compute_qualities(self, selection_method):
        #logger.info("Computing mrr")
        if len(self.mrr.qualities) > 0:
            self.mrr.compute_quality()
        else:
            self.mrr.mean_quality = 0

        #logger.info("Computing map")

        if len(self.map.qualities) > 0:
            self.map.compute_quality()
        else:
            self.map.mean_quality = 0

        #logger.info("Computing hr")
        if len(self.hr.qualities) > 0:
            self.hr.compute_quality()
        else:
            self.hr.mean_quality = 0

        #logger.info("Computing hr_single")
        if len(self.hr_single.qualities) > 0:
            self.hr_single.compute_quality()
        else:
            self.hr_single.mean_quality = 0

        #logger.info("Computing recall single")
        if len(self.recall_single.qualities) > 0:
            self.recall_single.compute_quality()
        else:
            self.recall_single.mean_quality = 0

        #logger.info("Computing mrr single")
        if len(self.mrr_single.qualities) > 0:
            self.mrr_single.compute_quality()
        else:
            self.mrr_single.mean_quality = 0

        #logger.info("Computing ndcg single")
        if len(self.ndcg_single.qualities) > 0:
            self.ndcg_single.compute_quality()
        else:
            self.ndcg_single.mean_quality = 0

        #logger.info("Computing div")
        if len(self.div.qualities) > 0:
            self.div.compute_quality()
            if selection_method != SelectionMethods.RANDOM or selection_method != SelectionMethods.FIRST:
                self.div.compute_inter_div()
        else:
            self.div.mean_quality = 0

        if len(self.ndcg.qualities) > 0:
            self.ndcg.compute_quality()
        else:
            self.ndcg.mean_quality = 0
        logger.info("Finished computing qualities")

