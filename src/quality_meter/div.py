import collections
import itertools
import logging
import streamlit as st

import pandas as pd

from src.quality_meter.quality_measure import QualityMeasure

logger = logging.getLogger(__name__)


class Div(QualityMeasure):
    def __init__(self, y):
        super().__init__(y)
        self.mean_inter_div = None
        self.recommendations = []

    def add_sequence(self, sequence=None, recommendations: pd.Series = None):
        # print(f"Number of combinations: {len(list(itertools.combinations(recommendations, r=2)))}")
        divs = []
        self.recommendations.append(recommendations)
        if len(recommendations) > 1:
            for comb in itertools.combinations(recommendations, r=2):
                divs.append(1 - (len(self.intersection(comb[0], comb[1])) / len(self.union(comb[0], comb[1]))))
            self.qualities.append(sum(divs) / len(divs))
        # print(self.mean_divs)

    def compute_quality(self):
        self.mean_quality = sum(self.qualities) / len(self.qualities)

    @staticmethod
    def union(lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    @staticmethod
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    def reset_values(self):
        self.qualities = []
        self.mean_quality = None
        self.mean_inter_div = None

    @staticmethod
    def is_equal(l1, l2):
        if len(l1) != len(l2):
            return False
        return collections.Counter(l1) == collections.Counter(l2)

    def compute_inter_div(self):
        inter_divs = []
        #logger.info(self.recommendations)
        for succ_a, succ_b in zip(self.recommendations, self.recommendations[1:]):
            rec_comb_equality = []
            for x, y in zip(succ_a, succ_b):
                rec_comb_equality.append(self.is_equal(x, y))
            inter_divs.append(rec_comb_equality.count(True) / len(rec_comb_equality))

        #for i, comb in enumerate(itertools.combinations(self.recommendations, r=2)):
        #    rec_comb_equality = []
        #    for x in comb[0]:
        #        for y in comb[1]:
        #            rec_comb_equality.append(self.is_equal(x, y))
        #    inter_divs.append(rec_comb_equality.count(True) / len(rec_comb_equality))

        if len(inter_divs) > 0:
            self.mean_inter_div = sum(inter_divs) / (len(inter_divs))
        else:
            self.mean_inter_div = 0

