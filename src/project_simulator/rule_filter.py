"""
__author__ = "Ludwig Zellner"
__copyright__ = "Copyright (C) 2022 Ludwig Zellner"
"""

import itertools
import pandas as pd
import streamlit as st
import constants
import utils
from src.enums.filter import Filter
from src.enums.rule import Rule


class RuleFilter:
    """
    1. Return top-k rules (cut after kth position)
    2. Return maximal pattern (consequent) if rule is relevant (antecedent matches)
    3. Return closed pattern (consequent) if rule is relevant (antecedent matches)
    """
    def __init__(self, rules, filt):
        self.rules = rules
        self.filter = filt
        #self.filtered_rules = None
        self.rel_div_aft = None

    def run(self):
        # filter rules by number (sorted by quality measure e.g. support or confidence)

        if self.filter == Filter.MAX_CONS:
            st.write("Filtering for maximal patterns on consequent")
            self.filter_patterns(self.filter_maximal, side=Rule.ANTECEDENT)
        elif self.filter == Filter.GEN_ANT:
            st.write("Filtering for generator patterns on antecedent")
            self.filter_patterns(self.filter_generators, side=Rule.CONSEQUENT)
        elif self.filter == Filter.GEN_MAX_COMB:  # does not work yet, index missing when applying iloc
            st.write("Filtering for maximal patterns and generator patterns (consequent and antecedent respectively)")
            self.filter_patterns(self.filter_maximal, side=Rule.ANTECEDENT)
            self.rules = self.rules.reset_index()
            self.filter_patterns(self.filter_generators, side=Rule.CONSEQUENT)
        filtered_rules = self.rules
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    logger.info(f"Rules after filtering: {rules.sort_values(['support', 'confidence'], ascending=False)}")
        self.rel_div_aft = utils.get_relative_diversity(filtered_rules['consequent'].tolist())
        #self.filtered_rules = filtered_rules
        return filtered_rules



    def filter_antecedent_supersets(self):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.rules)
        by_consequent = self.rules.groupby('consequent')
        for consequent, frame in by_consequent:
            print(f"First 2 entries for {consequent}")
            print("------------------------")
            print(frame.head(2), end="\n\n")
        return self.rules


    def filter_patterns(self, func, side):
        """
           Search for equal patterns in "side"

           :param func: decides which pruning strategy is applied (closed, maximal, generator)
           :param side: decides on which side of a rule is checked for commonalities
           :return: return indexes of dataframe which remain
       """
        c = []
        found = []
        for i1, row1 in self.rules.iterrows():
            if i1 not in found:
                t = [i1]
                for i2, row2 in self.rules.iterrows():
                    if i1 < i2 and row1[side] == row2[side]:
                        t.append(i2)
                        found.append(i2)
                c.append(t)
        filter_side = Rule.CONSEQUENT
        if side==Rule.CONSEQUENT:
            filter_side = Rule.ANTECEDENT
        return func(c, filter_side)


    def filter_generators(self, common_patterns, side):
        keep_indices = set()
        for common_pattern in common_patterns:
            if len(common_pattern) == 1:
                keep_indices.add(common_pattern[0] - 1)
            if len(common_pattern) > 1:
                for ind1, ind2 in list(itertools.combinations(common_pattern, 2)):
                    row1 = self.rules.iloc[ind1 - 1]
                    row2 = self.rules.iloc[ind2 - 1]
                    cons1 = set(row1[side])
                    cons2 = set(row2[side])
                    if cons2.issubset(cons1) and row2["support"] == row1["support"]:
                        keep_indices.add(ind2 - 1)
                        if ind1 - 1 in keep_indices:
                            keep_indices.remove(ind1 - 1)
                    elif cons1.issubset(cons2) and row1["support"] == row2["support"]:
                        keep_indices.add(ind1 - 1)
                        if ind2 - 1 in keep_indices:
                            keep_indices.remove(ind2 - 1)
        self.rules = self.rules.iloc[list(keep_indices)]


    def filter_maximal(self, common_patterns, side):
        keep_indices = set()
        for common_pattern in common_patterns:
            if len(common_pattern) == 1:
                keep_indices.add(common_pattern[0] - 1)
            if len(common_pattern) > 1:
                for ind1, ind2 in list(itertools.combinations(common_pattern, 2)):
                    cons1 = set(self.rules.iloc[ind1 - 1][side])
                    cons2 = set(self.rules.iloc[ind2 - 1][side])
                    if cons1.issubset(cons2):
                        keep_indices.add(ind2 - 1)
                        if ind1 - 1 in keep_indices:
                            keep_indices.remove(ind1 - 1)
                    elif cons2.issubset(cons1):
                        keep_indices.add(ind1 - 1)
                        if ind2 - 1 in keep_indices:
                            keep_indices.remove(ind2 - 1)
        self.rules = self.rules.iloc[list(keep_indices)]

    def filter_closed(self, common_patterns, side):
        keep_indices = set()
        for common_pattern in common_patterns:
            if len(common_pattern) == 1:
                keep_indices.add(common_pattern[0] - 1)
            if len(common_pattern) > 1:
                for ind1, ind2 in list(itertools.combinations(common_pattern, 2)):
                    row1 = self.rules.iloc[ind1 - 1]
                    row2 = self.rules.iloc[ind2 - 1]
                    cons1 = set(row1[side])
                    cons2 = set(row2[side])
                    if cons1.issubset(cons2) and row2["support"] == row1["support"]:
                        keep_indices.add(ind2 - 1)
                        if ind1 - 1 in keep_indices:
                            keep_indices.remove(ind1 - 1)
                    elif cons2.issubset(cons1) and row1["support"] == row2["support"]:
                        keep_indices.add(ind1 - 1)
                        if ind2 - 1 in keep_indices:
                            keep_indices.remove(ind2 - 1)
        self.rules = self.rules.iloc[list(keep_indices)]


