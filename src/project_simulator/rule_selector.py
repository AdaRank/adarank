import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity

import utils
from src.enums.selection_methods import SelectionMethods
import streamlit as st
import altair as alt


logger = logging.getLogger(__name__)

class RuleSelector:
    """
    1. Filters rules which antecedent matches to the given sequence that already exists
    2. Selects rule which consequent will be used to expand the given sequence by
        a) Sampling one rule randomly
        b) Taking the first rule by using the sorted list of rules (sort by support and confidence)
    * Implement dynamic ranking with discrete gaps
        - Number of gaps is required
    * Implement dynamic ranking with continuous gaps
        - Timestamps at which the events are activated are required
    Todo
    * Weight the rules to be applied by frequency with which the consequent occurs in the training set
    """
    def __init__(self, topk, bandwidth, rules, sequence=None, sequence_ts=None, method=None):
        self.method = method
        self.sequence = sequence
        self.sequence_ts = sequence_ts
        self.bandwidth = bandwidth
        self.rules = rules
        self.rules.reset_index(drop=True, inplace=True)
        self.stepwise_rules = None
        self.selection_size = []
        self.topk = topk
        if method == SelectionMethods.FIRST:
            self.rules = self.rules.nlargest(1, "support")
        elif method == SelectionMethods.NAIVE:
            self.rules = self.rules.nlargest(topk, "support")
        elif method == SelectionMethods.RANDOM or method == SelectionMethods.CGAP or method == SelectionMethods.DGAP:
            # Corresponds to shuffling rules and picking topk
            #self.rules = self.rules.sample(n=topk)
            # SelectionMethod == RANDOM: recommendation gets picked randomly
            # SelectionMethod == CGAP: recommendation gets picked by best result including timestamps
            # SelectionMethod == DGAP: recommendation gets picked by best result including gaps
            pass

    def __iter__(self):
        return self

    def __next__(self):
        #logger.info("Copying rules")
        if self.method == SelectionMethods.NAIVE or self.method == SelectionMethods.FIRST:
            self.stepwise_rules = self.rules.copy(deep=True)
        elif self.method == SelectionMethods.RANDOM:
            self.stepwise_rules = self.rules.copy(deep=True).sample(n=self.topk)
        elif (self.method == SelectionMethods.CGAP and len(self.sequence) > 0) or (self.method == SelectionMethods.DGAP and len(self.sequence) > 0):
            # filter rules which can be applied
            self.stepwise_rules = self.filter_matching_antecedents().copy(deep=True)
        elif (self.method == SelectionMethods.CGAP and len(self.sequence) == 0) or (self.method == SelectionMethods.DGAP and len(self.sequence) == 0):
            self.stepwise_rules = self.rules.copy(deep=True)

        self.selection_size.append(len(self.stepwise_rules.index))

        if self.method == SelectionMethods.DGAP:
            #logger.info("Ranking rules")
            row = self.dynamic_ranking_discrete()
            if row is None:
                return None
            return row["consequent"].values[0]
        elif self.method == SelectionMethods.CGAP:
            row = self.dynamic_ranking_continuous()
            if row is None:
                return None, None
            ts_list = row["cgaps"].values[0]
            return row["consequent"].values[0], (sum(ts_list)/len(ts_list))
        elif self.method == SelectionMethods.RANDOM or self.method == SelectionMethods.NAIVE or self.method == SelectionMethods.FIRST:
            if self.stepwise_rules.empty:
                return None
            return self.select_consequent(self.stepwise_rules)

    __call__ = __next__

    def filter_matching_antecedents(self):
        ind = []
        for index, row in self.rules.iterrows():
            a = row["antecedent"]
            if self.is_multi_subset(self.sequence, a):
                ind.append(index)
        return self.rules.iloc[ind]

    def select_consequent(self, rules):
        if self.method == SelectionMethods.RANDOM or self.method == SelectionMethods.NAIVE:
            #logger.info(f"Sampling on dataframe: {topk}")
            sample = rules.sample()
            #logger.info(f"Sampling following rule: {sample['rule']}")
            return sample["consequent"].values[0]
        elif self.method == SelectionMethods.FIRST:
            return rules.head(1)["consequent"].values[0]


    def dynamic_ranking_discrete(self):
        seq_len = len(self.sequence)
        ranking = dict()
        for index, row in self.stepwise_rules.iterrows():
            #logger.info(row)
            if seq_len > 0:
                indices = [i for i, x in enumerate(self.sequence) if x in row["antecedent"]]
            else:
                indices = [seq_len]
            ranking_values = [row["ms_dgaps"][seq_len - (i+1)] for i in indices if seq_len - (i+1) in row["ms_dgaps"]]
            if len(ranking_values) > 0:
                ranking[index] = max(ranking_values)
            #for index, row in self.stepwise_rules.iterrows():
            #    ms = self.create_multiset(row["dgaps"])
            #logging.info(ms)
            #source = pd.DataFrame({
            #    '|G|': ms.keys(),
            #    'Number of occurences': ms.values()
            #})
            #c = alt.Chart(source).mark_bar().encode(
            #    x='|G|',
            #    y='Number of occurences'
            #)
            #st.write(c)
        #logger.info("Sort ranking")
        sorted_ranking = sorted(ranking, key=ranking.get, reverse=True)
        #logger.info(f"Ranking: {sorted_ranking}")
        #df1 = self.stepwise_rules.set_index('Tm')
        #st.write("Ranking rules")
        #st.write(self.stepwise_rules)
        #logger.info("1")
        self.stepwise_rules = self.stepwise_rules.loc[sorted_ranking].head(self.topk)
        #st.write("Ranking rules")
        #st.write(self.stepwise_rules)
        #logger.info("2")
        max_indices = [key for key, value in ranking.items() if value == max(ranking.values())][:self.topk]
        #st.write(f"Max indices: {max_indices}")
        if len(max_indices) == 0:
            # no rule could be found -- abort simulation of current sequence
            return None
        #logger.info("3")
        min_index = min(max_indices)
        #st.write(f"Min index: {min_index}")
        #print(rules)
        return self.stepwise_rules.loc[[min_index]]

    def dynamic_ranking_continuous(self):
        ranking = dict()
        #print("Rules")
        #logger.debug(self.stepwise_rules)
        #print(f"Analysing sequence: {self.sequence} with ts: {self.sequence_ts}")
        #for i, el in enumerate(self.sequence):
        for index, row in self.stepwise_rules.iterrows():
            cgap = utils.retrieve_cgap(self.sequence, self.sequence_ts, row['antecedent'])
            ts_gaps = row["cgaps"]
            ts_arr = np.array(ts_gaps)
            if self.bandwidth:
                h = self.bandwidth
            else:
                # if bandwidth is not passed as a parameter we use Silverman's rule of thumb
                h = np.std(ts_arr) * (4 / 3 / len(ts_arr)) ** (1 / 5)
                if h == 0.0:
                    h = 1.0
            #st.write(f"Bandwidth={h}")
            # sns.kdeplot(ts_arr, bw=h)
            # plt.show()
            kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(ts_arr.reshape(-1, 1))
            prob = np.exp(kde.score_samples([[cgap.get_gap()]]))
            #st.write(cgap.get_gap())
            #X_plot = np.linspace(0, 2000000, 1000)[:, np.newaxis]
            #log_dens = kde.score_samples(X_plot)
            #fig, ax = plt.subplots()
            #ax.plot(
            #    X_plot[:, 0],
            #    np.exp(log_dens),
            #    color="blue",
            #    lw=2,
            #    linestyle="-",
            #    label="kernel = '{0}'".format("gaussian"),
            #)
            #st.pyplot(fig)
            #logging.info(f"Got probability: {prob[0]}")
            ranking[index] = prob[0]

        #logger.info(f"Ranking: {ranking}")
        sorted_ranking = sorted(ranking, key=ranking.get, reverse=True)
        #st.write("Ranking rules")
        #st.write(self.stepwise_rules)
        self.stepwise_rules = self.stepwise_rules.loc[sorted_ranking].head(self.topk)
        #st.write("Ranking rules")
        #st.write(self.stepwise_rules)
        max_indices = [key for key, value in ranking.items() if value == max(ranking.values())][:self.topk]
        #st.write(f"Max indices: {max_indices}")
        if len(max_indices) == 0:
            # no rule could be found -- abort simulation
            return None
        min_index = min(max_indices)
        #st.write(f"Min index: {min_index}")
        #print("Rules")
        #print(rules)
        #print(f"Select the following rule: {rules.loc[[min_index]]}")
        return self.stepwise_rules.loc[[min_index]]

    def is_multi_subset(self, l, to_check):
        d = utils.create_multiset(l)
        c = utils.create_multiset(to_check)
        matching = True
        for k, v in c.items():
            if k not in d:
                matching = False
                break
            else:
                if v > d[k]:
                    matching = False
                    break
        return matching



    def get_first_item(self):
        return self.rules.head(1)["consequent"].values[0]

    def get_first_timestamp(self):
        return [0]



