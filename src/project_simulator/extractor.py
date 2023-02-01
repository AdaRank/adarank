from sklearn.neighbors import KernelDensity

import utils
from src.enums.selection_methods import SelectionMethods
import streamlit as st
import numpy as np



class Extractor:
    def __init__(self, prelim, rules, selection_method):
        self.prelim = prelim
        self.rules = rules
        self.selection_method = selection_method

    def run(self):
        if self.selection_method == SelectionMethods.DGAP:
            # check if cgap or dgap has alreday been calculated, if so then do not do it again
            # if cgap has not been calculated then do also not calculate dgap again
            st.write("Extracting DGaps")
            self.extract_dgaps()
        elif self.selection_method == SelectionMethods.CGAP:
            st.write("Extracting CGaps")
            self.extract_cgaps()
        elif self.selection_method == SelectionMethods.NAIVE or self.selection_method == SelectionMethods.RANDOM:
            st.write("Nothing to extract")

    def extract_cgaps(self):
        self.rules["cgaps"] = None
        nr_rules = len(self.rules.index)
        counter = 0
        with st.empty():
            for index, row in self.rules.iterrows():
                counter += 1
                st.write(f"Extracting CGaps for rule #{counter}/{nr_rules}")
                ts_gaps = []
                for i, sequence in enumerate(self.prelim["train"]):
                    cgap = utils.retrieve_cgap(sequence, self.prelim["train_ts"][i], row["antecedent"])
                    # if gap was not computed because not all items matched
                    if cgap is None:
                        continue
                    # if discrete gap was negative because the consequent matched before the antecedent
                    if cgap.get_gap() is None:
                        continue

                    ts_gaps.append(cgap.get_gap())

                if len(ts_gaps) > 0:
                    self.rules.at[index, "cgaps"] = ts_gaps
                # print(f"Histogram for this rule {self.rules['rule'][index]}:\n{self.rules['gaps'][index]}")
                # print(self.rules.loc[[index]])
                # print(self.rules)

                # plt.hist(self.rules['gaps'][index], bins=[0, 1,2,3,4,5,6,7,8,9])
                # plt.show()


    def extract_dgaps(self):
        self.rules["dgaps"] = None
        nr_rules = len(self.rules.index)
        counter = 0
        with st.empty():
            for index, row in self.rules.iterrows():
                counter += 1
                st.write(f"Extracting DGaps for rule #{counter}/{nr_rules}")
                gaps = []
                for i, sequence in enumerate(self.prelim["train"]):
                    dgap = utils.retrieve_dgap(sequence, row["antecedent"], row["consequent"])

                    # if gap was not computed because not all items matched
                    if dgap is None:
                        continue
                    # if discrete gap was negative because the consequent matched before the antecedent
                    if dgap.get_gap() is None:
                        continue

                    gaps.append(dgap.get_gap())

                if len(gaps) > 0:
                    self.rules.at[index, "dgaps"] = gaps

