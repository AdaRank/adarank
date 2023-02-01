import logging
import random
import statistics

import utils
from src.enums.selection_methods import SelectionMethods
from src.project_simulator.qualities import Qualities
from src.project_simulator.rule_selector import RuleSelector

logger = logging.getLogger(__name__)


class FullSequenceSimulator:
    """
    * Creates n sequences by calling rule selector
    * Recommendation base: Uses consequent of first rule if there is no recommendation base
    * Stopping criterion: Mean length of sequence (from train set) +/- one standard deviation
    * Length of recommendation base is variable
    """

    def __init__(self, prelim, rules, filt, selection_method, bandwidth, db_size, rel_div_aft):
        self.kernel = 'gaussian'
        self.rules = rules
        self.sequences = []
        self.bandwidth = bandwidth
        self.prelim = prelim
        #logger.info(self.prelim["test"])
        self.std = prelim["std_seq"]
        self.l = prelim["mean_seq_len"]
        self.db_size = db_size
        self.selection_method = selection_method
        self.selection_sizes = []
        self.filter = filt
        self.rel_div_aft = rel_div_aft
        self.rel_recommendation_base = None
        self.topk = None
        self.qualities = Qualities(self.prelim["test"])

    def run(self):
        logger.info(f"Starting to create {self.db_size} sequences")
        ms = len(self.rules.index) * [None]
        if self.selection_method == SelectionMethods.DGAP:
            for index, row in self.rules.iterrows():
                # logger.info(index)
                #logger.info(row["rule"])
                #logger.info(row["dgaps"])
                if row["dgaps"]:
                    ms[index - 1] = utils.create_multiset(row["dgaps"])
                else:
                    ms[index - 1] = dict()
            self.rules["ms_dgaps"] = ms
        self.generate_sequences()
        logger.info(f"Cancelled recommendations: {self.qualities.cancelled_recommendations}")
        logger.info(f"Successful recommendations: {self.qualities.successful_recommendations}")

    def generate_sequences(self):
        for i in range(self.db_size):
            # logger.info(f"Detected length: {self.l}, {self.std} ")
            length = utils.compute_length(self.l, self.std)
            # length = len(random.choice(self.prelim["test"]))
            logger.info(f"Creating sequence #{i} with length {length} | {self.selection_method}")
            rule_selector = RuleSelector(self.topk, self.bandwidth, rules=self.rules, method=self.selection_method)
            gt_sequence = None
            # logger.info("Searching for valid test sequence")
            if length > 1:
                index = random.randint(0, len(self.prelim["test"])-1)
                gt_sequence = self.prelim["test"][index]
                new_sequence = self.recommendation_basis(index)  # recommendation base
                #logger.info(f"Valid test sequence found")
            else:
                new_sequence = []
                #logger.info("Starting new sequence")
            #logger.info(f"Using the following recommendation base: {new_sequence}")
            rule_selector.sequence = new_sequence
            if self.selection_method == SelectionMethods.CGAP:
                if length > 1:
                    new_sequence_ts = self.recommendation_basis_ts(index)  # recommendation base timestamp
                    #logger.debug(f"Corresponding recommendation base ts: {new_sequence_ts}")
                else:
                    new_sequence_ts = []
                rule_selector.sequence_ts = new_sequence_ts
                if self.bandwidth:
                    rule_selector.bandwidth = self.bandwidth
            # st.write("Creating rule iterator")
            rule_iterator = iter(rule_selector, self.l)
            while len(new_sequence) < length:
                # print("Getting next rule")
                if self.selection_method == SelectionMethods.CGAP:
                    next_item, next_ts = next(rule_iterator)
                    if next_item is None:
                        self.qualities.cancelled_recommendations += 1
                        break
                    # logger.info(f"Next item: {next_item}, next ts: {next_ts}")
                    new_sequence.extend(next_item)
                    # Add mean timestamp of predicted element to last timestamp
                    if len(new_sequence) == 1:
                        latest = 0.0
                    else:
                        latest = new_sequence_ts[-1]
                    for _ in range(len(next_item)):  # next recommendation can also be a batch of items
                        new_sequence_ts.append(latest + next_ts)
                elif self.selection_method == SelectionMethods.DGAP:
                    next_item = next(rule_iterator)
                    if next_item is None:
                        self.qualities.cancelled_recommendations += 1
                        break
                    new_sequence.extend(next_item)
                elif self.selection_method == SelectionMethods.RANDOM or self.selection_method == SelectionMethods.NAIVE or self.selection_method == SelectionMethods.FIRST:
                    next_item = next(rule_iterator)
                    if next_item is None:
                        self.qualities.cancelled_recommendations += 1
                        break
                    new_sequence.extend(next_item)

                self.qualities.successful_recommendations += 1

                #self.qualities.mrr.add_sequence(new_sequence, rule_selector.stepwise_rules["consequent"])
                #self.qualities.map.add_sequence(new_sequence, rule_selector.stepwise_rules["consequent"])
                self.qualities.hr.add_sequence_succ(new_sequence, gt_sequence,
                                                    rule_selector.stepwise_rules["consequent"])
                #if gt_sequence:
                #    self.qualities.hr.add_sequence_succ(new_sequence, gt_sequence, rule_selector.stepwise_rules["consequent"])
                #else:
                #    self.qualities.hr.add_sequence(new_sequence, rule_selector.stepwise_rules["consequent"])
                self.qualities.ndcg.add_sequence(new_sequence, rule_selector.stepwise_rules["consequent"])

                self.qualities.div.add_sequence(sequence=None,
                                                recommendations=rule_selector.stepwise_rules["consequent"])

                self.selection_sizes.extend(rule_selector.selection_size)

            if self.selection_method == SelectionMethods.CGAP:
                logger.debug(
                    f"{i}/{self.db_size} Created this sequence: {new_sequence[:length]} with timestamps {new_sequence_ts[:length]}")
            elif self.selection_method == SelectionMethods.DGAP:
                logger.debug(f"{i}/{self.db_size} Created this sequence: {new_sequence[:length]}")

            # st.write("Appending simulated sequence")
            # logger.info(f"Appending new sequence: {new_sequence[:length]}")
            self.sequences.append(
                new_sequence[:length])  # has to be pruned because last recommendation can be a batch of items

        self.qualities.compute_qualities(selection_method=self.selection_method)

        try:
            logger.info(f"Median of number of rules for selection is {statistics.median(self.selection_sizes)}")
        except statistics.StatisticsError:
            logger.info(f"Median of number of rules for selection is not available")
        if self.selection_method == SelectionMethods.RANDOM:
            logger.info(f"At every recommendation step {self.topk} where used of aforementioned number ")

    def replace_event(self, sequence, position):
        recommendation_base = sequence[:position]
        remainder = sequence[position:]
        rule_selector = iter(RuleSelector(self.topk, self.rules, recommendation_base), self.l)
        recommendation_base.extend(next(rule_selector))
        new_sequence = recommendation_base + remainder[1:]
        return new_sequence

    def recommendation_basis(self, index):
        #logger.info(f"Index: {index}, {self.rel_recommendation_base}")
        if self.rel_recommendation_base > 0.0 and index is not None:
            s = self.prelim["test"][index]
            l = int(len(s) * self.rel_recommendation_base)
            if l == 0:  # Use at least one item as recommendation basis
                return s[:1]
            return s[:l]
        else:
            s = self.prelim["test"][index]
            return s[:1]

    def recommendation_basis_ts(self, index):
        if self.rel_recommendation_base > 0.0 and index is not None:
            s = self.prelim["test_ts"][index]
            l = int(len(s) * self.rel_recommendation_base)
            if l == 0:
                return s[:1]
            return s[:l]
        else:
            return [0]

