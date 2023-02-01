import random
import streamlit as st

import utils
from src.data_analyzer import DataAnalyzer
from src.enums.rule import Rule
from src.enums.filter import Filter
from src.project_simulator.rule_filter import RuleFilter
from src.project_simulator.rule_reader import RuleReader
from src.project_simulator.successive_item_recommendation import FullSequenceSimulator

import logging

logger = logging.getLogger(__name__)



class Main:
    """
    Order of parameter application:
    1. Minsup/Minconf Threshold
    2. Filter
        2.1 Number of Rules (Top-K)
        2.2 Batch Size
    3. Recommendation Base
    4. Rule Selector
        4.1 Dynamic ranking with discrete gaps
        4.2 Dynamic ranking with continuous gaps
    """

    def __init__(self, db_size, selection_method):
        logger.info(f"Number of sequences per run: {db_size}")
        logger.info(f"Selection method for rules: {selection_method}\n")
        self.algo = None
        self.data = None
        self.std_seq = None
        self.mean_seq_len = None
        self.data_ts = None
        self.rules = None
        self.selection_method = selection_method
        self.session_title = None
        self.db_size = db_size
        self.split_data = None
        self.split_data_ts = None
        self.quality_measures = dict({"rel_div_bef": [], "rel_div_aft": []})
        self.session_mrr = None
        self.session_map = None
        self.session_div = None
        self.session_ndcg = None
        self.session_cr = None
        self.session_or = None
        self.simulators = dict()


    def prepare(self, rules, test_set, train_set, test_set_ts, train_set_ts):
        self.split_data = dict()
        self.split_data_ts = dict()
        self.split_data["train"] = utils.read_input_data(train_set)
        self.split_data["test"] = utils.read_input_data(test_set)
        self.split_data_ts["train"] = utils.read_input_data(train_set_ts)
        self.split_data_ts["test"] = utils.read_input_data(test_set_ts)
        logger.info("Analyzing data to decide how long the simulated sequence should be.")
        analyzer = DataAnalyzer(self.split_data["train"])
        self.mean_seq_len = analyzer.mean_seq_len[0]
        self.std_seq = analyzer.mean_seq_len[1]
        if self.std_seq > (self.mean_seq_len / 4):
            self.std_seq = self.mean_seq_len / 4

        # read rules from SPMF output (SPMF format; file reader)
        logger.info("Reading generated rules from file.")
        rule_reader = RuleReader(rules)
        self.rules = rule_reader.data
        logger.info(f"Number of rules: {len(self.rules.index)}")
        self.quality_measures['rel_div_bef'].append(utils.get_relative_diversity(self.rules['consequent'].tolist()))
        logger.info(f"Relative diversity: {self.quality_measures['rel_div_bef'][-1]}")


    def filter(self, filter):
        # filter rules by number (sorted by quality measure e.g. support or confidence)
        if filter != Filter.NO_FILTERING:
            logger.info("Filter rules by number (sorted by quality measure e.g. support or confidence)")
        else:
            logger.info("No filtering applied")
        rule_filter = RuleFilter(self.rules)
        if filter == Filter.MAX_CONS:
            st.write("Filtering for maximal patterns on consequent")
            rule_filter.filter_patterns(rule_filter.filter_maximal, side=Rule.ANTECEDENT)
        elif filter == Filter.GEN_ANT:
            st.write("Filtering for generator patterns on antecedent")
            rule_filter.filter_patterns(rule_filter.filter_generators, side=Rule.CONSEQUENT)
        elif filter == Filter.GEN_MAX_COMB:  # does not work yet, index missing when applying iloc
            st.write("Filtering for maximal patterns and generator patterns (consequent and antecedent respectively)")
            rule_filter.filter_patterns(rule_filter.filter_maximal, side=Rule.ANTECEDENT)
            rule_filter.rules = rule_filter.rules.reset_index()
            rule_filter.filter_patterns(rule_filter.filter_generators, side=Rule.CONSEQUENT)
        filtered_rules = rule_filter.rules
        logger.info(f"Number of rules after: {len(filtered_rules.index)}")
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    logger.info(f"Rules after filtering: {rules.sort_values(['support', 'confidence'], ascending=False)}")
        self.quality_measures['rel_div_aft'].append(utils.get_relative_diversity(filtered_rules['consequent'].tolist()))
        logger.info(f"Relative diversity after filtering: {self.quality_measures['rel_div_aft'][-1]}")

        # select specific rule and create/simulate project
        logger.info(
            f"Creating instance of simulator with sequence length = {self.mean_seq_len}, std = {self.std_seq}, selection method = {self.selection_method}")

        return FullSequenceSimulator(data=self.split_data, ts_data=self.split_data_ts,
                                     rules=self.rules, filtered_rules=filtered_rules, l=self.mean_seq_len,
                                     std=self.std_seq, rule_selection=None)

    def run(self, simulator, expander):
        print(f"Starting to create {self.db_size} sequences")
        simulator.generate_sequences(n=self.db_size, expander=expander)
        self.quality_measures["map"].append(simulator.get_map())
        self.quality_measures["mrr"].append(simulator.get_mrr())
        self.quality_measures["div"].append(simulator.get_div())
        self.quality_measures["ndcg"].append(simulator.get_ndcg())
        self.quality_measures["cancelled_recommendations"].append(simulator.cancelled_recommendations)
        self.quality_measures["successful_recommendations"].append(simulator.successful_recommendations)
        logger.info(f"Cancelled recommendations: {simulator.cancelled_recommendations}")
        logger.info(f"Overall recommendations: {simulator.successful_recommendations}")

        # Affiliation
        #result = self.create_subst_seq(simulator, data)

        self.session_mrr = sum(self.quality_measures['mrr'])/len(self.quality_measures['mrr'])
        self.session_map = sum(self.quality_measures['map'])/len(self.quality_measures['map'])
        self.session_div = sum(self.quality_measures['div'])/len(self.quality_measures['div'])
        self.session_ndcg = sum(self.quality_measures['ndcg'])/len(self.quality_measures['ndcg'])
        self.session_cr = sum(self.quality_measures['cancelled_recommendations'])/len(self.quality_measures['cancelled_recommendations'])
        self.session_or = sum(self.quality_measures['overall_recommendations'])/len(self.quality_measures['overall_recommendations'])
        self.session_rel_div_bef = sum(self.quality_measures['rel_div_bef'])/len(self.quality_measures['rel_div_bef'])
        if len(self.quality_measures['rel_div_aft']) > 0:
            self.session_rel_div_aft = sum(self.quality_measures['rel_div_aft'])/len(self.quality_measures['rel_div_aft'])
        else:
            self.session_rel_div_aft = -1

        logger.info("This session results in the following qualities")
        logger.info(f"Session MRR: {self.session_mrr}")
        logger.info(f"Session MAP: {self.session_map}")
        logger.info(f"Session DIV: {self.session_div}")
        logger.info(f"Session NDCG: {self.session_ndcg}")
        logger.info(f"Session CR: {self.session_cr}")
        logger.info(f"Session OR: {self.session_or}")


    def create_subst_seq(self, simulator, sequences):
        rep_sequences = []
        for seq in range(len(sequences)):
            pos = random.randint(0, len(seq)-1)
            new_seq = simulator.replace_event(seq, pos)
            rep_sequences.append(new_seq)
        return rep_sequences

    def set_title(self, title):
        self.session_title = title
        logger.info(f"\n\n-----------------\nSession title: {title}")

