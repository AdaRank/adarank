import copy
import multiprocessing
import numpy as np

import constants
import traceback
import streamlit as st
import time
from datetime import datetime as dt
import csv
import logging

import utils
from src.data_analyzer import DataAnalyzer
from src.enums.filter import Filter
from src.enums.selection_methods import SelectionMethods
from src.enums.setting import Setting
from src.project_simulator.extractor import Extractor
from src.project_simulator.output_writer import Writer
from src.project_simulator.rule_filter import RuleFilter
from src.project_simulator.rule_reader import RuleReader
from src.project_simulator.successive_item_recommendation import FullSequenceSimulator
from src.project_simulator.single_event_simulator import SingleEventSimulator

logger = logging.getLogger(__name__)


class Main:
    """
    * Read sequence database
    * Split input database into training and test set
    * Compute quality measures based on sequence data and groundtruth
        - Diversity
        - Coverage/Recall
        - Precision
        - Novelty
        - Affiliation
        - Connectivity
    """

    def __init__(self):
        self.top_k_range = None
        self.title = None
        self.rule_set = None
        self.support = None
        self.confidence = None
        self.decay = None
        self.max_antecedent = None
        self.max_consequent = None
        self.train_set = None
        self.train_set_path = None
        self.test_set = None
        self.train_set_ts = None
        self.test_set_ts = None
        self.mining_algorithms = None
        self.selection_methods = None
        self.db_size = None
        self.number_of_runs = None
        self.rel_recommendation_base_range = None
        self.filter = None
        self.sessions = []
        self.bandwidth = None
        self.prelim = dict()
        self.setting = None
        self.sample_size = None

    def create_gui(self):
        st.title('AdaRank GUI')
        st.sidebar.markdown("AdaRank GUI")
        st.sidebar.markdown(
            "*Simulate new sequences based on mined rules. Quality measures such as nDCG@k, IALD@k and IELD@k are calculated on the fly.*")
        st.info('Please take the following steps into consideration:\n'
                '* Split your data into training and test set first.\n'
                '* If you do not have a rule set ready for your simulation, start with mining rules first. Refer to ER-Miner implemented in [SPMF](https://www.philippe-fournier-viger.com/spmf/ERMiner.php) and [SCORER-Gap](https://github.com/nimbus262/scorer-gap) for this.\n'
                '* If you want to use *CGap* as a selection method, do not forget to separate your timestamp data from your event data. Both files should contain the information in [IBMGenerator format](https://www.philippe-fournier-viger.com/spmf/Converting_a_sequence_database_to_SPMF.php).',
                icon="ℹ️")

        self.rule_set = st.file_uploader('Upload rule set', type=['txt'])
        self.train_set = st.file_uploader('Upload training set (IBMGenerator format)', type=['txt'])
        self.test_set = st.file_uploader('Upload test set (IBMGenerator format)', type=['txt'])
        if st.checkbox("Use sample"):
            self.sample_size = st.number_input("State your sample size?", min_value=1, max_value=99, help="Percentage on input dataset.")
        if self.rule_set is None:
            st.error("Mined rules are required for sequence simulation!")
        if self.train_set is None or self.test_set is None:
            st.error("Dataset in IBMGenerator format split into training and test set is required for sequence simulation!")
        col1, col2 = st.columns(2)
        with col1:
            self.selection_methods = st.multiselect('Which selection methods would you like to use?',
                                                    [SelectionMethods.RANDOM.value, SelectionMethods.FIRST.value,
                                                     SelectionMethods.NAIVE.value, SelectionMethods.DGAP.value,
                                                     SelectionMethods.CGAP.value],
                                                    [SelectionMethods.DGAP.value])
            if SelectionMethods.CGAP.value in self.selection_methods:
                self.train_set_ts = st.file_uploader('Upload timestamps for training set (IBMGenerator format)',
                                                     type=['txt'])
                self.test_set_ts = st.file_uploader('Upload timestamps for test set (IBMGenerator format)', type=['txt'])
                if self.train_set_ts is None or self.test_set_ts is None:
                    st.error(
                        "A timestamp dataset split in training and test set is required for selection method *CGap*")
                if st.checkbox("Set custom bandwidth for KDE"):
                    self.bandwidth = st.number_input("Select your bandwidth:", min_value=1, max_value=1000000,
                                                     value=100000,
                                                     help="If you are not sure which bandwidth you are dealing with, take a look at you timestamp data.")
            self.db_size = st.number_input("Database Size for Simulation", value=100, min_value=1, max_value=10000,
                                           step=1, help="Number of sequences which will be simulated.")
            self.number_of_runs = st.number_input("How many runs would you like to simulate?", value=1, min_value=1,
                                                  max_value=1000, step=1,
                                                  help="Metrics get averaged over all runs.")
            #self.filter = st.multiselect("Which filter would yo like to use?",
            #                             [Filter.NO_FILTERING.value, Filter.GEN_ANT.value, Filter.MAX_CONS.value,
            #                              Filter.GEN_MAX_COMB.value], [Filter.NO_FILTERING.value])
        with col2:
            possible_topk = [1, 3, 5, 10]
            possible_rec_base = list(np.arange(0.1, 1.0, 0.1))
            topk_start, topk_end = st.select_slider("Top-K Rules", options=possible_topk, value=(5, 10))
            self.top_k_range = possible_topk[possible_topk.index(topk_start):possible_topk.index(topk_end) + 1]
            st.write(
                f"You choose the following Top-K value{'s' if len(self.top_k_range) > 1 else ''}: {self.top_k_range}.")
            if st.checkbox("Only recommend last event?"):
                self.setting = Setting.LAST_EVENT
                logging.info("Only recommending last event")
            else:
                self.setting = Setting.SIMULATION
                rec_base_start, rec_base_end = st.select_slider("Relative Recommendation Base",
                                                                options=possible_rec_base, value=(0.1, 0.1))
                self.rel_recommendation_base_range = possible_rec_base[possible_rec_base.index(
                    rec_base_start):possible_rec_base.index(rec_base_end) + 1]
                st.write(
                    f"You choose the following value{'s' if len(self.rel_recommendation_base_range) > 1 else ''} for Recommendation Base: {np.round(self.rel_recommendation_base_range, 1)}.")
        run = st.button("Run Simulation")
        if run:
            if self.test_set is not None and self.rule_set is not None:
                self.run_simulation()

    def load(self, rules, test_set, train_set, test_set_ts, train_set_ts):
        train_set = utils.read_input_data(train_set)
        test_set = utils.read_input_data(test_set)
        if self.sample_size:
            self.prelim["train"] = utils.get_sample(train_set, self.sample_size/100)
            self.prelim["test"] = utils.get_sample(test_set, self.sample_size/100)
            st.write(f"Using {len(self.prelim['train']) + len(self.prelim['test'])} of "
                     f"{(len(self.prelim['train']) / (self.sample_size / 100)) + (len(self.prelim['test']) / (self.sample_size / 100))} "
                     f"sequences")
        else:
            self.prelim["train"] = train_set
            self.prelim["test"] = test_set

        # logger.info(len(self.prelim["train"]))
        self.prelim["train_ts"] = utils.read_input_data(train_set_ts)
        self.prelim["test_ts"] = utils.read_input_data(test_set_ts)
        logger.info("Analyzing data to decide how long the simulated sequence should be.")
        analyzer = DataAnalyzer(self.prelim["train"])
        #        self.prelim["mean_seq_len"] = analyzer.mean_seq_len[0]
        #        self.prelim["mean_seq_len"] = analyzer.median_seq_len
        self.prelim["mean_seq_len"] = analyzer.mean_seq_len[0]
        self.prelim["std_seq"] = analyzer.mean_seq_len[1]
        # if self.prelim["std_seq"] > (self.prelim["mean_seq_len"] / 4):
        #    self.prelim["std_seq"] = self.prelim["mean_seq_len"] / 4
        if self.prelim["mean_seq_len"] < 4:
            self.prelim["std_seq"] = 1
        else:
            self.prelim["std_seq"] = self.prelim["mean_seq_len"] / 4

        logger.info(f"Mean seq len: {self.prelim['mean_seq_len']}, std: {self.prelim['std_seq']}")

        # read rules from SPMF output (IBMGenerator format; file reader)
        logger.info("Reading generated rules from file.")
        rule_reader = RuleReader(rules)
        self.prelim["rules"] = rule_reader.data
        logger.info(f"Number of rules: {len(self.prelim['rules'].index)}")
        self.prelim["rel_div_bef"] = utils.get_relative_diversity(self.prelim["rules"]['consequent'].tolist())
        self.prelim["filter"] = dict()
        # if one option is not to filter the rules then we should extract gap information first...
        self.prelim["simulators"] = []

    def mp_worker(self, simulator):
        for i in range(self.number_of_runs):
            print(f"Worker #{multiprocessing.current_process()} | Run #{i}")
            now = dt.now()
            s = now.strftime("%y%m%d-%H%M%S.%f")
            self.title = str('.'.join(self.rule_set.name.split(".")[:-1])) + \
                         "_" + str(simulator.selection_method) + \
                         "_" + str(simulator.topk) + \
                         "_" + str(s)
            s_time = time.time()

            f = open('results/qualities/out_' + self.title + '.csv', 'w', encoding='UTF8')
            writer = csv.writer(f)
            header = ["TITLE", "ALGORITHM", "#RULES", "DECAY", "METHOD", "DB_SIZE", "TOPK",
                      'REL_REC_BASE', "SUPPORT", "CONFIDENCE", "REL_DIV_BEF",
                      "REL_DIV_AFT", "MRR", "MRR_SINGLE", "MAP", "HR", "HR_SINGLE", "RECALL_SINGLE", "DIV", "INTER_DIV", "NDCG", "NDCG_SINGLE", "CR", "SR",
                      "DURATION IN SEC", "BANDWIDTH"]
            writer.writerow(header)
            writer = csv.writer(f)

            #simulator.qualities = Qualities(self.prelim["test"])

            logger.info(f"Creating experiments with title '{self.title}'")
            logger.info(f"### Current configuration:\n\n"
                        f"*Miner*: {str(self.rule_set.name.split('_')[0])}\n\n"
                        f"*Selection Method*: {simulator.selection_method}\n\n"
                        f"*Top-K*: {simulator.topk}\n\n"
                        f"*Recommendation Basis*: {simulator.rel_recommendation_base}")

            # ------------------------------------------------------------------------------------------------------
            simulator.run()
            # ------------------------------------------------------------------------------------------------------

            duration_in_s = time.time() - s_time
            logger.info(f"Took {duration_in_s}s")

            # Write simulated sequences to file
            logger.info(f"Writing simulated sequences to file | {simulator.selection_method}")
            seq_writer = Writer(constants.SEQUENCE_RESULTS_PATH, self.title + ".txt", simulator)
            seq_writer.write_output()

            row = [self.title,
                   str(self.rule_set.name.split("_")[0]),
                   len(simulator.rules.index),
                   self.decay if simulator.selection_method != SelectionMethods.RANDOM else -1,
                   simulator.selection_method,
                   self.db_size,
                   simulator.topk,
                   simulator.rel_recommendation_base,
                   self.support,
                   self.confidence,
                   self.prelim["rel_div_bef"],
                   simulator.rel_div_aft,
                   simulator.qualities.mrr.mean_quality,
                   simulator.qualities.mrr_single.mean_quality,
                   simulator.qualities.map.mean_quality,
                   simulator.qualities.hr.mean_quality,
                   simulator.qualities.hr_single.mean_quality,
                   simulator.qualities.recall_single.mean_quality,
                   simulator.qualities.div.mean_quality,
                   None if not simulator.qualities.div.mean_inter_div else simulator.qualities.div.mean_inter_div,
                   simulator.qualities.ndcg.mean_quality,
                   simulator.qualities.ndcg_single.mean_quality,
                   simulator.qualities.cancelled_recommendations,
                   simulator.qualities.successful_recommendations,
                   duration_in_s,
                   simulator.bandwidth]
            logger.info("Writing qualities to file.")
            writer.writerow(row)
            f.close()

    def run_simulation(self):
        try:
            self.load(self.rule_set, self.test_set, self.train_set, self.test_set_ts, self.train_set_ts)
            self.filter = [Filter.NO_FILTERING]
            if Filter.NO_FILTERING in self.filter:
                for selection_method in self.selection_methods:
                    extractor = Extractor(self.prelim, self.prelim["rules"].copy(deep=True), selection_method)
                    s_time = time.time()
                    extractor.run()
                    st.write(f"Took {time.time() - s_time}")
                    for filt in self.filter:
                        rules = extractor.rules.copy(deep=True)
                        rule_filter = RuleFilter(rules, filt)
                        rules = rule_filter.run()
                        rules.reset_index(drop=True, inplace=True)
                        simulator = None
                        # prelim, rules, filt, selection_method, recommendation_base, topk
                        if self.setting == Setting.SIMULATION:
                            simulator = FullSequenceSimulator(self.prelim, rules, filt, selection_method,
                                                              self.bandwidth, self.db_size, rule_filter.rel_div_aft)
                        if self.setting == Setting.LAST_EVENT:
                            simulator = SingleEventSimulator(self.prelim, rules, filt, selection_method, self.bandwidth,
                                                             self.db_size, rule_filter.rel_div_aft)
                        self.prelim["simulators"].append(simulator)
            else:
                # ...otherwise one does not know which rules are going to be kept after filtering
                # here the extraction step has to be done afterwards
                for filt in self.filter:
                    rules = self.prelim["rules"].copy(deep=True)
                    rule_filter = RuleFilter(rules, filt)
                    rules = rule_filter.run()
                    rules.reset_index(drop=True, inplace=True)
                    for selection_method in self.selection_methods:
                        extractor = Extractor(self.prelim, rules.copy(deep=True), selection_method)
                        s_time = time.time()
                        extractor.run()
                        st.write(f"Took {time.time() - s_time}")
                        # st.write(extractor.rules)
                        st.write("DONE")
                        simulator = None
                        if self.setting == Setting.SIMULATION:
                            simulator = FullSequenceSimulator(self.prelim, extractor.rules, filt, selection_method,
                                                              self.bandwidth, self.db_size, rule_filter.rel_div_aft)
                        if self.setting == Setting.LAST_EVENT:
                            simulator = SingleEventSimulator(self.prelim, extractor.rules, filt, selection_method,
                                                             self.bandwidth, self.db_size, rule_filter.rel_div_aft)
                        self.prelim["simulators"].append(simulator)
            st.write(f"\n\nStarting to run simulators...")

            simulator_list = []
            for prelim_sim in self.prelim["simulators"]:
                for topk in self.top_k_range:
                    if self.setting == Setting.SIMULATION:
                        for rec_base in self.rel_recommendation_base_range:
                            sim = copy.deepcopy(prelim_sim)
                            sim.topk = topk
                            sim.rel_recommendation_base = rec_base
                            sim.db_size = self.db_size
                            simulator_list.append(sim)
                    else:
                        sim = copy.deepcopy(prelim_sim)
                        sim.topk = topk
                        sim.db_size = self.db_size
                        simulator_list.append(sim)

            st.write(f"{len(simulator_list)} simulator(s) to execute.")
            num_cores = multiprocessing.cpu_count() if len(simulator_list) >= multiprocessing.cpu_count() else len(
                simulator_list)
            # with st.spinner(text="Executing simulators..."):
            now = dt.now()
            st.write(f"Started simulation at {now.strftime('%H:%M:%S')}")
            with multiprocessing.Pool(num_cores) as p:
                # p = multiprocessing.Pool(num_cores)
                st.write(f"Using {num_cores} cores to run experiments")
                p.map(self.mp_worker, simulator_list)
            # for i_exp, simulator in enumerate(simulator_list):
            now = dt.now()
            st.success(f"Successively conducted experiment(s) ({now.strftime('%H:%M:%S')}).")
            # p.close()
        except:
            st.error(f'Error:\n{traceback.format_exc()}', icon="🚨")


if __name__ == "__main__":
    Main().create_gui()
