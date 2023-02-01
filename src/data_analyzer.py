import logging
import statistics

import numpy as np
import seaborn
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    * Get number of sequences
    * Get number of distinct activities
    * Get average length of sequence
    * Get number of distinct activities per sequence
    * Plot distribution of activity frequency
    """

    nr_of_sequences = None
    nr_dist_items = None
    mean_seq_len = None
    nr_dist_items_per_seq = None

    def __init__(self, data):
        self.data = data
        self.nr_of_sequences = self.get_nr_of_seq()
        self.nr_dist_items = self.get_nr_of_dist_items()
        self.mean_seq_len = self.get_mean_seq_length()
        self.median_seq_len = self.get_median_seq_length()
        self.mode_seq_len = self.get_mode_seq_length()
        logger.info(f"Mode: {self.mode_seq_len}")
        self.nr_dist_items_per_seq = self.get_nr_of_dist_items_per_seq()
        #print(self.get_nr_of_seq())
        #print(self.get_nr_of_dist_items())
        #print(self.get_mean_seq_length())
        #print(self.get_nr_of_dist_items_per_seq())

    def get_nr_of_seq(self):
        return len(self.data)

    def get_nr_of_dist_items(self):
        flat_list = [item for sublist in self.data for item in sublist]
        db_as_set = set(flat_list)
        return len(db_as_set)

    def get_mean_seq_length(self):
        lengths = []
        for sequence in self.data:
            lengths.append(len(sequence))
        return sum(lengths) / len(self.data), np.std(lengths)

    def get_median_seq_length(self):
        lengths = []
        for sequence in self.data:
            lengths.append(len(sequence))
        return statistics.median(lengths)

    def get_nr_of_dist_items_per_seq(self):
        dist_items = []
        for sequence in self.data:
            seq_set = set(sequence)
            dist_items.append(len(seq_set))
        return sum(dist_items) / len(self.data), np.std(dist_items)

    def plot_activity_distribution(self):
        flat_list = [item for sublist in self.data for item in sublist]
        seaborn.displot(flat_list, kde=True)
        plt.show()

    def get_mode_seq_length(self):
        lengths = []
        for sequence in self.data:
            lengths.append(len(sequence))
        return max(set(lengths), key=lengths.count)

