import random
import logging
from io import StringIO

import streamlit as st

logger = logging.getLogger(__name__)

class InputReader:
    """
    Loads a given logfile and transforms the SPMF format into a list of given events.
    """
    def __init__(self, raw_data, tt_split, seed):
        self.data = []
        self.data_split = {}
        self.max_val = -1
        self.read_input(raw_data)
        self.train_test_split(tt_split, seed)

    def read_input(self, raw_data):
        lines = raw_data.readlines()
        self.transform_input(lines)

    def transform_input(self, d):
        i = 0
        for sequence in d:
            temp = []
            i += 1
            sequence = sequence.decode("utf-8")
            for x in sequence.strip().split(" "):
                if x != "-1" and x != "-2":
                    x_new = int(x)
                    if self.max_val < x_new:
                        self.max_val = x_new
                    temp.append(x_new)
            self.data.append(temp)

    def get_data(self):
        return self.data

    def get_split_data(self):
        return self.data_split

    def get_max_val(self):
        return self.max_val

    def train_test_split(self, tt_split, seed):
        random.Random(seed).shuffle(self.data)
        i = int(len(self.data)*tt_split)
        train = self.data[:i]
        test = self.data[i:]
        logger.info(f"Training set has length of {len(train)} sequences")
        logger.info(f"Test set has length of {len(test)} sequences")
        self.data_split["train"] = train
        self.data_split["test"] = test
