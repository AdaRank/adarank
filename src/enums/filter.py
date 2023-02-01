from enum import Enum


class Filter(str, Enum):
    NO_FILTERING = "No Filtering"
    GEN_ANT = 'Generators on Antecedent'
    MAX_CONS = "Maximal on Consequent"
    GEN_MAX_COMB = "Generators/Maximal Combination"