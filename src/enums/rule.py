from enum import Enum


class Rule(str, Enum):
    ANTECEDENT = "antecedent"
    CONSEQUENT = "consequent"