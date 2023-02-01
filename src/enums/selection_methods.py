from enum import Enum


class SelectionMethods(str, Enum):
    CGAP = "CGap"
    DGAP = "DGap"
    RANDOM = "Random"
    NAIVE = "Naive"
    FIRST = "First"