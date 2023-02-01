from enum import Enum


class Algorithm(str, Enum):
    ERMINER = 'ERMiner'
    SCORERGAP = 'ScorerGap'
    RULEGEN_WITH_PREFIX_SPAN = 'RuleGen with PrefixSpan'
    ASSOCIATION_RULES_WITH_FP_GROWTH = 'Association Rules with FPGrowth'