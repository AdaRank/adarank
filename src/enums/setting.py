from enum import Enum


class Setting(str, Enum):
    SIMULATION = "Simulation"
    LAST_EVENT = "Last Event"