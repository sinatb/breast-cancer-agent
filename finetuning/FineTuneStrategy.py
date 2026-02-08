from enum import Enum

class FineTuneStrategy(Enum):
    FULL_FT = 1
    LORA = 2
    LINEAR_PROBE = 3
    