from enum import Enum

class PoisonType(Enum):
    Disabled = 0
    Random = 1
    LabelFlip = 2
    LabelSwap = 3

    @staticmethod
    def make_type_from_cfg(config_entry):
        return {
            'disabled': PoisonType.Disabled,
            'random': PoisonType.Random,
            'labelflip': PoisonType.LabelFlip,
            'labelswap': PoisonType.LabelSwap
        }[config_entry]
