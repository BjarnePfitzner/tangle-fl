from enum import Enum


class ParticleSettings(Enum):
    USE_PARTICLES = 0
    PARTICLES_DEPTH_START = 1
    PARTICLES_DEPTH_END = 2
    NUM_PARTICLES = 3


class TipSelectorSettings(Enum):
    SELECTION_STRATEGY = 0
    CUMULATE_RATINGS = 1
    RATINGS_TO_WEIGHT = 2
    ALPHA = 3
    SELECT_FROM_WEIGHTS = 4