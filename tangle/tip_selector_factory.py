from tangle.core.tip_selection import TipSelector, AccuracyTipSelector, LazyAccuracyTipSelector
from tangle.core.tip_selection.tip_selection_settings import ParticleSettings, TipSelectorSettings


class TipSelectorFactory:
    def __init__(self, config):
        self.config = config
        
        self.particle_settings = {
            ParticleSettings.USE_PARTICLES: self.config.particles.enabled,
            ParticleSettings.PARTICLES_DEPTH_START: self.config.particles.depth_start,
            ParticleSettings.PARTICLES_DEPTH_END: self.config.particles.depth_end,
            ParticleSettings.NUM_PARTICLES: self.config.particles.number
        }

    def create(self, tangle):
        tip_selection_settings = {
            TipSelectorSettings.SELECTION_STRATEGY: self.config.strategy,
            TipSelectorSettings.CUMULATE_RATINGS: self.config.cumulate_ratings,
            TipSelectorSettings.RATINGS_TO_WEIGHT: self.config.ratings_to_weights,
            TipSelectorSettings.SELECT_FROM_WEIGHTS: self.config.select_from_weights,
            TipSelectorSettings.ALPHA: self.config.alpha
        }

        if self.config.type == 'random':
            return TipSelector(tangle, tip_selection_settings, self.particle_settings)

        elif self.config.type == 'accuracy':
            return AccuracyTipSelector(tangle, tip_selection_settings, self.particle_settings)

        elif self.config.type == 'lazy_accuracy':
            return LazyAccuracyTipSelector(tangle, tip_selection_settings, self.particle_settings)
