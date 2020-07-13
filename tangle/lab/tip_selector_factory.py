from ..core.tip_selection import TipSelector, AccuracyTipSelector
from ..core.tip_selection.accuracy_tip_selector import AccuracyTipSelectorSettings

class TipSelectorFactory:
    def __init__(self, config):
        self.config = config

    def create(self, tangle):
        if self.config.tip_selector == 'default':
            return TipSelector(tangle)

        elif self.config.tip_selector == 'accuracy':

            tip_selection_settings = {}
            tip_selection_settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] = self.config.acc_tip_selection_strategy
            tip_selection_settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS] = self.config.acc_cumulate_ratings
            tip_selection_settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] = self.config.acc_ratings_to_weights
            tip_selection_settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] = self.config.acc_select_from_weights
            tip_selection_settings[AccuracyTipSelectorSettings.ALPHA] = self.config.acc_alpha

            return AccuracyTipSelector(tangle, tip_selection_settings)
