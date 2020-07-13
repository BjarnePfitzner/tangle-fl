from ..core.tip_selection import AccuracyTipSelector

class RayAccuracyTipSelector(AccuracyTipSelector):
    def __init__(self, tangle, settings, precomputed_ratings):
        super().__init__(tangle, settings)
        self.precomputed_ratings = precomputed_ratings

    def _compute_ratings(self, node):
        return self.precomputed_ratings
