class LabConfiguration:
    seed: int
    tangle_dir: str
    tangle_tx_dir: str

    def __init__(self, seed, tangle_dir, tangle_tx_dir):
        self.seed = seed
        self.tangle_dir = tangle_dir
        self.tangle_tx_dir = tangle_tx_dir
        super().__init__()
