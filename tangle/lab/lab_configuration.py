class LabConfiguration:
    seed: int
    model_data_dir: str
    tangle_dir: str
    tangle_tx_dir: str

    def __init__(self, seed, model_data_dir, tangle_dir, tangle_tx_dir):
        self.seed = seed
        self.model_data_dir = model_data_dir
        self.tangle_dir = tangle_dir
        self.tangle_tx_dir = tangle_tx_dir
        super().__init__()
