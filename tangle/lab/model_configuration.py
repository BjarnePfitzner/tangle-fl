class ModelConfiguration:
    dataset: str
    model: str
    lr: float
    use_val_set: bool
    num_epochs: int
    batch_size: int

    def __init__(self, dataset, model, lr, use_val_set, num_epochs, batch_size):
        self.dataset = dataset
        self.model = model
        self.lr = lr
        self.use_val_set = use_val_set
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        super().__init__()
