class ModelConfiguration:
    dataset: str
    model: str
    lr: float
    num_epochs: int
    batch_size: int

    def __init__(self):
        self.dataset = None
        self.model = None
        self.lr = None
        self.num_epochs = None
        self.batch_size = None

    def define_args(self, parser):
        DATASETS = ['sent140', 'femnist', 'femnistclustered', 'shakespeare', 'celeba', 'synthetic', 'reddit']

        parser.add_argument('-dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        required=True)
        parser.add_argument('-model',
                        help='name of model;',
                        type=str,
                        required=True)
        parser.add_argument('-lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=-1,
                        required=False)

        # Minibatch doesn't support num_epochs, so make them mutually exclusive
        epoch_capability_group = parser.add_mutually_exclusive_group()
        epoch_capability_group.add_argument('--minibatch',
                        help='None for FedAvg, else fraction;',
                        type=float,
                        default=None)
        epoch_capability_group.add_argument('--num-epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)

        parser.add_argument('--batch-size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)

    def parse(self, args):
        self.dataset = args.dataset
        self.model = args.model
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
