class PeerConfiguration:
    create_genesis: bool
    broker: str
    timeout: int
    training_interval: int
    active_quota: float
    mock_model: bool

    def __init__(self):
        self.create_genesis = None
        self.storage = None
        self.broker = None
        self.timeout = None
        self.training_interval = None
        self.active_quota = None
        self.mock_model = None

    def define_args(self, parser):

        parser.add_argument('--create-genesis',
                            help='create a genesis transaction at /data/genesis.npy',
                            action='store_true')
        parser.add_argument('--storage', default='ipfs', help='sets the used storage')
        parser.add_argument('--broker', default='ipfs', help='sets the used message broker')
        parser.add_argument('--timeout', default=None, type=int_or_none, help='timeout for ipfs')
        parser.add_argument('--training_interval', default=20, type=int, help='training interval')
        parser.add_argument('--active_quota', type=float, default=0.01,
                            help='sets the quota of active pears, must be valid percentage (0.00, 0.01, ..., 0.99, 1.00')
        parser.add_argument('--mock-model', action='store_true', help='mock the model training and validation instead of using a TF model')

    def parse(self, args):
        self.create_genesis = args.create_genesis
        self.storage = args.storage
        self.broker = args.broker
        self.timeout = args.timeout
        self.training_interval = args.training_interval
        self.active_quota = args.active_quota
        self.mock_model = args.mock_model


def int_or_none(value):
    if isinstance(value, int) or value is None:
        return value
    elif isinstance(value, str) and value.isdigit():
        return int(value)
    elif value == 'None':
        return None
    else:
        msg = f'{value} is neither int nor None'
        raise argparse.ArgumentTypeError(msg)
