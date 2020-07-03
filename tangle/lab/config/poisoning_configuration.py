class PoisoningConfiguration:

    def define_args(self, parser):
        POISON_TYPES = ['none', 'random', 'labelflip']

        parser.add_argument('--poison-type',
                    help='type of malicious clients considered',
                    type=str,
                    choices=POISON_TYPES,
                    default='none',
                    required=False)
        parser.add_argument('--poison-fraction',
                        help='fraction of clients being malicious',
                        type=float,
                        default=0,
                        required=False)
        parser.add_argument('--poison-from',
                        help='epoch to start poisoning from',
                        type=float,
                        default=1,
                        required=False)

    def parse(self, args):
        self.poison_type = args.poison_type
        self.poison_fraction = args.poison_fraction
        self.poison_from = args.poison_from
