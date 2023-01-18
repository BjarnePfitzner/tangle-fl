class WandBConfiguration:
    run_id: int
    group: str
    name: str

    def __init__(self):
        self.run_id = None
        self.group = None
        self.name = None

    def define_args(self, parser):
        parser.add_argument('-run-id',
                    help='WandB ID for the current run',
                    type=str,
                    required=True)
        parser.add_argument('--group',
                    help='group for a multi-run',
                    type=str,
                    default=None,
                    required=False)
        parser.add_argument('--name',
                    help='name of the run on WandB',
                    type=str,
                    default=None,
                    required=False)

    def parse(self, args):
        self.run_id = args.run_id
        self.group = args.group
        self.name = args.name
