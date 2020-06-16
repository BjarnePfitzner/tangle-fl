import ray

from ..lab import Lab, LabConfiguration, ModelConfiguration, parse_args

class RayLab(Lab):
    def create_genesis(self):

        @ray.remote
        def _create_genesis(self):
            return super().create_genesis()

        return ray.get(_create_genesis.remote(self))


def main():
    args = parse_args()

    ray.init(webui_host='0.0.0.0')

    config = LabConfiguration(
        args.seed,
        args.start_from_round,
        args.tangle_dir,
        args.tangle_tx_dir
    )

    model_config = ModelConfiguration(
        args.dataset,
        args.model,
        args.lr
    )

    lab = RayLab(config, model_config)
