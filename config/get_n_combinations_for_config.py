import sys
from typing import MutableMapping
from functools import reduce

from omegaconf import OmegaConf


def main(config_path):
    config = OmegaConf.load(f'./experiment/{config_path}.yaml')
    params = config.hydra.sweeper.params
    flat_params = flatten_dict(params)
    n_combinations = reduce(lambda a, b: a*b, map(lambda x: len(x), flat_params.values()))
    print(n_combinations)


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


if __name__ == '__main__':
    main(sys.argv[1])