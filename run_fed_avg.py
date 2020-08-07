import subprocess

params = {
    'dataset': 'femnistclustered',   # is expected to be one value to construct default experiment name
    'model': 'cnn',       # is expected to be one value to construct default experiment name
    'num_rounds': 15,
    'eval_on_fraction': 0.05,
    'clients_per_round': 10,
    'model_data_dir': '../data/femnist-data-clustered-alt',
    'batch_size': 10,
    'learning_rate': 0.005,
}

def main():
    command = 'python -m fedavg ' \
                '-dataset %s ' \
                '-model %s ' \
                '--num-rounds %s ' \
                '--eval-on-fraction %s ' \
                '--clients-per-round %s ' \
                '--model-data-dir %s ' \
                '--batch-size %s ' \
                '-lr %s '
    parameters = (
        params['dataset'],
        params['model'],
        params['num_rounds'],
        params['eval_on_fraction'],
        params['clients_per_round'],
        params['model_data_dir'],
        params['batch_size'],
        params['learning_rate'])
    command = command % parameters

    training = subprocess.Popen(command.split(" "))
    training.wait()


if __name__ == '__main__':
    main()