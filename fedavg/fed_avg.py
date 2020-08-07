import time
import ray
import numpy as np

from tangle.lab import Lab
from tangle.core import Node

@ray.remote
def train_single(client_id, data, model_config, global_params, seed):
    model = Lab.create_client_model(seed, model_config)

    node = Node(None, None, None, client_id, None, data, None, model)
    new_params = node.train(global_params)

    return global_params - new_params, len(data)


def train(dataset, run_config, model_config, seed, lr=1.):
    model = Lab.create_client_model(seed, model_config)
    global_params = model.get_params()
    for epoch in range(run_config.num_rounds):
        start = time.time()
        clients = dataset.select_clients(epoch, run_config.clients_per_round)

        futures = [train_single.remote(c, dataset.remote_train_data[c], model_config, global_params, seed) for c in clients]

        param_update = 0
        total_weight = 0
        for param_diff, weight in ray.get(futures):
            param_update += param_diff + weight
            total_weight += weight

        param_update /= total_weight

        global_params += lr * param_update

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # compute average test error
    mean_accuracy = test(global_params, dataset, model_config, seed)
    print(mean_accuracy)


@ray.remote
def test_single(client_id, global_params, test_data, model_config, seed):
    model = Lab.create_client_model(seed, model_config)
    node = Node(None, None, None, client_id, None, None, test_data, model)

    results = node.test(global_params, 'test')

    return results['accuracy']


def test(global_params, dataset, model_config, seed, client_fraction=0.05):
    client_indices = np.random.choice(range(len(dataset.clients)),
                                      min(int(len(dataset.clients) * client_fraction), len(dataset.clients)),
                                      replace=False)
    validation_clients = [dataset.clients[i] for i in client_indices]
    futures = [test_single.remote(client_id, global_params, dataset.remote_train_data[client_id], model_config, seed)
               for client_id in validation_clients]

    return np.mean(ray.get(futures))

