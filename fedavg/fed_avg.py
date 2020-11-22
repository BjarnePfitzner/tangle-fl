import sys
import time
import ray
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tangle.lab import Lab
from tangle.core import Node

class DummyTipSelector():
    def compute_ratings(self, node):
        pass

@ray.remote
def train_single(client_id, data, model_config, global_params, seed):
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    model = Lab.create_client_model(seed, model_config)

    data = {'x': ray.get(data['x']), 'y': ray.get(data['y'])}

    node = Node(None, None, DummyTipSelector(), client_id, None, data, None, model)
    new_params = node.train(global_params)

    global_params = np.array(global_params)
    new_params = np.array(new_params)

    return global_params - new_params, len(data['y'])


def train(dataset, run_config, model_config, seed, lr=1.):
    model = Lab.create_client_model(seed, model_config)
    global_params = model.get_params()

    accuracies = []

    for epoch in range(run_config.num_rounds):
        print("Started training for round %d" % epoch)
        start = time.time()
        clients = dataset.select_clients(epoch, run_config.clients_per_round)

        futures = [train_single.remote(client_id, dataset.remote_train_data[client_id], model_config, global_params, seed) for (client_id, cluster_id) in clients]

        param_update = 0
        total_weight = 0
        for param_diff, weight in ray.get(futures):
            param_update += param_diff * weight
            total_weight += weight

        param_update /= total_weight

        global_params -= lr * param_update

        print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))

        # if evaluate on same clients each round
        if True:
            accuracies_this_round = test_this_round(global_params, dataset, model_config, clients, seed)
            accuracies.append(accuracies_this_round)
            print(f'Test Accuracy on selected clients: {accuracies_this_round}')

        if run_config.eval_every != -1 and epoch % run_config.eval_every == 0:
            mean_accuracy = test(global_params, dataset, model_config, run_config.eval_on_fraction, seed)
            print(f'Test Accuracy on {int(run_config.eval_on_fraction * len(dataset.clients))} clients: {mean_accuracy}')
        sys.stdout.flush()

    # compute average test error
    mean_accuracy = test(global_params, dataset, model_config, 1, seed)
    print(f'Test Accuracy on all Clients: {mean_accuracy}')

    plot_accuracy_boxplot(accuracies)

@ray.remote
def test_single(client_id, global_params, test_data, model_config, seed):
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = Lab.create_client_model(seed, model_config)
    test_data = {'x': ray.get(test_data['x']), 'y': ray.get(test_data['y'])}
    node = Node(None, None, DummyTipSelector(), client_id, None, None, test_data, model)

    results = node.test(global_params, 'test')

    return results['accuracy']

def test_this_round(global_params, dataset, model_config, clients_to_test_on, seed):
    futures = [test_single.remote(client_id, global_params, dataset.remote_train_data[client_id], model_config, seed)
               for (client_id, _) in clients_to_test_on]
    
    return ray.get(futures)

def test(global_params, dataset, model_config, eval_on_fraction, seed):
    client_indices = np.random.choice(range(len(dataset.clients)),
                                      min(int(len(dataset.clients) * eval_on_fraction), len(dataset.clients)),
                                      replace=False)
    validation_clients = [dataset.clients[i] for i in client_indices]
    futures = [test_single.remote(client_id, global_params, dataset.remote_train_data[client_id], model_config, seed)
               for (client_id, _) in validation_clients]

    return np.mean(ray.get(futures))

def plot_accuracy_boxplot(data, print_avg_acc=False):
    last_generation = len(data)

    plt.boxplot(data)

    if print_avg_acc:
        plt.plot([i for i in range(last_generation)], [np.mean(x) for x in data])
    
    # Settings for plot
    plt.title("Accuracy per round")
    
    plt.xlabel("Round")
    plt.xticks([i for i in range(last_generation)], [i if i % 5 == 0 else '' for i in range(last_generation)])
    
    plt.ylabel("")
    
    analysis_filepath = ("fed_avg_accuracy_per_round.png")
    plt.savefig(analysis_filepath)
    
    plt.clf()
