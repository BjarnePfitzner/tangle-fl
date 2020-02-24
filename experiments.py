import os
import shutil
from sklearn.model_selection import ParameterGrid

params = {
    'dataset': ['femnist'],
    'model': ['cnn'],
    'num_rounds': [150],
    'eval_every': [10],
    'clients_per_round':  [10, 35, 50],
    'num_tips':  [2, 3, 5],
    'sample_size':  [1, 2, 5],
    'reference_avg_top':  [1, 2, 10, 50],
    'target_accuracy':  [0.7],
    'learning_rate':  [0.06],
    'poison_type':  ['NONE'],
    'poison_fraction':  [0.3],
    'poison_from':  [0],
}

trials_per_setup = 2

file_name = 'results.txt'
    
for p in ParameterGrid(params):
    with open(file_name, 'a+') as file:
        file.write('\n----------STARTING NEW PARAMETER SETUP--------\n')
    for i in range(trials_per_setup):
        os.system('rm -rf tangle_data')
        p['sample_size'] = p['sample_size'] * p['num_tips']
        command = "python3 main.py -dataset %s -model %s --num-rounds %s --eval-every %s --clients-per-round %s --num-tips %s --sample-size %s --reference-avg-top %s --target-accuracy %s -lr %s --poison-type %s --poison-fraction %s --poison-from %s"
        parameters = (p['dataset'], p['model'], p['num_rounds'], p['eval_every'], p['clients_per_round'], p['num_tips'], p['sample_size'], p['reference_avg_top'], p['target_accuracy'], p['learning_rate'], p['poison_type'], p['poison_fraction'], p['poison_from'])
        command = command % parameters
        with open(file_name, 'a+') as file:
            file.write('\n\n' + command + '\n')
        os.system(command)
