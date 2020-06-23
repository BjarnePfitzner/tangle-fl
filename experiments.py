import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

from sklearn.model_selection import ParameterGrid

#############################################################################
############################# Parameter section #############################
#############################################################################

data_generation_path = "leaf/data/synthetic/"
data_generation_commands = ["rm -r data", "rm -r meta", "python main.py -num-tasks 1000 -num-classes 5 -num-dim 60 --prob-clusters 0.2 0.3 0.3 0.2"]
data_preprocessing_commands = ["bash ./preprocess.sh -s niid --sf 1.0 -k 5 -t sample --tf 0.85"]

params = {
    'dataset': ['synthetic'],   # is expected to be one value to construct default experiment name
    'model': ['log_reg'],       # is expected to be one value to construct default experiment name
    'num_rounds': [200],
    'eval_every': [-1],
    'clients_per_round': [10],
    'num_tips': [2],
    'sample_size': [2],
    'reference_avg_top': [1],
    'target_accuracy': [1],
    'learning_rate':  [0.005],
    'poison_type': ['NONE'],
    'poison_fraction': [0],
    'poison_from': [1],
    'acc-tip-selection-strategy': ['WALK'],
    'acc-cumulate-ratings': ['False', 'True'],
    'acc-ratings-to-weights': ['LINEAR'],
    'acc-select-from-weights': ['WEIGHTED_CHOICE'],
    'acc-alpha': [0.001]
}

tip_selector_start_rounds = {
    'tip_selector': [],
    'acc_tip_selector': [],
    'mal_tip_selector': []
}

##############################################################################
########################## End of Parameter section ##########################
##############################################################################

def main():
    traings_data_output_filename = 'data_generation.log'
    setup_filename = '1_setup.log'
    console_output_filename = '2_training.log'

    #exit_if_repo_not_clean()

    args = parse_args()
    experiment_folder = prepare_exp_folder(args)

    print("[Info]: Experiment results and log data will be stored at %s" % experiment_folder)

    git_hash = get_git_hash()
    cwd = get_current_working_directory()
    generate_and_preprocess_data(experiment_folder + '/' + traings_data_output_filename)
    change_working_directory_to(cwd)
    run_and_document_experiments(args, experiment_folder, setup_filename, console_output_filename, git_hash)

def exit_if_repo_not_clean():
    proc = subprocess.Popen(['git', 'status', '--porcelain'], stdout=subprocess.PIPE)

    try:
        dirty_files, errs = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        print('[Error]: Could not check git status!: %s' % errs, file=sys.stderr)
        exit(1)
    
    if dirty_files:
        print('[Error]: You have uncommited changes. Please commit them before continuing. No experiments will be executed.', file=sys.stderr)
        exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Run and document an experiment.')
    parser.add_argument('--name', help='The name of the ex1periment. Results will be stored under ../experiments/<name>. Default: <dataset>-<model>-<exp_number>')
    parser.add_argument('--overwrite_okay', type=bool, default=False, help='Overwrite existing experiment with same name. Default: False')
    args = parser.parse_args()

    return args

def prepare_exp_folder(args):
    experiments_base = '../experiments'
    os.makedirs(experiments_base, exist_ok=True)

    if not args.name:
        default_prefix = "%s-%s" % (params['dataset'][0], params['model'][0])

        # Find other experiments with default names
        all_experiments = next(os.walk(experiments_base))[1]
        default_exps = [exp for exp in all_experiments if re.match("^(%s-\d+)$" % default_prefix, exp)]

        # Find the last experiments with default name and increment id
        if len(default_exps) == 0:
            next_default_exp_id = 0
        else:
            default_exp_ids = [int(exp.split("-")[2]) for exp in default_exps]
            default_exp_ids.sort()
            next_default_exp_id = default_exp_ids[-1] + 1
        
        args.name = "%s-%d" % (default_prefix, next_default_exp_id)
    
    exp_name = args.name

    experiment_folder = experiments_base + '/' + exp_name

     # check, if existing experiment exists
    if (os.path.exists(experiment_folder) and not args.overwrite_okay):
        print('[Error]: Experiment "%s" already exists! To overwrite set --overwrite_okay to True' % exp_name, file=sys.stderr)
        exit(1)
    
    os.makedirs(experiment_folder, exist_ok=True)

    return experiment_folder

def get_git_hash():
    proc = subprocess.Popen(['git', 'rev-parse', '--verify', 'HEAD'], stdout=subprocess.PIPE)
    try:
        git_hash, errs = proc.communicate(timeout=3)
        git_hash = git_hash.decode("utf-8")
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        git_hash = 'Could not get Githash!: %s' % errs
    
    return git_hash

def get_current_working_directory():
    return os.getcwd()

def generate_and_preprocess_data(traings_data_output_file):
    with open(traings_data_output_file, 'w+') as file:
        print('Data generation commands: %s' % data_generation_commands, file=file)
        print('Data preprocessing commands: %s' % data_preprocessing_commands, file=file)
        print('', file=file)

    with open(traings_data_output_file, 'a+') as file:
        # change to data generation dir
        os.chdir(data_generation_path)

        print('Generating data...')
        for command in data_generation_commands:
            generation = subprocess.Popen(command.split(" "), stdout=file, stderr=file)
            generation.wait()


        print('Preprocessing data...')
        for command in data_preprocessing_commands:
            preprocessing = subprocess.Popen(command.split(" "), stdout=file, stderr=file)
            preprocessing.wait()

def change_working_directory_to(working_directory):
    os.chdir(working_directory)

def run_and_document_experiments(args, experiments_dir, setup_filename, console_output_filename, git_hash):
    parameter_grid = ParameterGrid(params)
    print(f'Starting experiments for {len(parameter_grid)} parameter combinations...')
    for idx, p in enumerate(parameter_grid):
        # Create folder for that run
        experiment_folder = experiments_dir + '/config_%s' % idx
        os.makedirs(experiment_folder, exist_ok=True)

        # Prepare execution command
        command = 'python main.py ' \
            '-dataset %s ' \
            '-model %s ' \
            '--num-rounds %s ' \
            '--eval-every %s ' \
            '--clients-per-round %s ' \
            '--num-tips %s ' \
            '--sample-size %s ' \
            '--reference-avg-top %s ' \
            '--target-accuracy %s ' \
            '-lr %s ' \
            '--poison-type %s ' \
            '--poison-fraction %s ' \
            '--poison-from %s '\
            '--acc-tip-selection-strategy %s ' \
            '--acc-cumulate-ratings %s ' \
            '--acc-ratings-to-weights %s ' \
            '--acc-select-from-weights %s ' \
            '--tangle-dir %s'
        parameters = (
            p['dataset'],
            p['model'],
            p['num_rounds'],
            p['eval_every'],
            p['clients_per_round'],
            p['num_tips'],
            p['sample_size'],
            p['reference_avg_top'],
            p['target_accuracy'],
            p['learning_rate'],
            p['poison_type'],
            p['poison_fraction'],
            p['poison_from'],
            p['acc-tip-selection-strategy'],
            p['acc-cumulate-ratings'],
            p['acc-ratings-to-weights'],
            p['acc-select-from-weights'],
            experiment_folder)
        command = command % parameters

        for start_round in tip_selector_start_rounds['tip_selector']:
            command += ' --tip-selector-from %s' %start_round
        for start_round in tip_selector_start_rounds['acc_tip_selector']:
            command += ' --acc-tip-selector-from %s' %start_round
        for start_round in tip_selector_start_rounds['mal_tip_selector']:
            command += ' --mal-tip-selector-from %s' %start_round

        start_time = datetime.datetime.now()

        # Print Parameters and command
        with open(experiment_folder + '/' + setup_filename, 'w+') as file:
            print('Data generation commands: %s' % data_generation_commands, file=file)
            print('Data preprocessing commands: %s' % data_preprocessing_commands, file=file)
            print('', file=file)
            print('StartTime: %s' % start_time, file=file)
            print('Githash: %s' % git_hash, file=file)
            print('Parameters:', file=file)
            print(json.dumps(p, indent=4), file=file)
            print('Command: %s' % command, file=file)

        # Execute training
        print('Training started...')
        with open(experiment_folder + '/' + console_output_filename, 'w+') as file:
            training = subprocess.Popen(command.split(" "), stdout=file, stderr=file)
            training.wait()

        # Document end of training
        print('Training finished. Documenting results...')
        with open(experiment_folder + '/' + setup_filename, 'a+') as file:
            end_time = datetime.datetime.now()
            print('EndTime: %s' % end_time, file=file)
            print('Duration Training: %s' % (end_time - start_time), file=file)

if __name__ == "__main__":
    main()
