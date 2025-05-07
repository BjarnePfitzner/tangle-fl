# Federated Learning DAG Experiments

This repository contains software artifacts to reproduce the experiments presented in the Middleware '21 paper "Implicit Model Specialization through DAG-based Decentralized Federated Learning"
It was later adapted for personal use by Bjarne Pfitzner to facilitate further experimentation for his doctoral thesis.

## General Usage

Install packages using conda: `conda env create -f environment.yml`.
Activate environment: `conda activate tangle-fl`.
Basic usage: `python main.py --experiment={experiment_file} {parameter.overwrite}={value}`
Experiment configuration is managed with the Hydra package, which reads the configuration from the `config` folder.

The project uses Weights and Biases (WandB) for logging. Either disable it by adding `wandb.disabled=True` to the main call, or alter the wandb setup in `main.py:58`.

To view a DAG (sometimes called a tangle) in a web browser, run `python -m http.server` in the repository root and open [http://localhost:8000/viewer/](http://localhost:8000/viewer/). Enter the name of your experiment run and adjust the round slider to see something.

## Obtaining the datasets

The contents of the `./data` directory can be obtained from https://data.osmhpi.de/ipfs/QmQMe1Bd8X7tqQHWqcuS17AQZUqcfRQmNRgrenJD2o8xsS/.

## Reproduction of the evaluation in the thesis

The experiements in the thesis can be reproduced by selecting different experiment scripts in the main call using the argument `--experiment` and selecting a yaml file from within `config/experiment`.
