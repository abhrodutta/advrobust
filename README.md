# Adversarial Attacks based on Semi Definite Programming Approaches

This code accompanies the NeurIPS 2019 Paper : "On Robustness to Adversarial Examples and Polynomial Optimization" by Pranjal Awasthi, Abhratanu Dutta and Aravindan Vijayaraghavan. 

## Getting Started

The config.json file contains various tunable parameters. The trained models are already present in the models directory. To retrain a model, run train.py.

## Running the tests

Run pgdfail.py first. This runs the PGD attack on the mnist dataset and creates a file that stores the indices where the PGD attack fails to come up with an adversarial example.

Then run full_sdp_attack_pgdfail.py. This takes in the file created by pgdfail.py, selects 100 mnist examples at random (where PGD attack fails) and runs SDP attack (described in the paper) on these examples. Each example takes around 3 to 4 minutes to run.

## Acknowledgments

Thanks to the mnist challenge and leaderboard put up by Alexander Madry's lab. Some of the parts were implemented from a fork of https://github.com/MadryLab/mnist_challenge.
