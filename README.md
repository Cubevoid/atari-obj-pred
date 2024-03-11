# Atari Object Prediction

## Introduction

See [proposal.pdf](proposal.pdf) for the full project proposal.

## Requirements

**Note:** Windows is not supported. Please use Linux.

- Install Pipenv
- Install dependencies using `pipenv install`
- Enter the environment using `pipenv shell`

## OCAtari

You can test OCAtari on the Assault game by running:
`python demo_ocatari.py -p assault_dqn.gz`

## Setup

Run `setup.sh` to download the model weights for SAM

## Running the code

The main executables are located under `src/scripts/`.
Use the following command to run the code:
`python -m src.scripts.<script_name>` (without the `.py` extension)
