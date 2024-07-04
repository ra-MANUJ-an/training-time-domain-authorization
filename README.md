# A Benchmark for Training-time Domain Authorization

**Project Goal:** A Public Reusable and Replicable Toolkit for Rigerously Evaluating Training-time domain authorization solutions.

After evaluating a defence on this benchmark, an evaluator should be able to say: This defence provide training time domain authorization with a high-bar of empirical evidence.

**Inspiration:** BIER, GEM, DecodingTrust, HarmBench

## Setup & Dev Stuff
```
$ curl -sSL https://install.python-poetry.org | python3 -
$ poetry install
```

The project uses Ruff with a Ruff pre-commit hook just for consistent styling.

### Development workflow

- Create or work on an existing issue in this repo
- Create a branch that is named after this issue
- When finished add a pull request to the `main` branch for review and tag the discord channel with the issue

## Adding dependencies

Use `poetry add` for all dependencies


## Project Structure
- `data` containts static data files
- `notebooks` contains jupyter notebooks
- `scripts` scripts used for analysis and other tasks
- `training_time_domain_authorization` contains the source code for the project
- `results` contains the results of the experiments
- `models` contains the trained models
- `experiments` contains the scripts to run the experiments
- `training_time_domain_authorization/datasets` - contains data loaders and evaluations for each dataset
- `training_time_domain_authorization/losses` - contains custom loss functions

Experiment scripts are located in the `experiments` directory. The scripts are named according to the experiment they run. The scripts are written in bash and are used to run the experiments. The scripts are used to run the experiments and save the results in the `results` directory. 

`main.py` is the main entry point into the project.

`arguments.py` contains all the arguments which should be a global variable to avoid code mess

## Experiments

Add documentation on all experiments added here:
- `experiments/train_viggo.sh`: a toy demonstration of how to train a model on the GEM Viggo dataset

