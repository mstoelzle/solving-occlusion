# Solving Occlusion

## Instructions for running

### 1. Installation:
All git submodules need to be initialized and updated:
```
git submodule update --init
```

As the generation of a synthetic dataset relies on the TerrainDataGenerator by Takahiro Miki and different raisim plugins (which only run on Ubuntu),
the following installation instruction need to be followed recursively after the `src/dataset_generation/synthetic_terrain_data_generator` git submodule is initialised:
https://bitbucket.org/tamiki/terrain_data_generator

This framework requires **Python 3.7.7**

**Note:** Cuda 10.2 needs to be installed and available.

It is recommended to use a package manager like Conda (https://docs.conda.io/en/latest/) to manage the Python version 
and all required Python packages.

The required Python packages can be installed as follows (within the Conda environment) in the root directory:
```
pip install -r requirements.txt --user
```

### 2. Running an experiment

We use JSON config files to specify all settings and parameters of our experiments. 
All config files need to be placed in a subdirectory of `{DIR_TO_REPO}/configs`.
Subsequently, an experiment can be started by stating the path relative to the `{DIR_TO_REPO}/configs` directory:

```
python main.py {CONFIG_NAME}.json
```

## Important components

### Task Path
The main ingredient to path learning are what we call task paths. 
They consist of different "training regimes" applied sequentially to a machine learning model. 
In our implementation they create tasks from their configs and are used as iterators to yield these tasks.
### Task
A task consists of a set of dataloaders, which are created from a base dataset (i.e. MNIST) and transformers applied to it. 
Moreover it also specifies a loss function, loss aggregator and the batch size for the dataloader.
In our implementation tasks measure their own runtime by using the "with" keyword.

### Controller
The controller is a member of the learning classes. It is implemented as an iterator that yields epochs until convergence 
or a maximum number of epochs is reached.

### Experiment
The experiment contains all other objects introduced above. It manages device placement and logdir creation.


