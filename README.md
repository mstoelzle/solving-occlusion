# Solving Occlusion

## Instructions

### 1. Prerequisites
This framework requires **Python 3.8.5**. The generation of synthetic datasets requires an Ubuntu environment. 

**Note:** To use efficient neural network training, Cuda 10.2 needs to be installed and available.

It is recommended to use a package manager like Conda (https://docs.conda.io/en/latest/) to manage the Python version 
and all required Python packages.

### 2. Initialisation of git submodules
All git submodules need to be initialized and updated:
```
git submodule update --init --recursive
```

### 3. Installation:
#### 3.1 Install all required PIP packages
The required Python packages can be installed as follows (within the Conda environment) in the root directory:
```
pip install -r requirements.txt --user
```

#### 3.2 Install the TerrainDataGenerator
As the generation of a synthetic dataset relies on the TerrainDataGenerator by Takahiro Miki and different raisim plugins (which only runs on Ubuntu),
the following installation instruction need to be followed recursively after the `src/dataset_generation/synthetic_terrain_data_generator` git submodule is initialised:
https://bitbucket.org/tamiki/terrain_data_generator

After following the installation instructions, the build directory needs to be added to the CMAKE_PREFIX_PATH:
```
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$LOCAL_BUILD
```

Additionally, the `terrain_data_generator` submodule has a requirement on OpenCV:
```
sudo apt-get install libopencv-dev
```

Finally, install the `terrain_data_generator` package:
```
pip install --user -e src/dataset_generation/synthetic_terrain_data_generator
```

#### 3.3 Install PyPatchMatch
We use the PatchMatch [[1]](#1) algorithm as a (traditional) baseline for in-painting of the occluded elevation maps.
If this baseline is specified for use in the config, the following installation steps to use the dependency [PyPatchMatch](https://github.com/vacancy/PyPatchMatch) need to be taken:
1. Install the pkg-config package
for macOS: `brew install pkg-config` for Ubuntu: `sudo apt-get install pkg-config`.
2. Manual build of OpenCV for [macOS](https://docs.opencv.org/master/d0/db2/tutorial_macos_install.html) and [Linux](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) 
Run this command (while adjusting the paths):
```
cmake -D CMAKE_BUILD_TYPE=Release -D \
         OPENCV_EXTRA_MODULES_PATH="${WORKSPACE}/opencv_contrib/modules" \
         PYTHON3_EXECUTABLE=~/miniconda3/envs/rsl_solving_occlusion/bin/python \
         PYTHON_INCLUDE_DIR=~/miniconda3/envs/rsl_solving_occlusion/include/python3.8 \
         PYTHON3_LIBRARY=~/miniconda3/envs/rsl_solving_occlusion/lib/libpython3.so \
         PYTHON3_NUMPY_INCLUDE_DIRS=~/rock/install/pip/lib/python3.8/site-packages/numpy/core/include \
         PYTHON3_PACKAGES_PATH=~/rock/install/pip/lib/python3.8/site-packages \
         OPENCV_GENERATE_PKGCONFIG=ON \
         -S "${WORKSPACE}/opencv" \
         -B "${WORKSPACE}/opencv_build"
```
and then make & install the build: 
```
cd ${WORKSPACE}/opencv_build && make -j7 && sudo make install
```

3. Set the environmental variable `export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig`
4. Install the PyPatchMatch package: `cd src/learning/models/baseline/py_patch_match & make`

### 4. Generating a dataset

We use JSON config files to specify all settings and parameters of our experiments. 
All config files need to be placed in a subdirectory of `{DIR_TO_REPO}/configs`.
Subsequently, a dataset generation can be started by stating the path relative to the `{DIR_TO_REPO}` directory:

```
python dataset_generation.py configs/{CONFIG_NAME}.json
```

### 5. Learning

We use JSON config files to specify all settings and parameters of our experiments. 
All config files need to be placed in a subdirectory of `{DIR_TO_REPO}/configs`. 
The absolute or relative path to the dataset needs to be specified in the JSON config.
Subsequently, a learning experiment can be started by stating the path relative to the `{DIR_TO_REPO}` directory:

```
python learning.py configs/{CONFIG_NAME}.json
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

## Citations
<a id="1">[1]</a> Barnes, Connelly, et al. 
"PatchMatch: A randomized correspondence algorithm for structural image editing." 
ACM Trans. Graph. 28.3 (2009): 24.


