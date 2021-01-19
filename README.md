# Solving Occlusion

## Instructions

### 1. Prerequisites
This framework requires **Python 3.9.1**. The generation of synthetic datasets requires an Ubuntu environment. 

**Note:** To use efficient neural network training, Cuda 11 needs to be installed and available.

It is recommended to use a package manager like Conda (https://docs.conda.io/en/latest/) to manage the Python version 
and all required Python packages.

### 2. Initialisation of git submodule
After setting the environment variable `$WORKSPACE` to a folder you want to work in, you should clone this repo:
```
git clone https://github.com/mstoelzle/solving occlusion $WORKSPACE/solving-occlusion && cd $WORKSPACE/solving-occlusion
```

All git submodules need to be initialized and updated:
```
git submodule update --init --recursive
```

### 3. Installation:
#### 3.1 Install all required PIP packages
The required Python packages can be installed as follows (within the Conda environment) in the root directory:
```
pip install -r ${WORKSPACE}/solving-occlusion/requirements.txt --user
```

#### 3.2 Build OpenCV from source (not required for now):

1. Install the pkg-config package
for macOS: `brew install pkg-config` for Ubuntu: `sudo apt-get install pkg-config`.
2. Manual build of OpenCV for [macOS](https://docs.opencv.org/master/d0/db2/tutorial_macos_install.html) and [Linux](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html).
Run this command (while adjusting the paths):
```
cmake -D CMAKE_BUILD_TYPE=Release -D \
         OPENCV_EXTRA_MODULES_PATH="${WORKSPACE}/opencv_contrib/modules" \
         PYTHON3_EXECUTABLE=~/miniconda3/envs/rsl_solving_occlusion/bin/python \
         PYTHON_INCLUDE_DIR=~/miniconda3/envs/rsl_solving_occlusion/include/python3.8 \
         PYTHON3_LIBRARY=~/miniconda3/envs/rsl_solving_occlusion/lib/libpython3.so \
         PYTHON3_NUMPY_INCLUDE_DIRS=~/rock/install/pip/lib/python3.8/site-packages/numpy/core/include \
         PYTHON3_PACKAGES_PATH=~/rock/install/pip/lib/python3.8/site-packages \
         -DOPENCV_GENERATE_PKGCONFIG=ON \
         -S "${WORKSPACE}/opencv" \
         -B "${WORKSPACE}/opencv_build"
```
and then make & install the build: 
```
cd ${WORKSPACE}/opencv_build && make -j7 && sudo make install
```

3. Set the environmental variable `export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig`

#### 3.3 Install the TerrainDataGenerator
**System requirements:** Ubuntu >= 16.04, g++ / gcc >= 6, CMake >= 3.10, CPU with support for avx2 instructions (produced within last 6 years)

First, please install RaiSim including its plugins raisimLib and raisimOgre: https://raisim.com/sections/Installation.html. Please acquire the appropriate license.

Initialise the raisim build directory: `export LOCAL_INSTALL=$WORKSPACE/raisim_build && mkdir $LOCAL_INSTALL`

As the generation of a synthetic dataset relies on the TerrainDataGenerator by Takahiro Miki,
the following installation instruction need to be followed recursively after the `src/dataset_generation/synthetic_terrain_data_generator` git submodule is initialised:
https://bitbucket.org/tamiki/terrain_data_generator

After following the installation instructions, the build directory needs to be added to the CMAKE_PREFIX_PATH:
```
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$LOCAL_INSTALL
```

Additionally, the `terrain_data_generator` submodule has a requirement on OpenCV:
```
sudo apt-get install libopencv-dev
```

Finally, install the `terrain_data_generator` package:
```
pip install --user -e "${WORKSPACE}/solving-occlusion/src/dataset_generation/synthetic_terrain_data_generator"
```

#### 3.4 Install PyPatchMatch
We use the PatchMatch [[1]](#1) algorithm as a (traditional) baseline for in-painting of the occluded elevation maps.

If this baseline is specified for use in the config, the following installation steps to use the dependency [PyPatchMatch](https://github.com/vacancy/PyPatchMatch) need to be taken:

```
cd "${WORKSPACE}/solving-occlusion/src/learning/models/baseline/py_patch_match" && make
```

#### 3.5 Install Traversability module
We first need to install the dependencies CMake, OpenCV and PCL. On macOS:
```
brew install cmake opencv pcl eigen
```

Subsequently, we can install the Python wrapper for the rock `perception-traversability` component with:
```
pip install --user -e "${WORKSPACE}/solving-occlusion/src/traversability/perception-traversability-pybind"
```

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

### Working with the Tenerife Dataset
We evaluate our methods on a dataset which was collected in June 2017 at the "Minas de San José" site in Tenerife using 
the Heavy Duty Planetary Rover (HDPR) as a Lunar Analogue. This dataset consists of images from three stereo cameras, one of which on a pan tilt unit, 
LiDAR and a Time of FLight (ToF) camera. It also includes an onboard Inertial Measurements Unit (IMU), an additional laser gyro for heading estimation
and Global Navigation Satellite System (GNSS) antenna for ground-truth absolute positioning. 
The dataset is stored in serialized [Rock pocolog logs](https://github.com/rock-core/tools-pocolog). 
We apply the GA SLAM [[2]](#2) technique on the raw data to extract occluded Digital Elevation Maps (DEMs).
Subsequently, the pocolog logs can be transformed to [Msgpacks](https://msgpack.org) using the following command:
```
pocolog2msgpack -l ga_slam.0.log -o ga_slam.msg
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
"PatchMatch: A randomized correspondence algorithm for structural image editing." ACM Trans. Graph. 28.3 (2009): 24.

<a id="2">[2]</a> Geromichalos, Dimitrios, et al. "SLAM for autonomous planetary rovers with global localization." 
Journal of Field Robotics 37.5 (2020): 830-847.


