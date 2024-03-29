# Solving Occlusion in Terrain Mapping with Neural Networks
[![ci](https://github.com/mstoelzle/solving-occlusion/actions/workflows/main.yml/badge.svg)](https://github.com/mstoelzle/solving-occlusion/actions/workflows/main.yml)
## Abstract
Accurate and complete terrain maps enhance the awareness of autonomous robots and enable safe and optimal path planning. Rocks and topography often create occlusions and lead to missing elevation information in the Digital Elevation Map (DEM). Currently, mostly traditional inpainting techniques based on diffusion or patch-matching are used by autonomous mobile robots to fill-in incomplete DEMs. These methods cannot leverage the high-level terrain characteristics and the geometric constraints of line of sight we humans use intuitively to predict occluded areas. We propose to use neural networks to reconstruct the occluded areas in DEMs. We introduce a self-supervised learning approach capable of training on real-world data without a need for ground-truth information. We accomplish this by adding artificial occlusion to the incomplete elevation maps constructed on a real robot by performing ray casting. We first evaluate a supervised learning approach on synthetic data for which we have the full ground-truth available and subsequently move to several real-world datasets. These real-world datasets were recorded during autonomous exploration of both structured and unstructured terrain with a legged robot, and additionally in a planetary scenario on Lunar analogue terrain. We state a significant improvement compared to the Telea and Navier-Stokes baseline methods both on synthetic terrain and for the real-world datasets. Our neural network is able to run in real-time on both CPU and GPU with suitable sampling rates for autonomous ground robots.

## Paper and Link
Our work has been published in the IEEE Robotics and Automation Letters (RA-L). Please refer to the paper on [IEEE Xplore](https://ieeexplore.ieee.org/document/9676411) or on [ArXiv](https://arxiv.org/abs/2109.07150).

We invite you to see our method in action in a video, where we record the inference of our method on the Gonzen mine dataset recorded with the ANYmal C legged robot: https://youtu.be/2Khxeto62LQ

Please cite our paper if you use our method in your work:
```bibtex
@ARTICLE{stolzle2022reconstructing,
  author={Stolzle, Maximilian and Miki, Takahiro and Gerdes, Levin and Azkarate, Martin and Hutter, Marco},
  journal={IEEE Robotics and Automation Letters}, 
  title={Reconstructing occluded Elevation Information in Terrain Maps with Self-supervised Learning}, 
  year={2022},
  volume={7},
  number={2},
  pages={1697-1704},
  doi={10.1109/LRA.2022.3141662}
}
```

## Instructions

### 1. Prerequisites
This framework requires > **Python 3.9.2**. The generation of synthetic datasets requires an Ubuntu environment. 

**Note:** To use efficient neural network training, CUDA 11.* needs to be installed and available.

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
#### 3.0 Install C++ dependencies
Please install the following C++ dependencies to use this repo:

```bash
conda install cmake pybind11 eigen
```
or alternatively on macOS with homebrew:
```bash
brew install cmake pybind11 eigen
```
or on Ubuntu with:
```bash
sudo apt install cmake python-pybind11 libeigen3-dev
```

On Windows we need to additionally install Visual Studio C++ to build Python packages and subsequently install dlib via pip:
```
pip install dlib
```


#### 3.1 Install all required Conda and PIP packages
If we want to leverage a NVIDIA GPU to train and infer the neural network, we need to install PyTorch first using conda:
```
conda install pytorch=1.8 torchvision=0.9 torchaudio=0.8 cudatoolkit=11.1 -c pytorch -c conda-forge
```

The required Python packages can be installed as follows (within the Conda environment) in the root directory:
```
pip install -r ${WORKSPACE}/solving-occlusion/requirements.txt --user
```

#### 3.2 Install ROS
We rely on ROS Noetic to read datasets stored in rosbags and process them in our DatasetGeneration component.

On Ubuntu this can be done with:
```
sudo apt install ros-noetic-ros-base ros-noetic-grid-map
```

or on macOS with (requires Python 3.6.* or 3.8.* for now):
```
conda install -c robostack ros-noetic-ros-base ros-noetic-grid-map
```

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
pip install git+https://github.com/esa-prl/perception-traversability-pybind.git
```

#### 3.6 Install rock-tools-pocolog_pybind module
We rely on custom pocolog Python bindings to read pocolog logs containing the elevation maps generated by the GA Slam 
method on the Tenerife dataset (see [Tenerife Dataset](#tenerife-dataset)).
We rely on an existing [Rock](https://www.rock-robotics.org) installation to build these pocolog python bindings. 
Please add the [tools-pocolog_pybind](https://github.com/esa-prl/tools-pocolog_pybind) repository to your Rock installation in the folder `tools/pocolog_pybind`.

Subsequently, we can install the Python wrapper with:
```
pip install --user /home/user/rock/tools/pocolog_pybind
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
python main.py configs/{CONFIG_NAME}.json
```

### 6. Visualization

We immediately visualize the trained model after all tasks of an experiment (e.g. seed) have completed as specified in 
`experiment/visualization` section of the config. That said, you can also visualize an experiment manually. 
In this case, please specify the path to the `config.json` within the experiment log directory:
```
python visualization.py {PATH_TO_EXPERIMENT_LOGDIR}/config.json
```

## Important notes

### Task Path
The main ingredient to path learning are what we call task paths. 
They consist of different "training regimes" applied sequentially to a machine learning model. 
In our implementation they create tasks from their configs and are used as iterators to yield these tasks.
### Task
A task consists of a set of dataloaders, which are created from a base dataset and transformers applied to it. 
Moreover it also specifies a loss function, loss aggregator and the batch size for the dataloader.
In our implementation tasks measure their own runtime by using the "with" keyword.

### Controller
The controller is a member of the learning classes. It is implemented as an iterator that yields epochs until convergence 
or a maximum number of epochs is reached.

### Experiment
The experiment contains all other objects introduced above. It manages device placement and logdir creation.

### Tenerife Dataset
We evaluate our methods on a dataset which was collected in June 2017 at the "Minas de San José" site in Tenerife using 
the Heavy Duty Planetary Rover (HDPR) as a Lunar Analogue. This dataset consists of images from three stereo cameras, one of which on a pan tilt unit, 
LiDAR and a Time of Flight (ToF) camera. It also includes an onboard Inertial Measurements Unit (IMU), an additional laser gyro for heading estimation
and Global Navigation Satellite System (GNSS) antenna for ground-truth absolute positioning. 
The dataset is stored in serialized [Rock pocolog logs](https://github.com/rock-core/tools-pocolog). 
We apply the GA SLAM [[2]](#2) technique on the raw data to extract occluded Digital Elevation Maps (DEMs).

### Canadian Planetary Emulation Terrain Energy-Aware Rover Navigation Dataset
We benchmark or methods on the enav-planetary dataset [[3]](#3).
We need to project the stitched point clouds stored in the dataset to 2.5D elevation maps. 
We use the [GridMap ROS](https://github.com/mstoelzle/grid_map/tree/from_point_cloud) package to transform ROS 
`sensors_msgs/PointCloud2` messages to `grid_map_msgs/grid_map`.
1. Download rosbag with point clouds from: https://starslab.ca/enav-planetary-dataset/
2. Copy grid map PCL config: 
   ```bash
   cp {PATH_TO_SOLVING_OCCLUSION}/src/ros/configs/grid_map_pcl_params.yml {PATH_TO_GRID_MAP}/grid_map_pcl/parameters.yml
   ```
2. Generate rosbag with grid map messages from rosbag with point clouds. Run the following three ROS1 commands each in a separate terminal:

Converter ROS node:
```bash
roslaunch grid_map_pcl PointCloud2_to_GridMap_msg_node.launch
```
Save `/grid_map` topic to a new rosbag:
```bash
rosbag record /grid_map
```
Replay rosbag with point cloud 2 messages at rate of 1% of original speed:
```bash
rosbag play -r 0.01 run1_clouds_only.bag
```

## Sample commands
Generate a synthetic height map dataset:
```bash
python dataset_generation.py configs/dg/synthetic_height_map_dataset_generation.json
```
Train on the synthetic height map terrain using supervised learning (after adjusting the path to the dataset in the config):
````bash
python main.py configs/learning/height_map/unet_synthetic_height_map_learning_supervised.json
````
Generate a HDF5 solving occlusion dataset from a rosbag containing GridMap messages (in this case for ENAV planetary dataset):
```bash
python dataset_generation.py configs/dg/enav_planetary.json
```
Train using self-supervised learning on a real-world solving occlusion HDF5 dataset:
```bash
python main.py configs/learning/enav/unet_enav_learning_self_supervision_raycasting_seed_101.json
```

## Citations
<a id="1">[1]</a> Barnes, Connelly, et al. 
"PatchMatch: A randomized correspondence algorithm for structural image editing." 
ACM Trans. Graph. 28.3 (2009): 24.

<a id="2">[2]</a> Geromichalos, Dimitrios, et al. "SLAM for autonomous planetary rovers with global localization." 
Journal of Field Robotics 37.5 (2020): 830-847.

<a id="3">[3]</a> Lamarre, Olivier, et al. 
"The canadian planetary emulation terrain energy-aware rover navigation dataset." 
The International Journal of Robotics Research 39.6 (2020): 641-650.


