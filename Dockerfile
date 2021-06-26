FROM pytorch/pytorch:latest

# Fix timezone issue
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# System packages
RUN apt-get update && apt-get install -y curl wget git

# Dependencies
RUN apt-get update && apt-get install -y cmake libeigen3-dev libopencv-dev pcl-tools

# install pybind11
RUN conda install pybind11

# install ROS1 Noetic
RUN apt update && sudo apt install -y ros-noetic-ros-base ros-noetic-grid-map

# RUN git clone https://github.com/mstoelzle/solving-occlusion
COPY . ./solving-occlusion
WORKDIR solving-occlusion

# install pip requirements
RUN pip3 install -r requirements.txt --user

# install PyPatchMatch
RUN echo pwd
RUN cd "src/learning/models/baseline/py_patch_match" && make




