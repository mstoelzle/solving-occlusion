FROM pytorch/pytorch:latest

# Fix timezone issue
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# System packages
RUN apt-get update && apt-get install -y curl wget git

# Dependencies
RUN apt-get update && apt-get install -y cmake libeigen3-dev libopencv-dev pcl-tools

# install ROS1 Noetic
# https://github.com/osrf/docker_images/blob/11c613986e35a1f36fd0fa18b49173e0c564cf1d/ros/noetic/ubuntu/focal/ros-core/Dockerfile
# install support packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*
# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list
# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# install ros packages
RUN apt-get update && apt-get install -y ros-noetic-ros-base ros-noetic-grid-map && rm -rf /var/lib/apt/lists/*
# install ros grid map packages
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    ros-noetic-grid-map ros-noetic-grid-map-core ros-noetic-grid-map-cv \
#    ros-noetic-grid-map-demos ros-noetic-grid-map-filters ros-noetic-grid-map-loader \
#    ros-noetic-grid-map-msgs ros-noetic-grid-map-ros ros-noetic-grid-map-rviz-plugin ros-noetic-grid-map-visualization

# install pybind11
RUN conda install pybind11

# RUN git clone https://github.com/mstoelzle/solving-occlusion
COPY . ./solving-occlusion
WORKDIR solving-occlusion
RUN echo "WORKDIR={pwd}"

# install pip requirements
RUN pip3 install -r requirements.txt --user

# install PyPatchMatch
# RUN cd "src/learning/models/baseline/py_patch_match" && make




