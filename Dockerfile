FROM pytorch/pytorch:latest

# Fix timezone issue
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# System packages
RUN apt-get update && apt-get install -y curl wget git

# Dependencies
RUN apt-get update && apt-get install -y cmake libeigen3-dev libopencv-dev

# install pybind11
RUN conda install pybind11

# RUN git clone https://github.com/mstoelzle/solving-occlusion
COPY . ./solving-occlusion
WORKDIR solving-occlusion

RUN pip3 install -r requirements.txt --user




