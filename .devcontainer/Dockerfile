FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install build-essential git zip ffmpeg libsm6 libxext6 -y

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN useradd -ms /bin/bash docker_user

USER docker_user

WORKDIR /home/docker_user/workspace
ENV PYTHONPATH '${PYTHONPATH}:/home/docker_user/workspace'
