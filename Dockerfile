# FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install dependencies
# RUN apt-get update && \
#     apt-get install -y \
#     build-essential \
#     wget \
#     libssl-dev \
#     zlib1g-dev \
#     libbz2-dev \
#     libreadline-dev \
#     libsqlite3-dev \
#     curl \
#     llvm \
#     libncurses5-dev \
#     libncursesw5-dev \
#     xz-utils \
#     tk-dev \
#     libffi-dev \
#     liblzma-dev \
#     python-openssl
#     #git && \
#     #apt-get clean && \
#     #rm -rf /var/lib/apt/lists/*

# Install Python 3.6.15
# RUN wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz && \
#     tar xzf Python-3.6.15.tgz && \
#     cd Python-3.6.15 && \
#     ./configure --enable-optimizations && \
#     make altinstall && \
#     cd .. && \
#     rm -rf Python-3.6.15 Python-3.6.15.tgz
# 
# # Set Python 3.6.15 as the default python
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.6 1
# 
# # Install pip for Python 3.6.15
# RUN /usr/local/bin/python3.6 -m ensurepip && \
#     /usr/local/bin/python3.6 -m pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
   && pip install --no-cache-dir -r requirements.txt \
   && pip install --no-cache-dir wandb==0.15.11 \
   && pip install --no-cache-dir dataclasses   \
   && pip install --no-cache-dir click==7.1.0 \
   && pip install --no-cache-dir nibabel  \
   && pip install --no-cache-dir SimpleITK   

# Copy the rest of the application code into the container
COPY . /app

# Default command to run train.py followed by eval.py
RUN chmod +x /app/entry.sh
CMD ["/bin/bash", "/app/entry.sh"]


# docker run -it --name ams_seminar_bruno --gpus all -v /media/Fillmore/zigab/XX_teaching/brunoc/data/datasets:/app/datasets:ro -v /media/Fillmore/zigab/XX_teaching/brunoc/data/Teacher_deformations:/app/Teacher_deformations:ro -v /media/Fillmore/zigab/XX_teaching/brunoc/data/weights:/app/weights ams_bruno