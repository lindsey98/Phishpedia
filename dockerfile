# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04
# Prevent interactive prompts during installation
ARG DEBIAN_FRONTEND=noninteractive
# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh
# Add Miniconda to PATH
ENV PATH="/opt/miniconda/bin:${PATH}"
# Set working directory
WORKDIR /workspace
# Install git, unzip and other dependencies
RUN apt-get install -y git unzip libgl1-mesa-glx libglib2.0-0
# Clone the Phishpedia project from GitHub into the container
RUN git clone https://github.com/lindsey98/Phishpedia.git /workspace/Phishpedia
# Change to the project directory and run setup.sh to configure the environment
WORKDIR /workspace/Phishpedia
# Install dos2unix
RUN apt-get install -y dos2unix
# Convert setup.sh to Unix format and RUN it
RUN dos2unix setup.sh
RUN chmod +x setup.sh
RUN bash setup.sh || true
# Ensure Conda is initialized and phishpedia environment is activated by default
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh && conda activate phishpedia" >> ~/.bashrc
# Set the default command to execute when the container starts
CMD ["bash", "-c", "source /opt/miniconda/etc/profile.d/conda.sh && conda activate phishpedia && cd /workspace/Phishpedia && /bin/bash"]