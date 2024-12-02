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
# Install git
RUN apt-get install -y git
# Clone the Phishpedia project from GitHub into the container
RUN git clone https://github.com/lindsey98/Phishpedia.git /workspace/Phishpedia
# Change to the project directory and run setup.sh to configure the environment
WORKDIR /workspace/Phishpedia
# Install dos2unix
RUN apt-get install -y dos2unix
# Convert setup.sh to Unix format and RUN it
RUN dos2unix setup.sh
RUN chmod +x setup.sh
RUN bash setup.sh
# Set the default command to execute when the container starts
CMD ["bash", "-c", "cd /workspace/Phishpedia && /bin/bash"]