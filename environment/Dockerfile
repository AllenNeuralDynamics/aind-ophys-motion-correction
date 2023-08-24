# hash:sha256:99280683fb763da41f3d9cf24ffdb942310963bc393cfb11c0445b591bf7a603
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /
RUN pip3 install -U --no-cache-dir \
    aind-ophys-utils \
    argschema \
    h5py \
    imageio-ffmpeg \
    numpy \
    pandas \
    sbxreader \
    suite2p==0.13.1
    aind-ophys-pipeline-utils
