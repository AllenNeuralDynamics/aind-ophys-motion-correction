# hash:sha256:99280683fb763da41f3d9cf24ffdb942310963bc393cfb11c0445b591bf7a603
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PIPELINE_URL="https://codeocean.allenneuraldynamics.org/capsule/5619253/tree"
ENV PIPELINE_VERSION="1.0"
ENV VERSION="8.0"
ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]
RUN pip3 install -U --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /
RUN pip3 install -U --no-cache-dir \
    aind-ophys-utils \
    sbxreader \
    suite2p \
    aind-data-schema==1.1.0 \
    pydantic==2.8.1 \
    aind-qcportal-schema \
    aind-log-utils
