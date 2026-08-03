FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV PIPELINE_URL="https://codeocean.allenneuraldynamics.org/capsule/5619253/tree"
ENV PIPELINE_VERSION="1.0"
ENV VERSION="8.0"

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]
RUN pip3 install -U --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /
RUN pip3 install -U --no-cache-dir \
    aind-ophys-utils \
    sbxreader \
    suite2p==0.14.4 \
    aind-data-schema==1.1.0 \
    pydantic==2.8.1 \
    aind-qcportal-schema \
    aind-log-utils \
    pydantic-settings

RUN mkdir -p /data /results /scratch /capsule/code
COPY code/ /capsule/code/
