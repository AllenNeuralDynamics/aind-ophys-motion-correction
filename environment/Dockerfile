# hash:sha256:3f85ba3eda075c8c65fe40d150332b5c17bd68fd1092e80ec8a04d7021277c89
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/py-r:python3.10-R4.2.2-IRkernel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==1.1.0 \
    aind-ophys-utils==0.0.7 \
    pydantic==2.8.1 \
    sbxreader==0.2.2 \
    torch==2.5.0 \
    git+https://github.com/arielleleon/suite2p@bug-pull-from-defined-h5py \

