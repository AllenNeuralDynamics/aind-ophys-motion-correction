# hash:sha256:04027cc7cf1b157dd90bb64a19d401606686cc8aa4be6b4eaf6e0cd8a21b1edd
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==2.6.0 \
    aind-data-schema-models==5.4.1 \
    aind-ophys-utils==0.0.8 \
    aind-qcportal-schema==0.6.4 \
    h5py==3.16.0 \
    numpy \
    pandas==3.0.2 \
    pydantic-settings==2.13.1 \
    pytz==2026.1.post1 \
    pyyaml==6.0.3 \
    suite2p==0.14.6
