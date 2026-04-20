# hash:sha256:90f08f182742f2276d5924c25e59a5d6a2e3a8400336366166705f54da3092f1
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
    numpy==2.4.4 \
    pandas==3.0.2 \
    pydantic-settings==2.13.1 \
    pytz==2026.1.post1 \
    pyyaml==6.0.3 \
    suite2p==0.14.6
