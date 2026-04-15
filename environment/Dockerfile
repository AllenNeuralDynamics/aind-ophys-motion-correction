# hash:sha256:748aa74ce54358177c4a4781ff53215288207d8998bd2e2f9db7fc2d8eaea851
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip3 install -U --no-cache-dir \
    aind-data-schema \
    aind-data-schema-models==5.4.1 \
    aind-ophys-utils==0.0.8 \
    aind-qcportal-schema==0.6.4 \
    h5py==3.16.0 \
    pandas==3.0.2 \
    pydantic-settings==2.13.1 \
    pytz==2026.1.post1 \
    pyyaml==6.0.3 \
    suite2p==0.14.6
