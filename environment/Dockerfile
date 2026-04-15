# hash:sha256:db5ac44914ddc425cdeb750b88349aa42e05184b164290215db08bd3753a443c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==2.6.0 \
    aind-data-schema-models==5.4.1 \
    aind-ophys-utils \
    aind-qcportal-schema==0.6.4 \
    h5py \
    pandas \
    pydantic-settings \
    pytz==2026.1.post1 \
    pyyaml \
    suite2p==0.14.6
