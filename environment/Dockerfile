# hash:sha256:e13b3e2bb960dce7416fe2a3e34ac3eb1914230c3bee99ccc466520de2c043dc
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==2.6.0 \
    aind-data-schema-models==5.4.1 \
    aind-log-utils==0.2.6 \
    aind-qcportal-schema==0.6.4 \
    pytz \
    suite2p==0.14.6
