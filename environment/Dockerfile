# hash:sha256:01675296066a3cc32ccbad1fc60304c0c5d5bf1281225124f677c4aebac49613
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
    pytz==2026.1.post1 \
    suite2p==0.14.6
