FROM mcr.microsoft.com/devcontainers/python:0-3.10

COPY .devcontainer/caadmin.netskope.com.cer /tmp/certs/caadmin.netskope.com.cer
RUN openssl x509 -inform DER -in /tmp/certs/caadmin.netskope.com.cer >> /etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY requirements.txt* /tmp/pip-tmp/
RUN umask 0002 && \
    pip install --upgrade pip && \
    pip install -r /tmp/pip-tmp/requirements.txt && \
    rm -rf /tmp/pip-tmp

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
ENV AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV GIT_SSL_CAPATH=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
