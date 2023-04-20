FROM python:3.10

COPY requirements.txt* /tmp/pip-tmp/
RUN umask 0002 && \
    pip install --upgrade pip && \
    pip install -r /tmp/pip-tmp/requirements.txt && \
    rm -rf /tmp/pip-tmp
