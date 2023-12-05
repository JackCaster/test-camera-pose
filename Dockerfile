FROM continuumio/miniconda3:23.10.0-1
SHELL ["/bin/bash", "-l", "-c"]
USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN \
    # https://stackoverflow.com/a/68804294
    apt-get --allow-releaseinfo-change update \
    && apt-get install -y --no-install-recommends \
    curl \
    git \
    wget \
    g++ \
    gcc \
    screen \
    ca-certificates \
    build-essential \
    # opencv
    libgl1 \
    libsm6 \
    libxext6 \
    # PIL display images
    imagemagick \
    # Clean up
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

ADD env.yml /tmp

ENV CONDA_ENV_NAME="project"

RUN conda env create --name ${CONDA_ENV_NAME} --file /tmp/env.yml \
    && conda clean --all

RUN echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
