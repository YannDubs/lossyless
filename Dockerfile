# ==================================================================
# module list
# ------------------------------------------------------------------
# ubuntu                18.04    
# cuda                  10.2
# cudnn7                7    
# python                3.8    (apt)
# nodejs                15     (apt)
# pytorch               latest (pip)
# jupyterlab (+ ext)    latest (pip)
# requirements          latest (pip)
# ==================================================================
# Credits : modified from https://github.com/ufoym/deepo
# ------------------------------------------------------------------

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8
ARG PYTHON_VERSION=3.8

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \

# ==================================================================
# nodejs
# ------------------------------------------------------------------

    curl -sL https://deb.nodesource.com/setup_15.x | bash - \
        && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        nodejs \
        && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python"$PYTHON_VERSION" \
        python"$PYTHON_VERSION"-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python"$PYTHON_VERSION" ~/get-pip.py && \
        ln -s /usr/bin/python"$PYTHON_VERSION" /usr/local/bin/python3 && \
        ln -s /usr/bin/python"$PYTHON_VERSION" /usr/local/bin/python && \
        $PIP_INSTALL \
            setuptools \
            && \

# ==================================================================
# jupyter lab
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyterlab \
        ipywidgets \
        jupyterlab_code_formatter \
        black \
        isort \
        && \

    jupyter labextension install \
        @jupyter-widgets/jupyterlab-manager \
        @ryantam626/jupyterlab_code_formatter \
        @jupyterlab/toc \
        jupyterlab-topbar-extension \
        jupyterlab-system-monitor \
        && \

    jupyter nbextension enable --py widgetsnbextension && \
    jupyter nbextension enable --py jupyterlab_code_formatter \
        && \

# ==================================================================
# pytorch
# ------------------------------------------------------------------

    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
        torch torchvision 

# ==================================================================
# requirements
# ------------------------------------------------------------------

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt 

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
        apt-get clean && \
        apt-get autoremove && \
        rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 8888 8889 6006



