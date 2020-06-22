# Don't worry, this also works on Intel
FROM ibmcom/powerai:1.6.2-tensorflow-ubuntu18.04-py36
ENV IBM_POWERAI_LICENSE_ACCEPT=yes

RUN bash -c ' \
    if [ "$HOSTTYPE" = "powerpc64le" ]; then \
        ${CONDA_INSTALL_DIR}/bin/conda install --name wmlce -c powerai ray ; \
        sudo apt-get update && sudo apt-get install -y --no-install-recommends build-essential && sudo rm -rf /var/lib/apt/lists/* ; \
        ${CONDA_INSTALL_DIR}/envs/wmlce/bin/pip install aiohttp psutil setproctitle grpcio ; \
    else \
        ${CONDA_INSTALL_DIR}/envs/wmlce/bin/pip install ray; \
    fi'
