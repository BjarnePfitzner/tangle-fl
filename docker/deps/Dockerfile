# Don't worry, this also works on Intel
FROM ibmcom/powerai:1.6.2-tensorflow-ubuntu18.04-py36
ENV IBM_POWERAI_LICENSE_ACCEPT=yes

RUN bash -c " \
    if [ '$HOSTTYPE' = 'powerpc64le' ]; then \
        ${CONDA_INSTALL_DIR}/bin/conda install -c powerai ray ; \
    else \
        ${CONDA_INSTALL_DIR}/envs/wmlce/bin/pip install ray; \
    fi"
