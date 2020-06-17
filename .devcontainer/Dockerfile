#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

FROM learning-tangle/deps

USER root

# Copy environment.yml (if found) to a temp locaition so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/

# Configure apt and install packages
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git openssh-client less iproute2 procps iproute2 lsb-release \
    #
    # Install pylint
    && ${CONDA_INSTALL_DIR}/bin/pip install pylint \
    #
    # Update Python environment based on environment.yml (if present)
    && if [ -f "/tmp/conda-tmp/environment.yml" ]; then ${CONDA_INSTALL_DIR}/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
    
RUN echo "source ${CONDA_INSTALL_DIR}/etc/profile.d/conda.sh" >> /home/${PWRAI_USER}/.bashrc
RUN echo "conda activate wmlce" >> /home/${PWRAI_USER}/.bashrc
