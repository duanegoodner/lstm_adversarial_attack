FROM ubuntu:22.04


ARG PRIMARY_USER
ARG PRIMARY_USER_HOME=/home/${PRIMARY_USER}
ARG CONTAINER_DEVSPACE
ARG CONTAINER_PROJECT_ROOT
ARG COMPONENTS=./app_resources/components

SHELL [ "/bin/bash", "--login", "-c" ]

COPY ${COMPONENTS}/create_sudo_user /tmp/create_sudo_user
RUN /tmp/create_sudo_user/main.sh ${PRIMARY_USER} \
    && sudo rm -rf /tmp/create_sudo_user

USER ${PRIMARY_USER}
ARG BUILD_DIR=${PRIMARY_USER_HOME}/docker_build
WORKDIR ${BUILD_DIR}

COPY ${COMPONENTS}/install_apt_pkgs ${BUILD_DIR}/install_apt_pkgs
COPY ./app_resources/pkglist ${BUILD_DIR}/pkglist
RUN ${BUILD_DIR}/install_apt_pkgs/main.sh ${BUILD_DIR}/pkglist \
    && sudo rm -rf ${BUILD_DIR}/*

COPY ${COMPONENTS}/zsh_setup ${BUILD_DIR}/zsh_setup
RUN ${BUILD_DIR}/zsh_setup/main.sh ${PRIMARY_USER} \
    && sudo rm -rf ${BUILD_DIR}/*

# ARG CONTAINER_DEVSPACE=/home/devspace
RUN sudo mkdir -p ${CONTAINER_DEVSPACE} \
    && sudo chown ${PRIMARY_USER}:${PRIMARY_USER} ${CONTAINER_DEVSPACE}
ARG CONDA_INSTALL_DIR=${CONTAINER_DEVSPACE}/miniconda3
ENV PATH=$CONDA_INSTALL_DIR/bin:$PATH

COPY ${COMPONENTS}/install_miniconda ${BUILD_DIR}/install_miniconda
RUN ${BUILD_DIR}/install_miniconda/main.sh \
    ${CONDA_INSTALL_DIR} \
    ${PRIMARY_USER} \
    && sudo rm -rf ${BUILD_DIR}/install_miniconda

COPY ./app_resources/environment.yml ${BUILD_DIR}/environment.yml
COPY ${COMPONENTS}/create_conda_env ${BUILD_DIR}/create_conda_env
RUN ${BUILD_DIR}/create_conda_env/main.sh \
    ${BUILD_DIR}/environment.yml \
    ${CONTAINER_DEVSPACE}/env \
    ${PRIMARY_USER} \
    ${CONDA_INSTALL_DIR} \
    && sudo rm -rf ${BUILD_DIR}/*
ENV CONDA_BIN_DIR=${CONTAINER_DEVSPACE}/env/bin

# #####################################################
# Done installing things in container
# #####################################################

RUN sudo mkdir ${CONTAINER_PROJECT_ROOT}
WORKDIR ${CONTAINER_PROJECT_ROOT}
RUN sudo rm -rf ${BUILD_DIR}

RUN sudo mkdir -p /usr/local/entrypoints/jupyter_lab
COPY ./app_resources/entrypoints/jupyter_lab /usr/local/entrypoints/jupyter_lab

