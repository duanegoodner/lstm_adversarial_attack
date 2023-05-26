FROM ubuntu:22.04


ARG PRIMARY_USER
ARG PRIMARY_USER_HOME=/home/${PRIMARY_USER}
ARG JETBRAINS_USER
ARG JETBRAINS_USER_HOME=/home/${JETBRAINS_USER}
ARG WORK_GROUP=${WORK_GROUP}
ARG CONTAINER_DEVSPACE
ARG CONTAINER_PROJECT_ROOT
ARG CONTAINER_SRC_DIR
ARG CONTAINER_DATA_DIR
ARG COMPONENTS=./_build_resources/components

SHELL [ "/bin/bash", "--login", "-c" ]

COPY ${COMPONENTS}/create_sudo_user /tmp/create_sudo_user
RUN /tmp/create_sudo_user/main.sh ${PRIMARY_USER} \
    && sudo rm -rf /tmp/create_sudo_user

USER ${PRIMARY_USER}
ARG BUILD_DIR=${PRIMARY_USER_HOME}/docker_build
WORKDIR ${BUILD_DIR}

COPY ${COMPONENTS}/install_apt_pkgs ${BUILD_DIR}/install_apt_pkgs
COPY ./pkglist ${BUILD_DIR}/pkglist
RUN ${BUILD_DIR}/install_apt_pkgs/main.sh ${BUILD_DIR}/pkglist \
    && sudo rm -rf ${BUILD_DIR}/*

COPY ${COMPONENTS}/install_snoopy ${BUILD_DIR}/install_snoopy
RUN ${BUILD_DIR}/install_snoopy/main.sh && sudo rm -rf ${BUILD_DIR}/*

COPY ${COMPONENTS}/add_ssh_user ${BUILD_DIR}/add_ssh_user
RUN --mount=type=secret,id=jetbrains_user_pubkey \
    ${BUILD_DIR}/add_ssh_user/main.sh ${JETBRAINS_USER} \
    /run/secrets/jetbrains_user_pubkey \
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

COPY ./environment.yml ${BUILD_DIR}/environment.yml
COPY ${COMPONENTS}/create_conda_env ${BUILD_DIR}/create_conda_env
RUN ${BUILD_DIR}/create_conda_env/main.sh \
    ${BUILD_DIR}/environment.yml \
    ${CONTAINER_DEVSPACE}/env \
    ${PRIMARY_USER} \
    ${CONDA_INSTALL_DIR} \
    && sudo rm -rf ${BUILD_DIR}/*

COPY ./_build_resources/local_python_packages/accessjd ${BUILD_DIR}/accessjd
RUN sudo chown -R ${PRIMARY_USER}:${PRIMARY_USER} ${BUILD_DIR}/accessjd \
    && ${CONTAINER_DEVSPACE}/env/bin/python -m pip install ${BUILD_DIR}/accessjd

# #####################################################
# Done installing things in container
# #####################################################

WORKDIR ${PRIMARY_USER_HOME}
RUN sudo rm -rf ${BUILD_DIR}

# Create rest of directory structure expected of local project
# ARG CONTAINER_PROJECT_ROOT=${CONTAINER_DEVSPACE}/project
# ARG CONTAINER_WORK_DIR=${CONTAINER_PROJECT_ROOT}/work

RUN sudo mkdir ${CONTAINER_PROJECT_ROOT} \
    # && sudo mkdir ${CONTAINER_SRC_DIR} ${CONTAINER_DATA_DIR} \
    && sudo groupadd ${WORK_GROUP} \
    && sudo usermod -a -G ${WORK_GROUP} ${PRIMARY_USER} \
    && sudo chown ${PRIMARY_USER}:${WORK_GROUP} ${CONTAINER_PROJECT_ROOT}

RUN sudo mkdir -p /usr/local/entrypoints/jbs
COPY ./_build_resources/entrypoints/jbs /usr/local/entrypoints/jbs
