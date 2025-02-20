# Setup

## Requirements

* git
* Docker
* CUDA-enabled GPU
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html#installation-guide)
  * NOTE: After installing the NVIDIA Container Toolkit, you may need to reboot.



## Procedure

### 1.  Build a PostgreSQL MIMIC-III database inside a Docker container

Go to https://github.com/duanegoodner/docker_postgres_mimiciii, and follow that repository's Getting Started steps 1 through 6 to build the following:
- A PostgreSQL MIMIC-III database in a named Docker volume
- A Docker image with a PostgreSQL instance that can access the named volume.

> **Note**  ~60 GB storage is needed to build the database, and the entire process may take ~60 minutes on a typical desktop system.

### 2. Clone the lstm_adversarial_attack repository to your machine:

```shell
git clone https://github.com/duanegoodner/lstm_adversarial_attack
```


### 3 Set the `LOCAL_PROJECT_ROOT` environment variable (to be used by Docker)

In file `lstm_adversarial_attack/docker/app/.env`, set the value of `LOCAL_PROJECT_ROOT` to the **absolute path** of the `lstm_adversarial_attack` root directory. Leave the values of `PROJECT_NAME`, `CONTAINER_DEVSPACE`, and `CONTAINER_PROJECT_ROOT` unchanged.

For example, if in you ran the command in step 2.3 from directory `/home/my_user/projects` causing the cloned repo root to be `/home/my_user/projects/lstm_adversarial_attack`  your `lstm_adversarial_attack/docker/app/.env` file would look like this:

```shell
LOCAL_PROJECT_ROOT=/home/my_user/projects/lstm_adversarial_attack

PROJECT_NAME=lstm_adversarial_attack
CONTAINER_DEVSPACE=/home/devspace
CONTAINER_PROJECT_ROOT=${CONTAINER_DEVSPACE}/project
```


### 4. Build the `lstm_aa_app` Docker image

> **Note** The size of the `lstm_aa_app` image will be ~10 GB.

From directory `lstm_adversarial_attack/docker`, run:

```shell
UID=${UID} GID=${GID} docker compose build
```
Image `lstm_aa_app` includes an installation of Miniconda3 and has a conda environment created in `/home/devspace/env`. All of the Python dependencies needed for the project will be installed in this environment and are included in `lstm_adversarial_attack/docker/app/environment.yml`. 

### 5. Run the `lstm_aa_app` and `postgres_mimiciii` containers

From directory `lstm_adversarial_attack/docker` run:

```shell
$ UID=${UID} GID=${GID} docker compose up -d
```
The output should look like this:

```bash
[+] Running 4/4
 ✔ Network app_default              Created                                                                                                  0.2s 
 ✔ Container postgres_mimiciii_dev  Started                                                                                                  0.7s 
 ✔ Container postgres_optuna        Started                                                                                                  0.7s 
 ✔ Container lstm_aa_app            Started 
```

Container `lstm_aa_app` will run Python modules, container `postres_mimiciii` runs the PostgreSQL instance with the MIMIC-III database, and container `postgres_optuna` container runs another PostgreSQL instance that will store data generated during hyperparameter tuning.


### 6. Exec into the `lstm_aa_app` container

The `lstm_aa_app` image has a sudo-privileged non-root user,  named `gen_user`.  The conda environment in `/home/devspace/env` gets activated whenever a `bash` or `zsh` shell is launched under `gen_user`.

Run the following command to launch a `zsh` shell in the container:

```bash 
docker exec -it lstm_aa_app /bin/zsh
```

You will then be at a `zsh` prompt in the container. Run the following commands to double-check our user name and working directory.

```shell
whoami
# gen_user

pwd
# /home/devspace/lstm_adversarial_attack
```
> **Note** Our `docker-compose.yml` maps container directory `/home/devspace/lstm_adversarial_attack` to the local lstm_adversarial_attack repository root.


### 7. Launch Jupyter Lab

From a `zsh` prompt in the container, run the following command to start a Jupyter Lab server:

```
> jupyter lab --no-browser --ip=0.0.0.0
```

Among the various text output to the terminal, you should some lines that look like this:

```
To access the server, open this file in a browser:
        file:///home/gen_user/.local/share/jupyter/runtime/jpserver-504-open.html
Or copy and paste one of these URLs:
        http://f0e281ad30a4:8888/lab?token=0447a83987dfdc124e4f13df65caf307dccb0198fd92460f
        http://127.0.0.1:8888/lab?token=0447a83987dfdc124e4f13df65caf307dccb0198fd92460f
```

You will have a different value for the token in the last two urls. In your browser, go to the bottom url (`http://127.0.0.1:8888/lab?token=...`) to open Jupyter Lab.



### 8 Run the Project Jupyter Notebook

In the Jupyter Lab file explorer, navigate to the `/notebooks` directory, and open `icu_deep_learning.ipynb`. Read through this notebook file, and use code cells within it to run the project.
