# Setup

## Requirements

* git
* Docker
* CUDA-enabled GPU
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html#installation-guide)
  * NOTE: After installing the NVIDIA Container Toolkit, you may need to reboot.



## Procedure

### 1.  Build a PostgreSQL MIMIC-III database inside a Docker container

Go to https://github.com/duanegoodner/docker-postgres-mimiciii, and follow that repository's Getting Started steps 1 through 6 to build the following:
- A PostgreSQL MIMIC-III database in a named Docker volume
- A Docker image with a PostgreSQL instance that can access the named volume.

> **Note**  ~60 GB storage is needed to build the database, and the entire process may take ~60 minutes on a typical desktop system.

### 2. Clone the `icu-deep-learning` repository to your machine:

```shell
git clone https://github.com/duanegoodner/icu-deep-learning
```


### 3. Set the `LOCAL_PROJECT_ROOT` in `.env` file for Docker



Open file `icu-deep-learning/docker/app/.env`, and set `LOCAL_PROJECT_ROOT` to the **absolute path** of the local `icu-deep-learning` root directory.

```shell
LOCAL_PROJECT_ROOT=/path/to/icu-deep-learning
# replace '/path/to/icu-deep-learning' with absolute path to local repo root

# OTHER VARIABLES (do not change)
# ...
# ...
```

Leave the values of other variables in this `.env` file unchanged


### 4. Build the `lstm_aa_app` Docker image

From directory `icu-deep-learning/docker/app/`, run:

```shell
UID=${UID} GID=${GID} docker compose build
```
This will build Docker image `lstm_aa_app` that is based on `ubuntu:22.04`. It will have Miniconda3 environment at `/home/devspace/env`containing all of the Python dependencies needed to run the project. See local file  `icu-deep-learning/docker/app/environment.yml` for a list these dependencies.

> **Note** The size of the `lstm_aa_app` image will be ~10 GB.

### 5. Use `docker compose` to start all project containers
From directory `icu-deep-learning/docker/app/` run:

```shell
UID=${UID} GID=${GID} docker compose up -d
```
The output should look like this:

```bash
[+] Running 5/5
 ✔ Network app_default              Created                                          0.2s 
 ✔ Volume "app_optuna_db"           Created                                          0.0s 
 ✔ Container postgres_optuna        Started                                          0.6s 
 ✔ Container postgres_mimiciii_dev  Started                                          0.6s 
✔ Container lstm_aa_app             Started                                          0.8s
```
There should be three running containers, as specified by `icu-deepl-learning/docker/app/docker-compose.yml`:

- `lstm_aa_app` will run Python modules
- `postres_mimiciii` runs the PostgreSQL instance with the MIMIC-III database
- `postgres_optuna` container runs another PostgreSQL instance that will store data generated during hyperparameter tuning.


### 6. Exec into the `lstm_aa_app` container

The `lstm_aa_app` image has a sudo-privileged non-root user,  named `gen_user`.  The conda environment in `/home/devspace/env` gets activated whenever a **bash** or **zsh** shell is launched under `gen_user`.

Run the following to launch a **zsh** shell in the container:

```bash 
docker exec -it lstm_aa_app /bin/zsh
```

You should then be at a **zsh** prompt in the container. Run the following commands to confirm the setup.

```shell
whoami
# gen_user
pwd
# /home/devspace/project
which python
# /home/devspace/env/bin/python
echo $0
# /bin/zsh
ls
# README.md  SETUP.md  config.toml  data  docker  docs  logs  notebooks  src
```
> **Note** The `docker-compose.yml` maps container directory `/home/devspace/icu-deep-learning` to the local `icu-deep-learning` root. 


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



### 8. Run the Project Jupyter Notebook

In the Jupyter Lab file explorer, navigate to the `notebooks/` directory, and open `icu_deep_learning.ipynb`. Read through this notebook file, and use code cells within it to run the project.
