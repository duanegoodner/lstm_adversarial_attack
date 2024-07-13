# lstm_adversarial_attack


## 1. Overview

This project builds upon work in [[1](#ref_01)] and [[2](#ref_02)] that used Long Short-Term Memory (LSTM) time series models of lab and vital sign measurements from the  Medical Information Mart for Intensive Care (MIMIC-III) database [[3](#ref_03), [4](ref_04)] to predict Intensive Care Unit (ICU) patient outcomes. The work in [[1](#ref_01)] also included adversarial attacks on a trained LSTM model.



The current work follows the general approach of the prior studies, with the following modifications / enhancements


  -  Predictive performance of the LSTM is improved (Prior work F1 scores: 0.53 - 0.63. Current study: > 0.96). This improvement is primarily due to hyperparameter tuning of the predictive model..
  - A preprocess package that reduces the time needed to convert database query output to model input features by 90%
  - A GPU-compatible implementation of the adversarial attack algorithm
  - The attack algorithm was allowed to continue running after the first adversarial example was found for a given input. In most cases, these additional search iterations led to the discovery of adversarial examples that exhibited greater sparsity and smaller perturbations than the initially discovered examples.
- Exploration of multiple parameters and objective functions during hyperparameter tuning of the attack algorithm
- Implementation in PyTorch (instead of Keras).



## 2. Documentation

See the Jupyter notebook [here](https://github.com/duanegoodner/lstm_adversarial_attack/blob/main/notebooks/lstm_adversarial_attack.ipynb) for additional background information, implementation details, and results.

> **Note**  Some of the cell outputs included in this notebook are quite long. Instead of just going to the Github link in a browser, it is recommended to use Jupyter Lab, Jupyter Notebook, VS Code with the Jupyter extension, or some other program with features for managing how cell output is displayed.



## 3. How to run this project

Follow the instructions in this section if you want to run the project code. (These steps are not necessary if you just want to view documentation and results in this Jupyter notebook).

### 2.1 Requirements

* git
* Docker
* CUDA-enabled GPU
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html#installation-guide)
  * NOTE: After installing the NVIDIA Container Toolkit, you may need to 




### 2.2 Build a  PostgreSQL MIMIC-III database inside a Docker container

> **Note**  ~ 60 GB storage is needed to build the database, and the entire process may take 1.5 - 2.0 hours.

Go to https://github.com/duanegoodner/docker_postgres_mimiciii, and follow that repository's Getting Started steps 1 through 6 to build a PostgreSQL MIMIC-III database in a named Docker volume, and a Docker image with a PostgreSQL database that can access the named volume.



### 2.3 Clone the lstm_adversarial_attack repository to your machine:

```shell
git clone https://github.com/duanegoodner/lstm_adversarial_attack
```



### 2.4 Set the `LOCAL_PROJECT_ROOT` environment variable (to be used by Docker)

In file `lstm_adversarial_attack/docker/app/.env`, set the value of `LOCAL_PROJECT_ROOT` to the absolute path of the `lstm_adversarial_attack` root directory. Leave the values of `PROJECT_NAME`, `CONTAINER_DEVSPACE`, and `CONTAINER_PROJECT_ROOT` unchanged. For example, if in you ran the command in step 2.3 from directory `/home/my_user/projects` causing the cloned repo root to be `/home/my_user/projects/lstm_adversarial_attack`  your `lstm_adversarial_attack/docker/app/.env` file would look like this:

```shell
LOCAL_PROJECT_ROOT=/home/my_user/projects/lstm_adversarial_attack

PROJECT_NAME=lstm_adversarial_attack
CONTAINER_DEVSPACE=/home/devspace
CONTAINER_PROJECT_ROOT=${CONTAINER_DEVSPACE}/project
```



### 2.5 Build the `lstm_aa_app` Docker image

> **Note** The size of the `lstm_aa_app` image will be ~10 GB.

`cd` into directory `lstm_adversarial_attack/docker`, and run:

```shell
docker compose build
```

Image `lstm_aa_app` includes an installation of Miniconda3 and has a conda environment created in `/home/devspace/env`. All of the Python dependencies needed for the project are installed in this environment. All of these dependencies are shown in  `lstm_adversarial_attack/docker/app/environment.yml`. 

### 2.6 Run the `lstm_aa_app` and `postgres_mimiciii` containers

From directory `lstm_adversarial_attack/docker` run:

```
docker compose up -d
```
Now confirm that the project containers are running with:
```
docker ps
```
You should see something like this:
```
CONTAINER ID   IMAGE               COMMAND                  CREATED       STATUS       PORTS                                                                        NAMES
e0790e9c2156   lstm_aa_app_dev     "/bin/bash /usr/loca…"   2 hours ago   Up 2 hours   127.0.0.1:6006->6006/tcp, 127.0.0.1:8888->8888/tcp, 127.0.0.1:2200->22/tcp   lstm_aa_app_dev
860dba19f68f   postgres_mimiciii   "docker-entrypoint.s…"   2 hours ago   Up 2 hours   0.0.0.0:5555->5432/tcp, :::5555->5432/tcp                                    postgres_mimiciii_dev
3fea80bc8e85   postgres            "docker-entrypoint.s…"   2 hours ago   Up 2 hours   0.0.0.0:5556->5432/tcp, :::5556->5432/tcp                                    postgres_optuna
```


### 2.7 Exec into the `lstm_aa_app` container

The `lstm_aa_app` image has a sudo-privileged non-root user,  named `gen_user`.  The conda environment in `/home/devspace/env` gets activated whenever a `bash` or `zsh` shell is launched under `gen_user`. It will be convenient to use `docker exec` to launch a shell in the container for a couple steps later on.

To launch a `zsh` shell in the container run:

``` 
$ docker exec -it lstm_aa_app /bin/zsh
```

You will then be at a `zsh` prompt in the container.

```shell
$ whoami
gen_user

$ pwd
/home/devspace/lstm_adversarial_attack
# docker-compose maps directory /home/devspace/lstm_adversarial_attack to the local lstm_adversarial_attack repository root
```

> **Note**: `gen_user`'s `zsh`profile uses OhMyZsh with a Powerlevel10k theme. The shell features provided by these settings are not reflected in this Markdown file. 



### 2.8 Launch Jupyter Lab

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



### 2.9 Run the Project Jupyter Notebook

In the Jupyter Lab file explorer, navigate to the `/notebooks` directory, and open `lstm_adversarial_attack.ipynb`. Read through this notebook file, and use code cells within it to run the project.



# References



<a id="ref_01">1.</a> [Sun, M., Tang, F., Yi, J., Wang, F. and Zhou, J., 2018, July. Identify susceptible locations in medical records via adversarial attacks on deep predictive models. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 793-801).](https://dl.acm.org/doi/10.1145/3219819.3219909)

<a id="ref_02">2.</a> [Tang, F., Xiao, C., Wang, F. and Zhou, J., 2018. Predictive modeling in urgent care: a comparative study of machine learning approaches. *Jamia Open*, *1*(1), pp.87-98.](https://academic.oup.com/jamiaopen/article/1/1/87/5032901)

<a><a id="ref_03">3.</a> </a>[Johnson, A., Pollard, T., and Mark, R. (2016) 'MIMIC-III Clinical Database' (version 1.4), *PhysioNet*.](https://doi.org/10.13026/C2XW26) 

<a id="ref_04">4.</a> [Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.](https://www.nature.com/articles/sdata201635)



