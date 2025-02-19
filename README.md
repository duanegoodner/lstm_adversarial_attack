# lstm_adversarial_attack


## Overview

This project implements and optimizes **Long Short-Term Memory (LSTM)** time series models of intensive care unit (ICU) patient lab and vital sign data to predict patient outcomes. Additionally, an **adversarial attack** algorithm is used to identify vulnerabilities in trained models. These studies build upon similar work published in [[1](#ref_01)] and [[2](#ref_02)].

The predictive model input consists of data from 13 lab measurements and 6 vital signs collected during the first 48 hours after patient admission to the ICU. The prediction target is a binary variable representing in-hospital mortality. This type of model can supplement standard heuristics used by care providers in identifying high-risk patients.

The results of adversarial attacks on trained LSTM models provide a gauge of model stability. Additionally, these results offer an opportunity to compare adversarial vulnerabilities in LSTMs with the well-documented adversarial behaviors observed in Convolutional Neural Networks (CNNs) used for computer vision. Unlike the adversarial examples found in CNNs — where perturbations imperceptible to the human eye can drastically alter model predictions — the adversarial examples discovered for our LSTM models exhibit a higher degree of plausibility, aligning more closely with human intuition.


## Tech Stack  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)   ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?logo=pytorch&logoColor=white)   ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)   ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?logo=postgresql&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)   ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)   ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)   ![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?logo=tensorflow&logoColor=white)   ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white)   ![Optuna](https://img.shields.io/badge/Optuna-7C3AED?logo=python&logoColor=white) ![msgspec](https://img.shields.io/badge/msgspec-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&logo=Jupyter&logoColor=white)





## Data Pipeline

- SQL queries to extract raw data from PostgreSQL instance containig the Medical Information Mart for Intensive Care (MIMIC-III) database
- Preprocessing 




## Highlights



This project builds upon work in [[1](#ref_01)] and [[2](#ref_02)] that used Long Short-Term Memory (LSTM) time series models of ICU patient lab and vital sign measurements to predict patient outcomes and then used an adversarial attack algorithm to identify model vulnerabilities.



The current work follows the general approach of the prior studies, with the following modifications / enhancements


  -  Predictive performance of the LSTM is improved (Prior work F1 scores: 0.53 - 0.63. Current study: > 0.96). This improvement is primarily due to hyperparameter tuning of the predictive model..
  - A preprocess package that reduces the time needed to convert database query output to model input features by 90%
  - A GPU-compatible implementation of the adversarial attack algorithm
  - The attack algorithm was allowed to continue running after the first adversarial example was found for a given input. In most cases, these additional search iterations led to the discovery of adversarial examples that exhibited greater sparsity and smaller perturbations than the initially discovered examples.
- Exploration of multiple parameters and objective functions during hyperparameter tuning of the attack algorithm
- Implementation in PyTorch (instead of Keras).



## 2. Documentation

Detailed documentation can be viewed in the project Jupyter notebook available on NBViewer: 

[![View in nbviewer](https://img.shields.io/badge/Open%20in-nbviewer-orange)](https://nbviewer.org/github/duanegoodner/lstm_adversarial_attack/blob/main/notebooks/lstm_adversarial_attack.ipynb)


## 3. How to run this project

Follow the instructions in this section if you want to run the project code. (These steps are not necessary if you just want to view documentation and results in  the Jupyter notebook linked above).

### 2.1 Requirements

* git
* Docker
* CUDA-enabled GPU
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html#installation-guide)
  * NOTE: After installing the NVIDIA Container Toolkit, you may need to reboot.




### 2.2 Build a  PostgreSQL MIMIC-III database inside a Docker container

> **Note**  ~ 60 GB storage is needed to build the database, and the entire process may take 1.5 - 2.0 hours.

Go to https://github.com/duanegoodner/docker_postgres_mimiciii, and follow that repository's Getting Started steps 1 through 6 to build a PostgreSQL MIMIC-III database in a named Docker volume, and a Docker image with a PostgreSQL database that can access the named volume.



### 2.3 Clone the lstm_adversarial_attack repository to your machine:

```shell
$ git clone https://github.com/duanegoodner/lstm_adversarial_attack
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
$ UID=${UID} GID=${GID} docker compose build
```
Image `lstm_aa_app` includes an installation of Miniconda3 and has a conda environment created in `/home/devspace/env`. All of the Python dependencies needed for the project are installed in this environment. All of these dependencies are shown in  `lstm_adversarial_attack/docker/app/environment.yml`. 

### 2.6 Run the `lstm_aa_app` and `postgres_mimiciii` containers

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

LSTM modeling and adversarial attack work will run in container `lstm_aa_app`. `postres_mimiciii` hosts a database with the healthcare data that will feed our model, and the `postgres_optuna` container holds a database that will store data generatd during model and adversarial attack tuning.


### 2.7 Exec into the `lstm_aa_app` container

The `lstm_aa_app` image has a sudo-privileged non-root user,  named `gen_user`.  The conda environment in `/home/devspace/env` gets activated whenever a `bash` or `zsh` shell is launched under `gen_user`. It will be convenient to use `docker exec` to launch a shell in the container for a couple steps later on.

To launch a `zsh` shell in the container run:

```bash 
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



