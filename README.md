# lstm_adversarial_attack


## 1. Overview

This project builds on work by [Sun et al.](https://dl.acm.org/doi/10.1145/3219819.3219909) , and [Tang et al.](https://academic.oup.com/jamiaopen/article/1/1/87/5032901) that used Long Short-Term Memory (LSTM) time series models to predict Intensive Care Unit (ICU) patient outcomes and (in Sun et al) performed adversarial attacks on these models.

The original papers trained LSTM models using data from the Medical Information Mart for Intensive Care (MIMIC-III) database. Input features consisted of 13 lab measurements and 6 vital signs, and a binary variable representing in-hospital mortality was the prediction target. In [Sun et al.](https://dl.acm.org/doi/10.1145/3219819.3219909), an adversarial attack algorithm was used to identify small perturbations which, when applied to a real, correctly-classified input features, caused the trained model to misclassify the perturbed input. L1 regularization was applied to the adversarial attack loss function to favor adversarial examples with sparse perturbations that resemble the structure of data entry errors most likely to occur in real medical data. Attack susceptibility calculations were then performed to input feature space regions most vulnerable to adversarial attack.

The current work uses methods similar to those employed in [Sun et al.](https://dl.acm.org/doi/10.1145/3219819.3219909) and [Tang et al.](https://academic.oup.com/jamiaopen/article/1/1/87/5032901) , with following modifications / enhancements:

- A vectorized `preprocess` package that provides a 90% reduction in the time needed to convert database query output to model input features.

- Extensive hyperparameter tuning of the predictive model

- Improved predictive performance of the LSTM model, as indicated in the following table:

  |                 | AUC             | F1              | Precision       | Recall          |
  | --------------- | --------------- | --------------- | --------------- | --------------- |
  | Sun et al.      | 0.9094 (0.0053) | 0.5429 (0.0194) | 0.4100 (0.0272) | 0.8071 (0.0269) |
  | Current Project | 0.9657 (0.0035) | 0.9669 (0.0038) | 0.9888 (0.0009) | 0.9459 (0.0072) |

- A larger number of sample (41,951 patient ICU stays vs 37,559 in the original studies). For patients with multiple ICU stays, the original studies' code filtered out all stays after the initial stay. This removal appears to have been inadvertent.

- A GPU-compatible implementation of the adversarial attack algorithm to enable running attacks on batches of samples

- Tuning of attack algorithm hyperparameters to maximize the number of adversarial examples found and the sparsity of their perturbations.

- The attack algorithm was allowed to continue running after the first adversarial example was found for a given input. In most cases, these additional search iterations led to new adversarial examples that exhibited greater sparsity and smaller perturbations than the initially discovered examples.

- Implementation in PyTorch (instead of Tensorflow).

Further information can be found by running the code and reading the documentation in [this Jupyter notebook](https://github.com/duanegoodner/lstm_adversarial_attack/blob/main/notebooks/lstm_adversarial_attack.ipynb).

## 2. How to run this project

This section contains instructions on how to set up a development environment to run the project code. (These steps are not necessary if you just want to review the results in the completed Jupyter notebook [link])

### 2.1 Requirements

* git
* Docker
* CUDA-enabled GPU
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html#installation-guide)



### 2.2 Build a  PostgreSQL MIMIC-III database inside a Docker container

> **Note**  ~ 60 GB storage is needed to build the database, and the entire process may take 1.5 - 2.0 hours.

Go to https://github.com/duanegoodner/docker_postgres_mimiciii, and follow that repository's Getting Started steps 1 through 6 to build a PostgreSQL MIMIC-III database in a named Docker volume, and a Docker image with a PostgreSQL database that can access the named volume.



### 2.3 Clone the lstm_adversarial_attack repository to your machine:

```shell
$ git clone https://github.com/duanegoodner/lstm_adversarial_attack
```



### 2.4 Set the `LOCAL_PROJECT_ROOT` environment variable (to be used by Docker)



In file `lstm_adversarial_attack/docker/app/.env`, set the value of `LOCAL_PROJECT_ROOT` to the absolute path of the `lstm_adversarial_attack` root directory. Leave the values of `PROJECT_NAME`, `CONTAINER_DEVSPACE`, and `CONTAINER_PROJECT_ROOT` unchanged. For example, if in you ran the command in step 3 from directory `/home/my_user/projects` causing the cloned repo root to be `/home/my_user/projects/lstm_adversarial_attack`  your `lstm_adversarial_attack/docker/app/.env` file would look like this:

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
$ docker compose build
```

Image `lstm_aa_app` includes an installation of Miniconda3 and has a conda environment created in `/home/devspace/env`. All of the Python dependencies needed for the project are installed in this environment. All of these dependencies are shown in  `lstm_adversarial_attack/docker/app/environment.yml`. 

### 2.6 Run the `lstm_aa_app` and `postgres_mimiciii` containers

From directory `lstm_adversarial_attack/docker` run:

```
$ docker compose up -d
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

1. [Sun, M., Tang, F., Yi, J., Wang, F. and Zhou, J., 2018, July. Identify susceptible locations in medical records via adversarial attacks on deep predictive models. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 793-801).](https://dl.acm.org/doi/10.1145/3219819.3219909)
2. [Johnson, A., Pollard, T., and Mark, R. (2016) 'MIMIC-III Clinical Database' (version 1.4), *PhysioNet*.](https://doi.org/10.13026/C2XW26) 
3. [Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.](https://www.nature.com/articles/sdata201635)

4. [Tang, F., Xiao, C., Wang, F. and Zhou, J., 2018. Predictive modeling in urgent care: a comparative study of machine learning approaches. *Jamia Open*, *1*(1), pp.87-98.](https://academic.oup.com/jamiaopen/article/1/1/87/5032901)



