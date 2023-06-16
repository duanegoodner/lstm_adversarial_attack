# lstm_adversarial_attack
LSTM time series deep learning model and adversarial attacks



## Overview

## 1. Overview

### 1.1 Summary of Prior Work

This project builds on results originally published in:

[Sun, M., Tang, F., Yi, J., Wang, F. and Zhou, J., 2018, July. Identify susceptible locations in medical records via adversarial attacks on deep predictive models. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 793-801)](https://dl.acm.org/doi/10.1145/3219819.3219909)

The original paper trained a Long Short-Term Memory (LSTM) time series model sing data from the Medical Information Mart for Intensive Care (MIMIC-III) database to predict patient outcomes. The input features consisted of data from 13 lab measurements (Blood Urea Nitrogen, HCO<sub>3</sub>, Na, PaCO2, Glucose, Creatinine, Albumin, Mg, K, Ca, Platelets, and Lactate), and 6 vital signs (Heart Rate, Respiration Rate, Systolic Blood Pressure, Diastolic Blood Pressure, Oxygen Saturation, and Temperature). The prediction target was a binary variable representing in-hospital mortality. Input data for each patient were collected over time spans ranging from 6 to 48 hours. The dataset used for model training and evaluation contained 37,559 samples. Each sample consisted of  a single ICU stay by a patient and was represented by a 48 (hour) x 19 (measurements) input feature matrix. Samples with less than 48 hours of data were padded to 48 hours  the global mean value of each measurement parameter. 

The full LSTM model consisted of:

- A single-layer, bi-directional LSTM with an input size of 19, and 128 hidden states per direction. Tanh activation was applied to the outputs.
- A dropout layer with dropout probability = 50%
- A fully-connected layer with an input size of 256 (for the 2 x 128 LSTM outputs), 32 output nodes, and ReLU activation.
- A final 2-node layer with soft-max activation.

This model was trained using a Binary Cross Entropy loss, and an Adam optimizer (learning rate = 1e-4, momentum decay rate = 0.999, and moving average decay rate = 0.5). An adversarial attack algorithm was then used to identify small perturbations which, when applied to a real, correctly-classified input features, caused the trained model to misclassify the perturbed input. The attack algorithm used L1 regularization to favor adversarial examples with sparse perturbations that resemble the structure of data entry errors most likely to occur in real medical data. Attacks were run serially (one sample at a time), and the attack on a given sample was halted upon finding a single adversarial example.

### 1.2 Focus and Key Findings of Current Poject

The current project follows the general approach of Sun et al, and adds the modifications / extensions

- Implementation in PyTorch (instead of Tensorflow).

- A larger dataset (41,951 patient ICU stays vs 37,559 in the original study). For patients with multiple ICU stays, the original study's code filtered all all stays after the initial stay. This removal appears to have been indadvertent.

- A streamlined `preprocess` sub-package is used to convert database SQL query outputs to the tensor form used for model input. This package reduces RAM consumption by saving large intermediate data structures to disk (primarily Pandas dataframes saved as .pickle files) instead of returning them to global scope of an executing program. It also performs the query output to tensor conversion 90% faster than the original code, primarily through the use of vectorized operations.

- The feature matrices of samples with data collection time ranges less than the maximum observation window (typically 48 hours) are padded with zero values. The padded data are input to the LSTM as Pytorch PackedSequence objects, allowing the padding values to be ignored by the model.

- A GPU-compatible adversarial attack algorithm is implemented to enable running attacks on batches of samples. This enabled a 20X reduction in the the amount of time needed to perform attacks on the full dataset. 

- LSTM predictive performance is improved hyperparameter tuning with the the Optuna Tree-structued Parzen Estimator (TPE) algorithm

- Hyperparameter tuning of the attack algorithm (also using TPE) to find adversarial perturbations with higher sparsity and lower magnitude.

- The attack algorithm was allowed to continue running after the first adversarial example was found for a given input. In most cases, these additional search iterations led to new adversarial examples that exhibited greater sparsity and smaller perturbations than the initially discovered examples.

  



## How to run this project

This section contains instructions on how to set up a development environment to run the project code. (These steps are not necessary if you just want to review the results in the completed Jupyter notebook [link])

### 1. Requirements

* git
* Docker
* CUDA-enabled GPU
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html#installation-guide)



### 2. Build a  PostgreSQL MIMIC-III database inside a Docker container

> **Note**  ~ 60 GB storage is needed to build the database, and the entire process may take 1.5 - 2.0 hours.

Go to https://github.com/duanegoodner/docker_postgres_mimiciii, and follow that repository's Getting Started steps 1 through 6 to build a PostgreSQL MIMIC-III database in a named Docker volume, and a Docker image with a PostgreSQL database that can access the named volume.



### 3. Clone the lstm_adversarial_attack repository to your machine:

```
$ git clone https://github.com/duanegoodner/lstm_adversarial_attack
```



### 4. Set the `LOCAL_PROJECT_ROOT` environment variable (to be used by Docker)



In file `lstm_adversarial_attack/docker/app/.env`, set the value of `LOCAL_PROJECT_ROOT` to the absolute path of the `lstm_adversarial_attack` root directory. Leave the values of `PROJECT_NAME`, `CONTAINER_DEVSPACE`, and `CONTAINER_PROJECT_ROOT` unchanged. For example, if in you ran the command in step 3 from directory `/home/my_user/projects` causing the cloned repo root to be `/home/my_user/projects/lstm_adversarial_attack`  your `lstm_adversarial_attack/docker/app/.env` file would look like this:

```
LOCAL_PROJECT_ROOT=/home/my_user/projects/lstm_adversarial_attack

PROJECT_NAME=lstm_adversarial_attack
CONTAINER_DEVSPACE=/home/devspace
CONTAINER_PROJECT_ROOT=${CONTAINER_DEVSPACE}/project
```



### 5. Build the `lstm_aa_app` Docker image

> **Note** The size of the `lstm_aa_app` image will be ~10 GB.

`cd` into directory `lstm_adversarial_attack/docker`, and run:

```
$ docker compose build
```

Image `lstm_aa_app` includes an installation of Miniconda3 and has a conda environment created in `/home/devspace/env`. All of the Python dependencies needed for the project are installed in this environment. All of these dependencies are shown in  `lstm_adversarial_attack/docker/app/environment.yml`. 

### 6. Run the `lstm_aa_app` and `postgres_mimiciii` containers

From directory `lstm_adversarial_attack/docker` run:

```
$ docker compose up -d
```



### 7. Exec into the `lstm_aa_app` container

The `lstm_aa_app` image has a sudo-privileged non-root user,  named `gen_user`.  The conda environment in `/home/devspace/env` gets activated whenever a `bash` or `zsh` shell is launched under `gen_user`. It will be convenient to use `docker exec` to launch a shell in the container for a couple steps later on.

To launch a `zsh` shell in the container run:

``` 
$ docker exec -it lstm_aa_app /bin/zsh
```

You will then be at a `zsh` prompt in the container.

```
> whoami
gen_user

> pwd
/home/devspace/lstm_adversarial_attack
# docker-compose maps directory is mapped to the local lstm_adversarial_attack repository root
```

> **Note**: `gen_user`'s `zsh`profile uses OhMyZsh with a Powerlevel10k theme. The shell features provided by these settings are not reflected in this Markdown file. 



### 8. Launch Jupyter Lab

From a `zsh` prompt in the container, run the following command to start a Jupyter Lab server:

```
> jupyter lab --no-browser --ip 0.0.0.0
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



### 9. Run the Project Jupyter Notebook

In the Jupyter Lab file explorer, navigate to the `/notebooks` directory, and open `lstm_adversarial_attack.ipynb`. Read through this notebook file, and use code cells within it to run the project.



## Original Paper Code Repository

Although  the code repository for [[Sun, 2018](https://dl.acm.org/doi/10.1145/3219819.3219909)] was not published, some of the authors reported on predictive modeling (but not adversarial attack) with a similar LSTM in:

[Tang, F., Xiao, C., Wang, F. and Zhou, J., 2018. Predictive modeling in urgent care: a comparative study of machine learning approaches. *Jamia Open*, *1*(1), (pp.87-98)](https://academic.oup.com/jamiaopen/article/1/1/87/5032901)

The repository for this paper is available at: https://github.com/illidanlab/urgent-care-comparative



# References

1. [Sun, M., Tang, F., Yi, J., Wang, F. and Zhou, J., 2018, July. Identify susceptible locations in medical records via adversarial attacks on deep predictive models. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 793-801).](https://dl.acm.org/doi/10.1145/3219819.3219909)
2. [Johnson, A., Pollard, T., and Mark, R. (2016) 'MIMIC-III Clinical Database' (version 1.4), *PhysioNet*.](https://doi.org/10.13026/C2XW26) 
3. [Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.](https://www.nature.com/articles/sdata201635)

4. [Tang, F., Xiao, C., Wang, F. and Zhou, J., 2018. Predictive modeling in urgent care: a comparative study of machine learning approaches. *Jamia Open*, *1*(1), pp.87-98.](https://academic.oup.com/jamiaopen/article/1/1/87/5032901)



