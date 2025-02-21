# Setup

## Requirements

- `git`
- `Docker`
- **CUDA-enabled GPU**
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html#installation-guide)**  
  *Note: After installing the NVIDIA Container Toolkit, you may need to reboot.*

---

## Procedure

### **1. Build a PostgreSQL MIMIC-III Database Inside a Docker Container**

Follow the instructions in the repository **[docker-postgres-mimiciii](https://github.com/duanegoodner/docker-postgres-mimiciii)**, completing **Steps 1–6** to build:

- A **PostgreSQL MIMIC-III database** in a named Docker volume.
- A **Docker image** containing a PostgreSQL instance with access to the database.

> **Note:** ~60GB of storage is required, and the process may take ~60 minutes on a typical desktop system.

---

### **2. Clone the `icu-deep-learning` repository**
Run:

```shell
git clone https://github.com/duanegoodner/icu-deep-learning
 ```

 Then, `cd` into the local repo root:
 ```
 cd icu-deep-learning
 ```

---

### 3. Create and Populate `.env` and Password Files

### 3.1 `docker/app/.env`

```
cp docker/app/.env.example docker/app.env
```

Then, open `docker/app/.env`, and on this line:
```shell
LOCAL_PROJECT_ROOT=/absolute/path/to/icu-deep-learning
```
replace `/absolute/path/to/icu-deep-learning` with the actual **absolute path** of your local `icu-deep-learning` root directory.

> **Note:** Leave the values of other variables in this `.env` file unchanged


### 3.2 `docker/app/postgres_password.txt`

- Create file `./docker/app/postgres_password.txt`
- Copy the value of `POSTGRES_PASSWORD` in file `./postgres/.env` that you created in your local `docker-postgres-mimiciii` repo in **Step 1** above, and paste into your newly created `./docker/app/postgres_password.txt` file.


### 3.3 `docker/app/tuner_password.txt`

- Create and open file `./docker/app/tuner_password.txt`
- Enter whatever value you want to use for the **Optuna** database on the first line of this file, save, and close.


---

### 4. Build the `lstm_aa_app` Docker image

Navigate to `icu-deep-learning/docker/app/` and run:

```shell
UID=${UID} GID=${GID} docker compose build
```
This command builds the `lstm_aa_app` Docker image, based on ubuntu:22.04, with a **Miniconda3 environment** at `/home/devspace/env` containing all required Python dependencies.
A full list of dependencies is in `icu-deep-learning/docker/app/environment.yml`.

> **Note** The size of the `lstm_aa_app` image will be ~10 GB.

---

### 5. Start All Project Containers Using `docker compose`
From `icu-deep-learning/docker/app/` run:

```shell
UID=${UID} GID=${GID} docker compose up -d
```
Expected output:

```bash
[+] Running 5/5
 ✔ Network app_default              Created                                          0.2s 
 ✔ Volume "app_optuna_db"           Created                                          0.0s 
 ✔ Container postgres_optuna        Started                                          0.6s 
 ✔ Container postgres_mimiciii_dev  Started                                          0.6s 
✔ Container lstm_aa_app             Started                                          0.8s
```
TThe following **three containers** should now be running (`icu-deep-learning/docker/app/docker-compose.yml` defines these services):

- `lstm_aa_app` – Runs Python modules
- `postres_mimiciii`– Runs PostgreSQL with the MIMIC-III database
- `postgres_optuna` – Runs another PostgreSQL instance for hyperparameter tuning results

We can confirm which containers are running with:
```
docker ps --format "table {{.ID}}\t{{.Ports}}\t{{.Names}}"
```
Expected output:
```
CONTAINER ID   PORTS                                                NAMES
8b3b66e7181c   127.0.0.1:6006->6006/tcp, 127.0.0.1:8888->8888/tcp   lstm_aa_app
e481271381bf   0.0.0.0:5555->5432/tcp, [::]:5555->5432/tcp          postgres_mimiciii_dev
2e962c184265   0.0.0.0:5556->5432/tcp, [::]:5556->5432/tcp          postgres_optuna
```


---


### 6. Enter the `lstm_aa_app` container

The `lstm_aa_app` image contains a **sudo-privileged non-root user**, `gen_user`.  The conda environment (`/home/devspace/env`) activates automatically when a `bash` or `zsh` shell is launched.

Run:

```bash 
docker exec -it lstm_aa_app /bin/zsh
```
Verify the setup with:

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
> **Note** The `docker-compose.yml` maps container directory `/home/devspace/icu-deep-learning` to your local `icu-deep-learning` root.

---

### 7. Run Project Code Inside the `lstm_aa_app` Container

Now that we are at a command prompt inside the `lstm_aa_app` we have two options for running project code: directly from the command line, or in a Jupyter notebook. 

---

#### 7.1 Runnng from command line

Instructions for running from the command line are provided inside the Jupyter notebook. View these instructions with any one of the following methods:

- Navigating to the [notebook file on GitHub](https://github.com/duanegoodner/icu-deep-learning/blob/main/notebooks/icu_deep_learning.ipynb).
- Opening local file [`./notebooks/icu_deep_learning.ipynb`](notebooks/icu_deep_learning.ipynb) in a Jupyter viewer.
-  Viewing in Google Colab:  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duanegoodner/lstm_adversarial_attack/blob/main/notebooks/icu_deep_learning.ipynb)

 Then, follow the the **"Running from the command line"** section of the notebook

---

#### 7.2 Running Jupyter Lab

##### 7.2.1 Start Jupyter server

From a `zsh` prompt in the container, run:

```
jupyter lab --no-browser --ip=0.0.0.0
```
---

#### 7.2.2 Open Jupyter Lab in Your Browser


Look for a line in the terminal output that starts with:
```
http://127.0.0.1:8888/lab?token=...
```
Example:
```
 http://127.0.0.1:8888/lab?token=1c4722891835e04ffec392da92ec9b49a8962a370b261e36
```
Copy this **full URL** into your browser to launch Jupyter Lab.

---

### 7.2.3 Open the project notbook in Jupyter Lab

In Jupyter Lab's file explorer, open:

```
notebooks/icu_deep_learning.ipynb

```

Read through the notebook and execute its code cells to run the project.

---
