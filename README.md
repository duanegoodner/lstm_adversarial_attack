# ICU Deep Learning


## Overview

This project implements and optimizes **Long Short-Term Memory (LSTM)** time series models of intensive care unit (ICU) patient lab and vital sign data to predict patient outcomes. Additionally, an **adversarial attack** algorithm is used to discover adversarial examples for trained models.  

The predictive model input consists of data from 13 lab measurements and 6 vital signs collected during the first 48 hours after patient admission to the ICU. The prediction target is a binary variable representing in-hospital mortality. This type of model can supplement standard heuristics used by care providers in identifying high-risk patients.

The results of adversarial attacks on trained LSTM models provide a gauge of model stability. Additionally, these results offer an opportunity to compare adversarial vulnerabilities in LSTMs with the well-documented adversarial behaviors observed in Convolutional Neural Networks (CNNs) used for computer vision. Unlike the adversarial examples found in CNNs — where perturbations imperceptible to the human eye can drastically alter model predictions — the adversarial examples discovered for our LSTM models exhibit a higher degree of plausibility, aligning more closely with human intuition.

This project builds upon LSTM and adversarial attack studies of the same dataset published in [[1](#ref_01)] and [[2](#ref_02)]. A great deal of gratitude is owed the authors of those investigations for providing a starting point for the current work. The 


## Highlights

- Extensive hyperparameter tuning for predictive and attack models.
- Flexibile attack objectives allow targeting different types of adversarial perturbations.
- Fully containerized for easy, reproducible environment setup.
- Single `config.toml` file centralizes all parameters for streamlined modification and experimenation.
- Auto-generated data provenance ensures reproducibility and prevents losing track of "what worked" during experiments.
- Modular data pipeline eliminates need for redundant upstream runs when testing multiple downstream settings.
- Flexible execution. Each pipeline component can run from the command line or inside the project's Jupyter notebook.
- Efficient adversarial attacks. Implemented a custom PyTorch AdversarialAttacker module capable of attacking batches of samples.
- Compared to previous studies of the same dataset:
  - Higher predictive performance
  - 10x faster data preprocessing


## Documentation

Detailed documentation can be viewed in the project Jupyter notebook available on NBViewer: 

[![View in nbviewer](https://img.shields.io/badge/Open%20in-nbviewer-orange)](https://nbviewer.org/github/duanegoodner/lstm_adversarial_attack/blob/main/notebooks/lstm_adversarial_attack.ipynb)





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

Among the various text output to the terminal, you should see some lines that look like this:

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


## Tools Used 

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)   ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?logo=pytorch&logoColor=white)   ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)   ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?logo=postgresql&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)   ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) ![Apache Arrow](https://img.shields.io/badge/Apache%20Arrow-0E77B3?logo=apache) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)   ![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?logo=tensorflow&logoColor=white)   ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white)   ![Optuna](https://img.shields.io/badge/Optuna-7C3AED?logo=python&logoColor=white) ![msgspec](https://img.shields.io/badge/msgspec-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&logo=Jupyter&logoColor=white)



# References



<a id="ref_01">1.</a> [Sun, M., Tang, F., Yi, J., Wang, F. and Zhou, J., 2018, July. Identify susceptible locations in medical records via adversarial attacks on deep predictive models. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 793-801).](https://dl.acm.org/doi/10.1145/3219819.3219909)

<a id="ref_02">2.</a> [Tang, F., Xiao, C., Wang, F. and Zhou, J., 2018. Predictive modeling in urgent care: a comparative study of machine learning approaches. *Jamia Open*, *1*(1), pp.87-98.](https://academic.oup.com/jamiaopen/article/1/1/87/5032901)

<a><a id="ref_03">3.</a> </a>[Johnson, A., Pollard, T., and Mark, R. (2016) 'MIMIC-III Clinical Database' (version 1.4), *PhysioNet*.](https://doi.org/10.13026/C2XW26) 

<a id="ref_04">4.</a> [Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.](https://www.nature.com/articles/sdata201635)



