



## 3. Database Queries
We need to run four queries on the MIMIC-III PostgreSQL database. The paths to files containing the queries are stored in a list as `DB_QUERIES` in the project `config_paths` file:



These queries are modified versions  of  `.sql` files (originally written for Google Big Query) from https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/concepts/pivot . 

## 2. Preprocessing

* Brief description of preprocess module base class. 
* Avoid returning large objects to global scope (save checkpoints instead)
* Reasons why faster than original
* Filtering criteria (> 18 yrs, > 1 day LOS)

## 3. Dataset Details

* Size of dataset, number of unique stays, why we have more samples than original paper

## 4. LSTM Model

### 4.1 Hyperparameter Tuning

* General architecture
* Optuna objective function structure
* Figure with modules of final model

### 4.2 Assessing Predictive Model Performance

- 5-fold CV with extended number of epochs
- Reason for needing such large number of epochs: WeightedRandomSampler
- Comparison with other studies



## 5. Attacking the Trained Model

### 5.1 Attack Algorithm and Regularization

To perform an adversarial attack on a trained model, we start with the input features of a sample that the model correctly classifies. We then make small modifications to these features with the goal of finding features which, when provided as inputs to the model, are not predicted to be in the same class as the original sample. 

and attempt to find one or more set of modified features , when applied to the input,  to these inputs th we follow the approach of Sun et al. and use an adversarial loss function:





Our method of adversarial attack is similar to Chen et al.'s approch that uses an adversarial loss function and L1 regularization. When attacking a binary classification model with trained parameters $\theta$, we start with the input feature matrix $X$ of a sample that the model correctly predicts to be in class $t_{c}$, so  $M(X) = t_{c}$ where $M$ is the model's prediction function. We then search for a perturbation matrix $P$ that meets the condition:
$$
M(X + P) \ne t_{c}
$$
Since we are dealing with binary classification, this condition is equivalent to:
$$
M(X + P) = \neg{t_{c}}
$$
where $\neg{t_c}$ is the negation of $t_c$. Defining a perturbed feature matrix $\widetilde{X} = X + P$ , an adversarial loss function can be written as:
$$
max\{[Logit(\widetilde{X})]_{t_c} - [Logit(\widetilde{X})]_{\neg{t_c}}, - \kappa \}
$$

When running perturbed input $\widetilde{X}$ through a forward pass, $[Logit(\widetilde{X})]_{t_c}$ and $[Logit(\widetilde{X})]_{\neg{t_c}}$ are the **pre-activation** values of the 2-node final layer. We use $\kappa > 0$. $[Logit(\widetilde{X})]_{t_c} - [Logit(\widetilde{X})]_{\neg{t_c}} > \kappa$ corresponds to a successful adversarial attack.


$$
max\{[Logit(\widetilde{X})]_{y_\theta} - [Logit(X)]_{\widetilde{y}_\theta}, - \kappa \} + \lambda||\widetilde{X}-X||_1
$$



### 5.2 Attack Implementation Details

* Feature Perturber module
* How to convert standard trained model into logit out model



5.3 Attack Hyperparameter tuning



