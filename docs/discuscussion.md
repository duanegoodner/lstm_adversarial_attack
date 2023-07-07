



This project reproduces and expands upon work published in [1] and [2] on Long Short-Term Memory (LSTM) predictive models and adversarial attacks on those models.  The previous studies used Long Short-Term Memory (LSTM) time series classification models trained with data from the Medical Information Mart for Intensive Care (MIMIC-III) database to predict Intensive Care Unit (ICU) patient outcomes. Input features to the classification models consisted of 13 lab measurements and 6 vital signs. A binary variable representing in-hospital mortality was the prediction target.

In [1], an adversarial attack algorithm was used to identify small perturbations which, when applied to a real, correctly-classified input features, caused a trained model to misclassify the perturbed input. L1 regularization was applied to the adversarial attack loss function to favor adversarial examples with sparse perturbations that resemble the structure of data entry errors most likely to occur in real medical data. Samples were attacked serially (one a time), and the attack process on a sample was stopped upon finding a single adversarial perturbation to that samples input features. After attacking a full dataset, susceptibility calculations were  performed to identify input feature space regions most vulnerable to adversarial attack.

The current study follows an approach similar to that of the previous studies. We use the same dataset, input features, and prediction targets to train a LSTM binary classification model and subsequently search for adversarial examples using an L1 regularized attack algorithm. Aspects of the current work that expand upon the previous studies include a vectorized (faster) approach to data preprocessing, extensive hyperparameter tuning (of both the predictive model and attack algorithm), improved performance of the predictive model, implementation of a GPU-compatible attack algorithm that enables attacking samples in batches, and not halting the attack process upon finding a single adversarial perturbation for a sample (so that additional, lower loss adversarial perturbations can be discovered).







## 







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

When running perturbed input $\widetilde{X}$ through a forward pass, $[Logit(\widetilde{X})]_{t_c}$ and $[Logit(\widetilde{X})]_{\neg{t_c}}$ are the **pre-activation** values at the nodes corresponding to $t_c$ and $\neg{t_c}$ the in 2-node final layer. We use $\kappa > 0$. $[Logit(\widetilde{X})]_{t_c} - [Logit(\widetilde{X})]_{\neg{t_c}} > \kappa$ corresponds to a successful adversarial attack.


$$
max\{[Logit(\widetilde{X})]_{y_\theta} - [Logit(X)]_{\widetilde{y}_\theta}, - \kappa \} + \lambda||\widetilde{X}-X||_1
$$



### 5.2 Attack Implementation Details

* Feature Perturber module
* How to convert standard trained model into logit out model



5.3 Attack Hyperparameter tuning



