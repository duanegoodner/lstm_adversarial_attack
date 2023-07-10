



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





Our method of adversarial attack is similar to Chen et al.'s approach that uses an adversarial loss function and L1 regularization. When attacking a binary classification model with trained parameters $\theta$, we start with the input feature matrix $X$ of a sample that the model correctly predicts to be in class $t_{c}$, so  $M(X) = t_{c}$ where $M$ is the model's prediction function. We then search for a perturbation matrix $P$ that meets the condition:
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

When running perturbed input $\widetilde{X}$ through a forward pass, $[Logit(\widetilde{X})]_{t_c}$ and $[Logit(\widetilde{X})]_{\neg{t_c}}$ are the **pre-activation** values at the nodes corresponding to $t_c$ and $\neg{t_c}$ the in 2-node final layer. A value $\ge 0$ is chosen for $\kappa$. Using a small non-zero value of $\kappa$ will prevent an attack algorithm from optimizing toward an infinitesimally small gap between $[Logit(\widetilde{X})]_{t_c}$ and $[Logit(\widetilde{X})]_{\neg{t_c}}$ while still targeting the small difference we want for an adversarial example.

To encourage an attack algorithm to find sparse perturbations, the following regularized version of Equation  () is used  

$$
max\{[Logit(\widetilde{X})]_{y_\theta} - [Logit(X)]_{\widetilde{y}_\theta}, - \kappa \} + \lambda||\widetilde{X}-X||_1
$$

wher $\lambda$ is the L1 regularization constant. Equation () can be minimized by subgradient descent or by an Iterative Soft-Thresholding Algorithm (ISTA). The latter approach typically converges faster. 



### 5.2 Attack Implementation Details

Adversarial attacks on a particular model and dataset input features are managed by an `AdversarialAttackTrainer`. In the procedure outlined below, we discover an adversarial example any time we find $[Logit(\widetilde{X})]_{\neg{t_c}} > [Logit(\widetilde{X})]_{t_c}$, even if we have not converged near a minimum value of Equation (). We attack each batch of samples for a fixed number of iterations, regardless of how many (if any) adversarial examples are found.

1. A `LogitNoDropoutModelBuiler` creates a modified version of the target model. The modified model has all dropout probabilities set to zero, and does not have an activation function on the output layer.
2. Batches of input features are run through a `FeaturePerturber` (implemented in `attack.feature_perturber`) that generates slightly modified versions of original features
3. The perturbed features are run through the modified model that was built by the `LogitNoDropoutModelBuiler` to obtain values for $[Logit(\widetilde{X})]_{t_c}$ and $[Logit(\widetilde{X})]_{\neg{t_c}}$
4. An instance of custom PyTorch loss function ` AdversarialLoss`, which implements Equation (), calculates a loss tensor
5. The Pytorch `.backward()`  method of the loss tensor finds the gradient of the loss with respect to the elements of the `FeaturePerturber.perturbation` tensor

6. If the current $Logit$ values resulting from a sample's perturbed input features represent an adversarial example, and the example is either the first or lowest loss example for that sample, the perturbations and other details are stored in a `BatchResult` object.

7. A Pytorch optimizer uses the loss gradient to calculate and apply adjustments to the perturbations

8. The `AdversarialAttackTrainer.apply_soft_bounded_threshold()` method performs ISTA thresholding on the perturbations

9. The perturbations (which have been adjusted by the optimizer *and* ISTA thresholding, are used in step 1 of the next attack iteration.

Two key points worth noting in the above procedure are: (1) Unlike the procedure used in [], we do not stop attacking an example upon finding a single adversarial perturbation for it.  (2) We use a combination of subgradient descent (in step 7), and ISTA (in step 8) to minimize (or at least reduce) the value of equation (). We do not know if this approach is guaranteed to converge to a minimum in the adversarial loss function, but empirically, we find this subgradient descent + ISTA more effective at finding sparse adversarial examples than either method is on its own.





