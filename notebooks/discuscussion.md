









### 8. Attacking the Trained Model

To perform an adversarial attack on a trained model, we start with the input features of a sample that the model correctly classifies. We then make small modifications to these features with the goal of finding features which, when provided as inputs to the model, are not predicted to be in the same class as the original sample. 

and attempt to find one or more set of modified features , when applied to the input,  to these inputs th we follow the approach of Sun et al. and use an adversarial loss function:





Our method of adversarial attack is similar to the approaches of Chen et al. and Sun et al. When attacking a binary classification model with trained parameters $\theta$, we start with the input feature matrix $X$ of a sample that the model correctly predicts to be in class $t_{c}$, so  $M(X) = t_{c}$ where $M$ is the model's prediction function. We then search for a perturbation matrix $P$ that meets the condition:
$$
M(X + P) \ne t_{c}
$$
Since we are dealing with binary classification, this condition is equivalent to:
$$
M(X + P) = \neg{t_{c}}
$$
where $\neg{t_c}$ is the boolean opposite of $t_c$. Defining a perturbed feature matrix $\widetilde{X} = X + P$ , an adversarial loss function can be written as:
$$
max\{[Logit(\widetilde{X})]_{t_c} - [Logit(X)]_{\neg{t_c}}, - \kappa \}
$$

 $Logit(Z)$ is obtained by 
$$
max\{[Logit(\widetilde{X})]_{y_\theta} - [Logit(X)]_{\widetilde{y}_\theta}, - \kappa \} + \lambda||\widetilde{X}-X||_1
$$




