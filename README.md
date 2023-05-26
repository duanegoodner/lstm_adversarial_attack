# lstm_adversarial_attack
*A reproduction study of adversarial attacks on a LSTM time series model*



## Overview

This project aims to reproduce results originally published in:

Sun, M., Tang, F., Yi, J., Wang, F. and Zhou, J., 2018, July. Identify susceptible locations in medical records via adversarial attacks on deep predictive models. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining* (pp. 793-801).  (https://arxiv.org/abs/1802.04822)

The original paper trained a Long Short-Term Memory (LSTM) model using time series of Intensive Care Unit (ICU) patient vital-sign and lab measurements as the inputs, and in-hospital mortality as the prediction target. An adversarial attack algorithm was then used to identify small perturbations which, when applied to a real, correctly-classified input sample, result in misclassification of the perturbed input. Susceptibility calculations were then performed to quantify the attack vulnerability as functions of time and the measurement type within the input feature space.



## Original Paper Code Repository

The original authors did not publish a code repository for this particular work, but some of the authors reported on predictive modeling (but not adversarial attack) with a similar LSTM in:

Tang, F., Xiao, C., Wang, F. and Zhou, J., 2018. Predictive modeling in urgent care: a comparative study of machine learning approaches. *Jamia Open*, *1*(1), pp.87-98.

The repository for this paper is available at: https://github.com/illidanlab/urgent-care-comparative



## Dependencies

* Python 3.9+

* dill

* gensim

* matplotlib

* numpy

* pandas

* scikit-learn

* seaborn



## Instructions for running code and viewing results in this repository

### Step 1. Download data

This work uses a subset of the MIMIC-III database. The queries used to extract the necessary views and tables from MIMIC-III are saved in `src/docker_db/mimiciii_queries` . The outputs from running these queries are saved as `.targ.gz` files in in`data/mimiciii_query_results` . (So it is not necessary to download the MIMIC-III database. Just clone this repository.) After cloning the ehr_adversarial_attack repository, run the following command from the project root to extract the query results into .csv format:

```
$ find data/mimiciii_query_results -type f -name '*.tar.gz' -execdir tar -xzvf {} \;
```

Output:

```
pivoted_bg.csv
d_icd_diagnoses.csv
icustay_detail.csv
pivoted_vital.csv
admissions.csv
diagnoses_icd.csv
pivoted_gc.csv
pivoted_uo.csv
pivoted_lab.csv
```

### Step 2. Obtain pre-processed data

There are two options for obtaining data that have been converted from the SQL query output format to the feature and label tensors required for LSTM model input. Option A is faster.

#### Option A: Extract the existing preprocessed archive files

From the repository root directory, run the command:

```
$ find data/output_feature_finalizer -type f -name '*.tar.gz' -execdir tar -xzvf {} \;
```

Output:

```
in_hospital_mortality_list.pickle
measurement_data_list.pickle
measurement_col_names.pickle
```

#### Option B: Run the preprocessing code

From the repository root directory, run the following command to preprocess the SQL query outputs into the LSTM model input format:

```
$ python src/preprocess/main.py
```

Output:

```shell
Starting Prefilter
Done with Prefilter

Starting ICUStatyMeasurementCombiner
Done with ICUStatyMeasurementCombiner

Starting FullAdmissionListBuilder
Done with FullAdmissionListBuilder

Starting FeatureBuilder
Done processing sample 0/41960
Done processing sample 5000/41960
Done processing sample 10000/41960
Done processing sample 15000/41960
Done processing sample 20000/41960
Done processing sample 25000/41960
Done processing sample 30000/41960
Done processing sample 35000/41960
Done processing sample 40000/41960
Done with FeatureBuilder

Starting FeatureFinalizer
Done with FeatureFinalizer

All Done!
Total preprocessing time = 555.2803893089294 seconds

```



### Step 3. Obtain a trained LSTM model

This step also has a fast option (A), and a slow option (B).

#### Option A: Do nothing, except note the filename that will be needed for Step 4

Parameters of a pre-trained LSTM model are saved in `data/cross_validate_sun2018_full48m19_01/2023-05-07_23:32:09.938445.tar`. Take note of the filename `2023-05-07_23:32:09.938445.tar` (just the filename, not the full path), so you can use it as input when running adversarial attacks in Step 4.

#### Option B: Run the cross-validation script

From the repository root directory, run:

```
$ python src/LSTMSun2018_cross_validate.py
```

This will train LSTM model using cross-validation and and evaluate the predictive performance periodically throughout the training process. Note that this method (used in the original paper) has drawbacks due to the fact that it never evaluates the model on unseen data, so there is significant risk of overfitting.

Output:

```ll
# Many realtime updates on training loss and evaluation metrics
Model parameters saved in file: 2023-05-07_23:40:59.241609.tar
# your filename will be different, but will have the same format
```

Take note of the .tar filename. for use in Step 4.



### Step 4. Run adversarial attacks on the model

Now we are ready to run an adversarial attack on an the trained LSTM model. From the repository root, run:

```
$ python src/adv_attack_full48_m19.py -f <file_name_obained_from_step_3.tar>
```

For example, if you used Option A in Step 3, you would run `$ python src/adv_attack_full48_m19.py -f 2023-05-07_23:32:09.938445.tar`

Output:

```
# Updates on elapsed time and index of sample under attack
Attack summary data saved in k0.0-l10.15-lr0.1-ma100-ms1-2023-05-07_23:47:51.261865.pickle
# Your .tar filename will be different, but will have the same format.
```

Once again, only the filename (not full path), is printed, but that's all you need for the next step.



### Step 5. Plot results of the adversarial attacks

From the repo root directory, run:

```
$ python src/attack_results_analyzer.py -f <filename_from_step4_output.tar>
```

For example, if your output from Step 4 said: `Attack summary data saved in k0.0-l10.15-lr0.1-ma100-ms1-2023-05-07_23:47:51.261865.pickle`, you would run `$ python src/attack_results_analyzer.py -f k0.0-l10.15-lr0.1-ma100-ms1-2023-05-07_23:47:51.261865.pickle `

This will produce plots of attack susceptibilities of various measurements vs. time for 0-to-1 and 1-to-0 attacks.  (The plot for 1-to-0 should appear first. Depending on your environment, you may need to close this plot window before the 0-to-1 plot appears.)



## Key Results

#### LSTM predictive metrics

Here are the LSTM predictive evaluation metrics from the original paper and our work. The performance metric scores from our study are actually higher than those in the original work.

<img src="https://github.com/duanegoodner/ehr_adversarial_attack/blob/main/data/images/LStM_predictive_metrics.png"  width="40%" height="30%">



#### Adversarial attack susceptibility vs. measurement parameter

The table below indicates that in our study, we were unable to reproduce the attack susceptibilities reported in the original paper.

![](https://github.com/duanegoodner/ehr_adversarial_attack/blob/main/data/images/Table.png)



#### Adversarial attack susceptibility vs measurement time

These below plots also do NOT show the increase in susceptibility at later measurement times that were reported in the original paper.

![](https://github.com/duanegoodner/ehr_adversarial_attack/blob/main/data/images/plots.png)



