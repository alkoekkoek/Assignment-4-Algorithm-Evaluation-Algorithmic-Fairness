# Assignment-Algorithm-Evaluation-Algorithmic-Fairness
For this assignment I evaluate the effectiveness of the bias-mitigation algorithm reweighing as described by Kamiran and Calders (2012). For my code I used the AIF360 github: https://github.com/Trusted-AI/AIF360/tree/main. Specifically, I adapted/used code from two examples that use reweighing provided by the github: 
- https://github.com/Trusted-AI/AIF360/blob/main/examples/tutorial_credit_scoring.ipynb
- https://github.com/Trusted-AI/AIF360/blob/main/examples/demo_reweighing_preproc.ipynb

## How to run
1. Open `Assignment_4_Algorithm_Evaluation_Algorithmic_Fairness.ipynb`.
2. Run all cells. The first cell installs AIF360 (results are produced with version 0.6.1) and downloads the raw data from UCI (Adult, German) and ProPublica (COMPAS) into AIF360's data folder.
3. The final results table is saved as `results.csv`.

## Parameter settings
- Train/test split: 70/30 (train/test), stratified on the class labels, `random_state=0`.
- Reweighing: AIF360 `Reweighing`, fitted on the training set only (no parameters).
- Classifier: scikit-learn `LogisticRegression` with default settings, features scaled with `StandardScaler` (fitted on the training set only), default decision threshold 0.5.
- DP_Diff is reported as P(Y=1 | privileged) − P(Y=1 | unprivileged), instead of P(Y=1 | unprivileged) - P(Y=1 | privileged) (sign flipped to make DP_diff similar to Kamiran & Calders(2012)).

## Downloading the data
I evaluated the effectiveness of reweighing on three datasets, which are downloaded in the python notebook from UCI and ProPublica and placed in AIF360's data folders.
- Adult Census Income; Protected attribute is "sex", with sex = male being considered privileged. 
- German Credit Data; Protected attribute is "age", with age>= 25 being considered privileged.
- COMPAS Recidivism Racial Bias; Protected attribute is "sex", with sex = female being considered privileged. 

Warnings about "fairlearn" or "inFairness" can be ignored as those algorithms are not used.

## Computing fairness measures before mitigation and training any classifiers 
The demographic (statistical) parity difference (DP_Diff), the disparate impact ratio (DIR), and the consistency were calculated as fairness measures. The results were saved for later comparison. 

## Train/test split with stratification
AIF360 datasets cannot be passed to `train_test_split` directly, so the row numbers are split (stratified on the flattened class labels) and the matching rows are selected with `.subset()`.

## Reweighing the data, training a model and saving predictions
It is important that only the training data is reweighed and NOT the test data, as the test set must reflect the real data. For this assignment two logistic regression classifiers are trained: a baseline model using the original data that was not reweighed and a model with the reweighed data.

## Computing fairness measures after mitigation (test predictions)
The demographic (statistical) parity difference (DP_Diff), the disparate impact ratio (DIR), and the consistency were calculated as fairness measures. Additionally, the accuracy and balanced accuracy were calculated to measure the predictive performance of the baseline and reweighed model predictions. All fairness measures are then compared between different stages: before mitigation & classification (original labels), baseline, no reweighing (test predictions), and after reweighing (test predictions). 

## References
Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems, 33(1), 1–33. https://doi.org/10.1007/s10115-011-0463-8
https://github.com/Trusted-AI/AIF360/tree/main
- https://github.com/Trusted-AI/AIF360/blob/main/examples/tutorial_credit_scoring.ipynb
- https://github.com/Trusted-AI/AIF360/blob/main/examples/demo_reweighing_preproc.ipynb
