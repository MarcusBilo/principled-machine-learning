## Table of Contents

1. Document  
   1.1 Introduction  
   1.2 Methodology  
   1.2.1 Model Selection  
   1.2.2 Training and Evaluation Procedure  
   1.2.3 Statistical Framework  
   1.2.4 Bayesian Bootstrap  
   1.2.5 Region of Practical Equivalence  
   1.2.6 Evaluation Metrics  
   1.3 Results  

Bibliography  
Appendix  

---

# 1. Document

This work illustrates one way to tackle a machine learning task. In particular, this example is intended as a guidepost towards more reliable conclusions from machine learning through better methodology.

The document focuses on methodological reasoning and interpretation, while implementation details can be found in the Python script.

## 1.1 Introduction

When applying machine learning, the first and most fundamental question is: what exactly are we trying to achieve?

Here, we analyse the Wisconsin Breast Cancer Dataset with two objectives:
- Distinguish as accurately as possible between malignant and benign tumours  
- Identify interpretably how to reach this distinction  

We also want the models to be deterministic.

---

## 1.2 Methodology

Before training any model, it is essential to understand the dataset:

- Are any features irrelevant (e.g., ID columns)?
- What types of variables exist?
- Are there missing values?
- Are there implausible values?

For this dataset:
- No irrelevant columns  
- All variables numeric  
- No missing values  
- No faulty measurements  

### Outliers

Outliers may arise from:
1. Population mismatch  
2. Legitimate extreme observations  
3. Random error  
4. Measurement/data errors  

In general, observations should be retained unless clearly erroneous.

---

## 1.2.1 Model Selection

Models used:
- Support Vector Machine (linear kernel)  
- Logistic Regression  

Reasons:
- Interpretability (linear boundaries)
- Deterministic behavior
- Solid performance in classification tasks  

---

## 1.2.2 Training and Evaluation Procedure

- Multiple runs with different random seeds  
- Repeated stratified k-fold cross-validation  
- Prevents biased splits  

Pipeline design:
- Standardization within each training fold  
- No data leakage  

No hyperparameter tuning inside CV → avoids selection bias.

---

## 1.2.3 Statistical Framework

Traditional NHST has limitations.

Example misconceptions:
- p-value ≠ probability hypothesis is true  
- significance ≠ reliability  
- confidence intervals ≠ probability statements  

Bayesian methods address these issues.

---

## 1.2.4 Bayesian Bootstrap

- Uses Dirichlet-distributed weights  
- Estimates posterior distribution  

Outputs:
- Posterior mean  
- Credible intervals  

Model comparison:
- Shared bootstrap weights preserve correlations  
- Estimates probability each model is best  

### Winner’s Curse

Selecting the best model introduces upward bias in performance estimates.

---

## 1.2.5 Region of Practical Equivalence (ROPE)

Instead of strict comparison:
- Compare practical differences  

Example:
- ROPE = ±0.01 → differences <1% are negligible  

ROPE depends on:
- Application needs  
- Dataset variability  

Multiple ROPE values used for sensitivity analysis.

---

## 1.2.6 Evaluation Metrics

### Core Metrics

- **TPR (Recall):**
  TPR = TP / (TP + FN)

- **TNR:**
  TNR = TN / (TN + FP)

- **PPV:**
  PPV = TP / (TP + FP)

- **NPV:**
  NPV = TN / (TN + FN)

### Combined Metrics

- **Informedness (Youden’s J):**
  J = TPR + TNR - 1

- **Markedness (ΔP):**
  ΔP = PPV + NPV - 1

Interpretation:
- Informedness → class separation  
- Markedness → prediction reliability  

For this dataset → focus on **Informedness**

---

## 1.3 Results

- Both models perform similarly  
- Informedness ≈ 0.94–0.95  
- No meaningful difference under ROPE  

Feature importance:
- Tumour size  
- Texture  
- Concavity  

### Important Note

False negatives (missed malignant tumors) are the most critical errors → further analysis.

---

# Bibliography

[1] Orr et al. (1991)  
[2] Gress et al. (2018)  
[3] Wainer (2016)  
[4] Strang et al. (2018)  
[5] Shmuel et al. (2025)  
[6] Cawley & Talbot (2010)  
[7] Benavoli et al. (2017)  
[8] Gigerenzer (2004)  
[9] Rubin (1981)  
[10] Guesné et al. (2024)  
[11] Powers (2015)  
[12] Opitz (2024)  
[13] Foulle (2025)  

---

# Appendix

## Maximum Divergence Between Informedness and Markedness

Dataset:
- Malignant: 212  
- Benign: 357  

FN = 212 - TP  
FP = 357 - TN  

TPR = TP / 212  
TNR = TN / 357  

PPV = TP / (TP + 357 - TN)  
NPV = TN / (TN + 212 - TP)  

J = TP/212 + TN/357 - 1  
ΔP = TP/(TP + 357 - TN) + TN/(TN + 212 - TP) - 1  

Maximum at (TP, TN) = (212, 294)

J ≈ 0.8235  
ΔP ≈ 0.7709  
f_max ≈ 0.0526  

---

## Brute Force Solution (Python)

```python
max_f = float("-inf")
best_cases = []

for TP in range(213):
    FN = 212 - TP

    for TN in range(358):
        FP = 357 - TN

        if TP + FP == 0 or TN + FN == 0:
            continue

        TPR = TP / 212
        TNR = TN / 357
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)

        J = TPR + TNR - 1
        deltaP = PPV + NPV - 1

        f = J - deltaP

        if f <= 0 or J <= 0:
            continue

        if f > max_f:
            max_f = f
            best_cases = [(TP, TN, FP, FN, f)]
        elif abs(f - max_f) < 1e-12:
            best_cases.append((TP, TN, FP, FN, f))

print(max_f, best_cases)
```
