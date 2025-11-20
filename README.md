# Predicting motor recovery after apinal cord injury using a stacked ensemble
![R](https://img.shields.io/badge/R-4.5.1-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-orange)

## Overview
This project develops a supervised machine learning model to predict 12-month motor recovery outcomes for individuals with spinal cord injury (SCI). Using data collected during the first 30 days post-injury, the model estimates the International Standards for Neurological Classification of Spinal Cord Injury (ISNCSCI) total motor score at 12 months. View the full rendered project page here: **https://jcoa05.github.io/emsci-ml/**

## Objective
Predict the ISNCSCI total motor score (0-100) at 12 months post-injury using early clinical data, leveraging stacked ensemble learning for optimal predictive performance.

## Dataset
- **Source:** Simulated dataset modeled after the European Multicenter Study about Spinal Cord Injury (EMSCI).
- **Sample Size:** 1,500 patients.
- **Design:** Includes realistic heterogeneity, variance, and ceiling effects.
- **Input Features (Day 0-30):**
  - Age
  - Sex
  - Cause of Injury
  - Neurological Level of Injury (NLI)
  - Baseline AIS Grade
  - Lower Extremity Motor Scores (LEMS)
  - Sensory Scores (Light Touch, Pin Prick)
- **Target Variable:** ISNCSCI Total Motor Score at 12 months (0-100).

## Methodology
### Modeling Approach: Stacked Generalization
The project uses stacked ensemble learning to combine diverse algorithms for improved predictive performance.

### Level 0: Base Learners
- **XGBoost** (Gradient Boosting)
- **SVM (RBF kernel)** via `kernlab`
- **MARS** via `earth`
- **GLMNet** (Elastic Net / Lasso)

### Level 1: Meta-Learner
- **LASSO Regression** to aggregate Level 0 predictions.

### Validation Strategy
- **5-fold Cross-Validation** across all base learners and the meta-learner.

## Results
- **R-Squared:** ~0.74
- **RMSE:** ~14.8
- **MAE:** ~11.3

### Subgroup Performance
- **Paraplegic:** R² = 0.65
- **Tetraplegic:** R² = 0.78
- **AIS A:** R² = 0.45 (Most difficult subgroup to predict)
- **AIS D:** R² = 0.75

### Calibration
Overall calibration is strong, with slight underprediction in high recovery ranges (76-100).

## Interpretability
Model interpretability was assessed using permutation-based and SHAP-based approaches.

### Key Predictors
- Baseline Total Motor Score (strongest predictor)
- AIS Grade
- Lower Extremity Motor Score (LEMS)

## Tech Stack
- **Language:** R (v4.5.1)
- **Framework:** tidymodels
- **Key Libraries:**
  - `stacks`
  - `xgboost`
  - `ranger`
  - `earth`
  - `kernlab`
  - `DALEX`
  - `fastshap`
  - `tidyverse`

## Contact
For questions or collaborations, please contact me. Found an error? Open a ticket!
