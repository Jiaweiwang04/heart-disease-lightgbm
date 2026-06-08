# Heart Disease Prediction with LightGBM

## Project Overview

This project explores the UCI Heart Disease dataset and aims to build a binary classification model to predict whether a patient has heart disease.

## Current Progress

- [x] Data loading
- [x] Exploratory Data Analysis
- [ ] Data preprocessing
- [ ] Baseline models
- [ ] LightGBM model
- [ ] Model evaluation
- [ ] Feature importance and SHAP analysis

## Dataset

The original label column is `num`, which indicates disease severity.
A binary classification target is created as:

target = 1 if num > 0 else 0

## Repository Structure

- `data/` - raw and processed dataset files
- `notebooks/` - exploratory notebooks
- `README.md` - project overview and progress

## Next Steps

- Handle missing values
- Encode categorical variables
- Create train/test split
- Build baseline models
- Train LightGBM
