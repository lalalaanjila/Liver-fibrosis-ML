# Liver-fibrosis-ML
# Reproducible Code for Machine Learning Analysis

## Overview
This repository provides reproducible example scripts for the machine learning workflow used in this study, including data preprocessing, feature selection, hyperparameter tuning, final model training, and internal testing.

Because the original study data are subject to data access restrictions and/or institutional data-use policies, the raw data are not publicly distributed in this repository. A small example input file is provided to illustrate the expected data structure and column naming format.

---
## Repository structure
```text
code/
├── 01_data_preprocessing_imputation_split.R
├── 02_feature_selection_boruta.R
├── 02_feature_selection_lasso_and_intersection.py
├── 03_model_tuning_tidymodels.R
└── 04_model_training_and_testing.R

data_example/
└── raw_dataset_example.csv

Script descriptions
01_data_preprocessing_imputation_split.R
This script:
reads the raw dataset with missing values
splits the dataset into training, internal validation, and internal testing cohorts
performs missing-value handling after cohort splitting
exports processed datasets and missing-data summaries

02_feature_selection_boruta.R
This script:
performs Boruta-based feature selection using the training cohort only
exports variable importance statistics and confirmed features

02_feature_selection_lasso_and_intersection.py
This script:
performs LASSO-based feature selection using the training cohort
generates LASSO tuning plots
visualizes the overlap between Boruta-selected and LASSO-selected features

03_model_tuning_tidymodels.R
This script:
performs hyperparameter tuning for multiple machine learning models using the training cohort
exports tuning metrics and the best parameter combinations

04_model_training_and_testing.R
This script:
trains the final models using the imputed training cohort
evaluates model performance using cross-validation in the training cohort
generates predictions for the internal testing cohort
exports ROC curve data and summary performance metrics
Expected input variables

The example workflow assumes that the dataset contains the following variables:
group
Age
Gender
Educational
Height
Weight
BMI
Waist_circumference
Hip_circumference
Hypertension
DM

Additional variables may be included in the original analytical dataset depending on the feature-selection step.

Detailed variable mapping and descriptions are provided in the manuscript supplementary materials.

Running order

Please run the scripts in the following order:
01_data_preprocessing_imputation_split.R
02_feature_selection_boruta.R
02_feature_selection_lasso_and_intersection.py
03_model_tuning_tidymodels.R
04_model_training_and_testing.R
Example data

The file data_example/raw_dataset_example.csv is a synthetic example input file created only to illustrate the expected file structure, variable names, and coding format. It is not part of the original study dataset and should not be used for reproducing the reported results.

Software environment
The scripts generate session or package version files during preprocessing, feature selection, tuning, and final modeling to facilitate reproducibility.

The main software environment includes:
R
Python
tidymodels
ranger
xgboost
bonsai / lightgbm
naivebayes
kknn
kernlab
Boruta
missForest
scikit-learn
pandas
numpy
matplotlib
networkx

Please refer to the exported session information files for exact package versions.

Data availability
The original data used in this study are not publicly distributed in this repository because they are subject to data access restrictions and/or institutional policies. Researchers who obtain appropriate access to the source datasets may use the scripts in this repository to reproduce the analytical workflow.

Notes
This repository is intended to provide a transparent and reproducible analytical framework. File paths, variable names, and input filenames may need minor adaptation according to local data organization.
