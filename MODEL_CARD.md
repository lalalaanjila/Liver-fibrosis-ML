# Model Card

## Model name
Liver-fibrosis-ML: Machine learning models for predicting liver fibrosis risk in the UK Biobank cohort

## Model purpose
This repository supports the reproducible implementation of the machine learning workflow used in the study for non-invasive prediction of liver fibrosis risk using demographic, anthropometric, clinical, and laboratory variables derived from the UK Biobank.

The primary purpose of the models is risk prediction and pre-screening support rather than definitive diagnosis. The repository is intended to improve transparency and reproducibility of the analytical workflow, including preprocessing, feature selection, hyperparameter tuning, model fitting, and internal evaluation.

## Prediction task
Binary classification of liver fibrosis risk.

## Intended use
These models are intended for research use and methodological reproducibility. They are not intended to replace liver biopsy, transient elastography, or specialist clinical judgment. The models should be interpreted as pre-screening or risk-stratification tools within the context of the study design.

## Data source
The analytical workflow was developed using data derived from the UK Biobank.

Because the original study dataset is subject to data access restrictions and/or institutional data-use policies, the raw analytical data are not publicly redistributed in this repository. A synthetic example input file is provided only to illustrate the expected data structure and variable naming format.

## Input features
The workflow uses demographic, anthropometric, clinical, and laboratory variables. Example predictors include:

- Age
- Sex
- Educational level
- Height
- Weight
- Body mass index
- Waist circumference
- Hip circumference
- Hypertension
- Diabetes mellitus

Additional variables may be included depending on the feature-selection and model-development steps used in the study workflow.

Detailed variable mapping and definitions are provided in the Supplementary Materials:
- **Table S1**: Variable mapping and description for UK Biobank data
- **Table S2**: Description of the study variables

## Outcome
The target outcome is liver fibrosis risk as defined in the study protocol and manuscript.

## Preprocessing
The public workflow includes:
- cohort splitting into training, internal validation, and internal testing cohorts;
- missing-value handling after cohort splitting;
- export of processed datasets and missing-data summaries.

Preprocessing is implemented in:
- `01_data_preprocessing_imputation_split.R`

## Feature selection
Two feature-selection approaches are used:
- **Boruta-based feature selection**
- **LASSO-based feature selection**

The overlap between Boruta-selected and LASSO-selected features is visualized as part of the workflow.

Feature-selection scripts:
- `02_feature_selection_boruta.R`
- `02_feature_selection_lasso_and_intersection.py`

## Candidate models
The workflow evaluates ten machine learning models:

- Multilayer perceptron (MLP)
- Logistic regression (LR)
- XGBoost
- Bagging
- Random forest (RF)
- Naive Bayes (NB)
- LightGBM
- Support vector machine (SVM)
- k-nearest neighbors (KNN)
- Decision tree (DT)

## Model architecture / implementation
The models are implemented through an R-based machine learning workflow centered on `tidymodels` and related packages, with an auxiliary Python script for LASSO-based feature selection and feature-overlap visualization.

Supporting R packages include, but are not limited to:
- `tidymodels`
- `workflowsets`
- `baguette`
- `discrim`
- `naivebayes`
- `kknn`
- `ranger`
- `bonsai`
- `Boruta`
- `missForest`
- `caret`
- `dplyr`
- `readr`
- `ggplot2`
- `purrr`
- `tibble`

The Python-based feature-selection step uses packages documented separately in the repository version files.

## Hyperparameter tuning
Hyperparameter tuning is conducted on the training cohort only. Depending on the model, either random search or grid search is used.

Detailed tuning strategies are documented in:
- **Table S7**: Hyperparameter optimisation strategies for each machine learning model

Final selected hyperparameters are documented in:
- **Table S6**: Final hyperparameters adopted in the ten machine learning models

## Final selected hyperparameters
The final selected hyperparameters for each model are provided in the manuscript Supplementary Materials (Table S6), including model-specific settings such as:
- hidden units, penalty, and epochs for MLP;
- penalty and mixture for LR;
- tree number, depth, learning rate, loss reduction, and sample size for XGBoost and LightGBM;
- mtry, trees, and min_n for RF;
- cost and rbf_sigma for SVM;
- neighbors, weighting function, and distance power for KNN;
- cost-complexity and min_n for DT.

## Random seeds and reproducibility
Random seeds are explicitly fixed within the public scripts for major analytical steps, including cohort splitting, feature selection, hyperparameter tuning, and final model fitting where applicable.

The scripts should be executed sequentially to preserve the intended analytical logic and maximize reproducibility.

Recommended execution order:
1. `01_data_preprocessing_imputation_split.R`
2. `02_feature_selection_boruta.R`
3. `02_feature_selection_lasso_and_intersection.py`
4. `03_model_tuning_tidymodels.R`
5. `04_model_training_and_testing.R`

## Evaluation
The final workflow includes:
- cross-validation-based evaluation in the training cohort;
- model training on the processed training cohort;
- prediction on the held-out internal testing cohort;
- export of ROC-related outputs and summary performance metrics.

Model training and evaluation are implemented in:
- `03_model_tuning_tidymodels.R`
- `04_model_training_and_testing.R`

## Software environment
The workflow uses both R and Python.

Exact software and package-version information are documented in the repository files, including:
- `R_package_versions.txt`
- `sessionInfo_R.txt`
- `python_package_versions.txt` (if applicable)

These files are provided to facilitate transparent reporting of the computational environment.

## Data availability
The original analytical dataset is not publicly redistributed in this repository because it is subject to data access restrictions and/or institutional data-use policies. Researchers with appropriate access to the source dataset may use the public scripts and version files to reproduce the analytical workflow.

## Limitations
Several limitations should be considered:

1. The repository does not contain the original restricted-access analytical dataset.
2. The example input file is synthetic and intended only to illustrate data structure and coding format.
3. Minor adaptation of file paths or local computing settings may be necessary in different environments.
4. Exact numerical reproducibility may still be affected by platform-specific differences, backend implementations, and software-version differences if the documented environment is not preserved.
5. The models are intended for research reproducibility and pre-screening/risk-stratification purposes, not direct clinical deployment without further validation.

## Contact / repository note
This Model Card is intended to accompany the public code repository and Supplementary Materials to improve methodological transparency and reproducibility.
