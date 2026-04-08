# 0. Load required packages
required_packages <- c(
  "tidymodels",
  "workflowsets",
  "baguette",
  "discrim",
  "naivebayes",
  "kknn",
  "ranger",
  "bonsai",
  "readr",
  "dplyr",
  "ggplot2",
  "purrr",
  "tibble"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

library(tidymodels)
library(workflowsets)
library(baguette)
library(discrim)
library(naivebayes)
library(kknn)
library(ranger)
library(bonsai)
library(readr)
library(dplyr)
library(ggplot2)
library(purrr)
library(tibble)

tidymodels_prefer()


# 1. User-defined settings
train_file <- "processed_data/train_imputed.csv"
test_file  <- "processed_data/internal_testing_imputed.csv"
output_dir <- "final_model_results"
outcome_var <- "group"
random_seed <- 1234

final_features <- c(
  "Age",
  "Gender",
  "Educational",
  "Height",
  "Weight",
  "Waist_circumference",
  "Hip_circumference",
  "Hypertension",
  "DM"
)

categorical_vars <- c(
  "Gender",
  "Educational",
  "Hypertension",
  "DM",
  "group"
)


# 2. Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


# 3. Read data
train_data <- read_csv(train_file, show_col_types = FALSE) %>% as.data.frame()
test_data  <- read_csv(test_file,  show_col_types = FALSE) %>% as.data.frame()

cat("Training data dimensions:", dim(train_data), "\n")
cat("Testing data dimensions:", dim(test_data), "\n")


# 4. Basic checks
required_cols <- c(outcome_var, final_features)

missing_train <- setdiff(required_cols, colnames(train_data))
missing_test  <- setdiff(required_cols, colnames(test_data))

if (length(missing_train) > 0) {
  stop(paste("Missing columns in training data:",
             paste(missing_train, collapse = ", ")))
}

if (length(missing_test) > 0) {
  stop(paste("Missing columns in testing data:",
             paste(missing_test, collapse = ", ")))
}

train_data <- train_data[, required_cols]
test_data  <- test_data[, required_cols]


# 5. Convert categorical variables to factors
available_cat_train <- intersect(categorical_vars, colnames(train_data))
available_cat_test  <- intersect(categorical_vars, colnames(test_data))

train_data[available_cat_train] <- lapply(train_data[available_cat_train], as.factor)
test_data[available_cat_test]   <- lapply(test_data[available_cat_test], as.factor)

# Ensure the outcome variable has consistent factor levels
train_data[[outcome_var]] <- as.factor(train_data[[outcome_var]])
test_data[[outcome_var]]  <- factor(test_data[[outcome_var]],
                                    levels = levels(train_data[[outcome_var]]))

# Relevel outcome so that the second level is treated as the event
# Adjust if your positive class is different
if (nlevels(train_data[[outcome_var]]) != 2) {
  stop("This template currently assumes a binary outcome.")
}

event_level <- levels(train_data[[outcome_var]])[2]
cat("Event level used for ROC/AUC:", event_level, "\n")


# 6. Recipe
rec <- recipe(
  as.formula(paste(outcome_var, "~", paste(final_features, collapse = " + "))),
  data = train_data
) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())


# 7. Final model specifications
# MLP
mlp_mod <- mlp(
  hidden_units = 10,
  penalty = 1,
  epochs = 50
) %>%
  set_engine("nnet") %>%
  set_mode("classification")

# Logistic Regression
lr_mod <- logistic_reg(
  penalty = 0.001,
  mixture = 1
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# XGBoost
xgb_mod <- boost_tree(
  trees = 280,
  tree_depth = 3,
  learn_rate = 0.1,
  loss_reduction = 1,
  sample_size = 0.7
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Bagging
bag_mod <- bag_tree(
  cost_complexity = 0.001,
  min_n = 10
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# Random Forest
rf_mod <- rand_forest(
  mtry = 2,
  trees = 1366,
  min_n = 10
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Naive Bayes
nb_mod <- naive_Bayes(
  Laplace = 0
) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

# LightGBM
lgb_mod <- boost_tree(
  trees = 100,
  tree_depth = 6,
  learn_rate = 0.1,
  loss_reduction = 0,
  min_n = 5,
  sample_size = 1
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

# SVM
svm_mod <- svm_rbf(
  cost = 1.000693,
  rbf_sigma = 1.002305
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# KNN
knn_mod <- nearest_neighbor(
  neighbors = 15,
  weight_func = "rectangular",
  dist_power = 1.25
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Decision Tree
dt_mod <- decision_tree(
  cost_complexity = 0.01,
  min_n = 2
) %>%
  set_engine("rpart") %>%
  set_mode("classification")


# 8. Workflow set
wf_set <- workflow_set(
  preproc = list(main_recipe = rec),
  models = list(
    MLP = mlp_mod,
    LR = lr_mod,
    XGBoost = xgb_mod,
    Bagging = bag_mod,
    RF = rf_mod,
    NB = nb_mod,
    LightGBM = lgb_mod,
    SVM = svm_mod,
    KNN = knn_mod,
    DT = dt_mod
  )
)

print(wf_set)


# 9. Training resampling
metrics_used <- metric_set(
  roc_auc, accuracy, sens, spec, precision, f_meas, brier_class
)

control_obj <- control_resamples(save_pred = TRUE)

set.seed(random_seed)
folds <- vfold_cv(train_data, v = 10, strata = !!sym(outcome_var))


# 10. Fit resamples on training cohort
set.seed(random_seed)
wf_res <- wf_set %>%
  workflow_map(
    "fit_resamples",
    resamples = folds,
    metrics = metrics_used,
    control = control_obj
  )


# 11. Training metrics summary
train_metrics <- collect_metrics(wf_res)

write.csv(
  train_metrics,
  file = file.path(output_dir, "training_cv_metrics.csv"),
  row.names = FALSE
)


# 12. Training ROC data
train_preds <- collect_predictions(wf_res)

train_roc_df <- train_preds %>%
  group_by(wflow_id) %>%
  roc_curve(
    truth = !!sym(outcome_var),
    .pred_{{event_level}}
  ) %>%
  ungroup()

# tidyeval workaround for dynamic prob column
prob_col_name <- paste0(".pred_", event_level)

train_roc_df <- train_preds %>%
  group_by(wflow_id) %>%
  roc_curve(
    truth = !!sym(outcome_var),
    !!sym(prob_col_name)
  ) %>%
  ungroup()


# 13. Fit final models and predict testing cohort
test_auc_results <- list()
test_predictions_all <- list()
test_predictions_wide <- list()

for (model_name in wf_set$wflow_id) {
  message("Fitting and predicting: ", model_name)
  
  current_workflow <- wf_set %>% extract_workflow(id = model_name)
  
  set.seed(random_seed)
  current_fit <- current_workflow %>% fit(data = train_data)
  
  test_pred <- current_fit %>%
    predict(new_data = test_data, type = "prob") %>%
    bind_cols(test_data %>% dplyr::select(all_of(outcome_var)))
  
  test_predictions_all[[model_name]] <- test_pred %>%
    mutate(wflow_id = model_name)
  
  test_predictions_wide[[model_name]] <- test_pred %>%
    dplyr::select(all_of(prob_col_name)) %>%
    rename(!!paste0(model_name, "_pred") := all_of(prob_col_name))
  
  test_auc <- roc_auc(
    data = test_pred,
    truth = !!sym(outcome_var),
    !!sym(prob_col_name)
  )
  
  test_auc_results[[model_name]] <- test_auc$.estimate
}


# 14. Testing metrics summary
test_auc_df <- tibble(
  Model = names(test_auc_results),
  Test_AUC = unlist(test_auc_results)
) %>%
  arrange(desc(Test_AUC))

print(test_auc_df)

write.csv(
  test_auc_df,
  file = file.path(output_dir, "testing_auc_summary.csv"),
  row.names = FALSE
)


# 15. Save test predictions
test_predictions_df <- bind_rows(test_predictions_all)
test_predictions_wide_df <- bind_cols(test_data, bind_cols(test_predictions_wide))

write.csv(
  test_predictions_df,
  file = file.path(output_dir, "testing_predictions_long_format.csv"),
  row.names = FALSE
)

write.csv(
  test_predictions_wide_df,
  file = file.path(output_dir, "testing_predictions_wide_format.csv"),
  row.names = FALSE
)


# 16. Testing ROC data
test_roc_df <- test_predictions_df %>%
  group_by(wflow_id) %>%
  roc_curve(
    truth = !!sym(outcome_var),
    !!sym(prob_col_name)
  ) %>%
  ungroup()


# 17. Custom colors
custom_colors <- c(
  "#E41A1C",
  "#377EB8",
  "#4DAF4A",
  "#984EA3",
  "#FF7F00",
  "#FFFF33",
  "#A65628",
  "#F781BF",
  "#999999",
  "#00CED1"
)


# 18. Plot training ROC
p_train <- ggplot(train_roc_df, aes(x = 1 - specificity, y = sensitivity, color = wflow_id)) +
  geom_line(linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#888888", linewidth = 0.5) +
  scale_color_manual(values = custom_colors) +
  theme_bw() +
  theme(
    legend.position = c(.98, .57),
    legend.justification = c("right", "top"),
    legend.box.background = element_rect(color = "#CCC", fill = NA, linewidth = 0.5),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  labs(
    title = "Training Cohort ROC",
    x = "1 - Specificity",
    y = "Sensitivity",
    color = "Model"
  )

ggsave(
  filename = file.path(output_dir, "training_roc_curve.pdf"),
  plot = p_train,
  width = 8,
  height = 6
)


# 19. Plot testing ROC
p_test <- ggplot(test_roc_df, aes(x = 1 - specificity, y = sensitivity, color = wflow_id)) +
  geom_line(linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#888888", linewidth = 0.5) +
  scale_color_manual(values = custom_colors) +
  theme_bw() +
  theme(
    legend.position = c(.98, .57),
    legend.justification = c("right", "top"),
    legend.box.background = element_rect(color = "#CCC", fill = NA, linewidth = 0.5),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  labs(
    title = "Internal Testing Cohort ROC",
    x = "1 - Specificity",
    y = "Sensitivity",
    color = "Model"
  )

ggsave(
  filename = file.path(output_dir, "testing_roc_curve.pdf"),
  plot = p_test,
  width = 8,
  height = 6
)


# 20. Save final hyperparameters table
final_hyperparameters <- tibble(
  Model = c("MLP", "LR", "XGBoost", "Bagging", "RF", "NB", "LightGBM", "SVM", "KNN", "DT"),
  Hyperparameters = c(
    "hidden_units = 10; penalty = 1; epochs = 50",
    "penalty = 0.001; mixture = 1",
    "trees = 280; tree_depth = 3; learn_rate = 0.1; loss_reduction = 1; sample_size = 0.7",
    "cost_complexity = 0.001; min_n = 10",
    "mtry = 2; trees = 1366; min_n = 10",
    "Laplace = 0",
    "trees = 100; tree_depth = 6; learn_rate = 0.1; loss_reduction = 0; min_n = 5; sample_size = 1",
    "cost = 1.000693; rbf_sigma = 1.002305",
    "neighbors = 15; weight_func = rectangular; dist_power = 1.25",
    "cost_complexity = 0.01; min_n = 2"
  )
)

write.csv(
  final_hyperparameters,
  file = file.path(output_dir, "final_hyperparameters.csv"),
  row.names = FALSE
)


# 21. Save session info
writeLines(
  capture.output(sessionInfo()),
  con = file.path(output_dir, "sessionInfo_final_modeling.txt")
)

cat("\nFinal model training and testing completed successfully.\n")
cat("Results saved to:", output_dir, "\n")