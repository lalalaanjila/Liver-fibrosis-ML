# 0. Load required packages
required_packages <- c(
  "tidymodels",
  "baguette",
  "discrim",
  "naivebayes",
  "kknn",
  "ranger",
  "bonsai",
  "doParallel",
  "readr",
  "dplyr"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

library(tidymodels)
library(baguette)
library(discrim)
library(naivebayes)
library(kknn)
library(ranger)
library(bonsai)
library(doParallel)
library(readr)
library(dplyr)

tidymodels_prefer()


# 1. User-defined settings
input_file <- "processed_data/train_imputed.csv"
output_dir <- "model_tuning_results"
outcome_var <- "outcome"
random_seed <- 1234

# Final retained variables used in the manuscript
final_features <- c(
  "Age",
  "Gender",
  "Educational",
  "Height",
  "Weight",
  "BMI",
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
  "outcome"
)


# 2. Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


# 3. Read data
train_data <- read_csv(input_file, show_col_types = FALSE) %>%
  as.data.frame()

cat("Input data dimensions:", dim(train_data), "\n")
cat("Column names:\n")
print(colnames(train_data))


# 4. Basic validation checks
if (!(outcome_var %in% colnames(train_data))) {
  stop(paste("Outcome variable not found in dataset:", outcome_var))
}

missing_features <- setdiff(final_features, colnames(train_data))
if (length(missing_features) > 0) {
  stop(paste("The following required features are missing:",
             paste(missing_features, collapse = ", ")))
}

# Keep only outcome + final features
train_data <- train_data[, c(outcome_var, final_features)]


# 5. Convert categorical variables to factors
available_categorical <- intersect(categorical_vars, colnames(train_data))
train_data[available_categorical] <- lapply(train_data[available_categorical], as.factor)
train_data[[outcome_var]] <- as.factor(train_data[[outcome_var]])

str(train_data)


# 6. Unified recipe
rec <- recipe(
  as.formula(
    paste(outcome_var, "~", paste(final_features, collapse = " + "))
  ),
  data = train_data
) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())


# 7. Cross-validation folds
set.seed(random_seed)
cv_folds <- vfold_cv(train_data, v = 5, strata = !!sym(outcome_var))

metrics_used <- metric_set(roc_auc, accuracy, precision, recall, f_meas)


# 8. Parallel backend
n_cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(n_cores)
registerDoParallel(cl)

on.exit({
  try(stopCluster(cl), silent = TRUE)
}, add = TRUE)


# 9. Helper function for tuning
run_tuning <- function(workflow_obj, resamples_obj, grid_obj, metrics_obj, model_name) {
  cat("\n============================\n")
  cat("Tuning model:", model_name, "\n")
  cat("============================\n")
  
  set.seed(random_seed)
  tune_res <- tune_grid(
    workflow_obj,
    resamples = resamples_obj,
    grid = grid_obj,
    metrics = metrics_obj,
    control = control_grid(save_pred = TRUE, verbose = TRUE)
  )
  
  best_params <- select_best(tune_res, metric = "roc_auc")
  metrics_table <- collect_metrics(tune_res)
  
  write.csv(
    metrics_table,
    file = file.path(output_dir, paste0(model_name, "_tuning_metrics.csv")),
    row.names = FALSE
  )
  
  write.csv(
    best_params,
    file = file.path(output_dir, paste0(model_name, "_best_params.csv")),
    row.names = FALSE
  )
  
  return(list(
    tune_res = tune_res,
    best_params = best_params,
    metrics_table = metrics_table
  ))
}


# 10. Random Forest
rf_mod_tune <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

rf_grid <- grid_regular(
  mtry(range = c(2L, 6L)),
  trees(range = c(200L, 1500L)),
  min_n(range = c(2L, 10L)),
  levels = 4
)

rf_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_mod_tune)

rf_results <- run_tuning(
  workflow_obj = rf_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = rf_grid,
  metrics_obj = metrics_used,
  model_name = "random_forest"
)


# 11. Decision Tree
dt_mod_tune <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

dt_grid <- grid_regular(
  cost_complexity(range = c(-5, -1)),
  tree_depth(range = c(2L, 10L)),
  min_n(range = c(2L, 20L)),
  levels = 4
)

dt_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(dt_mod_tune)

dt_results <- run_tuning(
  workflow_obj = dt_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = dt_grid,
  metrics_obj = metrics_used,
  model_name = "decision_tree"
)


# 12. XGBoost
xgb_mod_tune <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  min_n = tune(),
  mtry = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_grid <- grid_random(
  trees(range = c(100L, 800L)),
  tree_depth(range = c(2L, 8L)),
  learn_rate(range = c(-3, -1), trans = scales::log10_trans()),
  loss_reduction(),
  sample_size = sample_prop(),
  min_n(range = c(2L, 20L)),
  finalize(mtry(), train_data[, final_features]),
  size = 20
)

xgb_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_mod_tune)

xgb_results <- run_tuning(
  workflow_obj = xgb_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = xgb_grid,
  metrics_obj = metrics_used,
  model_name = "xgboost"
)


# 13. Logistic Regression (glmnet)
log_mod_tune <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

log_grid <- grid_random(
  penalty(range = c(-5, 0), trans = scales::log10_trans()),
  mixture(),
  size = 20
)

log_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(log_mod_tune)

log_results <- run_tuning(
  workflow_obj = log_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = log_grid,
  metrics_obj = metrics_used,
  model_name = "logistic_regression"
)


# 14. MLP
mlp_mod_tune <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) %>%
  set_engine("nnet") %>%
  set_mode("classification")

mlp_grid <- grid_random(
  hidden_units(range = c(1L, 10L)),
  penalty(range = c(-5, -1), trans = scales::log10_trans()),
  epochs(range = c(50L, 300L)),
  size = 20
)

mlp_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(mlp_mod_tune)

mlp_results <- run_tuning(
  workflow_obj = mlp_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = mlp_grid,
  metrics_obj = metrics_used,
  model_name = "mlp"
)


# 15. Naive Bayes
nb_mod_tune <- naive_Bayes(
  Laplace = tune()
) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

nb_grid <- grid_regular(
  Laplace(range = c(0, 3)),
  levels = 4
)

nb_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(nb_mod_tune)

nb_results <- run_tuning(
  workflow_obj = nb_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = nb_grid,
  metrics_obj = metrics_used,
  model_name = "naive_bayes"
)


# 16. KNN
knn_mod_tune <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune(),
  dist_power = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

knn_grid <- grid_regular(
  neighbors(range = c(3L, 20L)),
  weight_func(values = c("rectangular", "triangular", "epanechnikov")),
  dist_power(range = c(1, 2)),
  levels = 4
)

knn_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(knn_mod_tune)

knn_results <- run_tuning(
  workflow_obj = knn_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = knn_grid,
  metrics_obj = metrics_used,
  model_name = "knn"
)


# 17. SVM-RBF
svm_mod_tune <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_grid <- grid_random(
  cost(),
  rbf_sigma(),
  size = 20
)

svm_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(svm_mod_tune)

svm_results <- run_tuning(
  workflow_obj = svm_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = svm_grid,
  metrics_obj = metrics_used,
  model_name = "svm_rbf"
)


# 18. Bagging
bag_mod_tune <- bag_tree(
  cost_complexity = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

bag_grid <- grid_regular(
  cost_complexity(range = c(-5, -1)),
  min_n(range = c(2L, 20L)),
  levels = 4
)

bag_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(bag_mod_tune)

bag_results <- run_tuning(
  workflow_obj = bag_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = bag_grid,
  metrics_obj = metrics_used,
  model_name = "bagging"
)


# 19. LightGBM
lgb_mod_tune <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  min_n = tune(),
  sample_size = tune()
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

lgb_grid <- grid_random(
  trees(range = c(100L, 500L)),
  tree_depth(range = c(2L, 8L)),
  learn_rate(range = c(-3, -1), trans = scales::log10_trans()),
  loss_reduction(),
  min_n(range = c(2L, 20L)),
  sample_size = sample_prop(),
  size = 20
)

lgb_wf_tune <- workflow() %>%
  add_recipe(rec) %>%
  add_model(lgb_mod_tune)

lgb_results <- run_tuning(
  workflow_obj = lgb_wf_tune,
  resamples_obj = cv_folds,
  grid_obj = lgb_grid,
  metrics_obj = metrics_used,
  model_name = "lightgbm"
)


# 20. Summary table of best parameters
best_params_summary <- bind_rows(
  mutate(rf_results$best_params, model = "random_forest"),
  mutate(dt_results$best_params, model = "decision_tree"),
  mutate(xgb_results$best_params, model = "xgboost"),
  mutate(log_results$best_params, model = "logistic_regression"),
  mutate(mlp_results$best_params, model = "mlp"),
  mutate(nb_results$best_params, model = "naive_bayes"),
  mutate(knn_results$best_params, model = "knn"),
  mutate(svm_results$best_params, model = "svm_rbf"),
  mutate(bag_results$best_params, model = "bagging"),
  mutate(lgb_results$best_params, model = "lightgbm")
)

write.csv(
  best_params_summary,
  file = file.path(output_dir, "best_params_summary.csv"),
  row.names = FALSE
)


# 21. Save session info
writeLines(
  capture.output(sessionInfo()),
  con = file.path(output_dir, "sessionInfo_model_tuning.txt")
)

cat("\nModel tuning completed successfully.\n")
cat("Results saved to:", output_dir, "\n")