# 0. Load required packages
required_packages <- c("missForest", "doParallel", "caret")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

library(missForest)
library(doParallel)
library(caret)


# 1. User-defined settings
# Input raw data file (with missing values)
input_file <- "raw_dataset.csv"

# Output directory
output_dir <- "processed_data"

# Outcome variable used for stratified splitting
outcome_var <- "outcome"

# Specify categorical variables (example names; replace as needed)
categorical_vars <- c("Gender", "Educational", "Occupation",
                      "Hypertension", "DM", "HBV", "HCV", "outcome")

# Random seed for reproducibility
random_seed <- 1234

# Data split ratios
train_ratio <- 0.70
valid_ratio <- 0.20
test_ratio  <- 0.10

# missForest parameters
mf_maxiter <- 10
mf_ntree   <- 500


# 2. Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


# 3. Read raw data
data_raw <- read.csv(input_file, stringsAsFactors = FALSE)

cat("Raw data dimensions:", dim(data_raw), "\n")
cat("Column names:\n")
print(colnames(data_raw))


# 4. Basic validation checks
if (!(outcome_var %in% colnames(data_raw))) {
  stop(paste("Outcome variable not found in dataset:", outcome_var))
}

missing_categorical <- setdiff(categorical_vars, colnames(data_raw))
if (length(missing_categorical) > 0) {
  warning("The following categorical variables were not found and will be ignored: ",
          paste(missing_categorical, collapse = ", "))
  categorical_vars <- intersect(categorical_vars, colnames(data_raw))
}


# 5. Convert categorical variables to factors
if (length(categorical_vars) > 0) {
  data_raw[categorical_vars] <- lapply(data_raw[categorical_vars], as.factor)
}


# 6. Summarize missingness
missing_summary <- data.frame(
  Variable = colnames(data_raw),
  Missing_Count = colSums(is.na(data_raw)),
  Missing_Rate_Percent = round(colMeans(is.na(data_raw)) * 100, 2)
)

missing_summary <- missing_summary[order(-missing_summary$Missing_Rate_Percent), ]

cat("\nMissing data summary:\n")
print(missing_summary)

write.csv(
  missing_summary,
  file = file.path(output_dir, "missing_data_summary.csv"),
  row.names = FALSE
)


# 7. Split dataset first
#    training : validation : testing = 7 : 2 : 1
set.seed(random_seed)

# Step 1: split training (70%) vs remaining (30%)
train_index <- createDataPartition(
  y = data_raw[[outcome_var]],
  p = train_ratio,
  list = FALSE
)

train_data <- data_raw[train_index, ]
temp_data  <- data_raw[-train_index, ]

# Step 2: split remaining into validation and testing
# remaining is 30%; validation should be 20% overall => 20/30 = 2/3
set.seed(random_seed)
valid_index <- createDataPartition(
  y = temp_data[[outcome_var]],
  p = valid_ratio / (valid_ratio + test_ratio),
  list = FALSE
)

valid_data <- temp_data[valid_index, ]
test_data  <- temp_data[-valid_index, ]

cat("\nDataset split summary:\n")
cat("Training cohort dimensions:          ", dim(train_data), "\n")
cat("Internal validation cohort dimensions:", dim(valid_data), "\n")
cat("Internal testing cohort dimensions:   ", dim(test_data), "\n")


# 8. Helper function for missForest imputation
run_missforest <- function(df, maxiter = 10, ntree = 500, seed = 1234) {
  set.seed(seed)
  
  # Determine number of cores
  available_cores <- parallel::detectCores()
  cores_to_use <- max(1, min(available_cores - 1, ncol(df)))
  
  cl <- makeCluster(cores_to_use)
  registerDoParallel(cl)
  
  on.exit({
    try(stopCluster(cl), silent = TRUE)
  }, add = TRUE)
  
  # mtry setting
  mtry_value <- max(1, floor(sqrt(ncol(df)) / 2))
  
  imputed_result <- missForest(
    xmis = df,
    maxiter = maxiter,
    ntree = ntree,
    mtry = mtry_value,
    verbose = TRUE,
    parallelize = "forests"
  )
  
  return(imputed_result)
}


# 9. Impute after splitting
cat("\nRunning missForest imputation on training cohort...\n")
train_imp <- run_missforest(
  df = train_data,
  maxiter = mf_maxiter,
  ntree = mf_ntree,
  seed = random_seed
)

cat("\nRunning missForest imputation on internal validation cohort...\n")
valid_imp <- run_missforest(
  df = valid_data,
  maxiter = mf_maxiter,
  ntree = mf_ntree,
  seed = random_seed
)

cat("\nRunning missForest imputation on internal testing cohort...\n")
test_imp <- run_missforest(
  df = test_data,
  maxiter = mf_maxiter,
  ntree = mf_ntree,
  seed = random_seed
)

train_imputed <- train_imp$ximp
valid_imputed <- valid_imp$ximp
test_imputed  <- test_imp$ximp


# 10. Save imputation error estimates
imputation_error_summary <- data.frame(
  Cohort = c("Training", "Internal_Validation", "Internal_Testing"),
  OOB_Error = c(
    paste(train_imp$OOBerror, collapse = "; "),
    paste(valid_imp$OOBerror, collapse = "; "),
    paste(test_imp$OOBerror, collapse = "; ")
  )
)

cat("\nImputation OOB error summary:\n")
print(imputation_error_summary)

write.csv(
  imputation_error_summary,
  file = file.path(output_dir, "imputation_oob_error_summary.csv"),
  row.names = FALSE
)

# 11. Save processed datasets
write.csv(
  train_imputed,
  file = file.path(output_dir, "train_imputed.csv"),
  row.names = FALSE
)

write.csv(
  valid_imputed,
  file = file.path(output_dir, "internal_validation_imputed.csv"),
  row.names = FALSE
)

write.csv(
  test_imputed,
  file = file.path(output_dir, "internal_testing_imputed.csv"),
  row.names = FALSE
)

# 12. Save raw split datasets (optional)
write.csv(
  train_data,
  file = file.path(output_dir, "train_raw_split.csv"),
  row.names = FALSE
)

write.csv(
  valid_data,
  file = file.path(output_dir, "internal_validation_raw_split.csv"),
  row.names = FALSE
)

write.csv(
  test_data,
  file = file.path(output_dir, "internal_testing_raw_split.csv"),
  row.names = FALSE
)