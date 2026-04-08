# 0. Load required packages
required_packages <- c("Boruta", "readr", "doParallel")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

library(Boruta)
library(readr)
library(doParallel)


# 1. User-defined settings
# Input file: imputed training cohort generated from preprocessing script
input_file <- "processed_data/train_imputed.csv"

# Output directory
output_dir <- "feature_selection_results"

# Outcome variable
outcome_var <- "outcome"

# Categorical variables used in this study
categorical_vars <- c(
  "Gender",
  "Educational",
  "Occupation",
  "Hypertension",
  "DM",
  "HBV",
  "HCV",
  "outcome"
)

# Random seed
random_seed <- 1234

# Boruta parameters
boruta_ntree <- 1000
boruta_pvalue <- 0.005
boruta_doTrace <- 0


# 2. Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# 3. Read data
data <- read_csv(input_file, show_col_types = FALSE)
data <- as.data.frame(data)

cat("Input data dimensions:", dim(data), "\n")
cat("Column names:\n")
print(colnames(data))

# 4. Basic validation checks
if (!(outcome_var %in% colnames(data))) {
  stop(paste("Outcome variable not found in dataset:", outcome_var))
}

missing_categorical <- setdiff(categorical_vars, colnames(data))
if (length(missing_categorical) > 0) {
  warning("The following categorical variables were not found and will be ignored: ",
          paste(missing_categorical, collapse = ", "))
  categorical_vars <- intersect(categorical_vars, colnames(data))
}


# 5. Convert categorical variables to factors
if (length(categorical_vars) > 0) {
  data[categorical_vars] <- lapply(data[categorical_vars], as.factor)
}

# Ensure outcome variable is factor
data[[outcome_var]] <- as.factor(data[[outcome_var]])

str(data)


# 6. Register parallel backend
available_cores <- parallel::detectCores()
cores_to_use <- max(1, available_cores - 1)

cl <- makeCluster(cores_to_use)
registerDoParallel(cl)

on.exit({
  try(stopCluster(cl), silent = TRUE)
}, add = TRUE)

# 7. Run Boruta on training cohort only
set.seed(random_seed)

boruta_formula <- as.formula(paste(outcome_var, "~ ."))

boruta_obj <- Boruta(
  formula = boruta_formula,
  data = data,
  doTrace = boruta_doTrace,
  ntree = boruta_ntree,
  pValue = boruta_pvalue
)

# 8. Print results
cat("\nBoruta result (raw):\n")
print(boruta_obj)

cat("\nBoruta result after TentativeRoughFix:\n")
boruta_fixed <- TentativeRoughFix(boruta_obj)
print(boruta_fixed)

# 9. Variable importance statistics
boruta_stats <- attStats(boruta_obj)

cat("\nBoruta importance statistics:\n")
print(boruta_stats)

write.csv(
  boruta_stats,
  file = file.path(output_dir, "boruta_importance_stats.csv"),
  row.names = TRUE
)

# 10. Save confirmed feature names
confirmed_features <- rownames(boruta_stats)[boruta_stats$decision == "Confirmed"]

write.csv(
  data.frame(Confirmed_Features = confirmed_features),
  file = file.path(output_dir, "boruta_confirmed_features.csv"),
  row.names = FALSE
)

cat("\nConfirmed features:\n")
print(confirmed_features)

# 11. Plot Boruta results
png(
  filename = file.path(output_dir, "boruta_importance_plot.png"),
  width = 1800,
  height = 1200,
  res = 200
)

opar <- par(no.readonly = TRUE)
par(mar = c(10, 4, 3, 1))

plot(
  boruta_obj,
  las = 2,
  xlab = "",
  ylab = "Importance (Z-score)",
  main = "Boruta Feature Importance"
)

legend(
  "topright",
  legend = c("Shadow", "Rejected", "Tentative", "Confirmed"),
  fill = c("blue", "red", "yellow", "green"),
  bty = "n"
)

par(opar)
dev.off()

# 12. Save formula and session info
writeLines(
  paste("Boruta formula:", deparse(boruta_formula)),
  con = file.path(output_dir, "boruta_formula.txt")
)

writeLines(
  capture.output(sessionInfo()),
  con = file.path(output_dir, "sessionInfo_boruta.txt")
)

cat("\nBoruta feature selection completed successfully.\n")
cat("Results saved to:", output_dir, "\n")