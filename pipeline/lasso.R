# lasso_parallel.R - Lasso Path script that automatically selects the family
# Auto-install and load required packages
if (!require("glmnet", quietly = TRUE)) {
  install.packages("glmnet", repos = "https://cloud.r-project.org")
  library(glmnet)
} else {
  library(glmnet)
}

if (!require("reticulate", quietly = TRUE)) {
  install.packages("reticulate", repos = "https://cloud.r-project.org")
  library(reticulate)
} else {
  library(reticulate)
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Two arguments required: 1. input file path 2. output folder path [3. data type: binary/continuous (optional)]")
}
input_file <- args[1]
output_folder <- args[2]
# Third parameter: binary / continuous / multinomial

data_type <- ifelse(length(args) >= 3, tolower(args[3]), "continuous")

dfmax_val <- ifelse(length(args) >= 4, as.integer(args[4]), 1300)

if (is.na(dfmax_val)) {

  dfmax_val <- 1300

}

# cat("Using dfmax:", dfmax_val, "\n")
cat("Using dfmax:", dfmax_val, "\n")
# ✅ 第三参数：确定是 binary 还是 continuous
if (!(data_type %in% c("binary", "continuous", "multinomial"))) {
  stop("Data type must be 'binary', 'continuous', or 'multinomial'")
}

family_type <- ifelse(
  data_type == "binary",
  "binomial",
  ifelse(data_type == "multinomial", "multinomial", "gaussian")
)

# cat("Using family type:", family_type, "\n")
cat("Using family type:", family_type, "\n")
# 添加错误处理
tryCatch({

  # 加载数据
  cat("加载数据:", input_file, "\n")
  flush.console()

  np <- import("numpy")
  data <- np$load(input_file, allow_pickle = TRUE)
  cat("加载的文件包含以下键:\n")
#   cat("Loaded file contains the following keys:\n")
   cat("Loaded file contains the following keys:\n")
  print(data$files)
  flush.console()

  # 提取数据
  X <- data['X']
  y <- data['Y']

  cat("X shape:", dim(X)[1], "x", dim(X)[2], "\n")
  cat("Y shape:", dim(y)[1], "x", ifelse(length(dim(y)) > 1, dim(y)[2], 1), "\n")
  flush.console()

  # 检查并清洗数据
  if (is.null(X) || length(X) == 0) stop("X 数据为空或无效")
  if (is.null(y) || length(y) == 0) stop("Y 数据为空或无效")
  X <- as.matrix(X)

  if (length(dim(y)) > 1 && dim(y)[2] > 1) {
    cat("Warning: Y has multiple columns, using the first column\n")
    y <- y[, 1]
  }
  if (is.null(dim(y))) {
    y <- matrix(y, ncol = 1)
  }
  if (any(is.na(X))) {
    cat("Warning: X contains NA, replacing with 0\n")
    X[is.na(X)] <- 0
  }
  if (any(is.na(y))) {
    cat("Warning: Y contains NA, replacing with 0\n")
    y[is.na(y)] <- 0
  }

  p <- dim(X)[2]
  cat("Total features:", p, "\n")
  flush.console()

  # ===============================
  #  Lasso Path fitting
  # ===============================
  alpha_val <- 1
  # X <- scale(X)
  y <- as.vector(y)  # glmnet requires y to be a vector
if (family_type == "binomial") {
  uniq_y <- sort(unique(y))

  if (length(uniq_y) != 2) {
    stop(paste(
      "binomial family requires exactly 2 classes, got:",
      paste(uniq_y, collapse = ",")
    ))
  }

  if (!all(uniq_y %in% c(0, 1))) {
    y <- as.integer(factor(y)) - 1
    cat("Binary labels were remapped to 0/1.\n")
  }

} else if (family_type == "multinomial") {
  uniq_y <- sort(unique(y))

  if (length(uniq_y) < 3) {
    stop(paste(
      "multinomial family requires at least 3 classes, got:",
      paste(uniq_y, collapse = ",")
    ))
  }

  y <- as.integer(factor(y)) - 1
  cat("Multiclass labels were remapped to 0,1,2,...\n")
}
  cat("Starting Lasso path fitting (family = ", family_type, ")...\n")
  flush.console()

  start_time <- Sys.time()
  # Set random seed and split train/validation
# Set random seed and split train/validation

array_task_id <- Sys.getenv("SLURM_ARRAY_TASK_ID", unset = "0")

seed_val <- as.integer(array_task_id)

if (is.na(seed_val)) {

  seed_val <- 0

}

cat("Using SLURM_ARRAY_TASK_ID as R seed:", seed_val, "\n")

set.seed(seed_val)

n_samples <- nrow(X)

train_idx <- sample(seq_len(n_samples), size = floor(0.8 * n_samples))
val_idx <- setdiff(seq_len(n_samples), train_idx)
if (length(val_idx) == 0) {
  stop("Validation set is empty. Please provide more samples or change the split ratio.")
}
X_train <- X[train_idx, , drop = FALSE]
y_train <- y[train_idx]

X_val <- X[val_idx, , drop = FALSE]
y_val <- y[val_idx]

cat("Train set shape:", dim(X_train)[1], "x", dim(X_train)[2], "\n")
cat("Validation set shape:", dim(X_val)[1], "x", dim(X_val)[2], "\n")
fit <- glmnet(
  X_train,
  y_train,
  alpha = alpha_val,
  nlambda = 2000,
  standardize = FALSE,
  intercept = FALSE,
  dfmax = dfmax_val,
  lambda.min.ratio = 1e-8,
  family = family_type
)

lambda_seq <- fit$lambda
intercepts <- fit$a0

if (family_type == "multinomial") {
  # fit$beta is a list: each class has a p x nlambda matrix
  beta_list <- fit$beta
  class_names <- names(beta_list)

  # convert to array: p x nlambda x n_classes
  coefs_array <- array(
    0,
    dim = c(nrow(beta_list[[1]]), ncol(beta_list[[1]]), length(beta_list))
  )

  for (k in seq_along(beta_list)) {
    coefs_array[, , k] <- as.matrix(beta_list[[k]])
  }

  # use the maximum absolute coefficient across classes as the feature-level path
  # coefs_matrix: p x nlambda
  coefs_matrix <- apply(abs(coefs_array), c(1, 2), max)

} else {
  coefs_matrix <- as.matrix(fit$beta)
}

cat("Actual number of lambdas:", length(lambda_seq), "\n")
cat("coefs_matrix shape:", dim(coefs_matrix)[1], "x", dim(coefs_matrix)[2], "\n")

# ===============================
#  Validation set: choose best lambda
# ===============================
cat("Start evaluating each lambda on the validation set...\n")
flush.console()

if (family_type == "gaussian") {

  pred_val <- predict(
    fit,
    newx = X_val,
    s = lambda_seq,
    type = "response"
  )

  val_metric <- colMeans((pred_val - y_val)^2)

  best_idx <- which.min(val_metric)
  best_metric <- val_metric[best_idx]
  val_metric_name <- "mse"

} else if (family_type == "binomial") {

  pred_prob <- predict(
    fit,
    newx = X_val,
    s = lambda_seq,
    type = "response"
  )

  eps <- 1e-15
  pred_prob <- pmin(pmax(pred_prob, eps), 1 - eps)

  val_metric <- -colMeans(
    y_val * log(pred_prob) + (1 - y_val) * log(1 - pred_prob)
  )

  best_idx <- which.min(val_metric)
  best_metric <- val_metric[best_idx]
  val_metric_name <- "logloss"

} else if (family_type == "multinomial") {

  # pred_prob shape usually: n_val x n_classes x n_lambda
  pred_prob <- predict(
    fit,
    newx = X_val,
    s = lambda_seq,
    type = "response"
  )
  cat("multinomial pred_prob dim:", paste(dim(pred_prob), collapse = " x "), "\n")
  eps <- 1e-15
  pred_prob <- pmin(pmax(pred_prob, eps), 1 - eps)

  n_lambda <- length(lambda_seq)
  val_metric <- numeric(n_lambda)

  # y_val is already 0,1,2,...; R arrays are 1-based, so add +1
  y_class_idx <- as.integer(y_val) + 1

for (l in seq_len(n_lambda)) {
  prob_l <- pred_prob[, , l]

  if (is.null(dim(prob_l))) {
    stop("Unexpected multinomial prediction shape.")
  }

  true_prob <- prob_l[cbind(seq_len(nrow(X_val)), y_class_idx)]
  val_metric[l] <- -mean(log(true_prob))
}

  best_idx <- which.min(val_metric)
  best_metric <- val_metric[best_idx]
  val_metric_name <- "multiclass_logloss"

} else {
  stop(paste("Unsupported family_type for validation:", family_type))
}

best_lambda <- lambda_seq[best_idx]
best_coef <- as.numeric(coefs_matrix[, best_idx])
best_nonzero_indices <- as.integer(which(best_coef != 0) - 1)

if (family_type == "multinomial") {
  best_intercept <- NA
} else {
  best_intercept <- intercepts[best_idx]
}

end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = "secs")

cat("Lasso fitting and validation completed, duration:", duration, "secs\n")
cat("Maximum nonzero features:", max(colSums(coefs_matrix != 0)), "\n")
cat("Validation metric:", val_metric_name, "\n")
cat("Validation best lambda index R 1-based:", best_idx, "\n")
cat("Validation best lambda index Python 0-based:", best_idx - 1, "\n")
cat("Validation best lambda:", best_lambda, "\n")
cat("Validation best metric:", best_metric, "\n")
cat("Best lambda nonzero feature count:", sum(best_coef != 0), "\n")

  flush.console()

  # ===============================
  # Save results
  # ===============================
  base_filename <- basename(input_file)
  output_file_npz <- file.path(output_folder, paste0("lasso_path_7_", base_filename))
  output_file_rdata <- file.path(output_folder, paste0("lasso_path_7_", tools::file_path_sans_ext(base_filename), ".RData"))
if (family_type == "multinomial") {
  intercepts_to_save <- as.matrix(intercepts)
} else {
  intercepts_to_save <- intercepts
}
cat("Saving Lasso results to:", output_file_npz, " (npz)\n")
np$savez(
  output_file_npz,
  coefs = coefs_matrix,
  lambdas = lambda_seq,
  intercept = intercepts_to_save,
  alpha = alpha_val,

  # validation: additional fields
  val_metric = val_metric,
  val_metric_name = np$array(val_metric_name),
  best_idx = as.integer(best_idx - 1),   # Python 0-based
  best_lambda = best_lambda,
  best_metric = best_metric,
  best_coef = best_coef,
  best_intercept = best_intercept,
  best_nonzero_indices = best_nonzero_indices
)
  # Save as R format
cat("Saving Lasso results to:", output_file_rdata, " (RData)\n")
save(
  coefs_matrix,
  lambda_seq,
  intercepts,
  alpha_val,
  val_metric,
  val_metric_name,
  best_idx,
  best_lambda,
  best_metric,
  best_coef,
  best_intercept,
  best_nonzero_indices,
  file = output_file_rdata
)
cat("Processing completed ✅\n")
  flush.console()

}, error = function(e) {
cat("Error during execution:", conditionMessage(e), "\n")
  quit(status = 1)
})