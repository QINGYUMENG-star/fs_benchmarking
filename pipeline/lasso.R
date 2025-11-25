# lasso.R — Lasso Path via glmnet (binary / continuous / multinomial)
# 依赖: glmnet, reticulate
# 用法:
#   Rscript lasso.R <input_npz> <output_dir> [data_type] [dfmax]
#   - input_npz:  包含 X, Y 的 .npz 文件
#   - output_dir: 输出目录（若不存在会创建）
#   - data_type:  "binary" | "continuous" | "multinomial"（默认 continuous）
#   - dfmax:      最大非零特征数（默认 3000）
#
# 输出:
#   在 output_dir 下生成: lasso_path_<dfmax>_<basename(input_npz)>
#   内容含: coefs, lambdas, intercept, alpha

# ---------------------------
# Packages
# ---------------------------
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

# ---------------------------
# Args
# ---------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("LASSO needs at least 2 parameters: 1) input npz  2) output folder  [3) data type binary/continuous/multinomial (optional)]  [4) dfmax (optional)]")
}
input_file   <- args[1]
output_folder<- args[2]
data_type    <- ifelse(length(args) >= 3, tolower(args[3]), "continuous")
dfmax        <- if (length(args) >= 4) as.integer(args[4]) else 3000
name <- if (length(args) >=5) as.character(args[5]) else "test"
if (!(data_type %in% c("binary", "continuous", "multinomial"))) {
  stop("only support data type as 'binary', 'continuous' or 'multinomial'")
}
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)
}

family_type <- if (data_type == "binary") {
  "binomial"
} else if (data_type == "continuous") {
  "gaussian"
} else if (data_type == "multinomial") {
  "multinomial"
} else {
  stop("only support data type as 'binary', 'continuous' or 'multinomial'")
}
cat("using family:", family_type, "\n")

# ---------------------------
# Main
# ---------------------------
tryCatch({

  cat("load dataset:", input_file, "\n")
  flush.console()

  np <- import("numpy")
  data <- np$load(input_file, allow_pickle = TRUE)
  cat("loaded file contains keys:\n")
  print(data$files)
  flush.console()

  # 读取
  X <- data['X']
  y <- data['Y']

  # 基础检查 & 填 NA
  if (any(is.na(X))) { cat("Warning: X contains NA, replace with 0\n"); X[is.na(X)] <- 0 }
  if (any(is.na(y))) { cat("Warning: Y contains NA, replace with 0\n"); y[is.na(y)] <- 0 }

  cat("The dimension of X:", dim(X)[1], "x", dim(X)[2], "\n")
  cat("The dimension of Y:", dim(y)[1], "x", ifelse(length(dim(y)) > 1, dim(y)[2], 1), "\n")
  flush.console()

  if (is.null(X) || length(X) == 0) stop("X is empty or invalid")
  if (is.null(y) || length(y) == 0) stop("Y is empty or invalid")

  # 矩阵与类型
  X <- as.matrix(X)
  storage.mode(X) <- "double"   # 确保 glmnet 用 double 更稳

  # 处理 y 的形状与类型
  if (data_type == "multinomial") {
    ydim <- dim(y)
    if (!is.null(ydim) && length(ydim) >= 2 && ydim[2] > 1) {
      # one-hot -> 类别索引（1..K）
      y <- max.col(y, ties.method = "first")
    }
    y <- as.factor(as.vector(y))
  } else if (data_type == "binary") {
    ydim <- dim(y)
    if (!is.null(ydim) && length(ydim) >= 2 && ydim[2] > 1) {
      cat("Warning: Y has multiple columns, using the first column (binary)\n")
      y <- y[, 1]
    }
    y <- as.vector(y)
    if (!all(unique(y) %in% c(0, 1))) {
      f <- as.factor(y)
      y <- as.integer(f) - 1
    }
  } else { # continuous
    ydim <- dim(y)
    if (!is.null(ydim) && length(ydim) >= 2 && ydim[2] > 1) {
      cat("Warning: Y has multiple columns, using the first column (continuous)\n")
      y <- y[, 1]
    }
    y <- as.vector(y)
  }

  p <- dim(X)[2]
  cat("The total number of features:", p, "\n")
  flush.console()

  # ---------------------------
  # Fit Lasso Path
  # ---------------------------
  alpha_val <- 1  # lasso
  cat("Starting Lasso path (family = ", family_type, ")...\n")
  flush.console()

  start_time <- Sys.time()

  set.seed(42)
  n_samples <- nrow(X)
  train_idx <- sample(seq_len(n_samples), size = floor(0.6 * n_samples))
  X_train <- X[train_idx, , drop = FALSE]
  y_train <- y[train_idx]

  cat("Training set:", dim(X_train)[1], "x", dim(X_train)[2], "\n")
  cat("Validation set size:", n_samples - length(train_idx), "\n")

  fit <- glmnet(
    x = X_train, y = y_train,
    alpha = alpha_val,
    nlambda = 2000,
    standardize = FALSE,
    intercept = FALSE,
    dfmax = dfmax,
    thresh = 1e-03,
    maxit = 1000,
    pmax  = 2000,
    family = family_type
  )

  # 系数提取（多分类为列表）
  if (family_type == "multinomial") {
    # fit$beta: list of length K (每类一个 p×nlambda 矩阵)
    beta_list   <- lapply(fit$beta, function(b) as.matrix(b))  # 保证 matrix
    arr         <- simplify2array(beta_list)                   # p × nlambda × K
    # 在类别维取 max|coef|，得到与二分类/回归相同的 p × nlambda
    coefs_matrix <- apply(abs(arr), c(1, 2), max)
  } else {
    coefs_matrix <- as.matrix(fit$beta)  # p × nlambda
  }

  lambda_seq <- fit$lambda
  intercepts <- fit$a0

  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units = "secs")

  cat("Lasso fitting completed, time:", duration, "seconds\n")
  cat("Max non-zero features:", max(colSums(coefs_matrix != 0)), "\n")
  flush.console()

  # ---------------------------
  # Save
  # ---------------------------
  # Save
  # ---------------------------
  base_filename <- basename(input_file)
  output_file <- file.path(output_folder, paste0("LASSO_",name,"_idx"))

  cat("Saving to:", output_file, "\n")
  np$savez(
    output_file,
    coefs    = coefs_matrix,
    lambdas  = lambda_seq,
    intercept= intercepts,
    # alpha    = alpha_val
  )

  cat("Done ✅\n")
  flush.console()

}, error = function(e) {
  cat("Error occurred during execution:", conditionMessage(e), "\n")
  quit(status = 1)
})