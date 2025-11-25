# bcor.R — 方案A：safe_ncol/has_multi_cols + 详细日志（不改原有算法/分支语义）

# ----------------------------
# 依赖
# ----------------------------
library(Ball)
library(parallel)
library(doParallel)
library(reticulate)

# ----------------------------
# 工具函数（方案A）
# ----------------------------
safe_ncol <- function(x) {
  d <- dim(x)
  if (is.null(d)) 1L else if (length(d) >= 2) d[2] else 1L
}
has_multi_cols <- function(x) safe_ncol(x) > 1L

# ----------------------------
# 参数打印
# ----------------------------
args <- commandArgs(trailingOnly = TRUE)
cat("收到的命令行参数(args):\n")
if (length(args) == 0) {
  cat("  <empty>\n")
} else {
  for (i in seq_along(args)) {
    cat(sprintf("  args[%d] = '%s'\n", i, args[[i]]))
  }
}
flush.console()

if (length(args) < 2) {
  stop("需要两个参数: 1.输入文件路径 2.输出文件夹路径")
}
input_file  <- args[1]
output_folder <- args[2]
task_type   <- as.character(args[3])
name        <- if (length(args) >=4) as.character(args[4]) else "test"
cat(sprintf("解析到 task_type = '%s'\n", task_type)); flush.console()

# ----------------------------
# 主流程（仅加日志/安全取列数）
# ----------------------------
tryCatch({

  cat(sprintf("RETICULATE_PYTHON='%s'\n", Sys.getenv("RETICULATE_PYTHON", ""))); flush.console()

  # 读取 .npz
  cat("加载数据:", input_file, "\n"); flush.console()
  np <- import("numpy")
  data <- np$load(input_file, allow_pickle=TRUE)

  cat("加载的文件包含以下键:\n"); print(data$files); flush.console()

  # 提取
  X <- data[['X']]
  y <- data[['Y']]

  # X/Y 基本信息
  cat("===== X 基本信息 =====\n")
  cat("class(X):", paste(class(X), collapse = ","), "\n")
  cat("is.matrix(X)=", is.matrix(X), "; typeof(X)=", typeof(X), "\n")
  cat("dim(X)=", paste(dim(X), collapse=" x "), "\n")
  flush.console()

  cat("===== Y 基本信息 =====\n")
  cat("class(y):", paste(class(y), collapse = ","), "\n")
  cat("typeof(y)=", typeof(y), "\n")
  d_y <- dim(y)
  if (!is.null(d_y)) {
    cat("dim(y)=", paste(d_y, collapse=" x "), "\n")
    nc <- safe_ncol(y)
    cat("ncol(y)=", nc, "\n")
    if (nc > 1) cat("注意：Y 有多列，后续按你原逻辑处理\n")
  } else {
    cat("Y 为向量（无 dim）\n")
  }
  y_show <- tryCatch({
    yy <- as.vector(y)
    head(yy, 12)
  }, error = function(e) NA)
  cat("y 前 12 项=", paste(y_show, collapse = ","), "\n")
  flush.console()

  # 确保 X 矩阵
  X <- as.matrix(X)

  # ---- 按你原始逻辑处理 y（仅改用 has_multi_cols/safe_ncol）----
  y_raw <- y
  if (!is.null(dim(y_raw)) && has_multi_cols(y_raw)) {
    cat("检测到多列标签，尝试将 one-hot 转为类标\n"); flush.console()
    y_vec <- max.col(as.matrix(y_raw), ties.method = "first")
  } else {
    y_vec <- as.vector(y_raw)
  }

  # y_vec 概况
  cat("y_vec 概况：length=", length(y_vec), "\n")
  y_vec_head <- tryCatch(head(y_vec, 12), error=function(e) NA)
  cat("y_vec 前 12 项=", paste(y_vec_head, collapse=","), "\n")
  suppressWarnings( y_vec_num <- as.numeric(y_vec) )
  cat("y_vec 数值化后摘要：NA个数=", sum(is.na(y_vec_num)),
      "; 值域(去NA)=", paste(range(y_vec_num, na.rm=TRUE), collapse=".."), "\n")
  flush.console()

  # 任务类型分支（与原逻辑一致）
  if (task_type == "regression") {
    y <- as.numeric(y_vec)
    cat("任务=regression; y 转 numeric 完成\n"); flush.console()

  } else if (task_type == "binary") {
    y <- factor(y_vec)
    cat("任务=binary; y 转 factor 完成, nlevels=", nlevels(y),
        "; levels=", paste(levels(y), collapse=","), "\n")
    if (nlevels(y) != 2) {
      stop(paste("任务类型 binary 需要恰好两个类别，但检测到", nlevels(y), "个"))
    }

  } else if (task_type == "multiclass") {
    y <- factor(y_vec)
    cat("任务=multiclass; y 转 factor 完成, nlevels=", nlevels(y),
        "; levels=", paste(levels(y), collapse=","), "\n")
    if (nlevels(y) < 3) {
      stop(paste("任务类型 multiclass 需要至少三个类别，但检测到", nlevels(y), "个"))
    }

  } else {
    stop(paste("未知的 task_type:", task_type, "，必须是 regression/binary/multiclass"))
  }

  # NA 检查
  cat("NA 检查： X_NA_count=", sum(is.na(X)),
      "; y_NA_count=", sum(is.na(y)), "\n"); flush.console()

  if (any(is.na(X))) {
    cat("警告: X 包含NA值，将被移除(置0)\n")
    X[is.na(X)] <- 0
  }
  if (any(is.na(y))) {
    cat("警告: Y 包含NA值，将被移除(置0)\n")
    y[is.na(y)] <- 0
  }

  # 特征总数
  p <- dim(X)[2]
  cat("特征总数:", p, "\n"); flush.console()

  cat("开始使用doParallel外部并行 (bcorsis的num_threads=1)...\n"); flush.console()

  # 并行设置
  num_cores <- 10
  cat("使用", num_cores, "个处理核心\n"); flush.console()

  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  clusterExport(cl, c("X", "y"))

  start_time_external <- Sys.time()

  # 分块
  chunk_size <- ceiling(p / num_cores)
  chunks <- lapply(1:num_cores, function(i) {
    start_idx <- (i-1) * chunk_size + 1
    end_idx <- min(i * chunk_size, p)
    start_idx:end_idx
  })

  cat("开始并行处理", length(chunks), "个数据块，示例块1范围：",
      paste(range(chunks[[1]]), collapse=".."), "\n"); flush.console()

  # 并行计算
  results <- parLapply(cl, chunks, function(chunk_indices) {
    library(Ball)
    cat(sprintf("[child] chunk range=%d..%d (len=%d)\n",
                min(chunk_indices), max(chunk_indices), length(chunk_indices)))
    flush.console()

    result <- bcorsis(x = X[, chunk_indices], y = y,
                      method = "standard",
                      d = length(chunk_indices),
                      weight = "constant",
                      num.threads = 1)

    if (!is.null(result$ix)) {
      cat(sprintf("[child] bcorsis ok; local ix len=%d; complete.info null? %s\n",
                  length(result$ix), is.null(result$complete.info)))
    } else {
      cat("[child] bcorsis result$ix is NULL!\n")
    }
    flush.console()

    local_indices  <- result$ix
    global_indices <- chunk_indices[local_indices]

    if(!is.null(result$complete.info) && !is.null(result$complete.info$statistic)) {
      bcor_column <- which(colnames(result$complete.info$statistic) == "bcor.constant")
      if(length(bcor_column) > 0) {
        bcor_values <- result$complete.info$statistic[local_indices, bcor_column]
      } else {
        bcor_values <- result$complete.info$statistic[local_indices, 1]
      }
    } else {
      bcor_values <- numeric(length(local_indices))
      for (i in seq_along(local_indices)) {
        bcor_values[i] <- bcor(X[, global_indices[i]], y)
      }
    }

    cat(sprintf("[child] return indices len=%d, values len=%d\n",
                length(global_indices), length(bcor_values))); flush.console()
    list(indices = global_indices, values = bcor_values)
  })

  cat("整合并排序全部结果...\n"); flush.console()

  all_indices <- unlist(lapply(results, function(r) r$indices))
  all_values  <- unlist(lapply(results, function(r) r$values))

  cat(sprintf("聚合后: all_indices len=%d, all_values len=%d\n",
              length(all_indices), length(all_values))); flush.console()

  if(!is.numeric(all_values)) {
    cat("警告: all_values不是数值型，将尝试转换\n"); flush.console()
    all_values <- as.numeric(all_values)
  }

  sorted_idx <- order(abs(all_values), decreasing = TRUE)
  selected_indices_external <- all_indices[sorted_idx]

  end_time_external <- Sys.time()
  time_external <- difftime(end_time_external, start_time_external, units = "secs")

  stopCluster(cl)
  cat("外部并行计算完成，用时:", as.numeric(time_external), "秒\n\n"); flush.console()

  base_filename <- basename(input_file)
  features_file <- file.path(output_folder, paste0("BCOR_", name,'_idx'))

  cat("保存特征索引到:", features_file, "\n"); flush.console()
  np$save(features_file, selected_indices_external-1L)  # 0-based

  cat("处理完成!\n"); flush.console()

}, error = function(e) {
  write(paste("执行过程中出错:", conditionMessage(e)), file=stderr())
  quit(status=1)
})