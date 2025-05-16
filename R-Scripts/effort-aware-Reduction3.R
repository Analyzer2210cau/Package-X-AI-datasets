############################################################
# Effort-Aware Fault Prediction (Effort Reduction Scenario)
# Modular R Script
# Models: LR, RF, XGBoost, AdaBoost, SVM, ANN
# 10x10 Cross Validation, BPP (Balance) threshold
############################################################

# -- LIBRARY SETUP --
library(caret)
library(pROC)
library(e1071)
library(randomForest)
library(xgboost)
library(adabag)
library(nnet)
library(dplyr)
library(stringr)

# -- DATA LOADING & FEATURE DETECTION --

# Detect effort (LOC/SLOC/pkgLOC) column in a data frame
detect_effort_col <- function(colnames) {
  effort_candidates <- c("pkgLOC", "SLOC", "pkgKLOC")
  for (c in effort_candidates) {
    match_col <- colnames[which(str_trim(tolower(colnames)) == tolower(c))]
    if (length(match_col) > 0) return(match_col[1])
  }
  return(NA)
}

# Find actual metric columns (case-insensitive, trimmed)
find_metric_cols <- function(df, canonical_metrics) {
  metric_map <- sapply(canonical_metrics, function(m) {
    col <- grep(paste0("^\\s*", m, "\\s*$"), names(df), ignore.case=TRUE, value=TRUE)
    if (length(col) > 0) return(col[1]) else return(NA)
  })
  return(metric_map[!is.na(metric_map)])
}

# -- MODEL TRAINING & PREDICTION MODULES --

# Fit a classifier and return predicted probabilities
get_model_probs <- function(model_name, train_x, train_y, test_x) {
  if (model_name == "LR") {
    model <- glm(train_y ~ ., data = data.frame(train_x, train_y), family = "binomial")
    probs <- predict(model, newdata = data.frame(test_x), type = "response")
  } else if (model_name == "RF") {
    model <- randomForest(x=train_x, y=as.factor(train_y), ntree=100)
    probs <- predict(model, newdata=test_x, type="prob")[, "1"]
  } else if (model_name == "XGBoost") {
    dtrain <- xgb.DMatrix(data=as.matrix(train_x), label=train_y)
    dtest <- xgb.DMatrix(data=as.matrix(test_x))
    model <- xgboost(data=dtrain, max_depth=3, nrounds=30, objective="binary:logistic", verbose=0)
    probs <- predict(model, newdata=dtest)
  } else if (model_name == "AdaBoost") {
    train_df <- data.frame(train_x, y=as.factor(train_y))
    model <- boosting(y ~ ., data=train_df, boos=TRUE, mfinal=20)
    pred <- predict.boosting(model, newdata = data.frame(test_x))
    if (!is.null(dim(pred$prob)) && "1" %in% colnames(pred$prob)) {
      probs <- pred$prob[, "1"]
    } else if (!is.null(pred$prob)) {
      probs <- as.numeric(pred$prob)
    } else {
      probs <- rep(0, nrow(test_x))
    }
  } else if (model_name == "SVM") {
    model <- svm(train_x, as.factor(train_y), probability=TRUE, kernel="radial")
    pred <- predict(model, test_x, probability=TRUE)
    prob_matrix <- attr(pred, "probabilities")
    if (!is.null(prob_matrix) && "1" %in% colnames(prob_matrix)) {
      probs <- prob_matrix[, "1"]
    } else if (!is.null(prob_matrix)) {
      probs <- as.numeric(prob_matrix[, 1])
    } else {
      probs <- as.numeric(pred == "1")
    }
  } else if (model_name == "ANN") {
    scaler <- function(x) (x - min(x)) / (max(x) - min(x) + 1e-8)
    train_x_scaled <- as.data.frame(lapply(train_x, scaler))
    test_x_scaled <- as.data.frame(lapply(test_x, scaler))
    model <- nnet::nnet(train_x_scaled, train_y, size=5, maxit=100, linout=FALSE, trace=FALSE)
    probs <- predict(model, test_x_scaled, type="raw")
    if (is.matrix(probs) && ncol(probs) == 2) {
      probs <- probs[, 2]
    }
    probs <- as.numeric(probs)
  } else {
    stop("Unknown model name!")
  }
  return(probs)
}

# -- EFFORT-REDUCTION EVALUATION MODULE --

# Compute ER and Balance metric given test predictions, effort and labels
compute_er <- function(df_test, pred_probs, effort, actual, loc_col, num_col) {
  # Only valid if both classes are present
  if (length(unique(actual)) < 2) stop("test_y contains only one class. Skipping ROC for this fold.")
  roc_obj <- pROC::roc(response=actual, predictor=pred_probs, quiet=TRUE)
  coords_obj <- pROC::coords(roc_obj, "all", ret=c("threshold", "specificity", "sensitivity"), transpose=FALSE)
  pf <- 1 - coords_obj$specificity
  pd <- coords_obj$sensitivity
  balance <- 1 - sqrt((0 - pf)^2 + (1 - pd)^2) / sqrt(2)
  bpp_idx <- which.max(balance)
  thresh <- coords_obj$threshold[bpp_idx]
  pred_bin <- as.integer(pred_probs >= thresh)
  data <- data.frame(
    REL = actual,
    PRE = pred_probs,
    LOC = effort,
    NUM = if (num_col %in% colnames(df_test)) df_test[[num_col]] else actual
  )
  total_LOC <- sum(data$LOC)
  total_LOC_M <- sum(pred_bin * data$LOC)
  Effort_M <- total_LOC_M / total_LOC
  recall <- sum((pred_bin == 1) & (actual == 1)) / sum(actual == 1)
  Effort_R <- recall
  ER <- Effort_R - Effort_M
  return(list(ER=ER, balance=balance[bpp_idx], threshold=thresh))
}

# -- MAIN CROSS-VALIDATION DRIVER MODULES --

# Cross-validation for a single model, single dataset
crossval_model <- function(x_data, y_data, effort, df, effort_col, num_col, model_name, k_folds=10, k_times=10) {
  ER_scores <- numeric()
  for (i in 1:k_times) {
    folds <- createFolds(y_data, k = k_folds)
    for (fold_idx in 1:k_folds) {
      test_idx <- folds[[fold_idx]]
      train_x <- x_data[-test_idx, , drop=FALSE]
      train_y <- y_data[-test_idx]
      test_x  <- x_data[test_idx, , drop=FALSE]
      test_y  <- y_data[test_idx]
      test_effort <- effort[test_idx]
      test_df <- df[test_idx, ]
      probs <- get_model_probs(model_name, train_x, train_y, test_x)
      # Check and align lengths before ROC
      if (is.null(probs) || length(probs) != length(test_y) || any(is.na(probs)) || length(test_y) < 2) {
        next
      }
      if (length(unique(test_y)) < 2) {
        next
      }
      er_res <- tryCatch(
        compute_er(test_df, probs, test_effort, test_y, loc_col=effort_col, num_col=num_col),
        error = function(e) { NULL }
      )
      if (!is.null(er_res)) {
        ER_scores <- c(ER_scores, er_res$ER)
      }
    }
  }
  return(list(mean=mean(ER_scores, na.rm=TRUE), sd=sd(ER_scores, na.rm=TRUE)))
}

# Cross-validation for all models, all datasets
crossval_all_datasets <- function(csv_files, canonical_metrics, target_col, models, k_folds=10, k_times=10) {
  results_er <- data.frame()
  for (file in csv_files) {
    df <- tryCatch(read.csv(file), error=function(e) NULL)
    dataset_name <- tools::file_path_sans_ext(basename(file))
    if (is.null(df)) next
    colnames(df) <- str_trim(colnames(df))
    effort_col <- detect_effort_col(colnames(df))
    metric_map <- find_metric_cols(df, canonical_metrics)
    if (is.na(effort_col) || is.null(metric_map) || !target_col %in% colnames(df)) next
    df[[target_col]] <- ifelse(df[[target_col]] > 0, 1, 0)
    x_data <- df[, metric_map]
    y_data <- df[[target_col]]
    effort <- df[[effort_col]]
    num_col <- if ("BUG" %in% colnames(df)) "BUG" else if ("NUM" %in% colnames(df)) "NUM" else target_col
    
    for (m in models) {
      perf <- crossval_model(x_data, y_data, effort, df, effort_col, num_col, m, k_folds, k_times)
      results_er <- rbind(results_er, data.frame(Dataset=dataset_name, Model=m,
                                                 ER=sprintf("%.4f ± %.4f", perf$mean, perf$sd)))
    }
  }
  return(results_er)
}

# -- MAIN SCRIPT RUNNER --

run_er_evaluation <- function() {
  data_dir <- "E:/Explainatable-AI-SE/Experimental-setup/Dataset"
  csv_files <- list.files(path = data_dir, pattern = "\\.csv$", full.names = TRUE)
  canonical_metrics <- c("FCOH", "CU", "CPC", "PF", "IPSC", "DCO", "CR", "pkgreuse",
                         "CyclicDQ", "CyclicCQ", "CH", "COHM", "COUM")
  target_col <- "post"
  models <- c("LR", "RF", "XGBoost", "AdaBoost", "SVM", "ANN")
  k_folds <- 10
  k_times <- 10
  
  results_er <- crossval_all_datasets(csv_files, canonical_metrics, target_col, models, k_folds, k_times)
  results_wide_er <- tidyr::pivot_wider(results_er, names_from=Model, values_from=ER)
  print(results_wide_er, row.names=FALSE)
  
}

# Run the script
run_er_evaluation()
   