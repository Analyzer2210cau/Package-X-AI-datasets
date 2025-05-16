library(caret)
library(e1071)
library(randomForest)
library(xgboost)
library(dplyr)
library(stringr)
library(adabag)
library(nnet)

data_dir <- "E:/Explainatable-AI-SE/Experimental-setup/Dataset"
csv_files <- list.files(path = data_dir, pattern = "\\.csv$", full.names = TRUE)

# Canonical metric names for your analysis
canonical_metrics <- c("FCOH", "CU", "CPC", "PF", "IPSC", "DCO", "CR", "pkgreuse",
                       "CyclicDQ", "CyclicCQ", "CH", "COHM", "COUM")
target_col <- "post"

# Function to detect the correct LOC/effort column
detect_effort_col <- function(colnames) {
  effort_candidates <- c("pkgLOC", "SLOC", "pkgKLOC")
  for (c in effort_candidates) {
    match_col <- colnames[which(str_trim(tolower(colnames)) == tolower(c))]
    if (length(match_col) > 0) return(match_col[1])
  }
  return(NA)
}

# Function to find actual metric columns by partial match
find_metric_cols <- function(df) {
  metric_map <- sapply(canonical_metrics, function(m) {
    col <- grep(paste0("^\\s*", m, "\\s*$"), names(df), ignore.case=TRUE, value=TRUE)
    if (length(col) > 0) return(col[1]) else return(NA)
  })
  return(metric_map[!is.na(metric_map)])
}

set.seed(123)
k_folds <- 10
k_times <- 10
models <- c("LR", "SVM", "RF", "XGBoost", "AdaBoost", "ANN")
results <- data.frame()

for (file in csv_files) {
  df <- tryCatch(read.csv(file), error=function(e) NULL)
  dataset_name <- tools::file_path_sans_ext(basename(file))
  
  if (is.null(df)) {
    warning(paste("Could not read file:", file))
    next
  }
  colnames(df) <- str_trim(colnames(df)) # Remove whitespace from colnames
  
  # Detect columns
  effort_col <- detect_effort_col(colnames(df))
  metric_map <- find_metric_cols(df)
  if (is.na(effort_col) || is.null(metric_map) || !target_col %in% colnames(df)) {
    warning(paste("Missing required columns in:", file))
    next
  }
  # Prepare data
  df[[target_col]] <- ifelse(df[[target_col]] > 0, 1, 0) # Binary target
  x_data <- df[, metric_map]
  y_data <- df[[target_col]]
  effort <- df[[effort_col]]
  
  # Cross-validation function
  run_model <- function(model_name) {
    CE_scores <- numeric()
    for (i in 1:k_times) {
      folds <- createFolds(y_data, k = k_folds)
      for (fold_idx in 1:k_folds) {
        test_idx <- folds[[fold_idx]]
        train_x <- x_data[-test_idx, , drop=FALSE]
        train_y <- y_data[-test_idx]
        test_x  <- x_data[test_idx, , drop=FALSE]
        test_y  <- y_data[test_idx]
        test_effort <- effort[test_idx]
        
        # Model fitting
        if (model_name == "LR") {
          model <- glm(train_y ~ ., data = data.frame(train_x, train_y), family = "binomial")
          probs <- predict(model, newdata = data.frame(test_x), type = "response")
        } else if (model_name == "SVM") {
          model <- svm(train_x, as.factor(train_y), probability=TRUE, kernel="radial")
          probs <- attr(predict(model, test_x, probability=TRUE), "probabilities")[, "1"]
        } else if (model_name == "RF") {
          model <- randomForest(x=train_x, y=as.factor(train_y), ntree=100)
          probs <- predict(model, newdata=test_x, type="prob")[, "1"]
        } else if (model_name == "XGBoost") {
          dtrain <- xgb.DMatrix(data=as.matrix(train_x), label=train_y)
          dtest <- xgb.DMatrix(data=as.matrix(test_x))
          model <- xgboost(data=dtrain, max_depth=3, nrounds=30, objective="binary:logistic", verbose=0)
          probs <- predict(model, newdata=dtest)
        } else if (model_name == "AdaBoost") {
          # Prepare data frame for adabag (response as factor)
          train_df <- data.frame(train_x, y=as.factor(train_y))
          model <- boosting(y ~ ., data=train_df, boos=TRUE, mfinal=20)
          pred <- predict.boosting(model, newdata = data.frame(test_x))
          # If two classes, use column '1' for probability; otherwise fallback to first column
          if (!is.null(dim(pred$prob)) && "1" %in% colnames(pred$prob)) {
            probs <- pred$prob[, "1"]
          } else if (!is.null(pred$prob)) {
            probs <- as.numeric(pred$prob)
          } else {
            probs <- rep(0, length(test_y)) # fallback: all zeros
          }
        } else if (model_name == "ANN") {
          # For neural nets, scale inputs between 0 and 1 is helpful
          # nnet requires matrix inputs; size=5 is hidden layer size, can adjust
          train_x_scaled <- as.data.frame(scale(train_x))
          test_x_scaled <- as.data.frame(scale(test_x, center=attr(scale(train_x), "scaled:center"), scale=attr(scale(train_x), "scaled:scale")))
          model <- nnet(x = train_x_scaled, y = train_y, size = 5, linout = FALSE, rang = 0.1, maxit = 200, trace = FALSE)
          probs <- predict(model, test_x_scaled, type = "raw")
          # nnet sometimes gives output as vector if only 1 node; ensure is vector
          probs <- as.vector(probs)
        }
        
        # Effort-aware ranking
        df_test <- data.frame(prob=probs, effort=test_effort, target=test_y)
        df_test <- df_test[order(-df_test$prob / (df_test$effort + 1e-6)), ]
        df_test$cum_effort <- cumsum(df_test$effort)
        df_test$cum_faults <- cumsum(df_test$target)
        total_effort <- sum(df_test$effort)
        loc_20 <- 0.2 * total_effort
        faults_found <- sum(df_test$target[df_test$cum_effort <= loc_20])
        total_faults <- sum(df_test$target)
        ce_20 <- if (total_faults == 0) NA else faults_found / total_faults
        CE_scores <- c(CE_scores, ce_20)
      }
    }
    return(list(mean=mean(CE_scores, na.rm=TRUE), sd=sd(CE_scores, na.rm=TRUE)))
  }
  
  # Run all models
  for (m in models) {
    perf <- run_model(m)
    results <- rbind(results, data.frame(Dataset=dataset_name, Model=m,
                                         CE_at_0.2=sprintf("%.4f ± %.4f", perf$mean, perf$sd)))
  }
}

# Print results
results_wide <- tidyr::pivot_wider(results, names_from=Model, values_from=CE_at_0.2)
print(results_wide, row.names=FALSE)
