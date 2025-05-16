# Load required libraries
library(e1071)
library(dplyr)

# Path to datasets
data_dir <- "E:/Explainatable-AI-SE/Experimental-setup/Dataset"
csv_files <- list.files(path = data_dir, pattern = "\\.csv$", full.names = TRUE)

# Define the columns to extract
selected_cols <- c("FCOH", "CU", "CPC", "PF", "IPSC", "DCO", "CR", "pkgreuse",
                   "CyclicDQ", "CyclicCQ", "CH", "COHM", "COUM", "post")

# Normalize function
normalize_range <- function(x) {
  x[x < 0] <- 0
  x[x > 1] <- 1
  return(x)
}

# Function to compute skewness and kurtosis as formatted strings
summary_stats <- function(data, dataset_name) {
  metrics <- colnames(data)
  skew <- apply(data, 2, skewness, na.rm = TRUE)
  kurt <- apply(data, 2, kurtosis, na.rm = TRUE)
  values <- paste0(round(skew, 2), " / ", round(kurt, 2))
  df <- data.frame(Metric = metrics, stringsAsFactors = FALSE)
  df[[dataset_name]] <- values
  return(df)
}

# Initialize the result container
merged_result <- NULL

# Loop through each dataset
for (file in csv_files) {
  dataset_name <- tools::file_path_sans_ext(basename(file))
  df <- read.csv(file, header = TRUE)
  
  if (all(selected_cols %in% colnames(df))) {
    df_selected <- df[, selected_cols]
    df_selected_normalized <- as.data.frame(lapply(df_selected, normalize_range))
    independent_vars <- df_selected_normalized[, setdiff(names(df_selected_normalized), "post")]
    
    result <- summary_stats(independent_vars, dataset_name)
    
    if (is.null(merged_result)) {
      merged_result <- result
    } else {
      merged_result <- full_join(merged_result, result, by = "Metric")
    }
  } else {
    warning(paste("Missing columns in:", file))
  }
}

# Show the result
print(merged_result)

# Save to CSV
write.csv(merged_result, "E:/Explainatable-AI-SE/Experimental-setup/skew_kurtosis_summary.csv", row.names = FALSE)
