# Summary Statistics for all independent variables

# Step 1: Read the CSV file (update path if needed)
df <- read.csv("E:/Explainatable-AI-SE/Experimental-setup/Dataset/Eclipse-2.0.csv", header = TRUE)
#"E:/Sabina/Survey-Results/DataSet1.csv"
#E:/Explainatable-AI-SE/Experimental-setup/Dataset

# Step 2: Extract selected columns
selected_cols <- c("FCOH", "CU", "CPC", "PF", "IPSC", "DCO", "CR", "pkgreuse",
                   "CyclicDQ", "CyclicCQ", "CH", "COHM", "COUM", "post")
df_selected <- df[, selected_cols]

# Step 3: Normalize the data (clip negative values to 0 and values > 1 to 1)
normalize_range <- function(x) {
  x[x < 0] <- 0
  x[x > 1] <- 1
  return(x)
}

df_selected_normalized <- as.data.frame(lapply(df_selected, normalize_range))

# Step 4: Separate the dependent variable
post_variable <- df_selected_normalized$post
independent_vars <- df_selected_normalized[, setdiff(names(df_selected_normalized), "post")]

# Optional: Print a preview
print(head(independent_vars))
print(head(post_variable))



summary_stats <- function(data) {
  summary(data)
  library(e1071)
  skewness_kurtosis <- data.frame(
    Metric = colnames(data),
    Skewness = apply(data, 2, skewness, na.rm = TRUE),
    Kurtosis = apply(data, 2, kurtosis, na.rm = TRUE)
  )
  return(skewness_kurtosis)
}
summary_stats(independent_vars)

# Correlation Matrix and Heatmap
correlation_analysis <- function(data) {
  cor_matrix <- cor(data, use = "complete.obs")
  library(corrplot)
  corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)
  return(cor_matrix)
}
# Histograms, Boxplots, Density Plots
distribution_visualization <- function(data) {
  library(ggplot2)
  for (col in colnames(data)) {
    p1 <- ggplot(data, aes_string(x = col)) + 
      geom_histogram(bins = 30, fill = "skyblue", color = "black") + 
      ggtitle(paste("Histogram of", col))
    print(p1)
    
    p2 <- ggplot(data, aes_string(y = col)) + 
      geom_boxplot(fill = "lightgreen") + 
      ggtitle(paste("Boxplot of", col))
    print(p2)
    
    p3 <- ggplot(data, aes_string(x = col)) + 
      geom_density(fill = "lightcoral", alpha = 0.5) + 
      ggtitle(paste("Density Plot of", col))
    print(p3)
  }
}

# T-Tests based on binary 'post' values
t_tests <- function(data, dependent_col = "post") {
  results <- list()
  binary_post <- data[[dependent_col]] > 0
  for (col in setdiff(colnames(data), dependent_col)) {
    results[[col]] <- t.test(data[[col]] ~ binary_post)
  }
  return(results)
}
# ANOVA analysis
anova_analysis <- function(data, dependent_col = "post") {
  results <- list()
  for (col in setdiff(colnames(data), dependent_col)) {
    formula <- as.formula(paste(col, "~ as.factor(", dependent_col, ")"))
    results[[col]] <- summary(aov(formula, data = data))
  }
  return(results)
}

# Bar Charts for Binned Metric vs. Post
bar_charts <- function(data, dependent_col = "post") {
  library(ggplot2)
  for (col in setdiff(colnames(data), dependent_col)) {
    data[[paste0(col, "_bin")]] <- cut(data[[col]], breaks = 4, labels = FALSE)
    p <- ggplot(data, aes_string(x = paste0(col, "_bin"), fill = dependent_col)) + 
      geom_bar(position = "dodge") + 
      ggtitle(paste("Bar Chart of", col, "vs", dependent_col)) + 
      xlab(col)
    print(p)
  }
}


# Outlier and Normality Check
outlier_normality <- function(data) {
  library(car)
  diagnostics <- list()
  for (col in colnames(data)) {
    qqPlot(data[[col]], main = paste("Q-Q Plot for", col))
    diagnostics[[col]] <- shapiro.test(data[[col]])
  }
  return(diagnostics)
}
