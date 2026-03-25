# =========================================================
# Exploratory factor analysis (overall, between, within)
# =========================================================

# Load packages
library(psych)
library(GPArotation)
library(dplyr)

# ---------------------------------------------------------
# 1. Set working directory and load data
# ---------------------------------------------------------

# Set working directory to the folder containing this script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read dataset
dat_raw <- read.csv("../output/combined_output.csv")

# ---------------------------------------------------------
# 2. Define variables to include in factor analysis
# ---------------------------------------------------------

cols <- c(
  "bkt_bf_prediction",
  "BKT_forgetting_prediction",
  "PFA_prediction",
  "ELO_prediction",
  "KTM_prediction",
  "DKT_prediction",
  "DSAKT_prediction",
  "ATKT_prediction"
)

# ---------------------------------------------------------
# 3. Prepare overall EFA dataset
# ---------------------------------------------------------

# Keep only factor-analysis columns
efa_data <- dat_raw[, cols]

# Ensure all selected columns are numeric
efa_data[cols] <- lapply(efa_data[cols], function(x) as.numeric(as.character(x)))

# Remove rows with missing values
efa_data <- na.omit(efa_data)

# ---------------------------------------------------------
# 4. Overall factor analysis
# ---------------------------------------------------------

# Check whether factor analysis is appropriate
KMO(efa_data)
cortest.bartlett(cor(efa_data), n = nrow(efa_data))

# Parallel analysis to help choose number of factors
fa.parallel(efa_data, fa = "fa")

# Run overall EFA
fa_overall <- fa(
  efa_data,
  nfactors = 3,
  rotate = "oblimin",
  fm = "ml"
)

# Print loadings
print(fa_overall$loadings, cutoff = 0.50)

# Compute factor scores for each observation
scores_overall <- factor.scores(efa_data, fa_overall)$scores

# ---------------------------------------------------------
# 5. Prepare multilevel datasets using student ID ("user")
# ---------------------------------------------------------

# Keep user ID plus selected variables
dat <- dat_raw[, c("user", cols)]

# Ensure selected columns are numeric
dat[cols] <- lapply(dat[cols], function(x) as.numeric(as.character(x)))

# Remove rows with missing user
dat <- dat[!is.na(dat$user), ]

# ---------------------------------------------------------
# 6. Between-student dataset
#    One row per student: student-level means
# ---------------------------------------------------------

between_data <- dat %>%
  group_by(user) %>%
  summarise(across(all_of(cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

between_fa_data <- between_data[, cols]
between_fa_data <- na.omit(between_fa_data)

# ---------------------------------------------------------
# 7. Within-student dataset
#    Person-mean centered values
# ---------------------------------------------------------

within_data <- dat %>%
  group_by(user) %>%
  mutate(across(all_of(cols), ~ .x - mean(.x, na.rm = TRUE))) %>%
  ungroup()

within_fa_data <- within_data[, cols]
within_fa_data <- na.omit(within_fa_data)

# ---------------------------------------------------------
# 8. Separate parallel analyses
# ---------------------------------------------------------

fa.parallel(between_fa_data, fa = "fa")
fa.parallel(within_fa_data, fa = "fa")

# ---------------------------------------------------------
# 9. Run between-student and within-student EFAs
# ---------------------------------------------------------

fa_between <- fa(
  between_fa_data,
  nfactors = 3,
  rotate = "oblimin",
  fm = "ml"
)

fa_within <- fa(
  within_fa_data,
  nfactors = 3,
  rotate = "oblimin",
  fm = "ml"
)

# Print loadings
print(fa_between$loadings, cutoff = 0.50)
print(fa_within$loadings, cutoff = 0.50)

# Compute factor scores
scores_between <- factor.scores(between_fa_data, fa_between)$scores
scores_within  <- factor.scores(within_fa_data,  fa_within)$scores

# ---------------------------------------------------------
# 10. Save loadings to CSV
# ---------------------------------------------------------

# Convert loadings objects to data frames
loadings_overall_df <- as.data.frame(unclass(fa_overall$loadings))
loadings_between_df <- as.data.frame(unclass(fa_between$loadings))
loadings_within_df  <- as.data.frame(unclass(fa_within$loadings))

# Keep variable names as a column
loadings_overall_df$variable <- rownames(loadings_overall_df)
loadings_between_df$variable <- rownames(loadings_between_df)
loadings_within_df$variable  <- rownames(loadings_within_df)

# Move variable column to front
loadings_overall_df <- loadings_overall_df[, c("variable", setdiff(names(loadings_overall_df), "variable"))]
loadings_between_df <- loadings_between_df[, c("variable", setdiff(names(loadings_between_df), "variable"))]
loadings_within_df  <- loadings_within_df[, c("variable", setdiff(names(loadings_within_df), "variable"))]

# Save loadings
write.csv(loadings_overall_df, "factor_loadings_overall.csv", row.names = FALSE)
write.csv(loadings_between_df, "factor_loadings_between.csv", row.names = FALSE)
write.csv(loadings_within_df,  "factor_loadings_within.csv", row.names = FALSE)

# ---------------------------------------------------------
# 11. Save factor scores to CSV
# ---------------------------------------------------------

# Overall scores
scores_overall_df <- as.data.frame(scores_overall)
write.csv(scores_overall_df, "factor_scores_overall.csv", row.names = FALSE)

# Between-student scores: attach user ID
scores_between_df <- cbind(user = between_data$user[complete.cases(between_data[, cols])],
                           as.data.frame(scores_between))
write.csv(scores_between_df, "factor_scores_between.csv", row.names = FALSE)

# Within-student scores: attach user ID
within_complete <- within_data[complete.cases(within_data[, cols]), ]
scores_within_df <- cbind(user = within_complete$user,
                          as.data.frame(scores_within))
write.csv(scores_within_df, "factor_scores_within.csv", row.names = FALSE)

# ---------------------------------------------------------
# 12. Save factor correlations
# ---------------------------------------------------------

if (!is.null(fa_overall$Phi)) {
  write.csv(as.data.frame(fa_overall$Phi), "factor_correlations_overall.csv")
}

if (!is.null(fa_between$Phi)) {
  write.csv(as.data.frame(fa_between$Phi), "factor_correlations_between.csv")
}

if (!is.null(fa_within$Phi)) {
  write.csv(as.data.frame(fa_within$Phi), "factor_correlations_within.csv")
}

# ---------------------------------------------------------
# 13. Save correlations among the original variables
# ---------------------------------------------------------

# Correlation matrices
cor_overall  <- cor(efa_data, use = "pairwise.complete.obs")
cor_between  <- cor(between_fa_data, use = "pairwise.complete.obs")
cor_within   <- cor(within_fa_data, use = "pairwise.complete.obs")

# Save to CSV
write.csv(as.data.frame(cor_overall), "variable_correlations_overall.csv", row.names = TRUE)
write.csv(as.data.frame(cor_between), "variable_correlations_between.csv", row.names = TRUE)
write.csv(as.data.frame(cor_within), "variable_correlations_within.csv", row.names = TRUE)
