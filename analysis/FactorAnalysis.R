# =========================================================
# Confirmatory + Exploratory factor analysis
# overall, between-student, within-student
#
# Includes:
# - One-factor CFA for overall, between, within
# - KMO and Bartlett tests
# - Parallel analyses
# - EFA model comparison for 1-, 2-, 3-, and 4-factor solutions
# - Selected 3-factor EFA loadings, scores, and correlations
# =========================================================

# ---------------------------------------------------------
# 0. Load packages
# ---------------------------------------------------------

library(psych)
library(GPArotation)
library(dplyr)
library(lavaan)
library(rstudioapi)

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
# 3. Prepare overall EFA/CFA dataset
# ---------------------------------------------------------

# Keep only factor-analysis columns
efa_data <- dat_raw[, cols]

# Ensure all selected columns are numeric
efa_data[cols] <- lapply(efa_data[cols], function(x) as.numeric(as.character(x)))

# Remove rows with missing values
efa_data <- na.omit(efa_data)

# ---------------------------------------------------------
# 4. Prepare multilevel datasets using student ID "user"
# ---------------------------------------------------------

# Keep user ID plus selected variables
dat <- dat_raw[, c("user", cols)]

# Ensure selected columns are numeric
dat[cols] <- lapply(dat[cols], function(x) as.numeric(as.character(x)))

# Remove rows with missing user
dat <- dat[!is.na(dat$user), ]

# Between-student dataset: one row per student
between_data <- dat %>%
  group_by(user) %>%
  summarise(
    across(all_of(cols), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

between_fa_data <- between_data[, cols]
between_fa_data <- na.omit(between_fa_data)

# Within-student dataset: person-mean centered values
within_data <- dat %>%
  group_by(user) %>%
  mutate(
    across(all_of(cols), ~ .x - mean(.x, na.rm = TRUE))
  ) %>%
  ungroup()

within_fa_data <- within_data[, cols]
within_fa_data <- na.omit(within_fa_data)

# ---------------------------------------------------------
# 5. One-factor CFA
# ---------------------------------------------------------

one_factor_model <- '
KT_general =~ bkt_bf_prediction +
              BKT_forgetting_prediction +
              PFA_prediction +
              ELO_prediction +
              KTM_prediction +
              DKT_prediction +
              DSAKT_prediction +
              ATKT_prediction
'

# Use robust ML estimator because prediction variables may be non-normal
cfa_overall_1f <- cfa(
  one_factor_model,
  data = efa_data,
  std.lv = TRUE,
  estimator = "MLR"
)

cfa_between_1f <- cfa(
  one_factor_model,
  data = between_fa_data,
  std.lv = TRUE,
  estimator = "MLR"
)

cfa_within_1f <- cfa(
  one_factor_model,
  data = within_fa_data,
  std.lv = TRUE,
  estimator = "MLR"
)

# Print summaries
summary(cfa_overall_1f, fit.measures = TRUE, standardized = TRUE)
summary(cfa_between_1f, fit.measures = TRUE, standardized = TRUE)
summary(cfa_within_1f, fit.measures = TRUE, standardized = TRUE)

# Save key CFA fit indices
fit_table_1f <- data.frame(
  dataset = c("overall", "between", "within"),
  chisq = c(
    fitMeasures(cfa_overall_1f, "chisq"),
    fitMeasures(cfa_between_1f, "chisq"),
    fitMeasures(cfa_within_1f, "chisq")
  ),
  df = c(
    fitMeasures(cfa_overall_1f, "df"),
    fitMeasures(cfa_between_1f, "df"),
    fitMeasures(cfa_within_1f, "df")
  ),
  p_value = c(
    fitMeasures(cfa_overall_1f, "pvalue"),
    fitMeasures(cfa_between_1f, "pvalue"),
    fitMeasures(cfa_within_1f, "pvalue")
  ),
  cfi = c(
    fitMeasures(cfa_overall_1f, "cfi"),
    fitMeasures(cfa_between_1f, "cfi"),
    fitMeasures(cfa_within_1f, "cfi")
  ),
  tli = c(
    fitMeasures(cfa_overall_1f, "tli"),
    fitMeasures(cfa_between_1f, "tli"),
    fitMeasures(cfa_within_1f, "tli")
  ),
  rmsea = c(
    fitMeasures(cfa_overall_1f, "rmsea"),
    fitMeasures(cfa_between_1f, "rmsea"),
    fitMeasures(cfa_within_1f, "rmsea")
  ),
  srmr = c(
    fitMeasures(cfa_overall_1f, "srmr"),
    fitMeasures(cfa_between_1f, "srmr"),
    fitMeasures(cfa_within_1f, "srmr")
  ),
  aic = c(
    fitMeasures(cfa_overall_1f, "aic"),
    fitMeasures(cfa_between_1f, "aic"),
    fitMeasures(cfa_within_1f, "aic")
  ),
  bic = c(
    fitMeasures(cfa_overall_1f, "bic"),
    fitMeasures(cfa_between_1f, "bic"),
    fitMeasures(cfa_within_1f, "bic")
  )
)

write.csv(
  fit_table_1f,
  "cfa_one_factor_fit_summary.csv",
  row.names = FALSE
)

# Save standardized CFA loadings
std_loadings_overall <- standardizedSolution(cfa_overall_1f)
std_loadings_between <- standardizedSolution(cfa_between_1f)
std_loadings_within  <- standardizedSolution(cfa_within_1f)

write.csv(
  std_loadings_overall,
  "cfa_one_factor_loadings_overall.csv",
  row.names = FALSE
)

write.csv(
  std_loadings_between,
  "cfa_one_factor_loadings_between.csv",
  row.names = FALSE
)

write.csv(
  std_loadings_within,
  "cfa_one_factor_loadings_within.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 6. KMO, Bartlett, and parallel analysis: overall
# ---------------------------------------------------------

kmo_overall <- KMO(efa_data)
bart_overall <- cortest.bartlett(cor(efa_data), n = nrow(efa_data))

print(kmo_overall)
print(bart_overall)

capture.output(kmo_overall, file = "kmo_overall.txt")
capture.output(bart_overall, file = "bartlett_overall.txt")

png("parallel_analysis_overall.png", width = 1000, height = 800)
fa.parallel(efa_data, fa = "fa")
dev.off()

# ---------------------------------------------------------
# 7. KMO, Bartlett, and parallel analysis: between and within
# ---------------------------------------------------------

kmo_between <- KMO(between_fa_data)
bart_between <- cortest.bartlett(cor(between_fa_data), n = nrow(between_fa_data))

print(kmo_between)
print(bart_between)

capture.output(kmo_between, file = "kmo_between.txt")
capture.output(bart_between, file = "bartlett_between.txt")

png("parallel_analysis_between.png", width = 1000, height = 800)
fa.parallel(between_fa_data, fa = "fa")
dev.off()

kmo_within <- KMO(within_fa_data)
bart_within <- cortest.bartlett(cor(within_fa_data), n = nrow(within_fa_data))

print(kmo_within)
print(bart_within)

capture.output(kmo_within, file = "kmo_within.txt")
capture.output(bart_within, file = "bartlett_within.txt")

png("parallel_analysis_within.png", width = 1000, height = 800)
fa.parallel(within_fa_data, fa = "fa")
dev.off()

# ---------------------------------------------------------
# 8. Compare 1-, 2-, 3-, and 4-factor EFA solutions
# ---------------------------------------------------------

# Helper to safely extract named elements from psych::fa objects
safe_extract <- function(x, name) {
  if (!is.null(x[[name]])) {
    return(x[[name]])
  } else {
    return(NA)
  }
}

# Helper to extract EFA fit indices from psych::fa object
extract_efa_fit <- function(fa_obj, dataset_name, nfactors, n_obs) {
  
  rmsea_vals <- safe_extract(fa_obj, "RMSEA")
  
  data.frame(
    dataset = dataset_name,
    nfactors = nfactors,
    n_obs = n_obs,
    chisq = safe_extract(fa_obj, "STATISTIC"),
    df = safe_extract(fa_obj, "dof"),
    p_value = safe_extract(fa_obj, "PVAL"),
    TLI = safe_extract(fa_obj, "TLI"),
    CFI = safe_extract(fa_obj, "CFI"),
    RMSEA = ifelse(length(rmsea_vals) >= 1, rmsea_vals[1], NA),
    RMSEA_lower = ifelse(length(rmsea_vals) >= 2, rmsea_vals[2], NA),
    RMSEA_upper = ifelse(length(rmsea_vals) >= 3, rmsea_vals[3], NA),
    RMSR = safe_extract(fa_obj, "rms"),
    corrected_RMSR = safe_extract(fa_obj, "crms"),
    BIC = safe_extract(fa_obj, "BIC"),
    stringsAsFactors = FALSE
  )
}

# Helper to run 1-4 factor solutions for a dataset
run_efa_comparison <- function(data, dataset_name, factors = 1:4) {
  
  fit_list <- list()
  model_list <- list()
  
  for (nf in factors) {
    
    cat("\nRunning", dataset_name, "EFA with", nf, "factor(s)\n")
    
    fa_model <- tryCatch(
      fa(
        data,
        nfactors = nf,
        rotate = ifelse(nf == 1, "none", "oblimin"),
        fm = "ml"
      ),
      error = function(e) {
        message(
          "Model failed for ",
          dataset_name,
          ", ",
          nf,
          " factor(s): ",
          e$message
        )
        return(NULL)
      }
    )
    
    if (!is.null(fa_model)) {
      
      fit_list[[as.character(nf)]] <- extract_efa_fit(
        fa_obj = fa_model,
        dataset_name = dataset_name,
        nfactors = nf,
        n_obs = nrow(data)
      )
      
      model_list[[as.character(nf)]] <- fa_model
    }
  }
  
  list(
    fits = bind_rows(fit_list),
    models = model_list
  )
}

# Run EFA model comparisons
efa_compare_overall <- run_efa_comparison(
  data = efa_data,
  dataset_name = "overall",
  factors = 1:4
)

efa_compare_between <- run_efa_comparison(
  data = between_fa_data,
  dataset_name = "between",
  factors = 1:4
)

efa_compare_within <- run_efa_comparison(
  data = within_fa_data,
  dataset_name = "within",
  factors = 1:4
)

# ---------------------------------------------------------
# Save loadings for all 1-, 2-, 3-, and 4-factor EFA solutions
# ---------------------------------------------------------

save_all_efa_loadings <- function(efa_compare_object, dataset_name) {
  
  for (nf in names(efa_compare_object$models)) {
    
    fa_model <- efa_compare_object$models[[nf]]
    
    if (!is.null(fa_model)) {
      
      loadings_df <- as.data.frame(unclass(fa_model$loadings))
      
      loadings_df$variable <- rownames(loadings_df)
      
      loadings_df <- loadings_df[
        ,
        c("variable", setdiff(names(loadings_df), "variable"))
      ]
      
      filename <- paste0(
        "factor_loadings_",
        dataset_name,
        "_",
        nf,
        "_factor.csv"
      )
      
      write.csv(
        loadings_df,
        filename,
        row.names = FALSE
      )
    }
  }
}

save_all_efa_loadings(
  efa_compare_object = efa_compare_overall,
  dataset_name = "overall"
)

save_all_efa_loadings(
  efa_compare_object = efa_compare_between,
  dataset_name = "between"
)

save_all_efa_loadings(
  efa_compare_object = efa_compare_within,
  dataset_name = "within"
)

# Combine all fit indices into one comparison table
efa_fit_comparison <- bind_rows(
  efa_compare_overall$fits,
  efa_compare_between$fits,
  efa_compare_within$fits
)

# Rounded version for reporting
efa_fit_comparison_rounded <- efa_fit_comparison %>%
  mutate(
    across(
      c(
        chisq,
        p_value,
        TLI,
        CFI,
        RMSEA,
        RMSEA_lower,
        RMSEA_upper,
        RMSR,
        corrected_RMSR,
        BIC
      ),
      ~ round(.x, 3)
    )
  )

print(efa_fit_comparison_rounded)

write.csv(
  efa_fit_comparison,
  "efa_fit_comparison_1_to_4_factors.csv",
  row.names = FALSE
)

write.csv(
  efa_fit_comparison_rounded,
  "efa_fit_comparison_1_to_4_factors_rounded.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 9. Select 3-factor EFA solutions for loadings and scores
# ---------------------------------------------------------

fa_overall <- efa_compare_overall$models[["3"]]
fa_between <- efa_compare_between$models[["3"]]
fa_within  <- efa_compare_within$models[["3"]]

# Print selected 3-factor loadings
print(fa_overall$loadings, cutoff = 0.50)
print(fa_between$loadings, cutoff = 0.40)
print(fa_within$loadings, cutoff = 0.40)

# ---------------------------------------------------------
# 10. Save fit indices for selected 3-factor EFA solutions
# ---------------------------------------------------------

efa_fit_overall <- data.frame(
  dataset = "overall",
  nfactors = 3,
  chisq = fa_overall$STATISTIC,
  df = fa_overall$dof,
  p_value = fa_overall$PVAL,
  TLI = fa_overall$TLI,
  CFI = fa_overall$CFI,
  RMSEA = fa_overall$RMSEA[1],
  RMSEA_lower = fa_overall$RMSEA[2],
  RMSEA_upper = fa_overall$RMSEA[3],
  RMSR = fa_overall$rms,
  corrected_RMSR = fa_overall$crms,
  BIC = fa_overall$BIC
)

write.csv(
  efa_fit_overall,
  "efa_fit_overall_3_factor.csv",
  row.names = FALSE
)

efa_fit_multilevel <- data.frame(
  dataset = c("between", "within"),
  nfactors = c(3, 3),
  chisq = c(fa_between$STATISTIC, fa_within$STATISTIC),
  df = c(fa_between$dof, fa_within$dof),
  p_value = c(fa_between$PVAL, fa_within$PVAL),
  TLI = c(fa_between$TLI, fa_within$TLI),
  CFI = c(fa_between$CFI, fa_within$CFI),
  RMSEA = c(fa_between$RMSEA[1], fa_within$RMSEA[1]),
  RMSEA_lower = c(fa_between$RMSEA[2], fa_within$RMSEA[2]),
  RMSEA_upper = c(fa_between$RMSEA[3], fa_within$RMSEA[3]),
  RMSR = c(fa_between$rms, fa_within$rms),
  corrected_RMSR = c(fa_between$crms, fa_within$crms),
  BIC = c(fa_between$BIC, fa_within$BIC)
)

write.csv(
  efa_fit_multilevel,
  "efa_fit_between_within_3_factor.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 11. Save selected 3-factor loadings to CSV
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
loadings_overall_df <- loadings_overall_df[
  ,
  c("variable", setdiff(names(loadings_overall_df), "variable"))
]

loadings_between_df <- loadings_between_df[
  ,
  c("variable", setdiff(names(loadings_between_df), "variable"))
]

loadings_within_df <- loadings_within_df[
  ,
  c("variable", setdiff(names(loadings_within_df), "variable"))
]

# Save loadings
write.csv(
  loadings_overall_df,
  "factor_loadings_overall_3_factor.csv",
  row.names = FALSE
)

write.csv(
  loadings_between_df,
  "factor_loadings_between_3_factor.csv",
  row.names = FALSE
)

write.csv(
  loadings_within_df,
  "factor_loadings_within_3_factor.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 12. Save selected 3-factor scores to CSV
# ---------------------------------------------------------

# Compute factor scores
scores_overall <- factor.scores(efa_data, fa_overall)$scores
scores_between <- factor.scores(between_fa_data, fa_between)$scores
scores_within  <- factor.scores(within_fa_data, fa_within)$scores

# Overall scores
scores_overall_df <- as.data.frame(scores_overall)

write.csv(
  scores_overall_df,
  "factor_scores_overall_3_factor.csv",
  row.names = FALSE
)

# Between-student scores: attach user ID
between_complete <- between_data[complete.cases(between_data[, cols]), ]

scores_between_df <- cbind(
  user = between_complete$user,
  as.data.frame(scores_between)
)

write.csv(
  scores_between_df,
  "factor_scores_between_3_factor.csv",
  row.names = FALSE
)

# Within-student scores: attach user ID
within_complete <- within_data[complete.cases(within_data[, cols]), ]

scores_within_df <- cbind(
  user = within_complete$user,
  as.data.frame(scores_within)
)

write.csv(
  scores_within_df,
  "factor_scores_within_3_factor.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 13. Save selected 3-factor factor correlations
# ---------------------------------------------------------

if (!is.null(fa_overall$Phi)) {
  write.csv(
    as.data.frame(fa_overall$Phi),
    "factor_correlations_overall_3_factor.csv"
  )
}

if (!is.null(fa_between$Phi)) {
  write.csv(
    as.data.frame(fa_between$Phi),
    "factor_correlations_between_3_factor.csv"
  )
}

if (!is.null(fa_within$Phi)) {
  write.csv(
    as.data.frame(fa_within$Phi),
    "factor_correlations_within_3_factor.csv"
  )
}

# ---------------------------------------------------------
# 14. Save correlations among the original variables
# ---------------------------------------------------------

cor_overall <- cor(
  efa_data,
  use = "pairwise.complete.obs"
)

cor_between <- cor(
  between_fa_data,
  use = "pairwise.complete.obs"
)

cor_within <- cor(
  within_fa_data,
  use = "pairwise.complete.obs"
)

write.csv(
  as.data.frame(cor_overall),
  "variable_correlations_overall.csv",
  row.names = TRUE
)

write.csv(
  as.data.frame(cor_between),
  "variable_correlations_between.csv",
  row.names = TRUE
)

write.csv(
  as.data.frame(cor_within),
  "variable_correlations_within.csv",
  row.names = TRUE
)

# ---------------------------------------------------------
# 15. Print concise reporting table
# ---------------------------------------------------------

reporting_table <- efa_fit_comparison_rounded %>%
  select(
    dataset,
    nfactors,
    n_obs,
    chisq,
    df,
    p_value,
    TLI,
    CFI,
    RMSEA,
    RMSEA_lower,
    RMSEA_upper,
    RMSR,
    corrected_RMSR,
    BIC
  )

print(reporting_table)

write.csv(
  reporting_table,
  "efa_reporting_table_1_to_4_factors.csv",
  row.names = FALSE
)