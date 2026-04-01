# =========================================================
# Exploratory factor analysis by nth attempt within skill
# For each student-skill pair:
#   attempt 1 = first problem on that skill for that student
#   attempt 2 = second problem on that skill for that student
#   ...
#
# This script:
# 1. Loads the combined dataset
# 2. Defines attempt number within student-skill
# 3. Runs overall, between-student, and within-student EFAs
#    separately for each attempt number
# 4. Saves loadings, scores, correlations, and summaries
# 5. Computes AUC by attempt
# 6. Aligns factors across attempts and visualizes loadings
# =========================================================

# ---------------------------------------------------------
# 0. Load packages
# ---------------------------------------------------------

library(psych)
library(GPArotation)
library(dplyr)
library(rstudioapi)
library(pROC)
library(ggplot2)
library(tidyr)
library(purrr)
library(readr)
library(stringr)

# ---------------------------------------------------------
# 1. Set working directory and load data
# ---------------------------------------------------------

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

dat_raw <- read.csv("../output/combined_output.csv")

# ---------------------------------------------------------
# 2. Define prediction columns
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

required_cols <- c("user", "skill", "correct", cols)
missing_cols <- setdiff(required_cols, names(dat_raw))

if (length(missing_cols) > 0) {
  stop(
    paste(
      "Missing required columns in combined_output.csv:",
      paste(missing_cols, collapse = ", ")
    )
  )
}

# ---------------------------------------------------------
# 3. Prepare data: ensure ordering and define attempt number
# ---------------------------------------------------------

dat_raw <- dat_raw %>%
  mutate(.global_row_order = row_number()) %>%
  arrange(user, .global_row_order) %>%
  group_by(user, skill) %>%
  mutate(attempt_in_skill = row_number()) %>%
  ungroup()

# Ensure columns are numeric
dat_raw[cols] <- lapply(dat_raw[cols], function(x) as.numeric(as.character(x)))
dat_raw$correct <- as.numeric(as.character(dat_raw$correct))

# ---------------------------------------------------------
# 4. Helper function to run EFA pipeline for one attempt
# ---------------------------------------------------------

run_efa_for_attempt <- function(dat_attempt,
                                attempt_num,
                                cols,
                                nfactors = 3,
                                output_base = "efa_by_attempt") {
  
  cat("\n=====================================================\n")
  cat("Running EFA for attempt", attempt_num, "\n")
  cat("Rows in subset:", nrow(dat_attempt), "\n")
  cat("=====================================================\n")
  
  if (nrow(dat_attempt) == 0) {
    cat("No rows found for attempt", attempt_num, "- skipping.\n")
    return(NULL)
  }
  
  # Create output folder for this attempt
  attempt_dir <- file.path(output_base, paste0("attempt_", attempt_num))
  dir.create(attempt_dir, recursive = TRUE, showWarnings = FALSE)
  
  # -------------------------------------------------------
  # 4A. Overall EFA dataset
  # -------------------------------------------------------
  
  efa_data <- dat_attempt[, cols, drop = FALSE]
  efa_data[cols] <- lapply(efa_data[cols], function(x) as.numeric(as.character(x)))
  efa_data <- na.omit(efa_data)
  
  if (nrow(efa_data) < 10) {
    cat("Too few complete rows for overall EFA in attempt", attempt_num, "- skipping.\n")
    return(NULL)
  }
  
  # Save KMO and Bartlett results
  kmo_overall <- KMO(efa_data)
  bart_overall <- cortest.bartlett(cor(efa_data), n = nrow(efa_data))
  
  capture.output(kmo_overall, file = file.path(attempt_dir, "kmo_overall.txt"))
  capture.output(bart_overall, file = file.path(attempt_dir, "bartlett_overall.txt"))
  
  # Parallel analysis
  png(file.path(attempt_dir, "parallel_analysis_overall.png"), width = 1000, height = 800)
  fa.parallel(efa_data, fa = "fa")
  dev.off()
  
  # Run overall EFA
  fa_overall <- fa(
    efa_data,
    nfactors = nfactors,
    rotate = "oblimin",
    fm = "ml"
  )
  
  capture.output(
    print(fa_overall$loadings, cutoff = 0.40),
    file = file.path(attempt_dir, "loadings_overall.txt")
  )
  
  scores_overall <- factor.scores(efa_data, fa_overall)$scores
  
  # -------------------------------------------------------
  # 4B. Between-student dataset
  # -------------------------------------------------------
  
  dat_between <- dat_attempt[, c("user", cols), drop = FALSE]
  dat_between <- dat_between[!is.na(dat_between$user), ]
  
  between_data <- dat_between %>%
    group_by(user) %>%
    summarise(across(all_of(cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
  
  between_fa_data <- between_data[, cols, drop = FALSE]
  between_fa_data <- na.omit(between_fa_data)
  
  fa_between <- NULL
  scores_between <- NULL
  
  if (nrow(between_fa_data) >= 10) {
    kmo_between <- KMO(between_fa_data)
    bart_between <- cortest.bartlett(cor(between_fa_data), n = nrow(between_fa_data))
    
    capture.output(kmo_between, file = file.path(attempt_dir, "kmo_between.txt"))
    capture.output(bart_between, file = file.path(attempt_dir, "bartlett_between.txt"))
    
    png(file.path(attempt_dir, "parallel_analysis_between.png"), width = 1000, height = 800)
    fa.parallel(between_fa_data, fa = "fa")
    dev.off()
    
    fa_between <- fa(
      between_fa_data,
      nfactors = nfactors,
      rotate = "oblimin",
      fm = "ml"
    )
    
    capture.output(
      print(fa_between$loadings, cutoff = 0.40),
      file = file.path(attempt_dir, "loadings_between.txt")
    )
    
    scores_between <- factor.scores(between_fa_data, fa_between)$scores
  } else {
    cat("Too few complete rows for between-student EFA in attempt", attempt_num, "- skipping between.\n")
  }
  
  # -------------------------------------------------------
  # 4C. Within-student dataset
  # -------------------------------------------------------
  
  dat_within <- dat_attempt[, c("user", cols), drop = FALSE]
  dat_within <- dat_within[!is.na(dat_within$user), ]
  
  within_data <- dat_within %>%
    group_by(user) %>%
    mutate(across(all_of(cols), ~ .x - mean(.x, na.rm = TRUE))) %>%
    ungroup()
  
  within_fa_data <- within_data[, cols, drop = FALSE]
  within_fa_data <- na.omit(within_fa_data)
  
  fa_within <- NULL
  scores_within <- NULL
  
  if (nrow(within_fa_data) >= 10) {
    kmo_within <- KMO(within_fa_data)
    bart_within <- cortest.bartlett(cor(within_fa_data), n = nrow(within_fa_data))
    
    capture.output(kmo_within, file = file.path(attempt_dir, "kmo_within.txt"))
    capture.output(bart_within, file = file.path(attempt_dir, "bartlett_within.txt"))
    
    png(file.path(attempt_dir, "parallel_analysis_within.png"), width = 1000, height = 800)
    fa.parallel(within_fa_data, fa = "fa")
    dev.off()
    
    fa_within <- fa(
      within_fa_data,
      nfactors = nfactors,
      rotate = "oblimin",
      fm = "ml"
    )
    
    capture.output(
      print(fa_within$loadings, cutoff = 0.40),
      file = file.path(attempt_dir, "loadings_within.txt")
    )
    
    scores_within <- factor.scores(within_fa_data, fa_within)$scores
  } else {
    cat("Too few complete rows for within-student EFA in attempt", attempt_num, "- skipping within.\n")
  }
  
  # -------------------------------------------------------
  # 4D. Save loadings
  # -------------------------------------------------------
  
  loadings_overall_df <- as.data.frame(unclass(fa_overall$loadings))
  loadings_overall_df$variable <- rownames(loadings_overall_df)
  loadings_overall_df <- loadings_overall_df[, c("variable", setdiff(names(loadings_overall_df), "variable"))]
  write.csv(loadings_overall_df, file.path(attempt_dir, "factor_loadings_overall.csv"), row.names = FALSE)
  
  if (!is.null(fa_between)) {
    loadings_between_df <- as.data.frame(unclass(fa_between$loadings))
    loadings_between_df$variable <- rownames(loadings_between_df)
    loadings_between_df <- loadings_between_df[, c("variable", setdiff(names(loadings_between_df), "variable"))]
    write.csv(loadings_between_df, file.path(attempt_dir, "factor_loadings_between.csv"), row.names = FALSE)
  }
  
  if (!is.null(fa_within)) {
    loadings_within_df <- as.data.frame(unclass(fa_within$loadings))
    loadings_within_df$variable <- rownames(loadings_within_df)
    loadings_within_df <- loadings_within_df[, c("variable", setdiff(names(loadings_within_df), "variable"))]
    write.csv(loadings_within_df, file.path(attempt_dir, "factor_loadings_within.csv"), row.names = FALSE)
  }
  
  # -------------------------------------------------------
  # 4E. Save factor scores
  # -------------------------------------------------------
  
  scores_overall_df <- as.data.frame(scores_overall)
  write.csv(scores_overall_df, file.path(attempt_dir, "factor_scores_overall.csv"), row.names = FALSE)
  
  if (!is.null(scores_between)) {
    between_complete <- between_data[complete.cases(between_data[, cols]), ]
    scores_between_df <- cbind(user = between_complete$user, as.data.frame(scores_between))
    write.csv(scores_between_df, file.path(attempt_dir, "factor_scores_between.csv"), row.names = FALSE)
  }
  
  if (!is.null(scores_within)) {
    within_complete <- within_data[complete.cases(within_data[, cols]), ]
    scores_within_df <- cbind(user = within_complete$user, as.data.frame(scores_within))
    write.csv(scores_within_df, file.path(attempt_dir, "factor_scores_within.csv"), row.names = FALSE)
  }
  
  # -------------------------------------------------------
  # 4F. Save factor correlations
  # -------------------------------------------------------
  
  if (!is.null(fa_overall$Phi)) {
    write.csv(as.data.frame(fa_overall$Phi), file.path(attempt_dir, "factor_correlations_overall.csv"))
  }
  
  if (!is.null(fa_between) && !is.null(fa_between$Phi)) {
    write.csv(as.data.frame(fa_between$Phi), file.path(attempt_dir, "factor_correlations_between.csv"))
  }
  
  if (!is.null(fa_within) && !is.null(fa_within$Phi)) {
    write.csv(as.data.frame(fa_within$Phi), file.path(attempt_dir, "factor_correlations_within.csv"))
  }
  
  # -------------------------------------------------------
  # 4G. Save correlations among original variables
  # -------------------------------------------------------
  
  cor_overall <- cor(efa_data, use = "pairwise.complete.obs")
  write.csv(as.data.frame(cor_overall), file.path(attempt_dir, "variable_correlations_overall.csv"), row.names = TRUE)
  
  if (!is.null(fa_between)) {
    cor_between <- cor(between_fa_data, use = "pairwise.complete.obs")
    write.csv(as.data.frame(cor_between), file.path(attempt_dir, "variable_correlations_between.csv"), row.names = TRUE)
  }
  
  if (!is.null(fa_within)) {
    cor_within <- cor(within_fa_data, use = "pairwise.complete.obs")
    write.csv(as.data.frame(cor_within), file.path(attempt_dir, "variable_correlations_within.csv"), row.names = TRUE)
  }
  
  # -------------------------------------------------------
  # 4H. Save summary
  # -------------------------------------------------------
  
  summary_df <- data.frame(
    attempt = attempt_num,
    n_rows_raw = nrow(dat_attempt),
    n_rows_overall_complete = nrow(efa_data),
    n_rows_between_complete = ifelse(is.null(fa_between), NA, nrow(between_fa_data)),
    n_rows_within_complete = ifelse(is.null(fa_within), NA, nrow(within_fa_data)),
    n_unique_users = length(unique(dat_attempt$user))
  )
  
  write.csv(summary_df, file.path(attempt_dir, "summary.csv"), row.names = FALSE)
  
  cat("Finished attempt", attempt_num, "\n")
  
  return(list(
    fa_overall = fa_overall,
    fa_between = fa_between,
    fa_within = fa_within
  ))
}

# ---------------------------------------------------------
# 5. Run EFA for each attempt number
# ---------------------------------------------------------

dir.create("efa_by_attempt", recursive = TRUE, showWarnings = FALSE)

max_attempt <- max(dat_raw$attempt_in_skill, na.rm = TRUE)
cat("Maximum attempt number found:", max_attempt, "\n")

results <- list()

# Starts at 2 as KT models need the 1st attempt per skill
for (k in 2:min(max_attempt, 20)) {
  dat_k <- dat_raw %>%
    filter(attempt_in_skill == k)
  
  if (nrow(dat_k) == 0) next
  
  results[[paste0("attempt_", k)]] <- run_efa_for_attempt(
    dat_attempt = dat_k,
    attempt_num = k,
    cols = cols,
    nfactors = 3,
    output_base = "efa_by_attempt"
  )
}

# ---------------------------------------------------------
# 6. Save master summary across attempts
# ---------------------------------------------------------

master_summary <- dat_raw %>%
  group_by(attempt_in_skill) %>%
  summarise(
    n_rows = n(),
    n_users = n_distinct(user),
    n_skills = n_distinct(skill),
    .groups = "drop"
  )

write.csv(master_summary,
          file.path("efa_by_attempt", "master_attempt_summary.csv"),
          row.names = FALSE)

cat("\nAll attempt-specific EFAs completed.\n")

# =========================================================
# 7. Compute AUC by attempt number
# =========================================================

auc_results <- list()

for (k in 2:min(max_attempt, 20)) {
  
  dat_k <- dat_raw %>%
    filter(attempt_in_skill == k) %>%
    filter(!is.na(correct))
  
  if (nrow(dat_k) < 10) next
  
  aucs <- sapply(cols, function(col) {
    preds <- dat_k[[col]]
    truth <- dat_k$correct
    
    valid <- complete.cases(preds, truth)
    preds <- preds[valid]
    truth <- truth[valid]
    
    if (length(unique(truth)) < 2) return(NA)
    
    tryCatch({
      as.numeric(auc(truth, preds))
    }, error = function(e) NA)
  })
  
  auc_results[[as.character(k)]] <- data.frame(
    model = cols,
    auc = aucs,
    attempt = k
  )
}

auc_df <- bind_rows(auc_results)

auc_df <- auc_df %>%
  mutate(model_clean = gsub("_prediction", "", model)) %>%
  mutate(model_clean = recode(
    model_clean,
    "bkt_bf" = "BKT-BF",
    "BKT_forgetting" = "BKT-Forg",
    "PFA" = "PFA",
    "ELO" = "ELO",
    "KTM" = "KTM",
    "DKT" = "DKT",
    "DSAKT" = "DSAKT",
    "ATKT" = "ATKT"
  ))

auc_df$model_clean <- factor(
  auc_df$model_clean,
  levels = c("BKT-BF", "BKT-Forg", "PFA", "ELO", "KTM", "DKT", "DSAKT", "ATKT")
)

write.csv(auc_df, file.path("efa_by_attempt", "auc_by_attempt.csv"), row.names = FALSE)

p_auc_lines <- ggplot(
  auc_df,
  aes(x = attempt, y = auc, color = model_clean, group = model_clean)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  coord_cartesian(ylim = c(0.5, 1)) +
  theme_minimal(base_size = 14) +
  labs(
    title = "AUC across previous problems within skill",
    x = "Previous problems within skill",
    y = "AUC",
    color = "Model"
  )

print(p_auc_lines)

ggsave(
  file.path("efa_by_attempt", "auc_by_attempt_lines.png"),
  p_auc_lines,
  width = 10,
  height = 6,
  dpi = 300
)

# =========================================================
# 8. Align and visualize EFA loadings across attempts
# =========================================================

# ---------------------------------------------------------
# 8A. Settings for visualization
# ---------------------------------------------------------

attempts <- 2:min(max_attempt, 20)
base_dir <- "efa_by_attempt"

# choose one of: "overall", "between", "within"
loading_type <- "overall"

anchor_vars <- c(
  BKT = "bkt_bf_prediction",
  DKT = "DKT_prediction",
  ELO = "ELO_prediction"
)

# ---------------------------------------------------------
# 8B. Helper: read one attempt's loading file
# ---------------------------------------------------------

read_attempt_loadings <- function(attempt_num, type = "overall") {
  f <- file.path(
    base_dir,
    paste0("attempt_", attempt_num),
    paste0("factor_loadings_", type, ".csv")
  )
  
  if (!file.exists(f)) {
    message("File not found: ", f)
    return(NULL)
  }
  
  df <- read_csv(f, show_col_types = FALSE)
  
  if (!"variable" %in% names(df)) {
    stop("The file ", f, " does not contain a 'variable' column.")
  }
  
  df$attempt <- attempt_num
  df
}

# ---------------------------------------------------------
# 8C. Read all attempts
# ---------------------------------------------------------

all_loadings <- map_dfr(attempts, read_attempt_loadings, type = loading_type)

if (nrow(all_loadings) == 0) {
  stop("No loading files were found.")
}

factor_cols <- setdiff(names(all_loadings), c("variable", "attempt"))

if (length(factor_cols) == 0) {
  stop("No factor columns found in the loading files.")
}

# ---------------------------------------------------------
# 8D. Reshape to long format
# ---------------------------------------------------------

loadings_long <- all_loadings %>%
  pivot_longer(
    cols = all_of(factor_cols),
    names_to = "factor",
    values_to = "loading"
  )

# ---------------------------------------------------------
# 8E. Align factors across attempts using anchors
# ---------------------------------------------------------

aligned_list <- list()

for (att in attempts) {
  
  dat_att <- loadings_long %>% filter(attempt == att)
  if (nrow(dat_att) == 0) next
  
  anchor_table <- bind_rows(lapply(names(anchor_vars), function(label) {
    anchor <- anchor_vars[[label]]
    
    dat_att %>%
      filter(variable == anchor) %>%
      mutate(
        aligned_factor = label,
        anchor_variable = anchor,
        abs_loading = abs(loading)
      )
  }))
  
  if (nrow(anchor_table) == 0) next
  
  assignments <- data.frame()
  remaining_factors <- unique(anchor_table$factor)
  remaining_labels <- names(anchor_vars)
  
  for (i in seq_along(remaining_labels)) {
    best_row <- anchor_table %>%
      filter(
        aligned_factor %in% remaining_labels,
        factor %in% remaining_factors
      ) %>%
      arrange(desc(abs_loading)) %>%
      slice(1)
    
    if (nrow(best_row) == 0) break
    
    assignments <- bind_rows(assignments, best_row)
    remaining_labels <- setdiff(remaining_labels, best_row$aligned_factor)
    remaining_factors <- setdiff(remaining_factors, best_row$factor)
  }
  
  if (nrow(assignments) == 0) next
  
  att_results <- list()
  
  for (i in 1:nrow(assignments)) {
    chosen_factor  <- assignments$factor[i]
    label          <- assignments$aligned_factor[i]
    anchor         <- assignments$anchor_variable[i]
    chosen_loading <- assignments$loading[i]
    
    sign_flip <- ifelse(chosen_loading < 0, -1, 1)
    
    factor_df <- dat_att %>%
      filter(factor == chosen_factor) %>%
      mutate(
        aligned_factor = label,
        aligned_loading = sign_flip * .data$loading,
        anchor_variable = anchor,
        original_factor = chosen_factor
      )
    
    att_results[[label]] <- factor_df
  }
  
  aligned_list[[as.character(att)]] <- bind_rows(att_results)
}

aligned_loadings <- bind_rows(aligned_list)

if (nrow(aligned_loadings) == 0) {
  stop("No aligned loadings could be created.")
}

# ---------------------------------------------------------
# 8F. Save and inspect factor mapping
# ---------------------------------------------------------

factor_mapping_summary <- aligned_loadings %>%
  distinct(attempt, aligned_factor, original_factor, anchor_variable) %>%
  arrange(attempt, aligned_factor)

write_csv(
  factor_mapping_summary,
  file.path(base_dir, paste0("factor_mapping_summary_", loading_type, ".csv"))
)

print(factor_mapping_summary, n = 100)

duplicate_assignments <- factor_mapping_summary %>%
  count(attempt, original_factor) %>%
  filter(n > 1)

if (nrow(duplicate_assignments) > 0) {
  message("Warning: some original factors were assigned more than once.")
  print(duplicate_assignments)
} else {
  message("Factor assignment check passed: no duplicated factor assignments within attempts.")
}

# ---------------------------------------------------------
# 8G. Clean model names for plotting
# ---------------------------------------------------------

aligned_loadings <- aligned_loadings %>%
  mutate(variable_clean = gsub("_prediction", "", variable)) %>%
  mutate(variable_clean = recode(
    variable_clean,
    "bkt_bf" = "BKT-BF",
    "BKT_forgetting" = "BKT-Forg",
    "PFA" = "PFA",
    "ELO" = "ELO",
    "KTM" = "KTM",
    "DKT" = "DKT",
    "DSAKT" = "DSAKT",
    "ATKT" = "ATKT"
  )) %>%
  mutate(aligned_factor_label = recode(
    aligned_factor,
    "BKT" = "Factor 1: BKT / PFA",
    "DKT" = "Factor 2: DKT / DSAKT / ATKT",
    "ELO" = "Factor 3: ELO / KTM"
  ))

aligned_loadings$variable_clean <- factor(
  aligned_loadings$variable_clean,
  levels = c("BKT-BF", "BKT-Forg", "PFA", "ELO", "KTM", "DKT", "DSAKT", "ATKT")
)

aligned_loadings$aligned_factor_label <- factor(
  aligned_loadings$aligned_factor_label,
  levels = c(
    "Factor 1: BKT / PFA",
    "Factor 2: DKT / DSAKT / ATKT",
    "Factor 3: ELO / KTM"
  )
)

write_csv(
  aligned_loadings,
  file.path(base_dir, paste0("aligned_loadings_", loading_type, ".csv"))
)

# ---------------------------------------------------------
# 8H. Line plot: evolution of loadings
# ---------------------------------------------------------

p_lines <- ggplot(
  aligned_loadings,
  aes(
    x = attempt,
    y = aligned_loading,
    color = variable_clean,
    group = variable_clean
  )
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_wrap(~ aligned_factor_label, ncol = 1) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Evolution of factor loadings across problems solved",
    x = "Previous problems within skill",
    y = "Aligned loadings",
    color = "Model"
  )

print(p_lines)

ggsave(
  file.path(base_dir, "loadings_evolution_lines.png"),
  p_lines,
  width = 10,
  height = 12,
  dpi = 300
)

# ---------------------------------------------------------
# 8I. Heatmap
# ---------------------------------------------------------

p_heatmap <- ggplot(
  aligned_loadings,
  aes(
    x = attempt,
    y = variable_clean,
    fill = aligned_loading
  )
) +
  geom_tile() +
  facet_wrap(~ aligned_factor_label, ncol = 1) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Heatmap of factor loadings across problems solved",
    x = "Previous problems within skill",
    y = "Model",
    fill = "Loading"
  )

print(p_heatmap)

ggsave(
  file.path(base_dir, "loadings_evolution_heatmap.png"),
  p_heatmap,
  width = 10,
  height = 12,
  dpi = 300
)

cat("\nMerged attempt analysis completed. Files saved in:", base_dir, "\n")