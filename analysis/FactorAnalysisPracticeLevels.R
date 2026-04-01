# =========================================================
# Exploratory factor analysis by skill-level average attempts
# Quartile version: group skills by quartiles of the average
# number of attempts per student-skill pair, then run EFA
# separately for each quartile.
#
# This script:
# 1. Loads the combined dataset
# 2. Computes avg attempts per skill across students
# 3. Splits skills into 4 quartile groups
# 4. Runs overall, between-student, and within-student EFAs
#    separately for each quartile
# 5. Saves loadings, scores, correlations, and summaries
# 6. Aligns factors across quartiles and visualizes loadings
# =========================================================

# ---------------------------------------------------------
# 0. Load packages
# ---------------------------------------------------------

library(psych)
library(GPArotation)
library(dplyr)
library(rstudioapi)
library(tidyr)
library(ggplot2)
library(purrr)
library(readr)
library(stringr)
library(pROC)

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

required_cols <- c("user", "skill", cols)
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
# 3. Ensure prediction columns are numeric
# ---------------------------------------------------------

dat_raw[cols] <- lapply(dat_raw[cols], function(x) as.numeric(as.character(x)))

# Remove rows with missing user or skill
dat_raw <- dat_raw %>%
  filter(!is.na(user), !is.na(skill))

# ---------------------------------------------------------
# 4. Compute skill-level average number of attempts
# ---------------------------------------------------------
# Step 1: count number of rows for each (user, skill) pair
# Step 2: average those counts across students for each skill
# ---------------------------------------------------------

skill_attempts <- dat_raw %>%
  group_by(user, skill) %>%
  summarise(n_attempts = n(), .groups = "drop")

skill_summary <- skill_attempts %>%
  group_by(skill) %>%
  summarise(
    avg_attempts = mean(n_attempts, na.rm = TRUE),
    n_students = n(),
    total_attempts = sum(n_attempts, na.rm = TRUE),
    .groups = "drop"
  )

# ---------------------------------------------------------
# 5. Create quartile-based skill groups
# ---------------------------------------------------------

skill_summary <- skill_summary %>%
  mutate(
    attempt_group_num = ntile(avg_attempts, 4),
    attempt_group = recode(
      as.character(attempt_group_num),
      "1" = "Q1_lowest_attempts",
      "2" = "Q2_lower_mid_attempts",
      "3" = "Q3_upper_mid_attempts",
      "4" = "Q4_highest_attempts"
    )
  )

# Merge group labels back to the original data
dat_grouped <- dat_raw %>%
  left_join(skill_summary, by = "skill")

# Save skill summary
dir.create("efa_by_skill_attempt_quartile", recursive = TRUE, showWarnings = FALSE)
write.csv(
  skill_summary,
  "efa_by_skill_attempt_quartile/skill_attempt_summary.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 6. Helper function to run EFA pipeline for one subset
# ---------------------------------------------------------

run_efa_for_group <- function(dat_subset,
                              group_label,
                              cols,
                              nfactors = 3,
                              output_base = "efa_by_skill_attempt_quartile") {
  
  cat("\n=====================================================\n")
  cat("Running EFA for group:", group_label, "\n")
  cat("Rows in subset:", nrow(dat_subset), "\n")
  cat("=====================================================\n")
  
  if (nrow(dat_subset) == 0) {
    cat("No rows found for group", group_label, "- skipping.\n")
    return(NULL)
  }
  
  # Create output folder
  group_dir <- file.path(output_base, group_label)
  dir.create(group_dir, recursive = TRUE, showWarnings = FALSE)
  
  # -------------------------------------------------------
  # 6A. Overall EFA dataset
  # -------------------------------------------------------
  
  efa_data <- dat_subset[, cols, drop = FALSE]
  efa_data[cols] <- lapply(efa_data[cols], function(x) as.numeric(as.character(x)))
  efa_data <- na.omit(efa_data)
  
  if (nrow(efa_data) < 10) {
    cat("Too few complete rows for overall EFA in group", group_label, "- skipping.\n")
    return(NULL)
  }
  
  kmo_overall <- KMO(efa_data)
  bart_overall <- cortest.bartlett(cor(efa_data), n = nrow(efa_data))
  
  capture.output(
    kmo_overall,
    file = file.path(group_dir, "kmo_overall.txt")
  )
  capture.output(
    bart_overall,
    file = file.path(group_dir, "bartlett_overall.txt")
  )
  
  png(
    file.path(group_dir, "parallel_analysis_overall.png"),
    width = 1000,
    height = 800
  )
  fa.parallel(efa_data, fa = "fa")
  dev.off()
  
  fa_overall <- fa(
    efa_data,
    nfactors = nfactors,
    rotate = "oblimin",
    fm = "ml"
  )
  
  capture.output(
    print(fa_overall$loadings, cutoff = 0.40),
    file = file.path(group_dir, "loadings_overall.txt")
  )
  
  scores_overall <- factor.scores(efa_data, fa_overall)$scores
  
  # -------------------------------------------------------
  # 6B. Between-student dataset
  # -------------------------------------------------------
  
  dat_between <- dat_subset[, c("user", cols), drop = FALSE]
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
    
    capture.output(
      kmo_between,
      file = file.path(group_dir, "kmo_between.txt")
    )
    capture.output(
      bart_between,
      file = file.path(group_dir, "bartlett_between.txt")
    )
    
    png(
      file.path(group_dir, "parallel_analysis_between.png"),
      width = 1000,
      height = 800
    )
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
      file = file.path(group_dir, "loadings_between.txt")
    )
    
    scores_between <- factor.scores(between_fa_data, fa_between)$scores
  } else {
    cat("Too few complete rows for between-student EFA in group", group_label, "- skipping between.\n")
  }
  
  # -------------------------------------------------------
  # 6C. Within-student dataset
  # -------------------------------------------------------
  
  dat_within <- dat_subset[, c("user", cols), drop = FALSE]
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
    
    capture.output(
      kmo_within,
      file = file.path(group_dir, "kmo_within.txt")
    )
    capture.output(
      bart_within,
      file = file.path(group_dir, "bartlett_within.txt")
    )
    
    png(
      file.path(group_dir, "parallel_analysis_within.png"),
      width = 1000,
      height = 800
    )
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
      file = file.path(group_dir, "loadings_within.txt")
    )
    
    scores_within <- factor.scores(within_fa_data, fa_within)$scores
  } else {
    cat("Too few complete rows for within-student EFA in group", group_label, "- skipping within.\n")
  }
  
  # -------------------------------------------------------
  # 6D. Save loadings
  # -------------------------------------------------------
  
  loadings_overall_df <- as.data.frame(unclass(fa_overall$loadings))
  loadings_overall_df$variable <- rownames(loadings_overall_df)
  loadings_overall_df <- loadings_overall_df[, c("variable", setdiff(names(loadings_overall_df), "variable"))]
  write.csv(
    loadings_overall_df,
    file.path(group_dir, "factor_loadings_overall.csv"),
    row.names = FALSE
  )
  
  if (!is.null(fa_between)) {
    loadings_between_df <- as.data.frame(unclass(fa_between$loadings))
    loadings_between_df$variable <- rownames(loadings_between_df)
    loadings_between_df <- loadings_between_df[, c("variable", setdiff(names(loadings_between_df), "variable"))]
    write.csv(
      loadings_between_df,
      file.path(group_dir, "factor_loadings_between.csv"),
      row.names = FALSE
    )
  }
  
  if (!is.null(fa_within)) {
    loadings_within_df <- as.data.frame(unclass(fa_within$loadings))
    loadings_within_df$variable <- rownames(loadings_within_df)
    loadings_within_df <- loadings_within_df[, c("variable", setdiff(names(loadings_within_df), "variable"))]
    write.csv(
      loadings_within_df,
      file.path(group_dir, "factor_loadings_within.csv"),
      row.names = FALSE
    )
  }
  
  # -------------------------------------------------------
  # 6E. Save factor scores
  # -------------------------------------------------------
  
  scores_overall_df <- as.data.frame(scores_overall)
  write.csv(
    scores_overall_df,
    file.path(group_dir, "factor_scores_overall.csv"),
    row.names = FALSE
  )
  
  if (!is.null(scores_between)) {
    between_complete <- between_data[complete.cases(between_data[, cols]), ]
    scores_between_df <- cbind(
      user = between_complete$user,
      as.data.frame(scores_between)
    )
    write.csv(
      scores_between_df,
      file.path(group_dir, "factor_scores_between.csv"),
      row.names = FALSE
    )
  }
  
  if (!is.null(scores_within)) {
    within_complete <- within_data[complete.cases(within_data[, cols]), ]
    scores_within_df <- cbind(
      user = within_complete$user,
      as.data.frame(scores_within)
    )
    write.csv(
      scores_within_df,
      file.path(group_dir, "factor_scores_within.csv"),
      row.names = FALSE
    )
  }
  
  # -------------------------------------------------------
  # 6F. Save factor correlations
  # -------------------------------------------------------
  
  if (!is.null(fa_overall$Phi)) {
    write.csv(
      as.data.frame(fa_overall$Phi),
      file.path(group_dir, "factor_correlations_overall.csv")
    )
  }
  
  if (!is.null(fa_between) && !is.null(fa_between$Phi)) {
    write.csv(
      as.data.frame(fa_between$Phi),
      file.path(group_dir, "factor_correlations_between.csv")
    )
  }
  
  if (!is.null(fa_within) && !is.null(fa_within$Phi)) {
    write.csv(
      as.data.frame(fa_within$Phi),
      file.path(group_dir, "factor_correlations_within.csv")
    )
  }
  
  # -------------------------------------------------------
  # 6G. Save correlations among original variables
  # -------------------------------------------------------
  
  cor_overall <- cor(efa_data, use = "pairwise.complete.obs")
  write.csv(
    as.data.frame(cor_overall),
    file.path(group_dir, "variable_correlations_overall.csv"),
    row.names = TRUE
  )
  
  if (!is.null(fa_between)) {
    cor_between <- cor(between_fa_data, use = "pairwise.complete.obs")
    write.csv(
      as.data.frame(cor_between),
      file.path(group_dir, "variable_correlations_between.csv"),
      row.names = TRUE
    )
  }
  
  if (!is.null(fa_within)) {
    cor_within <- cor(within_fa_data, use = "pairwise.complete.obs")
    write.csv(
      as.data.frame(cor_within),
      file.path(group_dir, "variable_correlations_within.csv"),
      row.names = TRUE
    )
  }
  
  # -------------------------------------------------------
  # 6H. Save summary file
  # -------------------------------------------------------
  
  summary_df <- data.frame(
    group = group_label,
    n_rows_raw = nrow(dat_subset),
    n_rows_overall_complete = nrow(efa_data),
    n_rows_between_complete = ifelse(is.null(fa_between), NA, nrow(between_fa_data)),
    n_rows_within_complete = ifelse(is.null(fa_within), NA, nrow(within_fa_data)),
    n_unique_users = length(unique(dat_subset$user)),
    n_unique_skills = length(unique(dat_subset$skill))
  )
  
  write.csv(
    summary_df,
    file.path(group_dir, "summary.csv"),
    row.names = FALSE
  )
  
  cat("Finished group", group_label, "\n")
  
  return(list(
    fa_overall = fa_overall,
    fa_between = fa_between,
    fa_within = fa_within
  ))
}

# ---------------------------------------------------------
# 7. Run EFAs by quartile group
# ---------------------------------------------------------

group_levels <- c(
  "Q1_lowest_attempts",
  "Q2_lower_mid_attempts",
  "Q3_upper_mid_attempts",
  "Q4_highest_attempts"
)

results <- list()

for (g in group_levels) {
  dat_g <- dat_grouped %>%
    filter(attempt_group == g)
  
  if (nrow(dat_g) == 0) next
  
  results[[g]] <- run_efa_for_group(
    dat_subset = dat_g,
    group_label = g,
    cols = cols,
    nfactors = 3,
    output_base = "efa_by_skill_attempt_quartile"
  )
}

# ---------------------------------------------------------
# 8. Save master summary across quartiles
# ---------------------------------------------------------

master_summary <- dat_grouped %>%
  group_by(attempt_group) %>%
  summarise(
    n_rows = n(),
    n_users = n_distinct(user),
    n_skills = n_distinct(skill),
    mean_skill_avg_attempts = mean(avg_attempts, na.rm = TRUE),
    min_skill_avg_attempts = min(avg_attempts, na.rm = TRUE),
    max_skill_avg_attempts = max(avg_attempts, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  master_summary,
  "efa_by_skill_attempt_quartile/master_group_summary.csv",
  row.names = FALSE
)

cat("\nAll quartile-group EFAs completed.\n")

# =========================================================
# 9. Align and visualize loadings across quartile groups
# =========================================================

# ---------------------------------------------------------
# 9A. Settings for visualization
# ---------------------------------------------------------

base_dir <- "efa_by_skill_attempt_quartile"

# choose one of: "overall", "between", "within"
loading_type <- "overall"

anchor_vars <- c(
  BKT = "bkt_bf_prediction",
  DKT = "DKT_prediction",
  ELO = "ELO_prediction"
)

group_order <- c(
  "Q1_lowest_attempts",
  "Q2_lower_mid_attempts",
  "Q3_upper_mid_attempts",
  "Q4_highest_attempts"
)

# ---------------------------------------------------------
# 9B. Helper: read one group's loading file
# ---------------------------------------------------------

read_group_loadings <- function(group_label, type = "overall") {
  f <- file.path(
    base_dir,
    group_label,
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
  
  df$group_label <- group_label
  df
}

# ---------------------------------------------------------
# 9C. Read all groups
# ---------------------------------------------------------

all_loadings <- map_dfr(group_order, read_group_loadings, type = loading_type)

if (nrow(all_loadings) == 0) {
  stop("No loading files were found for visualization.")
}

factor_cols <- setdiff(names(all_loadings), c("variable", "group_label"))

if (length(factor_cols) == 0) {
  stop("No factor columns found in the group loading files.")
}

# ---------------------------------------------------------
# 9D. Reshape to long format
# ---------------------------------------------------------

loadings_long <- all_loadings %>%
  pivot_longer(
    cols = all_of(factor_cols),
    names_to = "factor",
    values_to = "loading"
  )

# ---------------------------------------------------------
# 9E. Align factors across groups using anchors
# ---------------------------------------------------------

aligned_list <- list()

for (grp in group_order) {
  
  dat_grp <- loadings_long %>% filter(group_label == grp)
  if (nrow(dat_grp) == 0) next
  
  anchor_table <- bind_rows(lapply(names(anchor_vars), function(label) {
    anchor <- anchor_vars[[label]]
    
    dat_grp %>%
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
  
  grp_results <- list()
  
  for (i in 1:nrow(assignments)) {
    chosen_factor  <- assignments$factor[i]
    label          <- assignments$aligned_factor[i]
    anchor         <- assignments$anchor_variable[i]
    chosen_loading <- assignments$loading[i]
    
    sign_flip <- ifelse(chosen_loading < 0, -1, 1)
    
    factor_df <- dat_grp %>%
      filter(factor == chosen_factor) %>%
      mutate(
        aligned_factor = label,
        aligned_loading = sign_flip * .data$loading,
        anchor_variable = anchor,
        original_factor = chosen_factor
      )
    
    grp_results[[label]] <- factor_df
  }
  
  aligned_list[[grp]] <- bind_rows(grp_results)
}

aligned_loadings <- bind_rows(aligned_list)

if (nrow(aligned_loadings) == 0) {
  stop("No aligned loadings could be created for the visualization step.")
}

# ---------------------------------------------------------
# 9F. Save and inspect factor mapping
# ---------------------------------------------------------

factor_mapping_summary <- aligned_loadings %>%
  distinct(group_label, aligned_factor, original_factor, anchor_variable) %>%
  arrange(group_label, aligned_factor)

write_csv(
  factor_mapping_summary,
  file.path(base_dir, paste0("factor_mapping_summary_", loading_type, ".csv"))
)

print(factor_mapping_summary, n = 100)

# ---------------------------------------------------------
# 9G. Clean labels for plotting
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
  mutate(group_clean = recode(
    group_label,
    "Q1_lowest_attempts" = "Q1: Least-practiced skills",
    "Q2_lower_mid_attempts" = "Q2: Lower-practice skills",
    "Q3_upper_mid_attempts" = "Q3: Higher-practice skills",
    "Q4_highest_attempts" = "Q4: Most-practiced skills"
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

aligned_loadings$group_clean <- factor(
  aligned_loadings$group_clean,
  levels = c(
    "Q1: Least-practiced skills",
    "Q2: Lower-practice skills",
    "Q3: Higher-practice skills",
    "Q4: Most-practiced skills"
  )
)

write_csv(
  aligned_loadings,
  file.path(base_dir, paste0("aligned_loadings_", loading_type, ".csv"))
)

# ---------------------------------------------------------
# 9H. Line plot comparing quartiles
# ---------------------------------------------------------

p_lines <- ggplot(
  aligned_loadings,
  aes(
    x = group_clean,
    y = aligned_loading,
    color = variable_clean,
    group = variable_clean
  )
) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~ aligned_factor_label, ncol = 1) +
  theme_minimal(base_size = 14) +
  labs(
    title = paste("Factor loadings by skill-attempt quartile -", str_to_title(loading_type)),
    x = "Skill practice level (quartiles)",
    y = "Aligned loading",
    color = "Model"
  )

print(p_lines)

ggsave(
  filename = file.path(base_dir, paste0("loadings_by_quartile_lines_", loading_type, ".png")),
  plot = p_lines,
  width = 11,
  height = 12,
  dpi = 300
)

# ---------------------------------------------------------
# 9I. Heatmap comparing quartiles
# ---------------------------------------------------------

p_heatmap <- ggplot(
  aligned_loadings,
  aes(
    x = group_clean,
    y = variable_clean,
    fill = aligned_loading
  )
) +
  geom_tile() +
  facet_wrap(~ aligned_factor_label, ncol = 1) +
  theme_minimal(base_size = 14) +
  labs(
    title = paste("Heatmap of factor loadings by skill-attempt quartile -", str_to_title(loading_type)),
    x = "Skill practice level (quartiles)",
    y = "Model",
    fill = "Loading"
  )

print(p_heatmap)

ggsave(
  filename = file.path(base_dir, paste0("loadings_by_quartile_heatmap_", loading_type, ".png")),
  plot = p_heatmap,
  width = 11,
  height = 12,
  dpi = 300
)

cat("\nVisualization completed. Files saved in:", base_dir, "\n")

# ---------------------------------------------------------
# Compute AUCs by quartile group
# ---------------------------------------------------------

auc_results <- list()

for (g in unique(dat_grouped$attempt_group)) {
  
  dat_g <- dat_grouped %>%
    filter(attempt_group == g) %>%
    filter(!is.na(correct))
  
  if (nrow(dat_g) < 10) next
  
  aucs <- sapply(cols, function(col) {
    
    preds <- dat_g[[col]]
    truth <- dat_g$correct
    
    # Remove NA pairs
    valid <- complete.cases(preds, truth)
    preds <- preds[valid]
    truth <- truth[valid]
    
    # Need both classes present
    if (length(unique(truth)) < 2) return(NA)
    
    tryCatch({
      as.numeric(auc(truth, preds))
    }, error = function(e) NA)
  })
  
  auc_results[[g]] <- data.frame(
    model = cols,
    auc = aucs,
    attempt_group = g
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
  )) %>%
  mutate(group_clean = recode(
    attempt_group,
    "Q1_lowest_attempts" = "Q1: Least-practiced skills",
    "Q2_lower_mid_attempts" = "Q2: Lower-practice skills",
    "Q3_upper_mid_attempts" = "Q3: Higher-practice skills",
    "Q4_highest_attempts" = "Q4: Most-practiced skills"
  ))

auc_df$model_clean <- factor(
  auc_df$model_clean,
  levels = c("BKT-BF", "BKT-Forg", "PFA", "ELO", "KTM", "DKT", "DSAKT", "ATKT")
)

auc_df$group_clean <- factor(
  auc_df$group_clean,
  levels = c(
    "Q1: Least-practiced skills",
    "Q2: Lower-practice skills",
    "Q3: Higher-practice skills",
    "Q4: Most-practiced skills"
  )
)

p_auc_lines <- ggplot(auc_df,
                      aes(x = group_clean, y = auc,
                          color = model_clean, group = model_clean)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  coord_cartesian(ylim = c(0.5, 1)) +   
  theme_minimal(base_size = 14) +
  labs(
    title = "AUC across skill practice levels",
    x = "Skill practice level (quartiles)",
    y = "AUC",
    color = "Model"
  )

print(p_auc_lines)

ggsave("efa_by_skill_attempt_quartile/auc_by_quartile_lines.png",
       p_auc_lines, width = 10, height = 6, dpi = 300)