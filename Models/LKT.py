# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 08:46:58 2026

@author: azamb

"""

import os
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegressionCV


# ============================================================
# Basic utilities
# ============================================================

def _sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _required_lkt_feature_cols(decay_values=(0.1, 0.5, 0.8)):
    required = [
        "user_id",
        "skill_id",
        "item",
        "correct",
        "row_id",

        "b4_correct",
        "b4_incorrect",

        "problem_b4_correct",
        "problem_b4_incorrect",

        "kc_trial_gap",
        "kc_log_trial_gap",
        "kc_recency",

        "problem_trial_gap",
        "problem_log_trial_gap",
        "problem_recency",
    ]

    for decay in decay_values:
        required.extend([
            f"kc_decayed_success_{decay}",
            f"kc_decayed_failure_{decay}",
            f"problem_decayed_success_{decay}",
            f"problem_decayed_failure_{decay}",
        ])

    return required


def _has_precomputed_lkt_features(df, decay_values=(0.1, 0.5, 0.8)):
    required = _required_lkt_feature_cols(decay_values)
    return all(col in df.columns for col in required)


def _default_lkt_cache_path(path):
    """
    Given an original tabular fold file path, create a safe cached LKT path.

    Example:
        data/processed/train/tabular/converted_fold_0.csv

    becomes:
        data/processed/train/tabular_lkt_auto/converted_fold_0_lkt.csv

    This avoids saving cached files inside the original tabular folder,
    because your experiment script lists all .csv files in that folder.
    """

    path = os.path.normpath(path)
    folder = os.path.dirname(path)
    filename = os.path.basename(path)

    name, ext = os.path.splitext(filename)

    parent = os.path.dirname(folder)
    folder_name = os.path.basename(folder)

    cache_folder = os.path.join(parent, f"{folder_name}_lkt_auto")
    cache_filename = f"{name}_lkt.csv"

    return os.path.join(cache_folder, cache_filename)


# ============================================================
# Feature standardization / precomputation
# ============================================================

def _standardize_columns(
    df: pd.DataFrame,
    decay_values=(0.1, 0.5, 0.8),
) -> pd.DataFrame:
    """
    Standardize KT data.

    Supports:
        user, skill, item, correct

    or:
        user_id, skill_id, item, correct

    Adds:
        KC-level prior counts
        problem-level prior counts
        sequence-based spacing/recency proxies
        decayed success/failure proxies
    """

    df = df.copy()

    if "user" in df.columns:
        user_col = "user"
    elif "user_id" in df.columns:
        user_col = "user_id"
    else:
        raise ValueError(f"No user column found. Columns: {list(df.columns)}")

    if "skill" in df.columns:
        skill_col = "skill"
    elif "skill_id" in df.columns:
        skill_col = "skill_id"
    else:
        raise ValueError(f"No skill column found. Columns: {list(df.columns)}")

    if "item" in df.columns:
        item_col = "item"
    elif "item_id" in df.columns:
        item_col = "item_id"
    else:
        raise ValueError(f"No item column found. Columns: {list(df.columns)}")

    if "correct" not in df.columns:
        raise ValueError(f"No correct column found. Columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["user_id"] = df[user_col].astype(str)
    out["skill_id"] = df[skill_col].astype(str)
    out["item"] = df[item_col].astype(str)
    out["correct"] = df["correct"].astype(int)

    if "row_id" in df.columns:
        out["row_id"] = df["row_id"].astype(float)
    else:
        out["row_id"] = np.arange(len(df), dtype=float)

    if "b4_correct" in df.columns and "b4_incorrect" in df.columns:
        out["b4_correct"] = df["b4_correct"].astype(float)
        out["b4_incorrect"] = df["b4_incorrect"].astype(float)
    else:
        out = _add_prior_counts(
            out,
            group_cols=("user_id", "skill_id"),
            correct_col="b4_correct",
            incorrect_col="b4_incorrect",
        )

    out = _add_prior_counts(
        out,
        group_cols=("user_id", "item"),
        correct_col="problem_b4_correct",
        incorrect_col="problem_b4_incorrect",
    )

    out = _add_sequence_spacing_features(
        out,
        group_cols=("user_id", "skill_id"),
        prefix="kc",
    )

    out = _add_sequence_spacing_features(
        out,
        group_cols=("user_id", "item"),
        prefix="problem",
    )

    out = _add_decayed_counts(
        out,
        group_cols=("user_id", "skill_id"),
        prefix="kc",
        decay_values=decay_values,
    )

    out = _add_decayed_counts(
        out,
        group_cols=("user_id", "item"),
        prefix="problem",
        decay_values=decay_values,
    )

    return out


def _add_prior_counts(
    df: pd.DataFrame,
    group_cols,
    correct_col: str,
    incorrect_col: str,
) -> pd.DataFrame:
    """
    Compute prior correct/incorrect counts by arbitrary grouping.
    Preserves current row order.
    """

    df = df.copy()

    counts = defaultdict(lambda: {"correct": 0, "incorrect": 0})

    prior_correct = []
    prior_incorrect = []

    for _, row in df.iterrows():
        key = tuple(row[col] for col in group_cols)

        prior_correct.append(counts[key]["correct"])
        prior_incorrect.append(counts[key]["incorrect"])

        if int(row["correct"]) == 1:
            counts[key]["correct"] += 1
        else:
            counts[key]["incorrect"] += 1

    df[correct_col] = prior_correct
    df[incorrect_col] = prior_incorrect

    return df


def _add_sequence_spacing_features(
    df: pd.DataFrame,
    group_cols,
    prefix: str,
) -> pd.DataFrame:
    """
    Adds sequence-based spacing/forgetting proxies.

    These are not real timestamp features.
    They use row/order distance as a proxy for spacing.
    """

    df = df.copy()

    last_seen = {}
    gaps = []

    for _, row in df.iterrows():
        key = tuple(row[col] for col in group_cols)
        current_pos = float(row["row_id"])

        if key in last_seen:
            gap = current_pos - last_seen[key]
            if gap < 1:
                gap = 1.0
        else:
            gap = np.nan

        gaps.append(gap)
        last_seen[key] = current_pos

    gap_arr = np.array(gaps, dtype=float)

    if np.all(np.isnan(gap_arr)):
        fill_gap = 1.0
    else:
        fill_gap = np.nanmedian(gap_arr)

    gap_arr = np.where(np.isnan(gap_arr), fill_gap, gap_arr)

    df[f"{prefix}_trial_gap"] = gap_arr
    df[f"{prefix}_log_trial_gap"] = np.log1p(gap_arr)
    df[f"{prefix}_recency"] = 1.0 / (1.0 + gap_arr)

    return df


def _add_decayed_counts(
    df: pd.DataFrame,
    group_cols,
    prefix: str,
    decay_values=(0.1, 0.5, 0.8),
) -> pd.DataFrame:
    """
    Adds opportunity-distance decayed success/failure counts.

    For each grouping:
        decayed_success = decay^gap * old_success + previous_correct
        decayed_failure = decay^gap * old_failure + previous_incorrect

    These are sequence-based forgetting proxies, not real-time forgetting.
    """

    df = df.copy()

    for decay in decay_values:
        success_col = f"{prefix}_decayed_success_{decay}"
        failure_col = f"{prefix}_decayed_failure_{decay}"

        success_state = defaultdict(float)
        failure_state = defaultdict(float)
        last_seen = {}

        success_values = []
        failure_values = []

        for _, row in df.iterrows():
            key = tuple(row[col] for col in group_cols)
            current_pos = float(row["row_id"])

            if key in last_seen:
                gap = current_pos - last_seen[key]
                if gap < 1:
                    gap = 1.0
            else:
                gap = 1.0

            success_state[key] *= decay ** gap
            failure_state[key] *= decay ** gap

            success_values.append(success_state[key])
            failure_values.append(failure_state[key])

            if int(row["correct"]) == 1:
                success_state[key] += 1.0
            else:
                failure_state[key] += 1.0

            last_seen[key] = current_pos

        df[success_col] = success_values
        df[failure_col] = failure_values

    return df


# ============================================================
# Optional manual precompute helpers
# ============================================================

def precompute_lkt_features_for_file(
    input_csv,
    output_csv=None,
    decay_values=(0.1, 0.5, 0.8),
):
    """
    Manually precompute expensive LKT history/spacing/decay features
    for one fold file.

    If output_csv is None, it uses the automatic cache path.
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    if output_csv is None:
        output_csv = _default_lkt_cache_path(input_csv)

    raw = pd.read_csv(input_csv)

    if _has_precomputed_lkt_features(raw, decay_values):
        featured = raw.copy()
        print(f"Already precomputed: {input_csv}")
    else:
        print(f"Precomputing LKT features for: {input_csv}")
        featured = _standardize_columns(
            raw,
            decay_values=decay_values,
        )

    output_dir = os.path.dirname(output_csv)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    featured.to_csv(output_csv, index=False)

    print(f"Saved precomputed LKT file: {output_csv}")
    print(f"Shape: {featured.shape}")

    return featured


def precompute_lkt_features_for_folds(
    input_dir,
    output_dir=None,
    kfold=5,
    input_pattern="converted_fold_{fold}.csv",
    output_pattern="converted_fold_{fold}_lkt.csv",
    decay_values=(0.1, 0.5, 0.8),
):
    """
    Manually precompute LKT features once for all fold files.

    If output_dir is None, it creates a sibling folder:
        tabular_lkt_auto
    """

    if output_dir is None:
        input_dir_norm = os.path.normpath(input_dir)
        parent = os.path.dirname(input_dir_norm)
        folder_name = os.path.basename(input_dir_norm)
        output_dir = os.path.join(parent, f"{folder_name}_lkt_auto")

    os.makedirs(output_dir, exist_ok=True)

    output_paths = []

    for fold in range(kfold):
        input_csv = os.path.join(
            input_dir,
            input_pattern.format(fold=fold),
        )

        output_csv = os.path.join(
            output_dir,
            output_pattern.format(fold=fold),
        )

        print("\n" + "=" * 60)
        print(f"Precomputing fold {fold}")
        print("=" * 60)

        precompute_lkt_features_for_file(
            input_csv=input_csv,
            output_csv=output_csv,
            decay_values=decay_values,
        )

        output_paths.append(output_csv)

    return output_paths


def _load_and_prepare_file(
    path,
    decay_values,
    use_precomputed=True,
    auto_cache=True,
):
    """
    Load one CSV.

    Behavior:
        1. If the input file already has LKT features, use it directly.
        2. Else, check for an automatically cached LKT version.
        3. If cached version exists, load it.
        4. Else, compute LKT features, save cache, and return it.

    This lets the main experiment script stay unchanged.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    # First, check whether the given file itself is already precomputed.
    raw = pd.read_csv(path)

    if use_precomputed and _has_precomputed_lkt_features(raw, decay_values):
        print(f"Using precomputed LKT features directly: {path}")
        return raw

    # Second, check whether an automatic cached version exists.
    if auto_cache:
        cache_path = _default_lkt_cache_path(path)

        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path)

            if _has_precomputed_lkt_features(cached, decay_values):
                print(f"Using cached LKT features: {cache_path}")
                return cached

            print(f"Cache exists but is invalid/incomplete, recomputing: {cache_path}")

    # Third, compute features from scratch.
    print(f"Calculating LKT features from scratch: {path}")

    featured = _standardize_columns(
        raw,
        decay_values=decay_values,
    )

    # Save automatic cache.
    if auto_cache:
        cache_path = _default_lkt_cache_path(path)
        cache_dir = os.path.dirname(cache_path)

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        featured.to_csv(cache_path, index=False)
        print(f"Saved cached LKT features: {cache_path}")

    return featured


def _combine_train_files(
    train_data,
    decay_values,
    use_precomputed=True,
    auto_cache=True,
):
    """
    Load and combine training files.

    With auto_cache=True:
        each original fold file is computed once, saved, and reused later.
    """

    if isinstance(train_data, (str, os.PathLike)):
        train_data = [train_data]

    frames = []

    for path in train_data:
        frame = _load_and_prepare_file(
            path,
            decay_values=decay_values,
            use_precomputed=use_precomputed,
            auto_cache=auto_cache,
        )
        frames.append(frame)

    if not frames:
        raise ValueError("No training files were provided.")

    return pd.concat(frames, ignore_index=True)


# ============================================================
# Logistic fitting
# ============================================================

@dataclass
class FitResult:
    weights: np.ndarray
    nll: float
    bic: float
    auc: float
    rmse: float


def _fit_logistic_mle(X, y, max_iter=500):
    """
    Unregularized logistic regression using scipy L-BFGS.

    BIC = k * log(n) + 2 * negative_log_likelihood.
    """

    n, k = X.shape
    y = y.astype(float)

    def objective(w):
        z = X @ w
        p = _sigmoid(z)

        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)

        nll = -np.sum(
            y * np.log(p) +
            (1.0 - y) * np.log(1.0 - p)
        )

        grad = X.T @ (p - y)

        return nll, np.asarray(grad).ravel()

    w0 = np.zeros(k)

    result = minimize(
        fun=lambda w: objective(w)[0],
        x0=w0,
        jac=lambda w: objective(w)[1],
        method="L-BFGS-B",
        options={
            "maxiter": max_iter,
            "ftol": 1e-7,
            "gtol": 1e-5,
        },
    )

    if not result.success:
        print(
            f"Warning: logistic optimizer stopped early: {result.message}. "
            "Using best weights found so far."
        )

    w = result.x
    z = X @ w
    p = _sigmoid(z)

    eps = 1e-12
    p_clip = np.clip(p, eps, 1.0 - eps)

    nll = -np.sum(
        y * np.log(p_clip) +
        (1.0 - y) * np.log(1.0 - p_clip)
    )

    bic = k * np.log(n) + 2.0 * nll

    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, p)
    else:
        auc = np.nan

    rmse = np.sqrt(mean_squared_error(y, p))

    return FitResult(
        weights=w,
        nll=float(nll),
        bic=float(bic),
        auc=float(auc),
        rmse=float(rmse),
    )


# ============================================================
# Feature block builder
# ============================================================

class LKTFeatureBuilder:
    """
    Builds LKT-style feature blocks.

    Important:
        fit() must be called only on the current training data.
    """

    def __init__(self, decay_values=(0.1, 0.5, 0.8)):
        self.decay_values = tuple(decay_values)

        self.skill_categories_ = None
        self.skill_to_idx_ = None

        self.item_categories_ = None
        self.item_to_idx_ = None

    def fit(self, train_df):
        skills = sorted(train_df["skill_id"].astype(str).unique())
        self.skill_categories_ = skills
        self.skill_to_idx_ = {
            skill: i
            for i, skill in enumerate(skills)
        }

        items = sorted(train_df["item"].astype(str).unique())
        self.item_categories_ = items
        self.item_to_idx_ = {
            item: i
            for i, item in enumerate(items)
        }

        return self

    def _skill_matrix(self, df):
        rows = []
        cols = []

        for row_idx, skill in enumerate(df["skill_id"].astype(str).values):
            col_idx = self.skill_to_idx_.get(skill)

            if col_idx is not None:
                rows.append(row_idx)
                cols.append(col_idx)

        data = np.ones(len(rows), dtype=float)

        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(df), len(self.skill_categories_)),
        )

    def _item_matrix(self, df):
        rows = []
        cols = []

        for row_idx, item in enumerate(df["item"].astype(str).values):
            col_idx = self.item_to_idx_.get(item)

            if col_idx is not None:
                rows.append(row_idx)
                cols.append(col_idx)

        data = np.ones(len(rows), dtype=float)

        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(df), len(self.item_categories_)),
        )

    def transform_blocks(self, df):
        kc_b4_correct = df["b4_correct"].astype(float).to_numpy()
        kc_b4_incorrect = df["b4_incorrect"].astype(float).to_numpy()
        kc_b4_total = kc_b4_correct + kc_b4_incorrect

        kc_logsuc = np.log1p(kc_b4_correct)
        kc_logfail = np.log1p(kc_b4_incorrect)
        kc_logafm = np.log1p(kc_b4_total)

        problem_b4_correct = df["problem_b4_correct"].astype(float).to_numpy()
        problem_b4_incorrect = df["problem_b4_incorrect"].astype(float).to_numpy()
        problem_b4_total = problem_b4_correct + problem_b4_incorrect

        problem_logsuc = np.log1p(problem_b4_correct)
        problem_logfail = np.log1p(problem_b4_incorrect)
        problem_logafm = np.log1p(problem_b4_total)

        kc_recency = df["kc_recency"].astype(float).to_numpy()
        kc_loggap = df["kc_log_trial_gap"].astype(float).to_numpy()

        problem_recency = df["problem_recency"].astype(float).to_numpy()
        problem_loggap = df["problem_log_trial_gap"].astype(float).to_numpy()

        skill_mat = self._skill_matrix(df)
        item_mat = self._item_matrix(df)

        blocks = {}

        blocks["intercept-KC"] = skill_mat
        blocks["intercept-Problem"] = item_mat

        blocks["lineafm-KC"] = sparse.csr_matrix(kc_b4_total.reshape(-1, 1))
        blocks["logafm-KC"] = sparse.csr_matrix(kc_logafm.reshape(-1, 1))
        blocks["logsuc-KC"] = sparse.csr_matrix(kc_logsuc.reshape(-1, 1))
        blocks["logfail-KC"] = sparse.csr_matrix(kc_logfail.reshape(-1, 1))
        blocks["linesuc-KC"] = sparse.csr_matrix(kc_b4_correct.reshape(-1, 1))
        blocks["linefail-KC"] = sparse.csr_matrix(kc_b4_incorrect.reshape(-1, 1))

        blocks["logsuc$-KC"] = skill_mat.multiply(kc_logsuc.reshape(-1, 1))
        blocks["logfail$-KC"] = skill_mat.multiply(kc_logfail.reshape(-1, 1))
        blocks["linesuc$-KC"] = skill_mat.multiply(kc_b4_correct.reshape(-1, 1))
        blocks["linefail$-KC"] = skill_mat.multiply(kc_b4_incorrect.reshape(-1, 1))
        blocks["logafm$-KC"] = skill_mat.multiply(kc_logafm.reshape(-1, 1))

        blocks["lineafm-Problem"] = sparse.csr_matrix(problem_b4_total.reshape(-1, 1))
        blocks["logafm-Problem"] = sparse.csr_matrix(problem_logafm.reshape(-1, 1))
        blocks["logsuc-Problem"] = sparse.csr_matrix(problem_logsuc.reshape(-1, 1))
        blocks["logfail-Problem"] = sparse.csr_matrix(problem_logfail.reshape(-1, 1))
        blocks["linesuc-Problem"] = sparse.csr_matrix(problem_b4_correct.reshape(-1, 1))
        blocks["linefail-Problem"] = sparse.csr_matrix(problem_b4_incorrect.reshape(-1, 1))

        blocks["logsuc$-Problem"] = item_mat.multiply(problem_logsuc.reshape(-1, 1))
        blocks["logfail$-Problem"] = item_mat.multiply(problem_logfail.reshape(-1, 1))
        blocks["linesuc$-Problem"] = item_mat.multiply(problem_b4_correct.reshape(-1, 1))
        blocks["linefail$-Problem"] = item_mat.multiply(problem_b4_incorrect.reshape(-1, 1))
        blocks["logafm$-Problem"] = item_mat.multiply(problem_logafm.reshape(-1, 1))

        blocks["recency-KC"] = sparse.csr_matrix(kc_recency.reshape(-1, 1))
        blocks["loggap-KC"] = sparse.csr_matrix(kc_loggap.reshape(-1, 1))
        blocks["recency-Problem"] = sparse.csr_matrix(problem_recency.reshape(-1, 1))
        blocks["loggap-Problem"] = sparse.csr_matrix(problem_loggap.reshape(-1, 1))

        blocks["recency$-KC"] = skill_mat.multiply(kc_recency.reshape(-1, 1))
        blocks["loggap$-KC"] = skill_mat.multiply(kc_loggap.reshape(-1, 1))

        blocks["recency$-Problem"] = item_mat.multiply(problem_recency.reshape(-1, 1))
        blocks["loggap$-Problem"] = item_mat.multiply(problem_loggap.reshape(-1, 1))

        blocks["logsuc-by-Problem"] = item_mat.multiply(kc_logsuc.reshape(-1, 1))
        blocks["logfail-by-Problem"] = item_mat.multiply(kc_logfail.reshape(-1, 1))
        blocks["logafm-by-Problem"] = item_mat.multiply(kc_logafm.reshape(-1, 1))

        blocks["problem-logsuc-by-KC"] = skill_mat.multiply(problem_logsuc.reshape(-1, 1))
        blocks["problem-logfail-by-KC"] = skill_mat.multiply(problem_logfail.reshape(-1, 1))
        blocks["problem-logafm-by-KC"] = skill_mat.multiply(problem_logafm.reshape(-1, 1))

        for decay in self.decay_values:
            kc_ds = df[f"kc_decayed_success_{decay}"].astype(float).to_numpy()
            kc_dfail = df[f"kc_decayed_failure_{decay}"].astype(float).to_numpy()

            problem_ds = df[f"problem_decayed_success_{decay}"].astype(float).to_numpy()
            problem_dfail = df[f"problem_decayed_failure_{decay}"].astype(float).to_numpy()

            blocks[f"kc_decayed_success_{decay}-KC"] = sparse.csr_matrix(
                kc_ds.reshape(-1, 1)
            )
            blocks[f"kc_decayed_failure_{decay}-KC"] = sparse.csr_matrix(
                kc_dfail.reshape(-1, 1)
            )

            blocks[f"problem_decayed_success_{decay}-Problem"] = sparse.csr_matrix(
                problem_ds.reshape(-1, 1)
            )
            blocks[f"problem_decayed_failure_{decay}-Problem"] = sparse.csr_matrix(
                problem_dfail.reshape(-1, 1)
            )

            blocks[f"kc_decayed_success${decay}-KC"] = skill_mat.multiply(
                kc_ds.reshape(-1, 1)
            )
            blocks[f"kc_decayed_failure${decay}-KC"] = skill_mat.multiply(
                kc_dfail.reshape(-1, 1)
            )

        return blocks


# ============================================================
# Matrix helpers
# ============================================================

def _hstack_blocks(blocks):
    return sparse.hstack(blocks, format="csr")


def _build_design_matrix(base_intercept, selected_names, block_dict):
    matrices = [base_intercept]

    for name in selected_names:
        matrices.append(block_dict[name])

    return _hstack_blocks(matrices)


def _build_design_matrix_with_slices(base_intercept, candidate_names, block_dict):
    matrices = [base_intercept]
    slices = {}
    start = 1

    for name in candidate_names:
        block = block_dict[name]
        matrices.append(block)

        end = start + block.shape[1]
        slices[name] = slice(start, end)
        start = end

    X = _hstack_blocks(matrices)

    return X, slices


# ============================================================
# Stepwise / LASSO search
# ============================================================

def _stepwise_search(
    train_blocks,
    y_train,
    candidate_names,
    forv=100.0,
    bacv=100.0,
    max_steps=8,
    optimizer_max_iter=500,
):
    n = len(y_train)
    base_intercept = sparse.csr_matrix(np.ones((n, 1)))

    selected = []

    X_current = _build_design_matrix(
        base_intercept,
        selected,
        train_blocks,
    )

    current_fit = _fit_logistic_mle(
        X_current,
        y_train,
        max_iter=optimizer_max_iter,
    )

    print(
        f"LKT-Python start: BIC={current_fit.bic:.3f}, "
        f"AUC={current_fit.auc:.4f}, RMSE={current_fit.rmse:.4f}"
    )

    for step in range(1, max_steps + 1):
        changed = False

        remaining = [
            name
            for name in candidate_names
            if name not in selected
        ]

        best_add_name = None
        best_add_fit = None
        best_add_bic = current_fit.bic

        print(f"\nLKT-Python step {step} forward search")

        for name in remaining:
            trial_selected = selected + [name]

            X_trial = _build_design_matrix(
                base_intercept,
                trial_selected,
                train_blocks,
            )

            trial_fit = _fit_logistic_mle(
                X_trial,
                y_train,
                max_iter=optimizer_max_iter,
            )

            bic_gain = current_fit.bic - trial_fit.bic

            print(
                f"  try add {name}: "
                f"BIC={trial_fit.bic:.3f}, "
                f"gain={bic_gain:.3f}, "
                f"AUC={trial_fit.auc:.4f}, "
                f"RMSE={trial_fit.rmse:.4f}"
            )

            if trial_fit.bic < best_add_bic:
                best_add_bic = trial_fit.bic
                best_add_name = name
                best_add_fit = trial_fit

        add_gain = current_fit.bic - best_add_bic

        if best_add_name is not None and add_gain >= forv:
            selected.append(best_add_name)
            current_fit = best_add_fit
            changed = True

            print(
                f"  added {best_add_name}; "
                f"BIC={current_fit.bic:.3f}, "
                f"gain={add_gain:.3f}, "
                f"AUC={current_fit.auc:.4f}, "
                f"RMSE={current_fit.rmse:.4f}"
            )
        else:
            if best_add_name is not None:
                print(
                    f"  best add was {best_add_name} with gain={add_gain:.3f}, "
                    f"below forv={forv}. Not adding."
                )

        if selected:
            best_remove_name = None
            best_remove_fit = None
            best_remove_bic = current_fit.bic

            print(f"LKT-Python step {step} backward search")

            for name in list(selected):
                trial_selected = [
                    s
                    for s in selected
                    if s != name
                ]

                X_trial = _build_design_matrix(
                    base_intercept,
                    trial_selected,
                    train_blocks,
                )

                trial_fit = _fit_logistic_mle(
                    X_trial,
                    y_train,
                    max_iter=optimizer_max_iter,
                )

                bic_gain = current_fit.bic - trial_fit.bic

                print(
                    f"  try remove {name}: "
                    f"BIC={trial_fit.bic:.3f}, "
                    f"gain={bic_gain:.3f}, "
                    f"AUC={trial_fit.auc:.4f}, "
                    f"RMSE={trial_fit.rmse:.4f}"
                )

                if trial_fit.bic < best_remove_bic:
                    best_remove_bic = trial_fit.bic
                    best_remove_name = name
                    best_remove_fit = trial_fit

            remove_gain = current_fit.bic - best_remove_bic

            if best_remove_name is not None and remove_gain >= bacv:
                selected.remove(best_remove_name)
                current_fit = best_remove_fit
                changed = True

                print(
                    f"  removed {best_remove_name}; "
                    f"BIC={current_fit.bic:.3f}, "
                    f"gain={remove_gain:.3f}, "
                    f"AUC={current_fit.auc:.4f}, "
                    f"RMSE={current_fit.rmse:.4f}"
                )
            else:
                if best_remove_name is not None:
                    print(
                        f"  best removal was {best_remove_name} "
                        f"with gain={remove_gain:.3f}, "
                        f"below bacv={bacv}. Not removing."
                    )

        print(
            f"LKT-Python step {step} result: "
            f"selected={selected}, "
            f"BIC={current_fit.bic:.3f}, "
            f"AUC={current_fit.auc:.4f}, "
            f"RMSE={current_fit.rmse:.4f}"
        )

        if not changed:
            print("No sufficiently large BIC-improving add/remove found. Stopping search.")
            break

    return selected, current_fit


def _lasso_search_predict(
    train_blocks,
    test_blocks,
    y_train,
    y_test,
    candidate_names,
    Cs=(0.001, 0.01, 0.1, 1.0, 10.0),
    max_iter=2000,
):
    n_train = len(y_train)
    n_test = len(y_test)

    train_intercept = sparse.csr_matrix(np.ones((n_train, 1)))
    test_intercept = sparse.csr_matrix(np.ones((n_test, 1)))

    X_train, slices = _build_design_matrix_with_slices(
        train_intercept,
        candidate_names,
        train_blocks,
    )

    X_test, _ = _build_design_matrix_with_slices(
        test_intercept,
        candidate_names,
        test_blocks,
    )

    model = LogisticRegressionCV(
        Cs=list(Cs),
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        cv=3,
        max_iter=max_iter,
        n_jobs=-1,
        refit=True,
        fit_intercept=False,
    )

    model.fit(X_train, y_train)

    predictions = model.predict_proba(X_test)[:, 1]

    coef = model.coef_.ravel()

    selected = []

    for name, sl in slices.items():
        if np.any(np.abs(coef[sl]) > 1e-8):
            selected.append(name)

    if len(np.unique(y_test)) > 1:
        test_auc = roc_auc_score(y_test, predictions)
    else:
        test_auc = np.nan

    test_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"LKT-Python LASSO selected features: {selected}")
    print(
        f"LKT-Python LASSO held-out stats: "
        f"AUC={test_auc:.4f}, RMSE={test_rmse:.4f}"
    )

    return predictions, selected


# ============================================================
# Candidate feature presets
# ============================================================

def _make_candidate_names(decay_values, feature_preset="medium"):
    if feature_preset == "compact":
        candidates = [
            "intercept-KC",

            "logafm-KC",
            "logsuc-KC",
            "logfail-KC",

            "logsuc$-KC",
            "logfail$-KC",

            "recency-KC",
            "loggap-KC",
        ]

        for decay in decay_values:
            candidates.extend([
                f"kc_decayed_success_{decay}-KC",
                f"kc_decayed_failure_{decay}-KC",
            ])

        return candidates

    if feature_preset == "medium":
        candidates = [
            "intercept-KC",

            "logafm-KC",
            "logsuc-KC",
            "logfail-KC",
            "linesuc-KC",
            "linefail-KC",

            "logsuc$-KC",
            "logfail$-KC",
            "logafm$-KC",

            "logafm-Problem",
            "logsuc-Problem",
            "logfail-Problem",

            "problem-logsuc-by-KC",
            "problem-logfail-by-KC",
            "problem-logafm-by-KC",

            "recency-KC",
            "loggap-KC",
            "recency-Problem",
            "loggap-Problem",
        ]

        for decay in decay_values:
            candidates.extend([
                f"kc_decayed_success_{decay}-KC",
                f"kc_decayed_failure_{decay}-KC",

                f"problem_decayed_success_{decay}-Problem",
                f"problem_decayed_failure_{decay}-Problem",

                f"kc_decayed_success${decay}-KC",
                f"kc_decayed_failure${decay}-KC",
            ])

        return candidates

    if feature_preset == "full":
        candidates = [
            "intercept-KC",
            "intercept-Problem",

            "lineafm-KC",
            "logafm-KC",
            "logsuc-KC",
            "logfail-KC",
            "linesuc-KC",
            "linefail-KC",

            "logsuc$-KC",
            "logfail$-KC",
            "linesuc$-KC",
            "linefail$-KC",
            "logafm$-KC",

            "lineafm-Problem",
            "logafm-Problem",
            "logsuc-Problem",
            "logfail-Problem",
            "linesuc-Problem",
            "linefail-Problem",

            "logsuc$-Problem",
            "logfail$-Problem",
            "linesuc$-Problem",
            "linefail$-Problem",
            "logafm$-Problem",

            "recency-KC",
            "loggap-KC",
            "recency-Problem",
            "loggap-Problem",

            "recency$-KC",
            "loggap$-KC",
            "recency$-Problem",
            "loggap$-Problem",

            "logsuc-by-Problem",
            "logfail-by-Problem",
            "logafm-by-Problem",

            "problem-logsuc-by-KC",
            "problem-logfail-by-KC",
            "problem-logafm-by-KC",
        ]

        for decay in decay_values:
            candidates.extend([
                f"kc_decayed_success_{decay}-KC",
                f"kc_decayed_failure_{decay}-KC",
                f"problem_decayed_success_{decay}-Problem",
                f"problem_decayed_failure_{decay}-Problem",
                f"kc_decayed_success${decay}-KC",
                f"kc_decayed_failure${decay}-KC",
            ])

        return candidates

    raise ValueError(
        f"Unknown feature_preset={feature_preset}. "
        "Use 'compact', 'medium', or 'full'."
    )


# ============================================================
# Main training function
# ============================================================

def train_predict_LKT(
    train_data,
    test_data,
    forv=100.0,
    bacv=100.0,
    max_steps=8,
    optimizer_max_iter=500,
    search_method="stepwise",
    decay_values=(0.1, 0.5, 0.8),
    feature_preset="medium",
    use_precomputed=True,
    auto_cache=True,
):
    """
    Pure-Python LKT-style model.

    Main behavior:
        - If auto_cache=True, it automatically creates and reuses cached
          LKT feature files.
        - It still rebuilds sparse feature blocks per fold using train-only mappings.
        - It still runs feature selection per fold using only the training data.
    """

    print("Loading training data...")
    train_df = _combine_train_files(
        train_data,
        decay_values=decay_values,
        use_precomputed=use_precomputed,
        auto_cache=auto_cache,
    )
    print(f"Train data shape: {train_df.shape}")

    print("Loading test data...")
    test_df = _load_and_prepare_file(
        test_data,
        decay_values=decay_values,
        use_precomputed=use_precomputed,
        auto_cache=auto_cache,
    )
    print(f"Test data shape: {test_df.shape}")

    y_train = train_df["correct"].astype(int).to_numpy()
    y_test = test_df["correct"].astype(int).to_numpy()

    print("Fitting feature builder on training data only...")
    feature_builder = LKTFeatureBuilder(
        decay_values=decay_values,
    ).fit(train_df)

    print("Building train feature blocks...")
    train_blocks = feature_builder.transform_blocks(train_df)

    print("Building test feature blocks using train-only mapping...")
    test_blocks = feature_builder.transform_blocks(test_df)

    candidate_names = _make_candidate_names(
        decay_values=decay_values,
        feature_preset=feature_preset,
    )

    print(
        f"LKT-Python config: "
        f"search_method={search_method}, "
        f"feature_preset={feature_preset}, "
        f"decay_values={decay_values}, "
        f"candidate_blocks={len(candidate_names)}, "
        f"forv={forv}, "
        f"bacv={bacv}, "
        f"max_steps={max_steps}, "
        f"use_precomputed={use_precomputed}, "
        f"auto_cache={auto_cache}"
    )

    if search_method == "lasso":
        predictions, selected = _lasso_search_predict(
            train_blocks=train_blocks,
            test_blocks=test_blocks,
            y_train=y_train,
            y_test=y_test,
            candidate_names=candidate_names,
        )

        return predictions, y_test

    if search_method != "stepwise":
        raise ValueError(
            f"Unknown search_method={search_method}. "
            "Use 'stepwise' or 'lasso'."
        )

    selected, fit = _stepwise_search(
        train_blocks=train_blocks,
        y_train=y_train,
        candidate_names=candidate_names,
        forv=forv,
        bacv=bacv,
        max_steps=max_steps,
        optimizer_max_iter=optimizer_max_iter,
    )

    print(f"\nFinal Python LKT-stepwise selected features: {selected}")

    n_test = len(test_df)
    test_intercept = sparse.csr_matrix(np.ones((n_test, 1)))

    X_test = _build_design_matrix(
        test_intercept,
        selected,
        test_blocks,
    )

    predictions = _sigmoid(X_test @ fit.weights)

    if len(np.unique(y_test)) > 1:
        test_auc = roc_auc_score(y_test, predictions)
    else:
        test_auc = np.nan

    test_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(
        f"LKT-Python held-out stats: "
        f"AUC={test_auc:.4f}, RMSE={test_rmse:.4f}"
    )

    return predictions, y_test