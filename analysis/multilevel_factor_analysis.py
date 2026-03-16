import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning


DEFAULT_PREDICTION_COLUMNS = [
    "bkt_bf_prediction",
    "BKT_forgetting_prediction",
    "PFA_prediction",
    "ELO_prediction",
    "KTM_prediction",
    "DKT_prediction",
    "DSAKT_prediction",
    "ATKT_prediction",
]

CORE_OUTPUT_FILES = {
    "within_correlation_matrix.csv",
    "between_correlation_matrix.csv",
    "factor_model_comparison.csv",
    "within_factor_loadings1.csv",
    "within_factor_loadings2.csv",
    "between_factor_loadings1.csv",
    "between_factor_loadings2.csv",
    "icc_by_measure.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute mixed-effects within/between correlations and "
            "separate within/between factor analyses."
        )
    )
    parser.add_argument(
        "--input",
        default="output/combined_output.csv",
        help="Path to combined prediction table.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/factor_analysis",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--user-col",
        default="user",
        help="Student/user ID column.",
    )
    parser.add_argument(
        "--pred-cols",
        nargs="+",
        default=DEFAULT_PREDICTION_COLUMNS,
        help="Prediction variables used in analysis.",
    )
    parser.add_argument(
        "--missing",
        choices=["complete", "mean_impute"],
        default="complete",
        help=(
            "Missing handling for factor-analysis table. "
            "'complete' = keep rows where all prediction columns are present."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for FactorAnalysis.",
    )
    parser.add_argument(
        "--mixed-maxiter",
        type=int,
        default=200,
        help="Maximum iterations for each MixedLM fit.",
    )
    return parser.parse_args()


def assert_columns_exist(df: pd.DataFrame, user_col: str, pred_cols):
    missing_cols = [c for c in [user_col, *pred_cols] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def build_correlation_input(df: pd.DataFrame, user_col: str, pred_cols):
    work = df[[user_col, *pred_cols]].copy()
    for col in pred_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[user_col]).copy()
    work[user_col] = work[user_col].astype(str)
    return work


def clean_for_factor_analysis(df: pd.DataFrame, user_col: str, pred_cols, missing: str):
    work = df[[user_col, *pred_cols]].copy()
    for col in pred_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    missing_report = work[pred_cols].isna().sum().sort_values(ascending=False)
    initial_rows = len(work)
    initial_students = work[user_col].nunique(dropna=True)

    # avoid turning NaN into string "nan" IDs
    work = work.dropna(subset=[user_col]).copy()

    # complete case across all 8 models
    # if any of 8 columns contains a NaN, that row is removed
    if missing == "complete":
        work = work.dropna(subset=pred_cols)
    else:
        means = work[pred_cols].mean()
        work[pred_cols] = work[pred_cols].fillna(means)

    work[user_col] = work[user_col].astype(str)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=pred_cols + [user_col])

    summary = {
        "initial_rows": int(initial_rows),
        "rows_after_missing_handling": int(len(work)),
        "rows_dropped": int(initial_rows - len(work)),
        "initial_students": int(initial_students),
        "students_after_missing_handling": int(work[user_col].nunique()),
        "missing_strategy": missing,
        "missing_count_by_column": {k: int(v) for k, v in missing_report.items()},
    }
    return work, summary


def _extract_random_intercept_value(re_value):
    arr = np.asarray(re_value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.ravel()[0])


def compute_mixed_effects_correlations(df: pd.DataFrame, user_col: str, pred_cols, maxiter: int):
    """
    For each variable v, fit random-intercept model
        v_ij = beta0 + u_j + e_ij
    Compute: 
    within correlation from residuals e
    between correlation from random intercepts u
    """
    work = df[[user_col, *pred_cols]].copy()
    work[user_col] = work[user_col].astype(str)
    for col in pred_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    residuals = pd.DataFrame(index=work.index, columns=pred_cols, dtype=float)
    students = sorted(work[user_col].dropna().unique().tolist())
    random_intercepts = pd.DataFrame(index=students, columns=pred_cols, dtype=float)

    for col in pred_cols:
        sub = work[[user_col, col]].dropna().copy()
        n_obs = len(sub)
        n_students = sub[user_col].nunique()
        if n_obs < 10 or n_students < 3:
            continue

        exog = np.ones((n_obs, 1), dtype=float)
        endog = sub[col].to_numpy(dtype=float)
        groups = sub[user_col].to_numpy()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Random effects covariance is singular",
                )
                model = sm.MixedLM(endog=endog, exog=exog, groups=groups)
                result = model.fit(reml=True, method="lbfgs", maxiter=maxiter, disp=False)
            residuals.loc[sub.index, col] = result.resid
            for g, re_value in result.random_effects.items():
                random_intercepts.loc[str(g), col] = _extract_random_intercept_value(re_value)
        except Exception as exc:
            print(f"[warning] MixedLM failed for {col}: {exc}")

    within_corr = residuals.astype(float).corr()
    between_corr = random_intercepts.astype(float).corr()
    return within_corr, between_corr


def compute_level_data(df: pd.DataFrame, user_col: str, pred_cols):
    # Decomposition + separate FA (not jointly estimated full multilevel FA)
    group_means = df.groupby(user_col)[pred_cols].mean()
    within = df[pred_cols] - df.groupby(user_col)[pred_cols].transform("mean")
    between = group_means.copy()
    return within, between


#ICC = between-student variance / (between-student variance + within-student variance)
#how much of the total variance is due to differences between students
def estimate_icc(df: pd.DataFrame, user_col: str, pred_cols):
    group_means = df.groupby(user_col)[pred_cols].mean()
    centered = df[pred_cols] - df.groupby(user_col)[pred_cols].transform("mean")
    between_var = group_means.var(ddof=1)
    within_var = centered.var(ddof=1)
    icc = between_var / (between_var + within_var)
    return icc


def _factor_param_count(n_features: int, n_factors: int):
    return int(
        n_features * n_factors
        + n_features
        - (n_factors * (n_factors - 1)) / 2
    )


def fit_factor_model(X: pd.DataFrame, n_factors: int, random_state: int):
    if X.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 rows for factor analysis, got {X.shape[0]} rows."
        )
    std = X.std(ddof=1)
    near_zero = std[std < 1e-10].index.tolist()
    if near_zero:
        raise ValueError(
            f"Columns with near-zero variance cannot be factor analyzed: {near_zero}"
        )

    Xz = StandardScaler().fit_transform(X.values)
    model = FactorAnalysis(n_components=n_factors, random_state=random_state)
    model.fit(Xz)

    avg_loglike = float(model.score(Xz))
    n_samples, n_features = Xz.shape
    total_loglike = avg_loglike * n_samples
    n_params = _factor_param_count(n_features=n_features, n_factors=n_factors)
    aic = 2 * n_params - 2 * total_loglike
    bic = np.log(n_samples) * n_params - 2 * total_loglike

    loadings = pd.DataFrame(
        model.components_.T,
        index=X.columns,
        columns=[f"Factor{i + 1}" for i in range(n_factors)],
    )
    communalities = (loadings**2).sum(axis=1).rename("communality")
    uniqueness = pd.Series(model.noise_variance_, index=X.columns, name="uniqueness")
    loading_table = pd.concat([loadings, communalities, uniqueness], axis=1)

    fit_stats = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_factors": int(n_factors),
        "avg_loglike": float(avg_loglike),
        "total_loglike": float(total_loglike),
        "n_parameters": int(n_params),
        "AIC": float(aic),
        "BIC": float(bic),
    }
    return fit_stats, loading_table


def run_factor_comparison(within: pd.DataFrame, between: pd.DataFrame, random_state: int):
    rows = []
    loading_tables = {}
    for level_name, X in [("within", within), ("between", between)]:
        for n_factors in [1, 2]:
            fit_stats, loading_table = fit_factor_model(
                X=X, n_factors=n_factors, random_state=random_state
            )
            fit_stats["level"] = level_name
            rows.append(fit_stats)
            loading_tables[(level_name, n_factors)] = loading_table
    comparison = pd.DataFrame(rows).sort_values(["level", "n_factors"]).reset_index(drop=True)
    return comparison, loading_tables


def write_core_outputs(
    output_dir: Path,
    within_corr: pd.DataFrame,
    between_corr: pd.DataFrame,
    comparison: pd.DataFrame,
    loading_tables,
    icc: pd.Series,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.iterdir():
        if old_file.is_file() and old_file.name not in CORE_OUTPUT_FILES:
            old_file.unlink()

    within_corr.to_csv(output_dir / "within_correlation_matrix.csv")
    between_corr.to_csv(output_dir / "between_correlation_matrix.csv")
    comparison.to_csv(output_dir / "factor_model_comparison.csv", index=False)
    loading_tables[("within", 1)].to_csv(output_dir / "within_factor_loadings1.csv")
    loading_tables[("within", 2)].to_csv(output_dir / "within_factor_loadings2.csv")
    loading_tables[("between", 1)].to_csv(output_dir / "between_factor_loadings1.csv")
    loading_tables[("between", 2)].to_csv(output_dir / "between_factor_loadings2.csv")
    icc.rename("ICC").to_csv(output_dir / "icc_by_measure.csv", header=True)


def print_summary(summary: dict, comparison: pd.DataFrame, output_dir: Path):
    print("Data Summary")
    print(f"Initial rows: {summary['initial_rows']}")
    print(f"Rows after missing handling: {summary['rows_after_missing_handling']}")
    print(f"Rows dropped: {summary['rows_dropped']}")
    print(f"Initial students: {summary['initial_students']}")
    print(f"Students after missing handling: {summary['students_after_missing_handling']}")
    print(f"Missing strategy: {summary['missing_strategy']}")

    print("\nFactor Model Comparison")
    print(
        comparison[
            ["level", "n_factors", "n_samples", "avg_loglike", "AIC", "BIC"]
        ].to_string(index=False)
    )
    print(f"\nSaved core outputs to: {output_dir}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    assert_columns_exist(df=df, user_col=args.user_col, pred_cols=args.pred_cols)

    # Mixed-effects correlation uses all available rows per variable
    corr_input = build_correlation_input(df=df, user_col=args.user_col, pred_cols=args.pred_cols)
    within_corr, between_corr = compute_mixed_effects_correlations(
        df=corr_input,
        user_col=args.user_col,
        pred_cols=args.pred_cols,
        maxiter=args.mixed_maxiter,
    )

    # Factor analysis uses one aligned table according to selected missing strategy
    cleaned, summary = clean_for_factor_analysis(
        df=df,
        user_col=args.user_col,
        pred_cols=args.pred_cols,
        missing=args.missing,
    )
    if cleaned.empty:
        raise ValueError("No rows remain after missing-value handling.")
    if cleaned[args.user_col].nunique() < 3:
        raise ValueError("Need at least 3 students for analysis.")

    within_level, between_level = compute_level_data(
        df=cleaned,
        user_col=args.user_col,
        pred_cols=args.pred_cols,
    )
    comparison, loading_tables = run_factor_comparison(
        within=within_level,
        between=between_level,
        random_state=args.random_state,
    )
    icc = estimate_icc(df=cleaned, user_col=args.user_col, pred_cols=args.pred_cols)

    write_core_outputs(
        output_dir=output_dir,
        within_corr=within_corr,
        between_corr=between_corr,
        comparison=comparison,
        loading_tables=loading_tables,
        icc=icc,
    )
    print_summary(summary=summary, comparison=comparison, output_dir=output_dir)


if __name__ == "__main__":
    main()
