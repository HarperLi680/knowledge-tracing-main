import argparse
import shutil
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import FactorAnalysis
from sklearn.exceptions import ConvergenceWarning as SKConvergenceWarning
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning

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
    "within_factor_loadings2.csv",
    "between_factor_loadings2.csv",
    "icc_by_measure.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="KT multilevel decomposition + factor analysis + mixed model diagnostic"
    )
    parser.add_argument("--input", default="../output/combined_output.csv")
    parser.add_argument("--output-dir", default="output/factor_analysis")
    parser.add_argument("--user-col", default="user")
    parser.add_argument("--pred-cols", nargs="+", default=DEFAULT_PREDICTION_COLUMNS)
    parser.add_argument("--missing", choices=["complete", "mean_impute"], default="complete")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-factors", type=int, default=4)
    parser.add_argument("--fa-maxiter", type=int, default=800)
    parser.add_argument("--mixed-maxiter", type=int, default=200)
    parser.add_argument("--mixed-boundary-threshold", type=float, default=1e-8)
    return parser.parse_args()


def _to_scalar(value) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.ravel()[0])


def _zscore_series(s: pd.Series) -> pd.Series:
    std = s.std(ddof=1)
    if std is None or np.isnan(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - s.mean()) / std


def _require_columns(df: pd.DataFrame, user_col: str, pred_cols):
    missing = [c for c in [user_col, *pred_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_correlation_input(df: pd.DataFrame, user_col: str, pred_cols) -> pd.DataFrame:
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

    initial_rows = int(len(work))
    initial_students = int(work[user_col].nunique(dropna=True))

    # avoid turning NaN into string "nan" IDs
    work = work.dropna(subset=[user_col]).copy()

    # complete case across all 8 models
    # if any of 8 columns contains a NaN, that row is removed
    if missing == "complete":
        work = work.dropna(subset=pred_cols)
    else:
        means = work[pred_cols].mean()
        work[pred_cols] = work[pred_cols].fillna(means)

    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=pred_cols + [user_col])
    work[user_col] = work[user_col].astype(str)

    summary = {
        "initial_rows": initial_rows,
        "rows_after_missing_handling": int(len(work)),
        "rows_dropped": int(initial_rows - len(work)),
        "initial_students": initial_students,
        "students_after_missing_handling": int(work[user_col].nunique()),
        "missing_strategy": missing,
    }
    return work, summary


def compute_mixed_effects_correlations(
    df: pd.DataFrame,
    user_col: str,
    pred_cols,
    maxiter: int,
    boundary_threshold: float,
):
    work = df[[user_col, *pred_cols]].copy()
    for col in pred_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[user_col] = work[user_col].astype(str)

    residuals = pd.DataFrame(index=work.index, columns=pred_cols, dtype=float)
    users = sorted(work[user_col].dropna().unique().tolist())
    random_intercepts = pd.DataFrame(index=users, columns=pred_cols, dtype=float)

    fallback_models = set()
    boundary_models = set()
    diag_rows = []

    for col in pred_cols:
        sub = work[[user_col, col]].dropna().copy()
        n_obs = int(len(sub))
        n_students = int(sub[user_col].nunique())

        status = "fitted"
        reason = ""
        cov_re_var = float("nan")

        if n_obs < 10 or n_students < 3:
            diag_rows.append(
                {
                    "model": col,
                    "n_obs": n_obs,
                    "n_students": n_students,
                    "status": "skipped",
                    "fallback_reason": "insufficient_data",
                    "cov_re_var": cov_re_var,
                }
            )
            continue

        sub[col] = _zscore_series(sub[col].astype(float))

        try:
            endog = sub[col].to_numpy(dtype=float)
            exog = np.ones((len(sub), 1), dtype=float)
            groups = sub[user_col].to_numpy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SMConvergenceWarning)
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*covariance.*singular.*",
                )
                model = sm.MixedLM(endog=endog, exog=exog, groups=groups)
                result = model.fit(reml=True, method="lbfgs", maxiter=maxiter, disp=False)

            cov_re_var = _to_scalar(result.cov_re)

            if (not np.isfinite(cov_re_var)) or (cov_re_var < boundary_threshold):
                status = "fallback"
                reason = "boundary_cov_re"
                fallback_models.add(col)
                boundary_models.add(col)

                centered = sub[col] - sub.groupby(user_col)[col].transform("mean")
                residuals.loc[sub.index, col] = centered.to_numpy()
                re_proxy = sub.groupby(user_col)[col].mean() - sub[col].mean()
                random_intercepts.loc[re_proxy.index.astype(str), col] = re_proxy.to_numpy()
            else:
                residuals.loc[sub.index, col] = result.resid

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        warnings.filterwarnings(
                            "ignore",
                            category=UserWarning,
                            message=".*covariance.*singular.*",
                        )
                        re_map = result.random_effects
                    for uid, re_val in re_map.items():
                        random_intercepts.loc[str(uid), col] = _to_scalar(re_val)
                except Exception:
                    status = "fallback"
                    reason = "random_effects_exception"
                    fallback_models.add(col)
                    re_proxy = sub.groupby(user_col)[col].mean() - sub[col].mean()
                    random_intercepts.loc[re_proxy.index.astype(str), col] = re_proxy.to_numpy()

        except Exception:
            status = "fallback"
            reason = "fit_exception"
            fallback_models.add(col)

            centered = sub[col] - sub.groupby(user_col)[col].transform("mean")
            residuals.loc[sub.index, col] = centered.to_numpy()
            re_proxy = sub.groupby(user_col)[col].mean() - sub[col].mean()
            random_intercepts.loc[re_proxy.index.astype(str), col] = re_proxy.to_numpy()

        diag_rows.append(
            {
                "model": col,
                "n_obs": n_obs,
                "n_students": n_students,
                "status": status,
                "fallback_reason": reason,
                "cov_re_var": cov_re_var,
            }
        )

    within_corr = residuals.astype(float).corr()
    between_corr = random_intercepts.astype(float).corr()

    mixed_diag = {
        "fallback_models": sorted(fallback_models),
        "boundary_models": sorted(boundary_models),
        "boundary_threshold": float(boundary_threshold),
        "details": pd.DataFrame(diag_rows).sort_values("model").reset_index(drop=True),
    }
    return within_corr, between_corr, mixed_diag


def compute_level_data(df: pd.DataFrame, user_col: str, pred_cols):
    within = df[pred_cols] - df.groupby(user_col)[pred_cols].transform("mean")
    between = df.groupby(user_col)[pred_cols].mean()
    return within, between


def estimate_icc(df: pd.DataFrame, user_col: str, pred_cols) -> pd.Series:
    means = df.groupby(user_col)[pred_cols].mean()
    centered = df[pred_cols] - df.groupby(user_col)[pred_cols].transform("mean")
    between_var = means.var(ddof=1)
    within_var = centered.var(ddof=1)
    return between_var / (between_var + within_var)


def varimax(loadings: np.ndarray, gamma: float = 1.0, q: int = 50, tol: float = 1e-6):
    if loadings.shape[1] < 2:
        return loadings

    p, k = loadings.shape
    rot = np.eye(k)
    d_old = 0.0
    for _ in range(q):
        lam = loadings @ rot
        u, s, vh = np.linalg.svd(
            loadings.T @ (lam**3 - (gamma / p) * lam @ np.diag(np.diag(lam.T @ lam)))
        )
        rot = u @ vh
        d = np.sum(s)
        if d_old != 0.0 and d / d_old < 1.0 + tol:
            break
        d_old = d
    return loadings @ rot


def _factor_param_count(n_features: int, n_factors: int) -> int:
    return int(
        n_features * n_factors
        + n_features
        - (n_factors * (n_factors - 1)) / 2
    )


def fit_factor_model(X: pd.DataFrame, n_factors: int, random_state: int, fa_maxiter: int):
    if X.shape[0] < 3:
        raise ValueError(f"Need at least 3 rows, got {X.shape[0]}")

    std = X.std(ddof=1)
    near_zero = std[std < 1e-10].index.tolist()
    if near_zero:
        raise ValueError(f"Near-zero variance columns: {near_zero}")

    Xz = StandardScaler().fit_transform(X.values)
    model = FactorAnalysis(n_components=n_factors, random_state=random_state, max_iter=fa_maxiter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SKConvergenceWarning)
        model.fit(Xz)

    avg_loglike = float(model.score(Xz))
    n_samples, n_features = Xz.shape
    total_loglike = avg_loglike * n_samples
    n_params = _factor_param_count(n_features=n_features, n_factors=n_factors)

    fit_stats = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_factors": int(n_factors),
        "avg_loglike": avg_loglike,
        "total_loglike": float(total_loglike),
        "n_parameters": int(n_params),
        "AIC": float(2 * n_params - 2 * total_loglike),
        "BIC": float(np.log(n_samples) * n_params - 2 * total_loglike),
        "fa_converged": bool(getattr(model, "n_iter_", fa_maxiter) < fa_maxiter),
    }

    unrot = pd.DataFrame(
        model.components_.T,
        index=X.columns,
        columns=[f"Factor{i + 1}" for i in range(n_factors)],
    )
    rot = pd.DataFrame(
        varimax(model.components_.T),
        index=X.columns,
        columns=[f"Factor{i + 1}" for i in range(n_factors)],
    )
    rot["communality"] = (rot[[c for c in rot.columns if c.startswith("Factor")]] ** 2).sum(axis=1)
    rot["uniqueness"] = model.noise_variance_

    return fit_stats, unrot, rot


def _scree_and_kaiser(X: pd.DataFrame):
    Xz = StandardScaler().fit_transform(X.values)
    cov = np.cov(Xz, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return eigvals, int(np.sum(eigvals > 1.0))


def run_factor_comparison(
    within: pd.DataFrame,
    between: pd.DataFrame,
    max_factors: int,
    random_state: int,
    fa_maxiter: int,
):
    rows = []
    unrot_tables = {}
    rot_tables = {}

    for level_name, X in [("within", within), ("between", between)]:
        eigvals, kaiser_n = _scree_and_kaiser(X)
        scree_text = "|".join(f"{v:.6f}" for v in eigvals)

        for n_factors in range(1, min(max_factors, X.shape[1]) + 1):
            fit_stats, unrot, rot = fit_factor_model(X, n_factors, random_state, fa_maxiter)
            fit_stats["level"] = level_name
            fit_stats["kaiser_n"] = int(kaiser_n)
            fit_stats["scree_eigenvalues"] = scree_text
            rows.append(fit_stats)
            unrot_tables[(level_name, n_factors)] = unrot
            rot_tables[(level_name, n_factors)] = rot

    comparison = pd.DataFrame(rows).sort_values(["level", "n_factors"]).reset_index(drop=True)
    return comparison, unrot_tables, rot_tables


def outputs(
    output_dir: Path,
    within_corr: pd.DataFrame,
    between_corr: pd.DataFrame,
    comparison: pd.DataFrame,
    unrot_tables,
    rot_tables,
    icc: pd.Series,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in output_dir.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file() and p.name not in CORE_OUTPUT_FILES:
            p.unlink()

    within_corr.to_csv(output_dir / "within_correlation_matrix.csv")
    between_corr.to_csv(output_dir / "between_correlation_matrix.csv")
    comparison.to_csv(output_dir / "factor_model_comparison.csv", index=False)

    if ("within", 2) in rot_tables:
        rot_tables[("within", 2)].to_csv(output_dir / "within_factor_loadings2.csv")
    else:
        rot_tables[("within", 1)].to_csv(output_dir / "within_factor_loadings2.csv")

    if ("between", 2) in rot_tables:
        rot_tables[("between", 2)].to_csv(output_dir / "between_factor_loadings2.csv")
    else:
        rot_tables[("between", 1)].to_csv(output_dir / "between_factor_loadings2.csv")

    icc.rename("ICC").to_csv(output_dir / "icc_by_measure.csv", header=True)

    # supplemental tables for factor 3 and 4
    supplement_dir = output_dir / "supplement_factor_loadings"
    supplement_dir.mkdir(parents=True, exist_ok=True)
    for p in supplement_dir.iterdir():
        if p.is_file():
            p.unlink()

    for level_name in ("within", "between"):
        for n_factors in (3, 4):
            key = (level_name, n_factors)
            if key in rot_tables:
                rot_tables[key].to_csv(
                    supplement_dir / f"{level_name}_factor_loadings{n_factors}_varimax.csv"
                )


def print_summary(summary: dict, comparison: pd.DataFrame, mixed_diag: dict, icc: pd.Series, output_dir: Path):
    print("Data Summary")
    print(f"Initial rows: {summary['initial_rows']}")
    print(f"Rows after missing handling: {summary['rows_after_missing_handling']}")
    print(f"Rows dropped: {summary['rows_dropped']}")
    print(f"Initial students: {summary['initial_students']}")
    print(f"Students after missing handling: {summary['students_after_missing_handling']}")
    print(f"Missing strategy: {summary['missing_strategy']}")

    fallback_models = mixed_diag["fallback_models"]
    boundary_models = mixed_diag["boundary_models"]
    print(f"Mixed effects fallback used for {len(fallback_models)} variable")
    print(f"Fallback variables: {fallback_models}")
    print(
        "Boundary (near-singular cov_re) variables: "
        f"{boundary_models} (threshold={mixed_diag['boundary_threshold']})"
    )

    low_icc = icc[icc < 0.01].index.tolist()
    if low_icc:
        print(f"low ICC (<0.01) variables: {low_icc}")

    print("\nFactor Model Comparison")
    print(comparison[["level", "n_factors", "AIC", "BIC", "kaiser_n"]].to_string(index=False))
    print(f"\nSaved core outputs to: {output_dir}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    _require_columns(df, args.user_col, args.pred_cols)

    corr_input = prepare_correlation_input(df, args.user_col, args.pred_cols)
    within_corr, between_corr, mixed_diag = compute_mixed_effects_correlations(
        corr_input,
        args.user_col,
        args.pred_cols,
        args.mixed_maxiter,
        args.mixed_boundary_threshold,
    )

    cleaned, summary = clean_for_factor_analysis(df, args.user_col, args.pred_cols, args.missing)
    if cleaned.empty:
        raise ValueError("No rows remain after missing value handling")
    if cleaned[args.user_col].nunique() < 3:
        raise ValueError("Need at least 3 students for analysis")

    within_level, between_level = compute_level_data(cleaned, args.user_col, args.pred_cols)
    comparison, unrot_tables, rot_tables = run_factor_comparison(
        within_level,
        between_level,
        args.max_factors,
        args.random_state,
        args.fa_maxiter,
    )
    icc = estimate_icc(cleaned, args.user_col, args.pred_cols)

    outputs(output_dir, within_corr, between_corr, comparison, unrot_tables, rot_tables, icc)
    print_summary(summary, comparison, mixed_diag, icc, output_dir)


if __name__ == "__main__":
    main()