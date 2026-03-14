from pyBKT.models import Model
import pandas as pd
import numpy as np


DEFAULTS = {
    'user_id': 'user',
    'order_id': 'order_id',
    'skill_name': 'skill',
    'correct': 'correct'
}


def _prepare_bkt_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'user' in df.columns:
        df['user'] = df['user'].astype(str)
    if 'skill' in df.columns:
        df['skill'] = df['skill'].astype(str)
    if 'correct' in df.columns:
        df['correct'] = df['correct'].astype(int)

    # Keep row_id only for alignment after prediction
    if 'row_id' not in df.columns:
        df['row_id'] = np.arange(len(df))

    # Build a stable temporal order for pyBKT
    if 'order_id' not in df.columns:
        if {'b4_correct', 'b4_incorrect'}.issubset(df.columns):
            df['order_id'] = df['b4_correct'].astype(int) + df['b4_incorrect'].astype(int)
        else:
            df['order_id'] = df.groupby('user').cumcount()

    df = df.sort_values(['user', 'order_id', 'row_id']).reset_index(drop=True)
    return df


def train_BKT(train_data):
    """
    train_data: list of CSV file paths
    """
    model = Model(seed=42, num_fits=5, parallel=True)

    if isinstance(train_data, str):
        train_data = [train_data]

    df_total = pd.concat(
        [pd.read_csv(file) for file in train_data],
        ignore_index=True
    ).copy()

    df_total = _prepare_bkt_frame(df_total)

    print("Number of rows in training data:", len(df_total))
    print("Columns:", list(df_total.columns))
    print("Number of unique skills:", df_total['skill'].nunique())

    model.fit(data=df_total, defaults=DEFAULTS, forgets=True)
    return model, df_total


def train_predict_BKT(train_data, test_data):
    """
    train_data: list of CSV file paths
    test_data: single CSV file path

    Returns:
        predictions aligned to original test rows, with NaN for rows
        pyBKT could not score.
    """
    model, df_train = train_BKT(train_data=train_data)

    df_test = pd.read_csv(test_data).reset_index(drop=True)
    df_test['row_id'] = np.arange(len(df_test))
    df_test = _prepare_bkt_frame(df_test)

    # Debug unseen skills
    train_skills = set(df_train['skill'].dropna().unique())
    test_skills = set(df_test['skill'].dropna().unique())
    unseen_skills = test_skills - train_skills

    print(f"Test rows: {len(df_test)}")
    print(f"Test unique skills: {len(test_skills)}")
    print(f"Unseen test skills: {len(unseen_skills)}")
    if unseen_skills:
        preview = list(sorted(unseen_skills))[:20]
        print(f"First unseen skills: {preview}")

    preds_df = model.predict(data=df_test)

    print(f"Rows returned by pyBKT.predict(): {len(preds_df)}")
    print(f"Prediction columns: {list(preds_df.columns)}")

    # Keep only row_id + prediction and merge back to original test rows
    if 'row_id' not in preds_df.columns:
        raise ValueError(
            "pyBKT predict output does not contain 'row_id'; "
            "cannot align predictions back to the test set."
        )

    pred_map = preds_df[['row_id', 'correct_predictions']].copy()

    skill_priors = df_train.groupby('skill')['correct'].mean().to_dict()
    global_prior = float(df_train['correct'].mean())

    merged = df_test[['row_id', 'skill', 'correct']].merge(
        pred_map,
        on='row_id',
        how='left'
    ).sort_values('row_id')

    prior_pred = merged['skill'].map(skill_priors).fillna(global_prior)
    raw_pred = merged['correct_predictions']
    history_pred = None
    if {'b4_correct', 'b4_incorrect'}.issubset(df_test.columns):
        history_map = df_test[['row_id', 'b4_correct', 'b4_incorrect']].copy()
        history_map['history_pred'] = (
            (history_map['b4_correct'].astype(float) + 1.0)
            / (history_map['b4_correct'].astype(float)
               + history_map['b4_incorrect'].astype(float)
               + 2.0)
        )
        merged = merged.merge(
            history_map[['row_id', 'history_pred']],
            on='row_id',
            how='left'
        )
        history_pred = merged['history_pred']

    # pyBKT can occasionally collapse to near-constant predictions on sparse folds
    raw_vals = raw_pred.to_numpy(dtype=float)
    raw_mean = np.nanmean(raw_vals) if np.any(~np.isnan(raw_vals)) else np.nan
    raw_std = np.nanstd(raw_vals) if np.any(~np.isnan(raw_vals)) else np.nan

    if (
        raw_pred.notna().sum() == 0
        or np.isnan(raw_mean)
        or np.isnan(raw_std)
        or raw_std < 1e-5
        or raw_mean < 1e-3
        or raw_mean > 1 - 1e-3
    ):
        print("pyBKT predictions collapsed; falling back to history/skill priors.")
        if history_pred is not None:
            merged['correct_predictions'] = history_pred
        else:
            merged['correct_predictions'] = prior_pred
    else:
        if history_pred is not None:
            merged['correct_predictions'] = raw_pred.fillna(history_pred)
        else:
            merged['correct_predictions'] = raw_pred.fillna(prior_pred)

    merged['correct_predictions'] = np.clip(
        merged['correct_predictions'].to_numpy(dtype=float),
        1e-6,
        1 - 1e-6
    )

    return (
        merged['correct_predictions'].to_numpy(),
        merged['correct'].to_numpy()
    )