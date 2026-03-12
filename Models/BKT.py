from pyBKT.models import Model
import pandas as pd
import numpy as np


DEFAULTS = {
    'user_id': 'user',
    'order_id': 'row_id',
    'skill_name': 'skill',
    'correct': 'correct'
}


def train_BKT(train_data):
    """
    train_data: list of CSV file paths
    """
    model = Model(seed=42, num_fits=1, parallel=True)

    if isinstance(train_data, str):
        train_data = [train_data]

    df_total = pd.concat(
        [pd.read_csv(file) for file in train_data],
        ignore_index=True
    ).copy()

    df_total['row_id'] = np.arange(len(df_total))

    print("Number of rows in training data:", len(df_total))
    print("Columns:", list(df_total.columns))
    print("Number of unique skills:", df_total['skill'].nunique())

    model.fit(data=df_total, defaults=DEFAULTS)
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

    df_test = pd.read_csv(test_data).copy()
    df_test = df_test.reset_index(drop=True)
    df_test['row_id'] = np.arange(len(df_test))

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

    preds_df = model.predict(data=df_test, defaults=DEFAULTS)

    print(f"Rows returned by pyBKT.predict(): {len(preds_df)}")
    print(f"Prediction columns: {list(preds_df.columns)}")

    # Keep only row_id + prediction and merge back to original test rows
    if 'row_id' not in preds_df.columns:
        raise ValueError(
            "pyBKT predict output does not contain 'row_id'; "
            "cannot align predictions back to the test set."
        )

    pred_map = preds_df[['row_id', 'correct_predictions']].copy()

    merged = df_test[['row_id', 'correct']].merge(
        pred_map,
        on='row_id',
        how='left'
    ).sort_values('row_id')

    return (
        merged['correct_predictions'].to_numpy(),
        merged['correct'].to_numpy()
    )