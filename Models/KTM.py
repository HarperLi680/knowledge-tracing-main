import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np


def train_predict_KTM(train_data, test_data):
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['item', 'skill']),
        ('num', 'passthrough', ['b4_correct', 'b4_incorrect'])
    ])

    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('lr', LogisticRegression(solver='liblinear', max_iter=1000))
    ])

    df_train = pd.concat([pd.read_csv(file) for file in train_data], ignore_index=True)
    df_test = pd.read_csv(test_data)

    X_train = df_train[['item', 'skill', 'b4_correct', 'b4_incorrect']]
    y_train = df_train['correct']

    X_test = df_test[['item', 'skill', 'b4_correct', 'b4_incorrect']]
    y_test = df_test['correct']

    pipe.fit(X_train, y_train)

    preds = pipe.predict_proba(X_test)[:, 1]
    return np.array(preds), np.array(y_test)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train_csv', type=str, nargs='?',default='dataset/assist2009_KTM/assist2009_KTM_train1.csv')
    parser.add_argument(
        'test_csv', type=str, nargs='?',default='dataset/assist2009_KTM/assist2009_KTM_test1.csv')
    options = parser.parse_args()

    train_predict_KTM([options.train_csv], options.test_csv)
