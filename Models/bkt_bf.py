"""Brute force BKT

The code is based on Professor Ryan Baker's Java code of Brute force BKT
(http://www.upenn.edu/learninganalytics/ryanbaker/edmtools.html)
This purpose of this module to find the best combination of Bayesian Knowledge tracing's
parameters, by trying every single possible combination.

Addition of methods to generate and return the knolwedge state and contextual guess
and slip parameters were made by Anthony Botelho
"""

from sklearn.base import BaseEstimator
import numpy as np
from pathlib import Path
import pickle
import itertools
import sys
import os
from time import time
import pandas as pd

ALMOST_ONE = 0.999
ALMOST_ZERO = 0.001


class BKT(BaseEstimator):

    @staticmethod
    def load_saved_model(skill):
        try:
            return BKT().load(skill)
        except FileNotFoundError:
            return BKT().load('no_skill')

    def __init__(self, skill='NA', step=0.1, bounded=True, best_k0=True):
        self.skill = skill

        # init parameter values
        self.k0 = ALMOST_ZERO
        self.transit = ALMOST_ZERO
        self.guess = ALMOST_ZERO
        self.slip = ALMOST_ZERO
        self.forget = ALMOST_ZERO

        # set the limitation of all parameters
        self.k0_limit = ALMOST_ONE
        self.transit_limit = ALMOST_ONE
        self.guess_limit = ALMOST_ONE
        self.slip_limit = ALMOST_ONE
        self.forget_limit = ALMOST_ONE

        self.current_k = ALMOST_ZERO

        self.step = step
        self.best_k0 = best_k0

        # ceiling values from Corbett & Anderson (1995)
        if bounded:
            self.k0_limit = 0.85
            self.transit_limit = 0.3
            self.guess_limit = 0.3
            self.slip_limit = 0.1

    def fit(self, X, y=None):
        """fit parameters from data"""

        if self.best_k0:
            self.k0 = self._find_best_k0(X)
            self.k0_limit = self.k0

        # generate all combinations
        k0s = np.arange(self.k0,
            min(self.k0_limit + self.step, ALMOST_ONE),
            self.step)
        transits = np.arange(self.transit,
            min(self.transit_limit + self.step, ALMOST_ONE),
            self.step)
        guesses = np.arange(self.guess,
            min(self.guess_limit + self.step, ALMOST_ONE),
            self.step)
        slips = np.arange(self.slip,
            min(self.slip_limit + self.step, ALMOST_ONE),
            self.step)
        all_parameters = [k0s, transits, guesses, slips]
        parameter_pairs = list(itertools.product(*all_parameters))

        # find the combination with lowest error
        min_error = sys.float_info.max
        for (k, t, g, s) in parameter_pairs:
            error, _ = self._compute_error(X, k, t, g, s)
            if error < min_error:
                self.k0 = k
                self.transit = t
                self.guess = g
                self.slip = s
                min_error = error

        return self.k0, self.transit, self.guess, self.slip

    def _compute_error(self, X, k, t, g, s):
        """computer error from current combination and performance data"""
        error = 0.0
        n = 0
        predictions = []

        init_k = k

        for seq in X:
            current_pred = []
            pred = init_k
            k = init_k
            for i, res in enumerate(seq):
                n += 1
                current_pred.append(pred)
                error += (res - pred) ** 2
                if res == 1.0:
                    p = k * (1 - s) / (k * (1 - s) + (1 - k) * g)
                else:
                    p = k * s / (k * s + (1 - k) * (1 - g))
                k = (p + (1 - p) * t)
                pred = k * (1 - s) + (1 - k) * g
                # self.current_k = k
            # print(self.current_k)
            predictions.append(current_pred)

        return (error / n)**0.5, predictions

    def _contextual_states(self, X, k, t, g, s):
        """computer error from current combination and performance data"""
        error = 0.0
        n = 0
        knowledge = []
        contextual_guess = []
        contextual_slip = []

        init_k = k

        for seq in X:
            current_k = []
            context_s = []
            context_g = []
            k = init_k
            pred = k
            for i in range(len(seq)):
                res = seq[i]
                n += 1
                current_k.append(k)
                error += (res - pred) ** 2
                if res == 1.0:
                    p = k * (1 - s) / (k * (1 - s) + (1 - k) * g)
                else:
                    p = k * s / (k * s + (1 - k) * (1 - g))
                k = p + (1 - p) * t

                if i < len(seq)-2:
                    gs1 = (1 - s) if seq[i+1] == 1.0 else g
                    gs2 = (1 - s) if seq[i+2] == 1.0 else g

                    pA_given_L = gs1*gs2

                    if seq[i+1] == 1.0 and seq[i+2] == 1.0:
                        pA_given_nL = t * ((1 - s) * (1 - s)) + (1 - t) * t * g * (1 - s) + ((1 - t) * (1 - t)) * (
                            (g) * (g))
                    elif seq[i+1] == 1.0 and seq[i+2] == 0.0:
                        pA_given_nL = t * ((1 - s) * (s)) + (1 - t) * t * g * (s) + ((1 - t) * (1 - t)) * (
                            (g) * (1 - g))
                    elif seq[i+1] == 0.0 and seq[i+2] == 1.0:
                        pA_given_nL = t * ((s) * (1 - s)) + (1 - t) * t * (1 - g) * (1 - s) + ((1 - t) * (1 - t)) * (
                            (1 - g) * (g))
                    else:
                        pA_given_nL = t * ((s) * (s)) + (1 - t) * t * (1 - g) * (s) + ((1 - t) * (1 - t)) * (
                            (1 - g) * (1 - g))

                    pA = k * pA_given_L + (1-k) * pA_given_nL
                else:
                    pA = np.nan

                context_g.append(np.nan if res == 0.0 else pA)
                context_s.append(np.nan if res == 1.0 else pA)

                pred = k * (1 - s) + (1 - k) * g
                self.current_k = k
            knowledge.append(current_k)
            contextual_guess.append(context_g)
            contextual_slip.append(context_s)

        return knowledge,contextual_guess,contextual_slip

    def _find_best_k0(self, X):
        """find the best init knowledge level by computing the performance of all first responses"""
        res = [seq[0] for seq in X]
        return np.mean(res)

    def save(self, directory):
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        filename = f"bkt_{self.skill}.pkl"
        filepath = directory_path / filename

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(directory, skill):
        directory_path = Path(directory)
        filename = f"bkt_{skill}.pkl"
        filepath = directory_path / filename

        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance

    def predict(self, X, return_error=False):
        """make predictions"""
        error, predictions = self._compute_error(X, self.k0, self.transit, self.guess, self.slip)
        if return_error:
            return error, predictions
        else:
            return predictions

    def get_contextual_states(self, X):
        """get estimates of knowledge state and contextual guess and slip"""
        return self._contextual_states(X, self.k0, self.transit, self.guess, self.slip)

    def params(self):
        return {'prior': self.k0,
                'learn': self.transit,
                'guess': self.guess,
                'slip': self.slip,
                'forget': self.forget}


def add_lol_to_df(df, list_of_lists, group_col_name='user_id', new_col_name='prediction'):
    # Ensure df is sorted by user_id (and by any sequence if appropriate) before grouping
    # This step is only needed if the DataFrame isn't already sorted as required
    # df = df.sort_values(['user_id', 'other_sorting_columns'], ignore_index=True)

    # NOTE: while maybe causing one to "laugh out loud" when first reading this function, the
    # use of "lol" here is an abbreviation of "list of lists"

    # Extract the grouped correctness lists (or any column) to match the structure of predictions_list_of_lists
    grouped = df.groupby(group_col_name, sort=False).size()
    # grouped is a Series with user_id as index and count of rows per user as values

    # Check that the number of users and the order matches the predictions_list_of_lists
    if len(grouped) != len(list_of_lists):
        raise ValueError("Number of prediction sublists does not match the number of unique users in df.")

    # Create a Series of predictions keyed by user_id
    lol_series = pd.Series(list_of_lists, index=grouped.index)

    # Explode the predictions so that each prediction aligns with a single row
    lol_exploded = lol_series.explode().astype(float)

    # To merge these predictions back into the original dataframe, create a sequence index per user
    df = df.copy()  # Work on a copy to avoid modifying the original df
    df['group_seq'] = df.groupby(group_col_name, sort=False).cumcount()

    lol_exploded = lol_exploded.reset_index()  # user_id and prediction value
    lol_exploded['group_seq'] = lol_exploded.groupby(group_col_name, sort=False).cumcount()

    # Merge predictions back into df
    df = pd.merge(df, lol_exploded, on=[group_col_name, 'group_seq'], how='left')

    # The prediction values are now in a column without a name (it was the Series values)
    # Rename that column to the specified prediction_col_name
    # The predictions are in the last column after the merge
    val_col = df.columns[-1]
    df = df.rename(columns={val_col: new_col_name}).drop('group_seq', axis=1)

    return df



# Function to apply BKT
def apply_BKT(df, skill):
    """
    Applies Bayesian Knowledge Tracing to the given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'row_id', 'user_id', 'skill_id', 'correct'.

    Returns:
    - df_results (pd.DataFrame): Original DataFrame with added 'predicted_knowledge' and 'predicted_correct' columns.
    - model: Trained BKT model.
    """
    df_bkt = df.copy()

    # Step 1: Prepare the DataFrame
    # Ensure 'correct' is binary (0 for incorrect, 1 for correct)
    df_bkt['correct'] = df_bkt['correct'].astype(int)

    # Sort the DataFrame by 'student_id' and 'row_id' to maintain the correct sequence
    df_bkt.sort_values(['user_id', 'row_id'], inplace=True)

    # Convert correctness to binary: values < 1 become 0, otherwise 1
    df_bkt['correct_binary'] = (df_bkt['correct'] >= 1).astype(int)

    # Group by user_id and aggregate into lists
    grouped = df_bkt.groupby('user_id')['correct_binary'].apply(list)

    # Convert to list of lists
    bkt_correctness_data = grouped.tolist()

    # Step 2: Initialize the BKT model
    model = BKT(skill=skill, step = 0.1, bounded = False, best_k0 = True)

    # Step 3: Fit the model to the data
    model.fit(bkt_correctness_data)

    # Step 4: Make predictions
    predictions = model.predict(bkt_correctness_data)

    Pk, Cg, Cs = model.get_contextual_states(bkt_correctness_data)

    df_bkt = add_lol_to_df(df_bkt, predictions, group_col_name='user_id', new_col_name='BKT_pred')
    df_bkt = add_lol_to_df(df_bkt, Pk, group_col_name='user_id', new_col_name='P_Know')
    df_bkt = add_lol_to_df(df_bkt, Cg, group_col_name='user_id', new_col_name='Contextual_Guess')
    df_bkt = add_lol_to_df(df_bkt, Cs, group_col_name='user_id', new_col_name='Contextual_Slip')

    return df_bkt, model

def train_predict_BKT(train_data, test_data):
    """
    Train and predict using brute force BKT implementation.

    Parameters
    ----------
    train_data : list[str] or str
        Training CSV file path(s).
    test_data : str
        Test CSV file path.

    Returns
    -------
    predictions : np.ndarray
        Predictions aligned to the original row order of df_test.
    actual : np.ndarray
        Actual labels aligned to the original row order of df_test.
    """

    # Load training data
    if isinstance(train_data, list):
        df_train = pd.concat([pd.read_csv(file) for file in train_data], ignore_index=True)
    else:
        df_train = pd.read_csv(train_data)

    # Load test data
    df_test = pd.read_csv(test_data)

    # Preserve original row order explicitly
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train['row_id'] = np.arange(len(df_train))
    df_test['row_id'] = np.arange(len(df_test))
    df_test['original_row_id'] = np.arange(len(df_test))

    # Map column names to expected format
    column_mapping = {
        'user': 'user_id',
        'skill': 'skill_id',
        'correct': 'correct'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df_train.columns:
            df_train = df_train.rename(columns={old_name: new_name})
        if old_name in df_test.columns:
            df_test = df_test.rename(columns={old_name: new_name})

    required_cols = {'user_id', 'skill_id', 'correct', 'row_id'}
    missing_train = required_cols - set(df_train.columns)
    missing_test = required_cols - set(df_test.columns)
    if missing_train:
        raise ValueError(f"Training data missing required columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test data missing required columns: {missing_test}")

    print("Number of rows in training data:", len(df_train))
    print("Columns:", df_train.columns.tolist())
    print("Number of unique skills:", df_train['skill_id'].nunique())

    # Train one BKT model per skill
    trained_models = {}
    for skill in df_train['skill_id'].unique():
        skill_train_df = df_train[df_train['skill_id'] == skill].copy()

        print(f"Training BKT for skill {skill}")
        print(f"N Rows: {len(skill_train_df)}")
        print(f"N Students: {skill_train_df['user_id'].nunique()}")
        print(f"Percent Correct: {skill_train_df['correct'].mean()}")

        _, model = apply_BKT(skill_train_df, skill)
        trained_models[skill] = model

    # Allocate aligned outputs in original df_test row order
    aligned_predictions = np.full(len(df_test), np.nan, dtype=float)
    aligned_actual = df_test['correct'].astype(int).to_numpy()

    # Predict skill by skill, but write back by original_row_id
    for skill in df_test['skill_id'].unique():
        skill_test_df = df_test[df_test['skill_id'] == skill].copy()

        if skill in trained_models:
            model = trained_models[skill]

            # Keep deterministic order within each user sequence
            skill_test_df['correct'] = skill_test_df['correct'].astype(int)
            skill_test_df.sort_values(['user_id', 'row_id'], inplace=True)
            skill_test_df['correct_binary'] = (skill_test_df['correct'] >= 1).astype(int)

            # Build sequences in the same order they will be exploded back
            grouped = skill_test_df.groupby('user_id', sort=False)['correct_binary'].apply(list)
            test_data_lists = grouped.tolist()

            predictions = model.predict(test_data_lists)

            # Put predictions back on the sorted skill-specific dataframe
            skill_test_df = add_lol_to_df(
                skill_test_df,
                predictions,
                group_col_name='user_id',
                new_col_name='BKT_pred'
            )

            # Write predictions back to their original global row positions
            aligned_predictions[skill_test_df['original_row_id'].to_numpy()] = (
                skill_test_df['BKT_pred'].astype(float).to_numpy()
            )

        else:
            print(f"Warning: No trained model found for skill {skill}")
            mean_correct = float(df_train['correct'].mean())
            aligned_predictions[skill_test_df['original_row_id'].to_numpy()] = mean_correct

    if np.isnan(aligned_predictions).any():
        missing_count = int(np.isnan(aligned_predictions).sum())
        raise ValueError(f"BKT produced {missing_count} unassigned predictions.")

    return aligned_predictions, aligned_actual

def test_train_predict_BKT():
    """
    Test function to verify train_predict_BKT works properly
    """
    import tempfile
    import os
    
    print("Testing train_predict_BKT function...")
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Training data
    train_data = {
        'user': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'item': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'skill': ['math', 'math', 'math', 'math', 'math', 'math', 
                 'reading', 'reading', 'reading', 'reading', 'reading', 'reading'],
        'correct': [0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    }
    df_train = pd.DataFrame(train_data)
    
    # Test data
    test_data = {
        'user': [1, 1, 2, 2, 3, 3, 5, 5],  # Include new user (5)
        'item': [13, 14, 15, 16, 17, 18, 19, 20],
        'skill': ['math', 'math', 'math', 'math', 'reading', 'reading', 'science', 'science'],  # Include new skill
        'correct': [1, 1, 0, 1, 1, 0, 1, 0]
    }
    df_test = pd.DataFrame(test_data)
    
    # Create temporary CSV files
    with tempfile.TemporaryDirectory() as temp_dir:
        train_file = os.path.join(temp_dir, 'train.csv')
        test_file = os.path.join(temp_dir, 'test.csv')
        
        df_train.to_csv(train_file, index=False)
        df_test.to_csv(test_file, index=False)
        
        print(f"Created temporary files:")
        print(f"  Train: {train_file}")
        print(f"  Test: {test_file}")
        
        # Test 1: Single file input
        print("\n=== Test 1: Single file input ===")
        try:
            predictions, actual = train_predict_BKT(train_file, test_file)
            
            print(f"✓ Function executed successfully")
            print(f"✓ Predictions shape: {predictions.shape}")
            print(f"✓ Actual shape: {actual.shape}")
            print(f"✓ Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
            print(f"✓ All predictions are probabilities: {np.all((predictions >= 0) & (predictions <= 1))}")
            
            # Check that we have predictions for all test samples
            assert len(predictions) == len(df_test), f"Expected {len(df_test)} predictions, got {len(predictions)}"
            print(f"✓ Correct number of predictions")
            
        except Exception as e:
            print(f"✗ Test 1 failed: {e}")
            return False
        
        # Test 2: List of files input
        print("\n=== Test 2: List of files input ===")
        try:
            predictions2, actual2 = train_predict_BKT([train_file], test_file)
            
            # Should give same results as single file
            np.testing.assert_array_almost_equal(predictions, predictions2, decimal=6)
            np.testing.assert_array_equal(actual, actual2)
            print(f"✓ List input gives same results as single file")
            
        except Exception as e:
            print(f"✗ Test 2 failed: {e}")
            return False
        
        # Test 3: Check skill-specific behavior
        print("\n=== Test 3: Skill-specific behavior ===")
        try:
            # Check that we have predictions for known skills
            math_indices = df_test['skill'] == 'math'
            reading_indices = df_test['skill'] == 'reading'
            science_indices = df_test['skill'] == 'science'
            
            print(f"✓ Math skill predictions: {np.sum(math_indices)} samples")
            print(f"✓ Reading skill predictions: {np.sum(reading_indices)} samples")
            print(f"✓ Science skill predictions (unseen): {np.sum(science_indices)} samples")
            
            # Science skill should use fallback (mean prediction)
            science_preds = predictions[science_indices]
            if len(science_preds) > 1:
                # All science predictions should be similar (using mean fallback)
                science_std = np.std(science_preds)
                print(f"✓ Science predictions std: {science_std:.6f} (should be small for fallback)")
            
        except Exception as e:
            print(f"✗ Test 3 failed: {e}")
            return False
        
        # Test 4: Model parameters
        print("\n=== Test 4: Model training verification ===")
        try:
            # Train just one skill to check parameters
            math_train = df_train[df_train['skill'] == 'math'].copy()
            math_train = math_train.rename(columns={'user': 'user_id', 'item': 'row_id', 'skill': 'skill_id'})
            
            _, model = apply_BKT(math_train, 'math')
            params = model.params()
            
            print(f"✓ Model parameters: {params}")
            print(f"✓ Prior (k0): {params['prior']:.3f}")
            print(f"✓ Learn (transit): {params['learn']:.3f}")
            print(f"✓ Guess: {params['guess']:.3f}")
            print(f"✓ Slip: {params['slip']:.3f}")
            
            # Check that parameters are in valid ranges
            assert 0 <= params['prior'] <= 1, f"Invalid prior: {params['prior']}"
            assert 0 <= params['learn'] <= 1, f"Invalid learn: {params['learn']}"
            assert 0 <= params['guess'] <= 1, f"Invalid guess: {params['guess']}"
            assert 0 <= params['slip'] <= 1, f"Invalid slip: {params['slip']}"
            print(f"✓ All parameters in valid range [0,1]")
            
        except Exception as e:
            print(f"✗ Test 4 failed: {e}")
            return False
    
    print("\n🎉 All tests passed! train_predict_BKT is working properly.")
    return True

if __name__ == '__main__':
    # Run the test function
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_train_predict_BKT()
    else:
        ### TODO: add your data loading code here
        bkt_df = pd.read_csv('REPLACE WITH YOUR DATA FILE')

        updated_dfs = []
        for skill in bkt_df['skill_id'].unique():
            skill_df = bkt_df[bkt_df['skill_id']==skill]

            print(f'=== SKILL {skill} ===')
            print(f'N Rows: {len(skill_df)}')
            print(f'N Students: {len(skill_df["user_id"].unique())}')
            print(f'Percent Correct: {skill_df["correct"].mean()}')
            print('-------------------')

            df_results, bkt_model = apply_BKT(skill_df, skill)

            # Display the first few rows of the results
            #print(df_results.head())
            updated_dfs.append(df_results)

            # Optionally, inspect the model parameters
            params = bkt_model.params()
            print(params)
            print('-------------------')

            bkt_model.save(directory='BKT_Models')

        bkt_results_df = pd.concat(updated_dfs, ignore_index=True)
        bkt_results_df.to_csv('bkt_results_subcategory.csv', index=False)
