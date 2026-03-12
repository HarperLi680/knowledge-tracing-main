import os
import pandas as pd
import numpy as np
from sklearn import metrics
import math
from tqdm import tqdm
import json

from Models.bkt_bf import train_predict_BKT
#from Models.BKT import train_predict_BKT
from Models.PFA import train_predict_PFA
from Models.KTM import train_predict_KTM
from Models.Elo import train_predict_Elo
from Models.ATKT import train_predict_ATKT
from Models.DSAKT import train_predict_DSAKT  # Import the DSAKT model

def calculate_n_skill(data_files):
    all_skills = set()
    for file in data_files:
        with open(file, 'r') as f:
            for line_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if line_id % 4 == 1:
                    skills = [int(x) for x in line.split(',') if x != ""]
                    all_skills.update(skills)
    return max(all_skills) + 1 # +1 because skill IDs are 0-indexed

def process_folds(data_folder):
    print(f"Looking for data files in: {data_folder}")
    files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
    print(f"Found {len(files)} .csv files")
    return files

def train_and_predict_traditional(data_folder, model_func, train_folds, test_fold):
    all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    all_files.sort()
    
    train_data = [os.path.join(data_folder, all_files[i]) for i in train_folds]
    test_data = os.path.join(data_folder, all_files[test_fold])
    
    print(f"Train data: {train_data}")
    print(f"Test data: {test_data}")
    
    preds, _ = model_func(train_data, test_data)
    return preds

def train_and_predict_deep_learning(data_folder, train_folds, test_fold, model_name):
    all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    all_files.sort()
    
    train_data = [os.path.join(data_folder, all_files[i]) for i in train_folds[:-1]]
    valid_path = os.path.join(data_folder, all_files[train_folds[-1]])
    test_path = os.path.join(data_folder, all_files[test_fold])
    
    # Verify all paths exist
    for path in train_data + [valid_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
    
    if model_name == 'ATKT':
        # Calculate n_skill using all available data
        all_data_files = train_data + [valid_path, test_path]
        try:
            n_skill = calculate_n_skill(all_data_files)
            print(f"Calculated n_skill: {n_skill}")
        except Exception as e:
            print(f"Error calculating n_skill: {str(e)}")
            raise
        
        # ATKT parameters
        dl_params = {
            'lr': 0.001,
            'gamma': 0.5,
            'lr_decay': 50,
            'hidden_emb_dim': 64,
            'skill_emb_dim': 128,
            'answer_emb_dim': 64,
            'beta': 0.2,
            'epsilon': 10,
            'seqlen': 100,
            'max_iter': 1,
            'batch_size': 32,
            'n_skill': n_skill
        }
        
        try:
            preds, _ = train_predict_ATKT(
                train_data=train_data,
                valid_path=valid_path,
                test_path=test_path,
                **dl_params
            )
            return preds
        except Exception as e:
            print(f"Error in train_predict_ATKT: {str(e)}")
            raise
    
    elif model_name == 'DSAKT':
        # DSAKT parameters (you may need to adjust these)
        dsakt_params = {
            'lr': 0.3,
            'window_size': 50,
            'dim': 24,
            'dropout': 0.7,
            'heads': 8
        }
        
        try:
            preds, _ = train_predict_DSAKT(
                train_path=train_data,
                valid_path=valid_path,
                test_path=test_path,
                **dsakt_params
            )
            return preds
        except Exception as e:
            print(f"Error in train_predict_DSAKT: {str(e)}")
            raise
    
    else:
        raise ValueError(f"Unknown deep learning model: {model_name}")

def get_original_data_predictions(traditional_data_folder, dl_data_folder):
    traditional_files = process_folds(traditional_data_folder)
    dl_files = process_folds(dl_data_folder)
    num_folds_dl = len(dl_files)
    num_folds_trad = len(traditional_files)
    
    all_predictions = {
        'BKT': {},
        'PFA': {},
        'KTM': {},
        'Elo': {},
        'ATKT': {},
        'DSAKT': {}  # Add DSAKT to the predictions dictionary
    }
    
    traditional_models = {
        'BKT': train_predict_BKT,
        'PFA': train_predict_PFA,
        'KTM': train_predict_KTM,
        'Elo': train_predict_Elo
    }

    # Train and collect predictions for traditional models
    for model_name, model_func in tqdm(traditional_models.items(), desc="Training traditional models"):
        print(f"\nTraining and predicting with {model_name}...")
        for test_fold in range(num_folds_trad):
            train_folds = [i for i in range(num_folds_trad) if i != test_fold]
            preds = train_and_predict_traditional(traditional_data_folder, model_func, train_folds, test_fold)
            all_predictions[model_name][test_fold] = preds
    
    # Train and collect predictions for deep learning models
    deep_learning_models = ['ATKT','DSAKT']
    for model_name in deep_learning_models:
        print(f"\nTraining and predicting with {model_name}...")
        for test_fold in range(num_folds_dl):
            train_folds = [i for i in range(num_folds_dl) if i != test_fold]
            preds = train_and_predict_deep_learning(dl_data_folder, train_folds, test_fold, model_name)
            all_predictions[model_name][test_fold] = preds
    
    return all_predictions

def create_combined_csv(traditional_data_folder, traditional_files, all_predictions, num_folds, output_file):
    combined_data = []
    for fold in range(num_folds):
        fold_file = os.path.join(traditional_data_folder, traditional_files[fold])
        fold_data = pd.read_csv(fold_file)
        
        fold_data['fold'] = fold
        
        for model_name in all_predictions.keys():
            print(f"Fold {fold}, {model_name}: Predictions shape: {len(all_predictions[model_name][fold])}")
            
            if len(all_predictions[model_name][fold]) != len(fold_data):
                print(f"Mismatch in predictions for {model_name} in fold {fold}")
                # Align predictions with original data for all models
                aligned_preds = np.full(len(fold_data), np.nan)
                aligned_preds[-len(all_predictions[model_name][fold]):] = all_predictions[model_name][fold]
                fold_data[f'{model_name}_prediction'] = aligned_preds
            else:
                fold_data[f'{model_name}_prediction'] = all_predictions[model_name][fold]
        
        combined_data.append(fold_data)
    
    final_df = pd.concat(combined_data, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
    return final_df

def calculate_metrics(final_df, all_predictions):
    for model_name in all_predictions.keys():
        print(f"\nMetrics for {model_name}:")
        predictions = final_df[f'{model_name}_prediction']
        actual = final_df['correct']
        
        # Remove rows where either predictions or actual values are NaN
        valid_indices = (~predictions.isna()) & (~actual.isna())
        valid_predictions = predictions[valid_indices]
        valid_actual = actual[valid_indices]
        
        fpr, tpr, _ = metrics.roc_curve(valid_actual, valid_predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = metrics.accuracy_score(valid_actual, [1 if i >= 0.5 else 0 for i in valid_predictions])
        rmse = math.sqrt(metrics.mean_squared_error(valid_actual, valid_predictions))
        
        print(f"  AUC: {auc:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Valid data points: {sum(valid_indices)} out of {len(predictions)}")

def main(traditional_data_folder, dl_data_folder, output_file):
    all_predictions = get_original_data_predictions(
        traditional_data_folder, dl_data_folder
    )

    # Convert NumPy arrays to lists before JSON serialization
    serializable_predictions = {}
    for model, fold_predictions in all_predictions.items():
        serializable_predictions[model] = {
            fold: predictions.tolist() if hasattr(predictions, 'tolist') else predictions
            for fold, predictions in fold_predictions.items()
        }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_predictions, f, indent=4)


if __name__ == '__main__':
    traditional_data_folder = 'data/processed/train/tabular'
    dl_data_folder = 'data/processed/train/sequential'
    output_file = 'output/predictions.json'
    main(traditional_data_folder, dl_data_folder, output_file)