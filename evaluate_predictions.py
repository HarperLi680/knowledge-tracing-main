import os
import pandas as pd
import numpy as np
from sklearn import metrics
import math
import json

def load_predictions_from_json(json_file):
    """Load predictions from JSON file and convert back to numpy arrays"""
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    
    # Convert lists back to numpy arrays and ensure fold keys are integers
    converted_predictions = {}
    for model in predictions:
        converted_predictions[model] = {}
        for fold_str, pred_list in predictions[model].items():
            fold_int = int(fold_str)  # Convert string keys back to integers
            converted_predictions[model][fold_int] = np.array(pred_list)
    
    return converted_predictions

def process_folds(data_folder):
    print(f"Looking for data files in: {data_folder}")
    files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
    print(f"Found {len(files)} .csv files")
    return files

def align_atkt_predictions(seq_file, preds, seqlen):
    """
    Expand ATKT next-step predictions back to row-level alignment.
    Inserts NaN for the first interaction of each chunk because ATKT
    cannot predict it.
    """
    aligned = []
    pred_ptr = 0

    with open(seq_file, 'r') as f:
        line_id = 0
        current_skills = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line_id % 4 == 1:
                current_skills = [x for x in line.split(',') if x != ""]

                # reproduce the same chunking used in DATA.load_data()
                for start in range(0, len(current_skills), seqlen):
                    chunk_len = min(seqlen, len(current_skills) - start)

                    # first position in each chunk has no ATKT prediction
                    aligned.append(np.nan)

                    # remaining positions in chunk do have predictions
                    n_preds_this_chunk = chunk_len - 1
                    aligned.extend(preds[pred_ptr:pred_ptr + n_preds_this_chunk])
                    pred_ptr += n_preds_this_chunk

            line_id += 1

    if pred_ptr != len(preds):
        raise ValueError(
            f"ATKT alignment mismatch: consumed {pred_ptr} predictions, "
            f"but model returned {len(preds)}"
        )

    return np.array(aligned)

def align_dsakt_predictions(seq_file, preds):
    """
    Align DSAKT predictions back to row-level interactions.

    DSAKT predicts next-step correctness for each original sequence:
    for a sequence of length L, it returns L-1 predictions corresponding
    to original positions 1..L-1.

    So alignment is:
        [NaN, pred_0, pred_1, ..., pred_(L-2)]
    for each student sequence.
    """
    preds = np.asarray(preds, dtype=float)
    aligned = []
    pred_ptr = 0

    with open(seq_file, 'r') as f:
        line_id = 0

        for line in f:
            line = line.strip()
            if not line:
                continue

            # skills line
            if line_id % 4 == 1:
                skills = [x for x in line.split(',') if x != ""]
                seq_len = len(skills)

                if seq_len == 0:
                    continue

                # first interaction has no DSAKT prediction
                aligned.append(np.nan)

                n_preds = seq_len - 1
                if n_preds > 0:
                    aligned.extend(preds[pred_ptr:pred_ptr + n_preds])
                    pred_ptr += n_preds

            line_id += 1

    if pred_ptr != len(preds):
        raise ValueError(
            f"DSAKT alignment mismatch: consumed {pred_ptr} predictions, "
            f"but model returned {len(preds)}"
        )

    return np.array(aligned, dtype=float)

def align_dkt_predictions(seq_file, preds):
    """
    Align DKT predictions back to row-level interactions.

    DKT predicts next-step correctness for each original sequence:
    for a sequence of length L, it returns L-1 predictions corresponding
    to original positions 1..L-1.

    So alignment is:
        [NaN, pred_0, pred_1, ..., pred_(L-2)]
    for each student sequence.
    """
    preds = np.asarray(preds, dtype=float)
    aligned = []
    pred_ptr = 0

    with open(seq_file, 'r') as f:
        line_id = 0

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line_id % 4 == 1:
                skills = [x for x in line.split(',') if x != ""]
                seq_len = len(skills)

                if seq_len == 0:
                    continue

                aligned.append(np.nan)

                n_preds = seq_len - 1
                if n_preds > 0:
                    aligned.extend(preds[pred_ptr:pred_ptr + n_preds])
                    pred_ptr += n_preds

            line_id += 1

    if pred_ptr != len(preds):
        raise ValueError(
            f"DKT alignment mismatch: consumed {pred_ptr} predictions, "
            f"but model returned {len(preds)}"
        )

    return np.array(aligned, dtype=float)

def create_combined_csv(
    traditional_data_folder,
    traditional_files,
    dl_data_folder,
    dl_files,
    all_predictions,
    num_folds,
    output_file,
    atkt_seqlen=100
):
    combined_data = []

    for fold in range(num_folds):
        fold_file = os.path.join(traditional_data_folder, traditional_files[fold])
        fold_data = pd.read_csv(fold_file).reset_index(drop=True)
        fold_data['fold'] = fold

        seq_file = os.path.join(dl_data_folder, dl_files[fold])

        for model_name in all_predictions.keys():
            preds = np.asarray(all_predictions[model_name][fold])
            print(f"Fold {fold}, {model_name}: raw predictions = {len(preds)}")

            if model_name == 'ATKT':
                aligned_preds = align_atkt_predictions(
                    seq_file=seq_file,
                    preds=preds,
                    seqlen=atkt_seqlen
                )

            elif model_name == 'DSAKT':
                aligned_preds = align_dsakt_predictions(
                    seq_file=seq_file,
                    preds=preds
                )
            
            elif model_name == 'DKT':
                aligned_preds = align_dkt_predictions(
                    seq_file=seq_file,
                    preds=preds
                )

            else:
                aligned_preds = preds

            if len(aligned_preds) != len(fold_data):
                raise ValueError(
                    f"{model_name} aligned length mismatch in fold {fold}: "
                    f"{len(aligned_preds)} vs {len(fold_data)}"
                )

            fold_data[f'{model_name}_prediction'] = aligned_preds

        combined_data.append(fold_data)

    final_df = pd.concat(combined_data, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
    return final_df

def calculate_metrics(final_df, all_predictions, output_json='output/metrics.json'):
    metrics_dict = {}
    
    for model_name in all_predictions.keys():
        print(f"\nMetrics for {model_name}:")
        
        if f'{model_name}_prediction' not in final_df.columns:
            print(f"  No predictions found for {model_name}")
            metrics_dict[model_name] = {"error": "No predictions found"}
            continue
            
        predictions = final_df[f'{model_name}_prediction']
        actual = final_df['correct']
        
        # Remove rows where either predictions or actual values are NaN
        valid_indices = (~predictions.isna()) & (~actual.isna())
        valid_predictions = predictions[valid_indices]
        valid_actual = actual[valid_indices]
        
        if len(valid_predictions) == 0:
            print(f"  No valid predictions for {model_name}")
            metrics_dict[model_name] = {"error": "No valid predictions"}
            continue
        
        try:
            fpr, tpr, _ = metrics.roc_curve(valid_actual, valid_predictions, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            acc = metrics.accuracy_score(valid_actual, [1 if i >= 0.5 else 0 for i in valid_predictions])
            rmse = math.sqrt(metrics.mean_squared_error(valid_actual, valid_predictions))
            
            # Store metrics in dictionary
            metrics_dict[model_name] = {
                "AUC": float(auc),
                "Accuracy": float(acc),
                "RMSE": float(rmse),
                "Valid_data_points": int(sum(valid_indices)),
                "Total_data_points": int(len(predictions))
            }
            
            print(f"  AUC: {auc:.4f}")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Valid data points: {sum(valid_indices)} out of {len(predictions)}")
        except Exception as e:
            print(f"  Error calculating metrics: {str(e)}")
            metrics_dict[model_name] = {"error": str(e)}
    
    # Save metrics to JSON file
    with open(output_json, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nMetrics saved to {output_json}")
    
    return metrics_dict

def main():
    traditional_data_folder = 'data/processed/train/tabular'
    dl_data_folder = 'data/processed/train/sequential'
    json_file = 'output/predictions.json'
    output_csv = 'output/combined_output.csv'
    output_metrics = 'output/metrics.json'
    
    # Load predictions from JSON
    print("Loading predictions from JSON...")
    all_predictions = load_predictions_from_json(json_file)
    
    # Get traditional files info
    traditional_files = process_folds(traditional_data_folder)
    dl_files = process_folds(dl_data_folder)
    num_folds = len(traditional_files)
    
    print(f"Found {num_folds} folds in traditional data")
    print(f"Loaded predictions for models: {list(all_predictions.keys())}")
    
    # Create combined CSV
    print("\nCreating combined CSV...")
    final_df = create_combined_csv(
        traditional_data_folder, 
        traditional_files,
        dl_data_folder,
        dl_files,
        all_predictions, 
        num_folds, 
        output_csv
    )
    
    # Calculate and display metrics
    print("\nCalculating metrics...")
    metrics_dict = calculate_metrics(final_df, all_predictions, output_metrics)
    
    print(f"\nFinal CSV shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")

if __name__ == '__main__':
    main()