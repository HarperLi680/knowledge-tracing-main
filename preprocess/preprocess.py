"""
Preprocessing script for ASSISTments 2009 dataset.
Converts raw CSV data into formatted sequences for knowledge tracing models.
"""

import pandas as pd
import argparse
import os
import sys
from collections import defaultdict

# Add parent directory to path to allow absolute imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cleaning import (
    id_mapping, 
    generate_sequences, 
    save_id2idx, 
    get_max_concepts,
    extend_multi_concepts
)
from splitting import train_test_split, KFold_split


def read_and_format_raw_data(input_csv, min_seq_len=3):
    """
    Read raw ASSISTments CSV and convert to dataframe format.
    
    Args:
        input_csv: Path to raw CSV file
        min_seq_len: Minimum sequence length to keep
        
    Returns:
        DataFrame with uid, questions, concepts, responses columns
    """
    df = pd.read_csv(input_csv, encoding="ISO-8859-1", dtype=str, low_memory=False)
    print(f"Original data shape: {df.shape}")
    
    # Drop rows with missing critical values
    df['tmp_index'] = range(len(df))
    df = df.dropna(subset=["user_id", "problem_id", "skill_id", "correct", "order_id"])
    print(f"After dropping NA: {df.shape}")
    
    # Group by user and create sequences
    data = {'uid': [], 'questions': [], 'concepts': [], 'responses': []}
    
    for user_id, group in df.groupby('user_id', sort=False):
        group = group.sort_values(by=['order_id', 'tmp_index'])
        
        if len(group) < min_seq_len:
            continue
            
        data['uid'].append(str(user_id))
        data['questions'].append(','.join(group['problem_id'].astype(str)))
        data['concepts'].append(','.join(group['skill_id'].astype(str)))
        data['responses'].append(','.join(group['correct'].astype(str)))
    
    return pd.DataFrame(data)


def convert_to_tabular_with_features(df, fold_id=None):
    """
    Convert sequence data to tabular format with b4_correct and b4_incorrect features.
    
    Args:
        df: Input dataframe with sequences
        fold_id: Fold ID to filter (None means process all rows)
    """
    tabular_rows = []
    user_skill_counts = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    
    for _, row in df.iterrows():
        # Filter by fold if specified
        if fold_id is not None and 'fold' in row and row['fold'] != fold_id:
            continue
            
        uid = row['uid']
        questions = row['questions'].split(',')
        concepts = row['concepts'].split(',')
        responses = row['responses'].split(',')
        
        for q, c, r in zip(questions, concepts, responses):
            q, c, r = int(q), int(c), int(r)
            
            if q == -1 or c == -1 or r == -1:  # Skip padding
                continue
                
            user_skill = (uid, c)
            b4_correct = user_skill_counts[user_skill]['correct']
            b4_incorrect = user_skill_counts[user_skill]['incorrect']
            
            tabular_rows.append({
                'user': uid,
                'skill': c,
                'item': q,
                'correct': r,
                'b4_correct': b4_correct,
                'b4_incorrect': b4_incorrect
            })
            
            if r == 1:
                user_skill_counts[user_skill]['correct'] += 1
            else:
                user_skill_counts[user_skill]['incorrect'] += 1
    
    return pd.DataFrame(tabular_rows)


def convert_to_sequential_format(df, output_file, fold_id=None):
    """
    Convert to text-based sequential format for deep learning models.
    Format: user_id,length / skills / questions / responses
    
    Args:
        df: Input dataframe
        output_file: Path to output file
        fold_id: Fold ID to filter (None means process all rows)
    """
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            # Filter by fold if specified
            if fold_id is not None and 'fold' in row and row['fold'] != fold_id:
                continue
                
            uid = row['uid']
            questions = row['questions'].split(',')
            concepts = row['concepts'].split(',')
            responses = row['responses'].split(',')
            
            # Remove padding
            valid = [(q, c, r) for q, c, r in zip(questions, concepts, responses)
                    if int(q) != -1 and int(c) != -1 and int(r) != -1]
            
            if not valid:
                continue
                
            questions, concepts, responses = zip(*valid)
            
            f.write(f"{uid},{len(questions)}\n")
            f.write(",".join(concepts) + "\n")
            f.write(",".join(questions) + "\n")
            f.write(",".join(responses) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess ASSISTments 2009 dataset')
    parser.add_argument('--dataset_name', type=str, default='assistments09',
                       help='Name of the dataset')
    parser.add_argument('--raw_csv', type=str, 
                       default='data/raw/skill_builder_data_corrected_collapsed.csv',
                       help='Path to raw CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--min_seq_len', type=int, default=3,
                       help='Minimum sequence length')
    parser.add_argument('--maxlen', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--kfold', type=int, default=5,
                       help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    
    print("="*50)
    print("STEP 1: Reading and formatting raw data")
    print("="*50)
    total_df = read_and_format_raw_data(args.raw_csv, args.min_seq_len)
    print(f"Total sequences: {len(total_df)}")
    
    print("\n" + "="*50)
    print("STEP 2: Calculate max concepts BEFORE ID mapping")
    print("="*50)
    # Get effective keys from the dataframe
    effective_keys = set(total_df.columns)
    
    # Calculate max concepts on original string IDs
    max_concepts = get_max_concepts(total_df) if 'concepts' in total_df.columns else -1
    print(f"Max concepts per question: {max_concepts}")
    
    print("\n" + "="*50)
    print("STEP 3: Extend multi-concepts")
    print("="*50)
    total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
    print(f"After extending: {len(total_df)} sequences")
    
    print("\n" + "="*50)
    print("STEP 4: Mapping IDs to integers")
    print("="*50)
    total_df, dkeyid2idx = id_mapping(total_df)
    dkeyid2idx["max_concepts"] = max_concepts
    
    print(f"Number of users: {len(dkeyid2idx.get('uid', {}))}")
    print(f"Number of questions: {len(dkeyid2idx.get('questions', {}))}")
    print(f"Number of concepts: {len(dkeyid2idx.get('concepts', {}))}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_id2idx(dkeyid2idx, os.path.join(args.output_dir, "keyid2idx.json"))
    
    print("\n" + "="*50)
    print("STEP 5: Train/Test Split")
    print("="*50)
    train_df, test_df = train_test_split(total_df, 0.2)
    print(f"Train sequences: {len(train_df)}")
    print(f"Test sequences: {len(test_df)}")
    
    print("\n" + "="*50)
    print("STEP 6: K-Fold split on training data")
    print("="*50)
    train_df = KFold_split(train_df, args.kfold)
    train_df['fold'] = train_df['fold'].astype(int)
    
    print("\n" + "="*50)
    print("STEP 7: Generating padded sequences")
    print("="*50)
    # Add fold to effective keys
    effective_keys.add('fold')
    train_sequences = generate_sequences(train_df, effective_keys, args.min_seq_len, args.maxlen)
    print(f"Train sequences after padding: {len(train_sequences)}")
    
    # Generate test sequences (without fold)
    test_effective_keys = effective_keys - {'fold'}
    test_sequences = generate_sequences(test_df, test_effective_keys, args.min_seq_len, args.maxlen)
    print(f"Test sequences after padding: {len(test_sequences)}")
    
    print("\n" + "="*50)
    print("STEP 8: Creating training fold files")
    print("="*50)
    
    train_tabular_dir = os.path.join(args.output_dir, "train", "tabular")
    train_sequential_dir = os.path.join(args.output_dir, "train", "sequential")
    os.makedirs(train_tabular_dir, exist_ok=True)
    os.makedirs(train_sequential_dir, exist_ok=True)
    
    for fold in range(args.kfold):
        print(f"\nProcessing fold {fold}...")
        
        # Tabular format
        tabular_df = convert_to_tabular_with_features(train_sequences, fold_id=fold)
        tabular_file = os.path.join(train_tabular_dir, f"converted_fold_{fold}.csv")
        tabular_df.to_csv(tabular_file, index=False)
        print(f"  ✓ Tabular: {len(tabular_df)} interactions")
        
        # Sequential format
        sequential_file = os.path.join(train_sequential_dir, f"converted_fold_{fold}.csv")
        convert_to_sequential_format(train_sequences, sequential_file, fold_id=fold)
        print(f"  ✓ Sequential: saved")
    
    print("\n" + "="*50)
    print("STEP 9: Creating test set files")
    print("="*50)
    
    test_tabular_dir = os.path.join(args.output_dir, "test", "tabular")
    test_sequential_dir = os.path.join(args.output_dir, "test", "sequential")
    os.makedirs(test_tabular_dir, exist_ok=True)
    os.makedirs(test_sequential_dir, exist_ok=True)
    
    # Test tabular format
    test_tabular_df = convert_to_tabular_with_features(test_sequences)
    test_tabular_file = os.path.join(test_tabular_dir, "test.csv")
    test_tabular_df.to_csv(test_tabular_file, index=False)
    print(f"✓ Test tabular: {len(test_tabular_df)} interactions")
    
    # Test sequential format
    test_sequential_file = os.path.join(test_sequential_dir, "test.csv")
    convert_to_sequential_format(test_sequences, test_sequential_file)
    print(f"✓ Test sequential: saved")
    
    print("\n" + "="*50)
    print("Preprocessing complete!")
    print("="*50)
    print(f"Output structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── keyid2idx.json")
    print(f"    ├── train/")
    print(f"    │   ├── tabular/converted_fold_0..{args.kfold-1}.csv")
    print(f"    │   └── sequential/converted_fold_0..{args.kfold-1}.csv")
    print(f"    └── test/")
    print(f"        ├── tabular/test.csv")
    print(f"        └── sequential/test.csv")


if __name__ == "__main__":
    main()