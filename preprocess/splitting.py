import copy
import pandas as pd


def train_test_split(df, test_ratio=0.2, random_state=1024):
    """
    Splits the dataframe into train and test sets.
    
    Args:
        df: Input dataframe
        test_ratio: Ratio of test data (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Shuffle the dataframe
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Calculate split point
    test_num = int(len(df) * test_ratio)
    
    # Split
    test_df = df[:test_num]
    train_df = df[test_num:]
    
    print(f"Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


def KFold_split(df, k=5, random_state=1024):
    """
    Assigns a fold ID to each row in the dataframe for K-fold cross-validation.
    
    Args:
        df: Input dataframe
        k: Number of folds (default: 5)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with added 'fold' column
    """
    # Shuffle the dataframe
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Calculate fold sizes
    fold_size = len(df) // k
    remainder = len(df) % k
    
    # Assign folds
    folds = []
    for i in range(k):
        # First 'remainder' folds get one extra sample
        size = fold_size + (1 if i < remainder else 0)
        folds.extend([i] * size)
    
    df["fold"] = folds
    
    # Print fold distribution
    for i in range(k):
        count = folds.count(i)
        print(f"Fold {i}: {count} samples")
    
    return df