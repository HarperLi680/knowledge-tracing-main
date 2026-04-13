import copy
import pandas as pd


def KFold_split(df, k=6, random_state=1024):
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