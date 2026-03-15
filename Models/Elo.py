import argparse
import numpy as np
import pandas as pd
from elo_rating import Elo

DEFAULT_RATING = 0.0

def train_Elo(train_files, k: float = 1):
    df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    scores = df_train[['user', 'item', 'correct']].to_numpy()

    engine = Elo()

    for u, i, correct in scores:
        player_u = f"user_{int(u)}"
        player_i = f"item_{int(i)}"
        engine.add_match(player_u, player_i, float(correct), k=k)

    all_ratings = engine.ratings()

    abilities = {
        int(name.split("_", 1)[1]): r
        for name, r in all_ratings.items()
        if name.startswith("user_")
    }
    difficulties = {
        int(name.split("_", 1)[1]): r
        for name, r in all_ratings.items()
        if name.startswith("item_")
    }

    return abilities, difficulties

def train_predict_Elo(train_files, test_file, k: float = 1):
    abilities, difficulties = train_Elo(train_files, k=k)
    df_test = pd.read_csv(test_file)
    
    #Changing scale of original formula from elo
    def win_prob(u, i, scale=2.0):
        r_u = abilities.get(u, DEFAULT_RATING)
        r_i = difficulties.get(i, DEFAULT_RATING)
        return 1.0 / (1.0 + 10 ** ((r_i - r_u) / scale))

    rows = df_test[['user', 'item']].to_numpy()
    probs = np.array([win_prob(int(u), int(i)) for u, i in rows], dtype=float)

    for idx, (u, i) in enumerate(rows[:20]):
        u = int(u)
        i = int(i)

    return probs, df_test['correct'].to_numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv', nargs='+', help="One or more training fold CSVs")
    parser.add_argument('test_csv', help="Single test CSV")
    args = parser.parse_args()

    predictions, actual = train_predict_Elo(args.train_csv, args.test_csv)
    print("Predictions:", predictions[:10])
    print("Actuals:    ", actual[:10])