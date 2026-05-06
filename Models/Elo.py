import argparse
import numpy as np
import pandas as pd
import math
from collections import defaultdict

DEFAULT_RATING = 0.0


class Elo:
    """
    Vanilla Elo Knowledge Tracing (Pelánek)

    p(correct) = 1 / (1 + exp(-(q_s - d_i)))
    q_s <- q_s + k * (y - p)
    d_i <- d_i + k * (p - y)
    """

    def __init__(self, default_rating=0.0):
        self.default_rating = float(default_rating)
        self._ratings = defaultdict(lambda: self.default_rating)

    @staticmethod
    def _sigmoid(x):
        # numerically stable sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def add_match(self, player_u, player_i, correct, k=1.0):
        """
        player_u = "user_X"
        player_i = "item_Y"
        correct = 0 or 1
        """
        y = float(correct)

        q_s = self._ratings[player_u]  # student ability
        d_i = self._ratings[player_i]  # item difficulty

        p = self._sigmoid(q_s - d_i)

        # Vanilla KT updates
        self._ratings[player_u] = q_s + k * (y - p)
        self._ratings[player_i] = d_i + k * (p - y)

    def ratings(self):
        return dict(self._ratings)


def train_Elo(train_files, k: float = 1.0):
    df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    scores = df_train[['user', 'item', 'correct']].to_numpy()

    engine = Elo(default_rating=DEFAULT_RATING)

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


def train_predict_Elo(train_files, test_file, k: float = 1.0):
    abilities, difficulties = train_Elo(train_files, k=k)
    df_test = pd.read_csv(test_file)

    predictions = []
    actuals = []

    for u, i, correct in df_test[['user', 'item', 'correct']].to_numpy():
        u = int(u)
        i = int(i)
        y = float(correct)

        # Current ratings before seeing this test answer
        q_s = abilities.get(u, DEFAULT_RATING)
        d_i = difficulties.get(i, DEFAULT_RATING)

        # Predict first
        p = Elo._sigmoid(q_s - d_i)

        predictions.append(p)
        actuals.append(y)

        # Then update ONLY student ability.
        # Item difficulty remains fixed during test.
        abilities[u] = q_s + k * (y - p)

    return np.array(predictions, dtype=float), np.array(actuals, dtype=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv', nargs='+', help="One or more training fold CSVs")
    parser.add_argument('test_csv', help="Single test CSV")
    parser.add_argument('--k', type=float, default=1.0, help="Elo step size")
    args = parser.parse_args()

    predictions, actual = train_predict_Elo(args.train_csv, args.test_csv, k=args.k)
    print("Predictions:", predictions[:10])
    print("Actuals:    ", actual[:10])