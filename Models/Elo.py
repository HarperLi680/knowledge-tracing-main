import argparse
import numpy as np
import pandas as pd
import math
from collections import defaultdict

DEFAULT_RATING = 0.0


class Elo:
    """
    Elo Knowledge Tracing with Pelánek uncertainty.

    p(correct) = 1 / (1 + exp(-(q_s - d_i)))

    U(n) = a / (1 + b * n)

    Training:
        q_s <- q_s + U(n_s) * (y - p)
        d_i <- d_i + U(n_i) * (p - y)

    Test:
        q_s <- q_s + U(n_s) * (y - p)
        d_i is fixed during test
    """

    def __init__(self, default_rating=0.0, a=4.0, b=0.5):
        self.default_rating = float(default_rating)
        self.a = float(a)
        self.b = float(b)

        self._ratings = defaultdict(lambda: self.default_rating)
        self._counts = defaultdict(int)

    @staticmethod
    def _sigmoid(x):
        # numerically stable sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def uncertainty(self, n):
        return self.a / (1.0 + self.b * n)

    def add_match(self, player_u, player_i, correct):
        """
        Training update.

        player_u = "user_X"
        player_i = "item_Y"
        correct = 0 or 1

        Updates both student ability and item difficulty.
        """
        y = float(correct)

        q_s = self._ratings[player_u]
        d_i = self._ratings[player_i]

        n_s = self._counts[player_u]
        n_i = self._counts[player_i]

        p = self._sigmoid(q_s - d_i)

        u_s = self.uncertainty(n_s)
        u_i = self.uncertainty(n_i)

        self._ratings[player_u] = q_s + u_s * (y - p)
        self._ratings[player_i] = d_i + u_i * (p - y)

        self._counts[player_u] += 1
        self._counts[player_i] += 1

    def ratings(self):
        return dict(self._ratings)

    def counts(self):
        return dict(self._counts)


def train_Elo(train_files, a: float = 4.0, b: float = 0.5):
    df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    scores = df_train[['user', 'item', 'correct']].to_numpy()

    engine = Elo(default_rating=DEFAULT_RATING, a=a, b=b)

    for u, i, correct in scores:
        player_u = f"user_{int(u)}"
        player_i = f"item_{int(i)}"
        engine.add_match(player_u, player_i, float(correct))

    all_ratings = engine.ratings()
    all_counts = engine.counts()

    abilities = {
        int(name.split("_", 1)[1]): rating
        for name, rating in all_ratings.items()
        if name.startswith("user_")
    }

    difficulties = {
        int(name.split("_", 1)[1]): rating
        for name, rating in all_ratings.items()
        if name.startswith("item_")
    }

    student_counts = {
        int(name.split("_", 1)[1]): count
        for name, count in all_counts.items()
        if name.startswith("user_")
    }

    item_counts = {
        int(name.split("_", 1)[1]): count
        for name, count in all_counts.items()
        if name.startswith("item_")
    }

    return abilities, difficulties, student_counts, item_counts


def train_predict_Elo(train_files, test_file, a: float = 4.0, b: float = 0.5):
    abilities, difficulties, student_counts, item_counts = train_Elo(
        train_files,
        a=a,
        b=b,
    )

    df_test = pd.read_csv(test_file)

    predictions = []
    actuals = []

    def uncertainty(n):
        return a / (1.0 + b * n)

    for u, i, correct in df_test[['user', 'item', 'correct']].to_numpy():
        u = int(u)
        i = int(i)
        y = float(correct)

        # Current values before seeing this test answer
        q_s = abilities.get(u, DEFAULT_RATING)
        d_i = difficulties.get(i, DEFAULT_RATING)
        n_s = student_counts.get(u, 0)

        # Predict first
        p = Elo._sigmoid(q_s - d_i)

        predictions.append(p)
        actuals.append(y)

        # Online test-time adaptation:
        # update ONLY student ability using Pelánek uncertainty.
        # Item difficulty remains fixed during test.
        u_s = uncertainty(n_s)
        abilities[u] = q_s + u_s * (y - p)
        student_counts[u] = n_s + 1

    return np.array(predictions, dtype=float), np.array(actuals, dtype=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv', nargs='+', help="One or more training fold CSVs")
    parser.add_argument('test_csv', help="Single test CSV")
    parser.add_argument('--a', type=float, default=4.0, help="Uncertainty numerator")
    parser.add_argument('--b', type=float, default=0.5, help="Uncertainty decay")
    args = parser.parse_args()

    predictions, actual = train_predict_Elo(
        args.train_csv,
        args.test_csv,
        a=args.a,
        b=args.b,
    )

    print("Predictions:", predictions[:10])
    print("Actuals:    ", actual[:10])