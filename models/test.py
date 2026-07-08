import numpy as np

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def duplicate_dataset(X, y, k):
    """Duplicate the dataset k times."""
    X_dup = np.tile(X, (k, 1))
    y_dup = np.tile(y, k)
    return X_dup, y_dup


if __name__ == "__main__":
    # Create synthetic data
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        noise=0.1,
        random_state=0,
    )

    # Split ONCE
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
    )

    ks = [1, 2, 3, 5, 10, 20]
    n_seeds = 20

    print("=" * 70)
    print(f"{'k':<5} {'Mean R²':>10} {'Std R²':>10} {'Mean RMSE':>12}")
    print("=" * 70)

    for k in ks:
        # Duplicate ONLY the training set
        X_train_dup, y_train_dup = duplicate_dataset(X_train, y_train, k)

        r2s = []
        rmses = []

        for seed in range(n_seeds):
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=seed,
                n_jobs=-1,
            )

            model.fit(X_train_dup, y_train_dup)

            pred = model.predict(X_test)

            r2s.append(r2_score(y_test, pred))
            rmses.append(np.sqrt(mean_squared_error(y_test, pred)))

        print(
            f"{k:<5}"
            f"{np.mean(r2s):>10.4f}"
            f"{np.std(r2s):>10.4f}"
            f"{np.mean(rmses):>12.3f}"
        )

    print("=" * 70)