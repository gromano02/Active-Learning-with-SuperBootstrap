
"""
MultinomialWeightedRFRegressor
------------------------------
A RandomForestRegressor subclass where the bootstrap step is replaced by
a Multinomial-distribution weighting scheme:

  1. Draw kn counts from Multinomial(kn, [1/n, ..., 1/n]).
  2. Use those counts as sample_weight for the tree (no actual resampling).

This is the "Bayesian bootstrap" / Poisson-approximation trick:
  - k=1  →  expected weight per point ≈ classic bootstrap
  - k<1  →  more regularisation (sparser weights)
  - k>1  →  more variance (heavier tails)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed


def _fit_single_tree(tree, X, y, k, random_state):
    """Fit one decision tree with Multinomial-bootstrap weights."""
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    kn = max(1, int(round(k * n)))

    # Draw counts ~ Multinomial(kn, uniform)
    counts = rng.multinomial(kn, np.ones(n) / n)   # shape (n,)

    # Only pass points that were actually "sampled" (count > 0)
    mask = counts > 0
    X_sub = X[mask]
    y_sub = y[mask]
    w_sub = counts[mask].astype(float)

    tree.fit(X_sub, y_sub, sample_weight=w_sub)
    return tree


class MultinomialWeightedRFRegressor(RandomForestRegressor):
    """
    RandomForestRegressor with Multinomial-distribution bootstrap.

    Parameters
    ----------
    k : float, default=1.0
        Multiplier for the number of multinomial draws per tree.
        Each tree draws kn = round(k * n_samples) counts from
        Multinomial(kn, uniform), then uses those counts as
        sample weights.  k=1 approximates the classic bootstrap.
    *args, **kwargs
        All other parameters are forwarded to RandomForestRegressor.
        Note: `bootstrap` is ignored — weighting is always used.
    """

    def __init__(self, k: float = 1.0, **kwargs):
        # Force bootstrap=False so the parent never does its own resampling
        kwargs.pop("bootstrap", None)
        super().__init__(bootstrap=False, **kwargs)
        self.k = k

    # ------------------------------------------------------------------
    # Core fit override
    # ------------------------------------------------------------------
    def fit(self, X, y, sample_weight=None):
        """
        Build the forest with Multinomial-weighted trees.

        `sample_weight` passed here is ignored (the weighting comes
        from the Multinomial draws). Raise a warning if supplied.
        """
        if sample_weight is not None:
            import warnings
            warnings.warn(
                "MultinomialWeightedRFRegressor ignores external sample_weight "
                "because weights are determined by the Multinomial bootstrap.",
                UserWarning,
                stacklevel=2,
            )

        # Input validation
        from sklearn.utils.validation import validate_data
        X, y = validate_data(self, X, y, multi_output=True, y_numeric=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        # Build base tree template
        base_tree = DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            splitter="best",
        )

        # Seed management
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_estimators)

        # Fit trees (parallel, same as sklearn)
        trees = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(
            delayed(_fit_single_tree)(clone(base_tree), X, y, self.k, int(s))
            for s in seeds
        )

        self.estimators_ = trees
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]

        # Attributes sklearn's predict path needs
        self.estimators_features_ = [
            np.arange(X.shape[1]) for _ in trees
        ]

        return self

    # ------------------------------------------------------------------
    # Predict — delegate to each tree, average
    # ------------------------------------------------------------------
    def predict(self, X):
        check_is_fitted(self)
        from sklearn.utils.validation import validate_data
        X = validate_data(self, X, reset=False)
        preds = np.array([t.predict(X) for t in self.estimators_])
        return preds.mean(axis=0)

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    print("=" * 55)
    print(f"{'Model':<35} {'R²':>8} {'RMSE':>10}")
    print("=" * 55)

    for k in [0.5, 1.0, 2.0]:
        m = MultinomialWeightedRFRegressor(
            k=k, n_estimators=100, random_state=42, n_jobs=-1
        )
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        r2   = r2_score(y_te, pred)
        rmse = mean_squared_error(y_te, pred) ** 0.5
        print(f"  Multinomial k={k:<20} {r2:>8.4f} {rmse:>10.3f}")

    # Baseline: standard RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    r2   = r2_score(y_te, pred)
    rmse = mean_squared_error(y_te, pred) ** 0.5
    print(f"  Standard RF (bootstrap=True)         {r2:>8.4f} {rmse:>10.3f}")
    print("=" * 55)
