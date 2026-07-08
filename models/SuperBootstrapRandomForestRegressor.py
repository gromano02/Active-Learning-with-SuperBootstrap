"""
MultinomialWeightedRFRegressor
------------------------------
A RandomForestRegressor subclass where the bootstrap step is replaced by
a Multinomial-distribution weighting scheme:

  1. Draw kn = k*n counts from Multinomial(kn, [1/n, ..., 1/n]).
  2. Use those counts as sample_weight for the tree (no actual resampling).

Everything else — tree building, Cython internals, parallelism, predict,
OOB scoring, warm start — is identical to sklearn's RandomForestRegressor.
The only change is replacing _generate_sample_indices with a Multinomial draw.
"""

import threading
import numpy as np
from warnings import warn

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import (
    _accumulate_prediction,
    _check_sample_weight,
    _fit_context,
    _get_n_samples_bootstrap,
    _generate_unsampled_indices,
    _partition_estimators,
    MAX_INT,
    DTYPE,
    DOUBLE,
)
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.multiclass import type_of_target
from sklearn.utils._tags import get_tags
from sklearn.base import is_classifier
from scipy.sparse import issparse
from joblib import Parallel, delayed


def _multinomial_build_trees(
    tree, X, y, sample_weight, tree_idx, n_trees,
    verbose=0, class_weight=None, n_samples_bootstrap=None,
    missing_values_in_feature_mask=None, k=1.0,
):
    """
    Identical to sklearn's _parallel_build_trees except the bootstrap uses
    a Multinomial draw instead of randint + bincount.
    """
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    n_samples = X.shape[0]
    kn = max(1, int(round(k * n_samples)))

    if sample_weight is None:
        curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
    else:
        curr_sample_weight = sample_weight.copy()

    # --- Multinomial draw: only change from sklearn ---
    rng = check_random_state(tree.random_state)
    counts = rng.multinomial(kn, np.ones(n_samples) / n_samples)
    curr_sample_weight *= counts
    # --------------------------------------------------

    tree._fit(
        X,
        y,
        sample_weight=curr_sample_weight,
        check_input=False,
        missing_values_in_feature_mask=missing_values_in_feature_mask,
    )

    return tree


class MultinomialWeightedRFRegressor(RandomForestRegressor):
    """
    RandomForestRegressor with Multinomial-distribution bootstrap.

    Identical to RandomForestRegressor in every way except the bootstrap:
    instead of sampling n indices with replacement, draws kn = k*n counts
    from Multinomial(kn, uniform) and uses them as sample weights directly
    into sklearn's Cython tree-fitting code path.

    Parameters
    ----------
    k : float, default=1.0
        Multiplier for Multinomial draws. kn = round(k * n_samples).
        k=1 closely matches standard bootstrap behaviour.
    """

    def __init__(self, k: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """
        Build the forest. Identical to BaseForest.fit except the Parallel
        loop calls _multinomial_build_trees instead of _parallel_build_trees.
        """
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        X, y = validate_data(
            self, X, y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            ensure_all_finite=False,
        )

        estimator = type(self.estimator)(criterion=self.criterion)
        missing_values_in_feature_mask = (
            estimator._compute_missing_values_in_feature_mask(
                X, estimator_name=self.__class__.__name__
            )
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples,), for example "
                "using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is not allowed "
                    "for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which is necessary "
                    "for Poisson regression."
                )

        self._n_samples, self.n_outputs_ = y.shape

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if sample_weight is None:
            _sample_weight = expanded_class_weight
        elif expanded_class_weight is None:
            _sample_weight = sample_weight
        else:
            _sample_weight = sample_weight * expanded_class_weight

        self._sample_weight = _sample_weight

        # bootstrap must be True for the Multinomial path; max_samples is ignored
        # (kn = k*n already controls effective sample size)
        if self.max_samples is not None:
            warn(
                "max_samples is ignored by MultinomialWeightedRFRegressor; "
                "use k to control effective sample size.",
                UserWarning,
                stacklevel=2,
            )
        n_samples_bootstrap = int(round(self.k * X.shape[0]))
        self._n_samples_bootstrap = n_samples_bootstrap

        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )
        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for _ in range(n_more_estimators)
            ]

            # --- Only difference: call our function instead of _parallel_build_trees ---
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_multinomial_build_trees)(
                    t, X, y, _sample_weight, i, len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                    missing_values_in_feature_mask=missing_values_in_feature_mask,
                    k=self.k,
                )
                for i, t in enumerate(trees)
            )
            # --------------------------------------------------------------------------

            self.estimators_.extend(trees)

        if self.oob_score and (
            n_more_estimators > 0 or not hasattr(self, "oob_score_")
        ):
            y_type = type_of_target(y)
            if y_type == "unknown" or (
                is_classifier(self) and y_type == "multiclass-multioutput"
            ):
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            if callable(self.oob_score):
                self._set_oob_score_and_attributes(
                    X, y, scoring_function=self.oob_score
                )
            else:
                self._set_oob_score_and_attributes(X, y)

        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    X, y = make_regression(
        n_samples=500, n_features=10, noise=0.1, random_state=0
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ks = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    n_seeds = 20

    print("=" * 65)
    print(f"{'Model':<35} {'mean R²':>8} {'std R²':>8} {'mean RMSE':>10}")
    print("=" * 65)

    for k in ks:
        r2s, rmses = [], []
        for seed in range(n_seeds):
            m = MultinomialWeightedRFRegressor(
                k=k,
                n_estimators=100,
                random_state=seed,
                n_jobs=-1,
            )
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            r2s.append(r2_score(y_te, pred))
            rmses.append(mean_squared_error(y_te, pred) ** 0.5)

        print(
            f"  Multinomial k={k:<20} "
            f"{np.mean(r2s):>8.4f} "
            f"{np.std(r2s):>8.4f} "
            f"{np.mean(rmses):>10.3f}"
        )

    rf_r2, rf_rmse = [], []

    for seed in range(n_seeds):
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)
        rf_r2.append(r2_score(y_te, pred))
        rf_rmse.append(mean_squared_error(y_te, pred) ** 0.5)

    print(
        f"  Standard RF                      "
        f"{np.mean(rf_r2):>8.4f} "
        f"{np.std(rf_r2):>8.4f} "
        f"{np.mean(rf_rmse):>10.3f}"
    )
    print("=" * 65)