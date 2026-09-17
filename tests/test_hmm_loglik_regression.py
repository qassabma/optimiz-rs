"""Regression test: Baum-Welch must not stop after 2 iterations.

Before the fix the convergence check used log(sum(alpha[-1])) on a row that had just been
normalised to sum 1, i.e. always log(1) = 0, so iteration 2 always looked converged.
Synthetic two-regime data is used on purpose: the true parameters must be known.
Runs against the pure-Python trainer so it needs no compiled extension.
"""
import numpy as np
from optimizr.hmm import HMM


def _series():
    rng = np.random.default_rng(7)
    return np.concatenate([rng.normal(0, 0.2, 150), rng.normal(0, 2.0, 150),
                           rng.normal(0, 0.2, 150), rng.normal(0, 2.0, 150)])


def _fit(n_iterations):
    h = HMM(n_states=2)
    h._fit_python(_series(), n_iterations, 1e-9)
    return np.sort(h.emission_stds_), np.sort(np.diag(h.transition_matrix_))


def test_training_continues_past_two_iterations():
    s2, _ = _fit(2)
    s100, _ = _fit(100)
    assert not np.allclose(s2, s100), "100 iterations equal 2 iterations: training stops early"


def test_recovers_the_two_regimes():
    stds, stay = _fit(100)
    assert abs(stds[0] - 0.2) < 0.05 and abs(stds[1] - 2.0) < 0.3
    assert stay.min() > 0.95          # true stay probability is 149/150
