"""Part 2: the Table 1 settings reach every model, and partial_fit is used as the handout says."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

import part2_sklearn_exploration as part2
from common import generate_spirals

X_TRAIN, Y_TRAIN, _, _ = generate_spirals()


def test_settings_are_table_1():
    assert part2.SETTINGS == {
        "solver": "adam",
        "learning_rate_init": 0.01,
        "alpha": 1e-4,
        "batch_size": "auto",
        "max_iter": 1500,
        "n_iter_no_change": 1500,
        "random_state": 42,
    }
    assert part2.ARCHITECTURES == ((36,), (18, 18), (12, 12, 12), (6, 6, 6, 6, 6, 6))
    assert all(sum(arch) == 36 for arch in part2.ARCHITECTURES)
    assert part2.ACTIVATIONS == ("logistic", "relu")
    assert part2.WIDTHS == (2, 4, 16, 64)
    assert part2.EVOLUTION_EPOCHS == (1, 25, 100, 300, 1000)


def test_train_passes_the_settings_and_runs_every_epoch(monkeypatch):
    monkeypatch.setitem(part2.SETTINGS, "max_iter", 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = part2.train((4, 4), "relu", X_TRAIN, Y_TRAIN)
    params = clf.get_params()
    assert params["hidden_layer_sizes"] == (4, 4) and params["activation"] == "relu"
    assert all(params[key] == value for key, value in part2.SETTINGS.items())
    assert clf.n_iter_ == 20 and len(clf.loss_curve_) == 20


@pytest.mark.parametrize("M", part2.WIDTHS)
def test_parameter_count_formula(M):
    clf = MLPClassifier(hidden_layer_sizes=(M, M), max_iter=1).partial_fit(
        X_TRAIN, Y_TRAIN, classes=[0, 1]
    )
    assert part2.n_parameters(clf) == M * M + 5 * M + 1


def test_evolution_is_one_partial_fit_call_per_epoch(monkeypatch):
    """Epochs 1 and 3 of the loop against a model stepped by hand."""
    monkeypatch.setattr(part2, "EVOLUTION_EPOCHS", (1, 3))
    snapshots = part2.boundary_evolution(X_TRAIN, Y_TRAIN)
    assert sorted(snapshots) == [(a, t) for a in sorted(part2.ACTIVATIONS) for t in (1, 3)]

    grid = part2.make_grid(X_TRAIN)[2]
    for activation in part2.ACTIVATIONS:
        clf = MLPClassifier(hidden_layer_sizes=(16, 16), activation=activation, **part2.SETTINGS)
        for epoch in (1, 2, 3):
            clf.partial_fit(X_TRAIN, Y_TRAIN, classes=[0, 1])
            if epoch in (1, 3):
                P, train_acc = snapshots[activation, epoch]
                assert np.array_equal(P, clf.predict_proba(grid)[:, 1])
                assert train_acc == clf.score(X_TRAIN, Y_TRAIN)


def test_grid_covers_the_points():
    G1, G2, grid = part2.make_grid(X_TRAIN, n=50)
    assert G1.shape == G2.shape == (50, 50) and grid.shape == (2500, 2)
    assert grid.min(axis=0) == pytest.approx(X_TRAIN.min(axis=0) - 0.5)
    assert grid.max(axis=0) == pytest.approx(X_TRAIN.max(axis=0) + 0.5)
