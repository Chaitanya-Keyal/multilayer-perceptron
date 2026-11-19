"""ScratchMLP against the handout's specification, piece by piece."""

import numpy as np
import pytest

from common import generate_moons
from part1_mlp_scratch import ACTIVATIONS, ScratchMLP, accuracy, gradient_check, sigmoid

X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = generate_moons()


def test_initialisation_follows_the_handout():
    model = ScratchMLP(7, seed=3)
    rng = np.random.RandomState(3)
    W1 = rng.normal(0, 1 / np.sqrt(2), (2, 7))  # W1 first, then W2, from the same generator
    W2 = rng.normal(0, 1 / np.sqrt(7), (7, 1))
    assert np.array_equal(model.params["W1"], W1) and np.array_equal(model.params["W2"], W2)
    assert np.array_equal(model.params["b1"], np.zeros((1, 7)))
    assert np.array_equal(model.params["b2"], np.zeros((1, 1)))


def test_sigmoid_matches_the_plain_formula():
    z = np.linspace(-30, 30, 601)
    assert np.allclose(sigmoid(z), 1 / (1 + np.exp(-z)), rtol=1e-15, atol=0)
    assert sigmoid(np.array([-1000.0, 1000.0])).tolist() == [0.0, 1.0]  # and no overflow warning


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_loss_matches_a_plain_loop(activation):
    model = ScratchMLP(3, activation, reg_lambda=0.5)
    model.params["b1"][:] = [[0.3, -0.2, 0.1]]
    model.params["b2"][:] = 0.4
    X, y = X_TRAIN[:6], Y_TRAIN[:6]
    W1, b1, W2, b2 = (model.params[k] for k in ("W1", "b1", "W2", "b2"))
    total = 0.0
    for x, target in zip(X, y, strict=True):
        z1 = x @ W1 + b1[0]
        a1 = (
            np.tanh(z1)
            if activation == "tanh"
            else np.array([v if v > 0 else 0.05 * v for v in z1])
        )
        y_hat = 1 / (1 + np.exp(-(a1 @ W2[:, 0] + b2[0, 0])))
        total -= target * np.log(y_hat) + (1 - target) * np.log(1 - y_hat)
    expected = total / 6 + 0.5 / (2 * 6) * ((W1**2).sum() + (W2**2).sum())  # biases not penalised
    loss, _ = model.loss_and_grads(X, y)
    assert loss == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_gradients_match_finite_differences(activation):
    start, smallest_z, errors = gradient_check(X_TRAIN, Y_TRAIN, activation)
    assert start == 0 and smallest_z >= 1e-3  # the first five rows are a valid batch
    assert set(errors) == {"W1", "b1", "W2", "b2"}
    assert max(errors.values()) < 1e-6


def test_gradient_check_skips_a_batch_on_the_kink():
    """Make row 0 hit z = 0 for one hidden unit: the check must move on to rows 1-5."""
    model = ScratchMLP(3)
    X = X_TRAIN.copy()
    w = model.params["W1"][:, 0]
    X[0] = [w[1], -w[0]]  # perpendicular to the unit's weights, so its z is 0
    start, smallest_z, errors = gradient_check(X, Y_TRAIN, "leaky_relu")
    assert start == 1 and smallest_z >= 1e-3
    assert max(errors.values()) < 1e-6


def test_leaky_relu_slope_at_zero_is_the_small_one():
    model = ScratchMLP(1, "leaky_relu")
    assert model._phi_prime(np.array([-1.0, 0.0, 1.0])).tolist() == [0.05, 0.05, 1.0]
    assert model._phi(np.array([-2.0, 0.0, 2.0])).tolist() == [-0.1, 0.0, 2.0]


def test_labels_may_be_flat_or_a_column():
    model = ScratchMLP(4, "tanh")
    loss_flat, grads_flat = model.loss_and_grads(X_TRAIN, Y_TRAIN)
    loss_col, grads_col = model.loss_and_grads(X_TRAIN, Y_TRAIN.reshape(-1, 1))
    assert loss_flat == loss_col
    for key in grads_flat:
        assert grads_flat[key].shape == model.params[key].shape
        assert np.array_equal(grads_flat[key], grads_col[key])


def test_fit_records_every_epoch_and_the_snapshots():
    model = ScratchMLP(4, "tanh", n_epochs=30)
    first = {k: v.copy() for k, v in model.params.items()}
    loss0, grads0 = model.loss_and_grads(X_TRAIN, Y_TRAIN)
    model.fit(X_TRAIN, Y_TRAIN, snapshot_epochs=(1, 30))

    assert len(model.loss_history_) == 30
    assert model.loss_history_[0] == loss0  # the loss of epoch 1 is the loss at initialisation
    assert sorted(model.snapshots) == [1, 30]
    for key in first:  # one step of theta <- theta - lr * dtheta
        assert np.array_equal(model.snapshots[1][key], first[key] - 0.1 * grads0[key])
        assert np.array_equal(model.snapshots[30][key], model.params[key])
        assert model.snapshots[30][key] is not model.params[key]  # a copy, not the live array
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_predictions_are_flat_and_thresholded_at_half():
    model = ScratchMLP(8, "tanh", n_epochs=50).fit(X_TRAIN, Y_TRAIN)
    proba, labels = model.predict_proba(X_TEST), model.predict(X_TEST)
    assert proba.shape == labels.shape == Y_TEST.shape
    assert np.array_equal(labels, (proba >= 0.5).astype(int))
    assert accuracy(Y_TEST, labels) == (labels == Y_TEST).mean()
    # at a snapshot, predictions come from the stored parameters
    model.fit(X_TRAIN, Y_TRAIN, snapshot_epochs=(50,))
    assert np.array_equal(model.predict(X_TEST, model.snapshots[50]), model.predict(X_TEST))


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_one_hidden_node_gives_a_straight_boundary(activation):
    """With H = 1 the predicted class can only change once along the projection w1 . x."""
    model = ScratchMLP(1, activation, n_epochs=300).fit(X_TRAIN, Y_TRAIN)
    projection = (X_TEST @ model.params["W1"]).ravel()
    labels = model.predict(X_TEST)[np.argsort(projection)]
    assert np.count_nonzero(np.diff(labels)) == 1
