"""Part 1: a two-layer MLP (2 -> H -> 1) written from scratch in numpy, trained on Two-Moons.

    python part1_mlp_scratch.py

Prints the gradient-check errors and the rows of Table 2, and saves part1_boundaries.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import generate_moons

HIDDEN_SIZES = (1, 2, 8, 32)
ACTIVATIONS = ("leaky_relu", "tanh")
LEAK = 0.05  # slope of leaky ReLU for z <= 0
SNAPSHOT_EPOCHS = (10, 200, 2000)

CLASS_COLORS = ("#2a78d6", "#eb6834")
REGION_COLORS = ("#d9e7f8", "#fbe4da")
LINE_COLOR = "#222222"


def sigmoid(z):
    # same value as 1 / (1 + exp(-z)), but exp never gets a large positive argument
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def pct(x):
    return f"{100 * x:.2f}"


class ScratchMLP:
    """One hidden layer of n_hidden nodes and a sigmoid output, for 2 input features."""

    def __init__(
        self, n_hidden, activation="leaky_relu", lr=0.1, reg_lambda=1e-4, n_epochs=2000, seed=42
    ):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        self.n_hidden = n_hidden
        self.activation = activation
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        rng = np.random.RandomState(seed)
        W1 = rng.normal(0, 1 / np.sqrt(2), (2, n_hidden))  # drawn first
        W2 = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, 1))
        self.params = {
            "W1": W1,
            "b1": np.zeros((1, n_hidden)),
            "W2": W2,
            "b2": np.zeros((1, 1)),
        }
        self.loss_history_ = []
        self.snapshots = {}

    def _phi(self, Z):
        if self.activation == "tanh":
            return np.tanh(Z)
        return np.where(Z > 0, Z, LEAK * Z)

    def _phi_prime(self, Z):
        if self.activation == "tanh":
            return 1 - np.tanh(Z) ** 2
        return np.where(Z > 0, 1.0, LEAK)

    def _forward(self, X, params=None):
        p = self.params if params is None else params
        Z1 = X @ p["W1"] + p["b1"]
        A1 = self._phi(Z1)
        Z2 = A1 @ p["W2"] + p["b2"]
        return Z1, A1, sigmoid(Z2)

    def loss_and_grads(self, X, y):
        """Regularised BCE loss and its gradients at the current parameters."""
        p, lam, N = self.params, self.reg_lambda, len(X)
        y = np.reshape(y, (N, 1))
        Z1, A1, y_hat = self._forward(X)

        clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)
        bce = -np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
        penalty = lam / (2 * N) * (np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2))

        dZ2 = (y_hat - y) / N
        dA1 = dZ2 @ p["W2"].T
        dZ1 = dA1 * self._phi_prime(Z1)
        grads = {
            "W1": X.T @ dZ1 + lam / N * p["W1"],
            "b1": dZ1.sum(axis=0, keepdims=True),
            "W2": A1.T @ dZ2 + lam / N * p["W2"],
            "b2": dZ2.sum(axis=0, keepdims=True),
        }
        return bce + penalty, grads

    def fit(self, X, y, snapshot_epochs=None):
        """Full-batch gradient descent. loss_history_[t - 1] is the loss at the start of epoch t,
        snapshots[t] the parameters after the update of epoch t."""
        snapshot_epochs = snapshot_epochs or ()
        self.loss_history_, self.snapshots = [], {}
        for epoch in range(1, self.n_epochs + 1):
            loss, grads = self.loss_and_grads(X, y)
            self.loss_history_.append(loss)
            for key in self.params:
                self.params[key] -= self.lr * grads[key]
            if epoch in snapshot_epochs:
                self.snapshots[epoch] = {k: v.copy() for k, v in self.params.items()}
        return self

    def predict_proba(self, X, params=None):
        """y_hat for every row of X, shape (N,). params can be one of the snapshots."""
        return self._forward(X, params)[2].ravel()

    def predict(self, X, params=None):
        return (self.predict_proba(X, params) >= 0.5).astype(int)


def gradient_check(X, y, activation, n_rows=5, n_hidden=3, eps=1e-5):
    """Compare the analytic gradients with central differences on n_rows training rows.

    Returns the first row of the batch, its smallest |pre-activation| and the largest relative
    error of each gradient."""
    model = ScratchMLP(n_hidden, activation)
    start = 0
    # leaky ReLU has a kink at 0, so skip a batch that has a pre-activation too close to it
    while np.abs(model._forward(X[start : start + n_rows])[0]).min() < 1e-3:
        start += 1
    Xb, yb = X[start : start + n_rows], y[start : start + n_rows]
    smallest_z = np.abs(model._forward(Xb)[0]).min()

    _, grads = model.loss_and_grads(Xb, yb)
    errors = {}
    for key, theta in model.params.items():
        numeric = np.zeros_like(theta)
        for k in np.ndindex(*theta.shape):
            old = theta[k]
            theta[k] = old + eps
            loss_plus, _ = model.loss_and_grads(Xb, yb)
            theta[k] = old - eps
            loss_minus, _ = model.loss_and_grads(Xb, yb)
            theta[k] = old
            numeric[k] = (loss_plus - loss_minus) / (2 * eps)
        diff = np.abs(grads[key] - numeric)
        errors[key] = np.max(diff / np.maximum(np.abs(grads[key]) + np.abs(numeric), 1e-12))
    return start, smallest_z, errors


def logistic_regression(X, y, lr=0.1, reg_lambda=1e-4, n_epochs=2000):
    """Weights and bias of a linear classifier trained with the same loss and settings as the
    MLP. Used only to compare the H = 1 network with."""
    N = len(X)
    y = y.reshape(N, 1)
    w, b = np.zeros((2, 1)), 0.0
    for _ in range(n_epochs):
        dZ = (sigmoid(X @ w + b) - y) / N
        w -= lr * (X.T @ dZ + reg_lambda / N * w)
        b -= lr * dZ.sum()
    return w, b


def make_grid(X, n=300, margin=0.5):
    """An n x n grid covering the points X, as two meshgrid arrays and as an (n * n, 2) array."""
    x1 = np.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, n)
    x2 = np.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, n)
    G1, G2 = np.meshgrid(x1, x2)
    return G1, G2, np.column_stack([G1.ravel(), G2.ravel()])


def plot_boundary(ax, G1, G2, P, X, y):
    """P is P(class 1) on the grid: shade the two regions, draw the 0.5 line, plot the points."""
    P = P.reshape(G1.shape)
    ax.contourf(G1, G2, P, levels=[0, 0.5, 1], colors=REGION_COLORS)
    if P.min() < 0.5 < P.max():
        ax.contour(G1, G2, P, levels=[0.5], colors=LINE_COLOR, linewidths=1.2)
    for label, color in enumerate(CLASS_COLORS):
        pts = X[y == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color, edgecolors="white", linewidths=0.4)
    ax.tick_params(labelsize=8)


def add_legend(fig):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=c, label=f"class {k}")
        for k, c in enumerate(CLASS_COLORS)
    ]
    handles.append(plt.Line2D([], [], color=LINE_COLOR, linewidth=1.2, label="decision boundary"))
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False)


def boundary_figure(models, X, y):
    """2 x 3 grid: one row per activation, one column per snapshot epoch."""
    G1, G2, grid = make_grid(X)
    fig, axes = plt.subplots(
        2, 3, figsize=(11, 7.5), sharex=True, sharey=True, layout="constrained"
    )
    for row, activation in enumerate(ACTIVATIONS):
        model = models[activation]
        axes[row, 0].set_ylabel(activation, fontsize=13)
        for col, epoch in enumerate(SNAPSHOT_EPOCHS):
            params = model.snapshots[epoch]
            plot_boundary(axes[row, col], G1, G2, model.predict_proba(grid, params), X, y)
            acc = pct(accuracy(y, model.predict(X, params)))
            axes[row, col].set_title(f"epoch {epoch}, train acc. {acc}%", fontsize=12)
    add_legend(fig)
    fig.suptitle("ScratchMLP with H = 8 on Two-Moons: boundary and training points")
    return fig


def main():
    X_train, y_train, X_test, y_test = generate_moons()

    print("Gradient check (H = 3, eps = 1e-5, central differences)")
    checks = {}
    for activation in ACTIVATIONS:
        start, smallest_z, errors = gradient_check(X_train, y_train, activation)
        assert max(errors.values()) < 1e-6, f"gradient check failed for {activation}"
        worst = max(errors["W1"], errors["W2"])
        checks[activation] = {"start": start, "smallest_z": smallest_z, "worst": worst, **errors}
        print(f"  {activation}: training rows {start}-{start + 4}, smallest |z| = {smallest_z:.4f}")
        print(
            f"    max relative error {worst:.3e} (dW1 {errors['W1']:.3e}, dW2 {errors['W2']:.3e})"
        )
        print(f"    biases: db1 {errors['b1']:.3e}, db2 {errors['b2']:.3e}")

    print("\nTable 2: ScratchMLP on Two-Moons (T = 2000, lr = 0.1, lambda = 1e-4)")
    row = "  {:<10}  {:>2}  {:>16}  {:>14}  {:>13}  {:>22}"
    header = ("Activation", "H", "Final train loss", "Train acc. (%)", "Test acc. (%)")
    print(row.format(*header, "Loss after last update"))
    rows, models = [], {}
    for activation in ACTIVATIONS:
        for H in HIDDEN_SIZES:
            model = ScratchMLP(H, activation).fit(X_train, y_train, SNAPSHOT_EPOCHS)
            loss = model.loss_history_[-1]
            loss_after, _ = model.loss_and_grads(X_train, y_train)
            train_acc = accuracy(y_train, model.predict(X_train))
            test_acc = accuracy(y_test, model.predict(X_test))
            rows.append((activation, H, loss, train_acc, test_acc, loss_after))
            cells = (f"{loss:.4f}", pct(train_acc), pct(test_acc), f"{loss_after:.4f}")
            print(row.format(activation, H, *cells))
            if H == 8:
                models[activation] = model
    print("  Final train loss is loss_history_[-1], the loss of epoch 2000 (before its update).")

    w, b = logistic_regression(X_train, y_train)  # sigmoid(z) >= 0.5 is the same as z >= 0
    linear_acc = (
        accuracy(y_train, (X_train @ w + b >= 0).ravel()),
        accuracy(y_test, (X_test @ w + b >= 0).ravel()),
    )
    print("\nLinear classifier to compare H = 1 with (logistic regression, same settings)")
    print(f"  train acc. {pct(linear_acc[0])}%, test acc. {pct(linear_acc[1])}%")

    fig = boundary_figure(models, X_train, y_train)
    fig.savefig("part1_boundaries.png", dpi=150)
    plt.close(fig)
    print("\nSaved part1_boundaries.png (H = 8, epochs 10, 200 and 2000)")
    return {"gradient_check": checks, "rows": rows, "linear": linear_acc}


if __name__ == "__main__":
    main()
