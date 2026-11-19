"""Part 2: depth, width and activation with scikit-learn's MLPClassifier on Two-Spirals.

    python part2_sklearn_exploration.py

Prints the rows of Tables 3 and 4, and saves part2_depth_vs_width.png, part2_loss_curves.png
and part2_boundary_evolution.png.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from common import generate_spirals

# every model runs for the full max_iter epochs, so scikit-learn warns that it did not converge
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Table 1 of the handout, the same for every model in this part
SETTINGS = {
    "solver": "adam",
    "learning_rate_init": 0.01,
    "alpha": 1e-4,
    "batch_size": "auto",
    "max_iter": 1500,
    "n_iter_no_change": 1500,
    "random_state": 42,
}
ARCHITECTURES = ((36,), (18, 18), (12, 12, 12), (6, 6, 6, 6, 6, 6))
ACTIVATIONS = ("logistic", "relu")
WIDTHS = (2, 4, 16, 64)
EVOLUTION_EPOCHS = (1, 25, 100, 300, 1000)

CLASS_COLORS = ("#2a78d6", "#eb6834")
REGION_COLORS = ("#d9e7f8", "#fbe4da")
LINE_COLOR = "#222222"


def train(hidden_layer_sizes, activation, X, y):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, **SETTINGS)
    clf.fit(X, y)
    assert clf.n_iter_ == SETTINGS["max_iter"]  # no early stop
    return clf


def pct(x):
    return f"{100 * x:.2f}"


def n_parameters(clf):
    return sum(W.size for W in clf.coefs_) + sum(b.size for b in clf.intercepts_)


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


def depth_vs_width_figure(models, X, y):
    """4 x 2 grid: one row per architecture, one column per activation."""
    G1, G2, grid = make_grid(X)
    fig, axes = plt.subplots(4, 2, figsize=(8, 15), sharex=True, sharey=True, layout="constrained")
    for row, arch in enumerate(ARCHITECTURES):
        for col, activation in enumerate(ACTIVATIONS):
            clf = models[arch, activation]
            plot_boundary(axes[row, col], G1, G2, clf.predict_proba(grid)[:, 1], X, y)
            axes[row, col].set_title(
                f"{arch}, {activation} (train acc. {pct(clf.score(X, y))}%)", fontsize=10
            )
    add_legend(fig)
    fig.suptitle("36 hidden nodes on Two-Spirals: boundary and training points")
    return fig


def loss_curve_figure(models):
    """Training loss per epoch of the deepest architecture, one line per activation."""
    arch = ARCHITECTURES[-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for activation, color in zip(ACTIVATIONS, CLASS_COLORS, strict=True):
        curve = models[arch, activation].loss_curve_
        ax.plot(range(1, len(curve) + 1), curve, color=color, linewidth=2, label=activation)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (clf.loss_curve_)")
    ax.set_xlim(0, SETTINGS["max_iter"])
    ax.set_ylim(bottom=0)
    ax.grid(color="#e6e6e3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title(f"Loss curves of {arch} on Two-Spirals")
    return fig


def boundary_evolution(X, y):
    """Train (16, 16) with one partial_fit call per epoch. For every epoch in EVOLUTION_EPOCHS
    keep P(class 1) on the grid and the training accuracy."""
    grid = make_grid(X)[2]
    snapshots = {}
    for activation in ACTIVATIONS:
        clf = MLPClassifier(hidden_layer_sizes=(16, 16), activation=activation, **SETTINGS)
        for epoch in range(1, max(EVOLUTION_EPOCHS) + 1):
            clf.partial_fit(X, y, classes=[0, 1])
            if epoch in EVOLUTION_EPOCHS:
                snapshots[activation, epoch] = clf.predict_proba(grid)[:, 1], clf.score(X, y)
    return snapshots


def evolution_figure(snapshots, X, y):
    """2 x 5 grid: one row per activation, one column per epoch."""
    G1, G2, _ = make_grid(X)
    fig, axes = plt.subplots(
        2, 5, figsize=(15, 6.6), sharex=True, sharey=True, layout="constrained"
    )
    for row, activation in enumerate(ACTIVATIONS):
        axes[row, 0].set_ylabel(activation, fontsize=13)
        for col, epoch in enumerate(EVOLUTION_EPOCHS):
            P, train_acc = snapshots[activation, epoch]
            plot_boundary(axes[row, col], G1, G2, P, X, y)
            axes[row, col].set_title(f"epoch {epoch}, train acc. {pct(train_acc)}%", fontsize=12)
    add_legend(fig)
    fig.suptitle("(16, 16) on Two-Spirals, trained with partial_fit: boundary and training points")
    return fig


def main():
    X_train, y_train, X_test, y_test = generate_spirals()

    print("Table 3: 36 hidden nodes, depth vs. width (Two-Spirals)")
    row = "  {:<20}  {:<10}  {:>14}  {:>13}  {:>10}  {:>12}"
    header = ("Architecture", "Activation", "Train acc. (%)", "Test acc. (%)")
    print(row.format(*header, "Final loss", "Layer-1 norm"))
    table3, models = [], {}
    for arch in ARCHITECTURES:
        for activation in ACTIVATIONS:
            clf = train(arch, activation, X_train, y_train)
            models[arch, activation] = clf
            train_acc, test_acc = clf.score(X_train, y_train), clf.score(X_test, y_test)
            norm = np.linalg.norm(clf.coefs_[0], "fro")
            table3.append((arch, activation, train_acc, test_acc, clf.loss_, norm))
            cells = (pct(train_acc), pct(test_acc), f"{clf.loss_:.4f}", f"{norm:.4f}")
            print(row.format(str(arch), activation, *cells))

    fig = depth_vs_width_figure(models, X_train, y_train)
    fig.savefig("part2_depth_vs_width.png", dpi=150)
    plt.close(fig)
    fig = loss_curve_figure(models)
    fig.savefig("part2_loss_curves.png", dpi=150)
    plt.close(fig)
    print("\nSaved part2_depth_vs_width.png and part2_loss_curves.png")

    print("\nTable 4: width sweep, two hidden layers (M, M) (Two-Spirals)")
    row = "  {:<12}  {:>5}  {:<10}  {:>14}  {:>13}"
    print(row.format("Architecture", "P", "Activation", "Train acc. (%)", "Test acc. (%)"))
    table4 = []
    for M in WIDTHS:
        for activation in ACTIVATIONS:
            clf = train((M, M), activation, X_train, y_train)
            P = n_parameters(clf)
            assert P == M * M + 5 * M + 1
            train_acc, test_acc = clf.score(X_train, y_train), clf.score(X_test, y_test)
            table4.append((M, activation, P, train_acc, test_acc))
            print(row.format(str((M, M)), P, activation, pct(train_acc), pct(test_acc)))

    snapshots = boundary_evolution(X_train, y_train)
    fig = evolution_figure(snapshots, X_train, y_train)
    fig.savefig("part2_boundary_evolution.png", dpi=150)
    plt.close(fig)
    print("\nSaved part2_boundary_evolution.png ((16, 16), epochs 1, 25, 100, 300 and 1000)")
    evolution = [(a, epoch, acc) for (a, epoch), (_, acc) in snapshots.items()]
    deepest = {a: min(models[ARCHITECTURES[-1], a].loss_curve_) for a in ACTIVATIONS}
    return {"table3": table3, "table4": table4, "evolution": evolution, "deepest_min_loss": deepest}


if __name__ == "__main__":
    main()
