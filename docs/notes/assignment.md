# The assignment: what is asked and where it lives

## In short

Three parts. Part 1 builds a small neural network from scratch in numpy and trains it on
Two-Moons. Part 2 uses scikit-learn's `MLPClassifier` on Two-Spirals to compare depth, width and
activation functions. Part 3 is three written answers about our own results. The handout fixes
the data, every setting and the seed, so there is nothing to tune: the work is to implement the
specification exactly and to explain what the numbers show. The submission is one zip with nine
files: three scripts, four figures, `README.txt` and `report.pdf`.

## Rules that apply everywhere

- **Libraries.** Part 1 may import only numpy, plus matplotlib for the plot. No scikit-learn,
  SciPy, PyTorch or any other machine-learning library; a prohibited import in
  `part1_mlp_scratch.py` means zero for Part 1. Part 2 must use scikit-learn's `MLPClassifier`,
  with numpy and matplotlib.
- **Data.** Both datasets come from the generator functions the handout prints, copied exactly
  into `common.py`. `tests/test_common.py` compares the file with the handout's text.
- **No test-set leakage.** The test sets are used only for the final accuracies. Nothing is
  tuned on them, and the scaling statistics come from the training sets.
- **Fixed settings and seed 42** (the handout's Table 1):

  | Part | Settings |
  |---|---|
  | 1 | one hidden layer, H = 1, 2, 8, 32; full-batch gradient descent, 2000 epochs, learning rate 0.1, L2 penalty 1e-4; leaky ReLU (slope 0.05) and tanh |
  | 2.1 | 36 hidden nodes as `(36,)`, `(18, 18)`, `(12, 12, 12)`, `(6, 6, 6, 6, 6, 6)`; `logistic` and `relu`; Adam, learning rate 0.01, alpha 1e-4, default batch size, 1500 epochs with no early stop |
  | 2.2 | `(M, M)` for M = 2, 4, 16, 64, same solver settings; `(16, 16)` again with `partial_fit`, snapshots at epochs 1, 25, 100, 300, 1000 |

- **The scripts** must run from the command line with no arguments and print the numbers the
  tables need.
- **The report** is typed, with accuracies in % to two decimals and losses and norms to four.
  Each answer is at most 120 words and has to use our own numbers or figures.

## Where each item lives

| Handout item | Code | Printed or saved as |
|---|---|---|
| Dataset generators | `common.py` | |
| `ScratchMLP(n_hidden, activation, lr, reg_lambda, n_epochs, seed)` | `part1_mlp_scratch.py` | |
| `loss_and_grads`, `fit`, `predict_proba`, `predict`, `params`, `loss_history_`, `snapshots` | methods and attributes of `ScratchMLP` | |
| (a) finite-difference gradient check | `gradient_check` | "Gradient check" |
| (b) single-node linear ceiling | `logistic_regression`, the H = 1 rows | "Linear classifier to compare H = 1 with" |
| Part 1 task 1, Table 2 | `main` | "Table 2" |
| Part 1 task 2, boundaries for H = 8 | `boundary_figure` | `part1_boundaries.png` |
| Table 1 solver settings | `SETTINGS` in `part2_sklearn_exploration.py` | |
| Task 2.1, Table 3 with the layer-1 norm | `main` | "Table 3" |
| Task 2.1, 4 x 2 grid of boundaries | `depth_vs_width_figure` | `part2_depth_vs_width.png` |
| Task 2.1, loss curves of the deepest network | `loss_curve_figure` | `part2_loss_curves.png` |
| Task 2.2, Table 4 | `main` | "Table 4" |
| Task 2.2, boundary evolution with `partial_fit` | `boundary_evolution`, `evolution_figure` | `part2_boundary_evolution.png` |
| Q1 to Q3 and the filled-in tables | `report/report.tex` | `report.pdf` |
| Versions and commands | `README.txt` | |

`results/part1.txt` and `results/part2.txt` are the two scripts' output, word for word.

## Results at a glance

| | |
|---|---|
| Gradient check | largest relative error 3.5e-10 (leaky ReLU), 8.0e-11 (tanh); limit 1e-6 |
| Part 1, H = 1 | 87.25% train, 88.00% test for both activations, the same as logistic regression |
| Part 1, best | H = 32: 94.00% test with leaky ReLU, 95.00% with tanh |
| Part 2.1, best split | `(18, 18)`: 97.50% test with logistic, 99.17% with relu |
| Part 2.1, deepest | `(6, 6, 6, 6, 6, 6)`: 66.67% with logistic, 98.33% with relu |
| Part 2.2, best width | logistic M = 64 (98.33% test), relu M = 16 (99.17%) |
| Part 2.2, M = 64 | train minus test is +0.63 and +0.84 points: no clear overfitting |

## How we know the numbers are right

- **The gradient check** that the handout asks for, printed by Part 1 on every run. The script
  stops if it fails.
- **31 tests** (`uv run pytest`):
  - `common.py` against the code printed in the handout, and the sizes of the splits;
  - the starting weights against the same draws made by hand, in the same order;
  - the loss against a plain loop over the points, with the bias left out of the penalty;
  - one update step and the stored snapshots against the formulas;
  - the gradient check, including a case built to land on the corner of leaky ReLU;
  - an H = 1 network changing its prediction exactly once along its projection;
  - the Part 2 settings against Table 1, the parameter count against M^2 + 5M + 1, and the
    `partial_fit` loop against a model stepped by hand;
  - every statement the three answers make, and their word counts.
- **A clean run of the zip** (`uv run python -m tools.package`): unzipped into an empty folder
  with the figures deleted, both scripts print exactly what `results/` holds and write the same
  figures again, byte for byte.

## The submission

One zip, `<Student_ID>_A2.zip`, with these nine files at its root:

```
common.py
part1_mlp_scratch.py
part2_sklearn_exploration.py
part1_boundaries.png
part2_depth_vs_width.png
part2_loss_curves.png
part2_boundary_evolution.png
README.txt
report.pdf
```

`uv run python -m tools.package` builds it in `dist/` and runs the checks above. The tests, the
notebook and these notes are not part of it.
