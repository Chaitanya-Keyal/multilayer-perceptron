# Multi-layer Perceptron: From Scratch, then Depth, Width and Activation

A two-layer perceptron (2 -> H -> 1) written in numpy with its own forward pass, backpropagation
and gradient check, trained on a Two-Moons dataset; then scikit-learn's `MLPClassifier` on a
Two-Spirals dataset, to see what depth, width and the activation function do to accuracy.

| Part | What it does |
|------|--------------|
| 1 | `ScratchMLP`: Gaussian initialisation, leaky ReLU or tanh hidden layer, L2-regularised cross-entropy, full-batch gradient descent; finite-difference gradient check; H = 1, 2, 8, 32 |
| 2.1 | 36 hidden nodes split over 1, 2, 3 or 6 layers, with `logistic` and `relu` |
| 2.2 | Two layers of width M = 2, 4, 16, 64; the decision boundary of `(16, 16)` as it trains |
| 3 | Three short answers on what the numbers show, in `report.pdf` |

## 1. Layout

```
assignment.pdf                  the handout
common.py                       the handout's dataset generators, copied as printed
part1_mlp_scratch.py            Part 1: prints the gradient check and Table 2, saves its figure
part2_sklearn_exploration.py    Part 2: prints Tables 3 and 4, saves its three figures
part1_boundaries.png ...        the four figures, as the scripts save them
README.txt                      the submission's readme: versions, commands, open choices
report.pdf                      Tables 2 to 4 and the answers to Q1 to Q3
results/                        part1.txt, part2.txt (what each script prints), results.json, numbers.tex
report/report.tex               the source of report.pdf; its numbers come from results/numbers.tex
tools/                          run_all.py (results/), build_pdf.py (report.pdf), package.py (the zip)
tests/                          checks of the network against the handout's formulas, and of the report
notebooks/demo.ipynb            both parts step by step, by calling the two scripts
docs/notes/                     study notes: the ideas in plain language, the maths, the code
pyproject.toml, uv.lock         dependencies; .python-version pins Python 3.12
```

The repo root holds the files the submission zip is made of, so running a script here is the same
as running it in the unzipped submission.

## 2. Environment

```bash
uv sync                      # numpy, matplotlib and scikit-learn, plus pytest and ruff (the dev group)
uv sync --group notebook     # adds JupyterLab, for the demo notebook
```

`uv.lock` pins every version. Tests, then formatting and lint:

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
```

`part1_mlp_scratch.py` and `common.py` import only numpy and matplotlib; scikit-learn is used in
Part 2 alone.

## 3. Data

Nothing is downloaded. `common.py` generates both datasets with seed 42, splits them 80/20 and
standardises the features with the training set's mean and standard deviation:

| Dataset | Used in | Samples | Train | Test |
|---|---|---|---|---|
| Two-Moons, noise 0.20 | Part 1 | 500 | 400 | 100 |
| Two-Spirals, 1.5 turns, noise 0.05 | Part 2 | 600 | 480 | 120 |

The test sets are used only for the accuracies in the tables.

## 4. Running

```bash
uv run python part1_mlp_scratch.py           # about 3 s
uv run python part2_sklearn_exploration.py   # about 25 s
uv run python -m tools.run_all               # both, into results/
uv run python -m tools.build_pdf             # report.pdf, with latexmk
```

Every random draw is seeded with 42, so a rerun reproduces `results/` and the figures byte for
byte; the PDF carries no build date, so it does too.

## 5. Results

Part 1, `ScratchMLP` on Two-Moons after 2000 epochs (train / test accuracy, %):

| H | Leaky ReLU | tanh |
|---|---|---|
| 1 | 87.25 / 88.00 | 87.25 / 88.00 |
| 2 | 88.25 / 88.00 | 87.25 / 88.00 |
| 8 | 87.50 / 89.00 | 95.50 / 94.00 |
| 32 | 98.25 / 94.00 | 97.25 / 95.00 |

With one hidden node the network is a linear classifier and matches logistic regression
(87.25 / 88.00). The gradient check gives a largest relative error of 3.5e-10 (leaky ReLU) and
8.0e-11 (tanh) over dW1 and dW2.

Part 2.1, 36 hidden nodes on Two-Spirals (test accuracy, %):

| Architecture | logistic | relu |
|---|---|---|
| `(36,)` | 83.33 | 97.50 |
| `(18, 18)` | 97.50 | 99.17 |
| `(12, 12, 12)` | 75.83 | 97.50 |
| `(6, 6, 6, 6, 6, 6)` | 66.67 | 98.33 |

Part 2.2, two layers of width M (train / test accuracy, %):

| M | Parameters | logistic | relu |
|---|---|---|---|
| 2 | 15 | 62.92 / 64.17 | 66.88 / 60.83 |
| 4 | 37 | 61.25 / 58.33 | 76.04 / 79.17 |
| 16 | 337 | 76.67 / 75.00 | 99.17 / 99.17 |
| 64 | 4417 | 98.96 / 98.33 | 99.17 / 98.33 |

The full tables, the four figures and the answers are in `report.pdf`.

## 6. Demo

```bash
uv run --group notebook jupyter lab notebooks/demo.ipynb
```

The notebook has no model code of its own: it imports the two scripts, trains the same models
and shows the tables and figures inline. It is committed without outputs; running every cell
takes about 30 s.

## 7. Submission

```bash
uv run python -m tools.package          # dist/<ID>_A2.zip
```

The zip holds the nine files the handout names. The script first rebuilds the PDF in an empty
folder and checks that it matches `report.pdf` byte for byte (skipped, with a printed note, where
latexmk is missing). It then unzips the zip into an empty folder, deletes the figures there,
runs each script with a plain `python partN_*.py`, and checks that the output matches `results/`
and that the figures come back identical. It also checks that Part 1 and `common.py` import
nothing beyond numpy, matplotlib and the standard library.
