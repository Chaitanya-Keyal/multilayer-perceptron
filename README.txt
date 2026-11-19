BITS F464 Machine Learning, Assignment 2
Multi-layer Perceptron From Scratch & Hyperparameter dynamics

Group
  Chaitanya Keyal   2023B4A70727H
  Vishesh Agarwal   2023B4A70611H
  Deeksha Thakur    2023B4A71257H

Versions used
  Python 3.12.10, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2.
  Both scripts also print exactly the same numbers with numpy 2.0.2, scikit-learn 1.6.1 and
  matplotlib 3.10.0.

    pip install numpy==2.5.5 scikit-learn==1.9.1 matplotlib==3.11.2

How to reproduce the tables and figures
  Keep all the files in one folder and run the two scripts from that folder. They take no
  arguments and need no data files: common.py (the handout's code, unchanged) generates both
  datasets.

    python part1_mlp_scratch.py            about 3 s
    python part2_sklearn_exploration.py    about 25 s

  part1_mlp_scratch.py prints the gradient-check errors for both activations and the eight rows
  of Table 2, and saves part1_boundaries.png.

  part2_sklearn_exploration.py prints the eight rows of Table 3, the lowest loss of the deepest
  network, the eight train/test pairs of Table 4 with their gaps, and the training accuracy at
  the five snapshot epochs. It saves part2_depth_vs_width.png, part2_loss_curves.png and
  part2_boundary_evolution.png.

  The figures are written to the folder the script is run from. Accuracies are printed in % to
  two decimals, losses and norms to four, as in report.pdf.

Files
  common.py                      generate_moons() and generate_spirals() from the handout
  part1_mlp_scratch.py           Part 1: ScratchMLP, the gradient check, Table 2 and its figure
  part2_sklearn_exploration.py   Part 2: Tables 3 and 4 and the three Part 2 figures
  part1_boundaries.png           H = 8 on Two-Moons at epochs 10, 200 and 2000
  part2_depth_vs_width.png       the eight models of Table 3
  part2_loss_curves.png          loss curves of (6, 6, 6, 6, 6, 6), logistic against relu
  part2_boundary_evolution.png   (16, 16) with partial_fit at epochs 1, 25, 100, 300 and 1000
  report.pdf                     Tables 2 to 4 and the answers to Q1 to Q3

Choices the handout leaves open
  - Final train loss (Table 2) is loss_history_[-1], the loss computed in epoch 2000 before that
    epoch's update. The script also prints the loss after the last update. The two agree to four
    decimals in every row except leaky_relu with H = 8 (0.2607 and 0.2606).
  - A snapshot at epoch t holds the parameters after the update of epoch t, so the snapshot at
    epoch 2000 is the trained model.
  - predict_proba and predict return arrays of shape (N,), the shape of the labels common.py
    returns. y may be passed as (N,) or (N, 1).
  - The weights are drawn when the model is created. Calling fit a second time on the same
    object carries on from the current weights.
  - The gradient check uses a new ScratchMLP(3) with seed 42 on training rows 0 to 4. The
    smallest |pre-activation| in that batch is 0.0020, above the 1e-3 limit, so no other batch
    was needed. If it were below, the code would move on to rows 1 to 5, and so on.
  - For sanity check (b), Part 1 also trains a plain logistic regression (numpy only, same loss
    and settings) and prints its accuracy under Table 2.
  - The two partial_fit models get the same settings as every other model in Part 2. The
    accuracy in each figure title is the training accuracy.
  - A train-test gap is the difference between the two printed accuracies.
