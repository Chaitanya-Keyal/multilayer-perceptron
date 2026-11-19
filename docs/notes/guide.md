# The whole assignment in plain language

First the ideas, assuming nothing. Then each part: what was asked, what we did, what came out.
Then a walk through the notebook, and the questions most likely to be asked. Every number is
from our own run: `results/part1.txt` and `results/part2.txt` hold what the two scripts print.

---

# The ideas

## The setup

- **The data.** Every sample is a point on a plane, two numbers (x1, x2), with a class, 0 or 1.
  - **Two-Moons** (Part 1): two half circles that hook into each other. 500 samples, 400 for
    training and 100 for testing.
  - **Two-Spirals** (Part 2): two arms that wind around each other one and a half times. 600
    samples, 480 for training and 120 for testing.
- **The job.** Find a boundary that puts class 0 on one side and class 1 on the other. For both
  datasets a straight line is not enough.
- **Training and test.** The model only ever learns from the training samples. The test samples
  are used once, at the end, to see how it does on samples it has not seen.
- **Standardising.** Each feature has the training mean subtracted and is divided by the training
  standard deviation. `common.py` does this, and it is the handout's code, copied exactly.
- **Seed 42.** Every random draw uses seed 42, so everyone who follows the handout gets the same
  numbers. There is nothing to tune.

## Words that come up everywhere

- **Node (neuron).** Takes some numbers in, multiplies each by a weight, adds them up with a
  bias, and passes the sum through an activation function. `z = w . x + b`, then `a = phi(z)`.
- **Layer.** A group of nodes working side by side on the same inputs. A **hidden layer** is one
  between the input and the output.
- **Width and depth.** Width is the number of nodes in a layer. Depth is the number of hidden
  layers.
- **Activation function.** The bend inside each node. Without it, any number of layers still
  adds up to one straight-line rule. We use four:
  - tanh: a smooth S from -1 to 1.
  - logistic (sigmoid): a smooth S from 0 to 1.
  - relu: 0 for negative input, the input itself for positive input.
  - leaky ReLU: like relu, but negative input is multiplied by 0.05 and not cut to 0.
- **Forward pass.** Going from the input to the prediction.
- **Loss.** One number for how wrong the predictions are. We use binary cross-entropy. A model
  that always says "50/50" has a loss of ln 2 = 0.693, so anything near 0.69 means it has learnt
  almost nothing.
- **Gradient.** For each weight, how much the loss would change if that weight grew a little.
- **Backpropagation.** The way to get all the gradients in one sweep, from the output back to
  the input, using the chain rule.
- **Gradient descent.** Move every weight a small step against its gradient:
  `w = w - lr * dw`. `lr` is the learning rate. One **epoch** is one pass over the training set.
- **L2 penalty.** A small extra term in the loss that grows with the squared weights. It keeps
  the weights from getting large for no reason.
- **Decision boundary.** The line or curve where the prediction is exactly 0.5.
- **Logistic regression.** A sigmoid applied to `w . x + b`, with no hidden layer. Its boundary
  is always a straight line, so it is the usual linear classifier to compare with.
- **Underfitting and overfitting.** Underfitting: the model is poor even on the training set.
  Overfitting: it is much better on the training set than on the test set.

---

# Part 1: a network from scratch

**What was asked.** Build a network with one hidden layer (2 inputs, H hidden nodes, 1 output)
using only numpy: the forward pass, the loss, backpropagation and the weight updates. Check the
gradients, train it on Two-Moons for H = 1, 2, 8, 32 with leaky ReLU and tanh, and plot the
boundary for H = 8.

**The four steps of one epoch.**
1. Forward: `Z1 = X W1 + b1`, `A1 = phi(Z1)`, `Z2 = A1 W2 + b2`, `y_hat = sigmoid(Z2)`.
2. Loss: cross-entropy between `y_hat` and `y`, plus `lambda / (2N)` times the sum of the
   squared weights.
3. Backward: the gradients of the loss for `W1`, `b1`, `W2`, `b2`.
4. Update: each parameter moves by `-0.1` times its gradient.

That is repeated 2000 times on all 400 training samples at once.

**The gradient check.** Before training, we check the backward step. Nudge one weight up by
0.00001, then down by 0.00001, and see how the loss changes. That slope should match what
backpropagation says. The worst mismatch is 3.5e-10 for leaky ReLU and 8.0e-11 for tanh, over
dW1 and dW2. The mismatch is measured as the difference divided by the size of the two slopes.
The limit is 1e-6, so the gradients are right.

**The results (Table 2).**

| H | Leaky ReLU train / test | tanh train / test |
|---|---|---|
| 1 | 87.25 / 88.00 | 87.25 / 88.00 |
| 2 | 88.25 / 88.00 | 87.25 / 88.00 |
| 8 | 87.50 / 89.00 | 95.50 / 94.00 |
| 32 | 98.25 / 94.00 | 97.25 / 95.00 |

**What they show.**
- **H = 1 is a straight line.** One hidden node can only split the plane with one line. It gets
  87.25% / 88.00%, which is exactly what a logistic regression gets (the script prints that
  too).
- **More nodes, more bends.** With 32 nodes both activations reach 94 to 95% on the test set.
- **H = 8 with leaky ReLU is still at 87.50%.** That is a slow start, not a limit. It has not
  left the straight-line solution by epoch 2000. As an experiment outside the submission we let
  it run on: 97.25% train at epoch 5000. With seeds 0 to 4 it gets to about 97% within 2000.
- **The shapes differ.** In `part1_boundaries.png` the leaky ReLU boundary is made of straight
  pieces with corners. The tanh boundary is one smooth curve.

---

# Part 2.1: the same 36 nodes, arranged four ways

**What was asked.** With scikit-learn's `MLPClassifier` on Two-Spirals, spread 36 hidden nodes
over 1, 2, 3 or 6 layers, with the logistic and relu activations. Report accuracy, final loss
and the size of the first layer's weights (Table 3), plot the eight boundaries, and plot the
loss curves of the deepest one.

**How scikit-learn trains.** It uses an update rule called Adam, on mini-batches. Each epoch it
shuffles the 480 training samples, cuts them into batches of 200, 200 and 80, and updates the
weights after each batch. Every model gets 1500 epochs.

**The results (test accuracy, %).**

| Architecture | logistic | relu |
|---|---|---|
| `(36,)` 1 layer | 83.33 | 97.50 |
| `(18, 18)` 2 layers | 97.50 | 99.17 |
| `(12, 12, 12)` 3 layers | 75.83 | 97.50 |
| `(6, 6, 6, 6, 6, 6)` 6 layers | 66.67 | 98.33 |

**What they show.**
- **Two layers are best for both.** Deeper is not automatically better.
- **The relu models hardly care.** All four are within two test samples of each other (one test
  sample is 0.83 percentage points).
- **The logistic models fall apart beyond two layers.** Six layers get 66.67%. Their loss never
  goes below 0.65 in 1500 epochs (`part2_loss_curves.png`), which is close to the "knows
  nothing" value of 0.693.
- **Why.** Splitting 36 nodes over more layers makes every layer narrower: 6 nodes each. And
  the learning signal has to travel back through six layers. The slope of the logistic function
  is at most 0.25. Going back through six layers multiplies six such slopes, at most
  0.25^6 = 0.00024. The weights between the layers multiply in as well, and at the start they
  are small, so they do not make up for it. The slope of relu is 1 for every node whose input
  is positive.

---

# Part 2.2: making two layers wider

**What was asked.** Two hidden layers of width M = 2, 4, 16, 64. Report train and test accuracy
(Table 4). Then train `(16, 16)` one epoch at a time and draw the boundary after epochs 1, 25,
100, 300 and 1000.

**The results (train / test, %).** P is the number of weights and biases, M^2 + 5M + 1.

| M | P | logistic | relu |
|---|---|---|---|
| 2 | 15 | 62.92 / 64.17 | 66.88 / 60.83 |
| 4 | 37 | 61.25 / 58.33 | 76.04 / 79.17 |
| 16 | 337 | 76.67 / 75.00 | 99.17 / 99.17 |
| 64 | 4417 | 98.96 / 98.33 | 99.17 / 98.33 |

**What they show.**
- **Small networks underfit.** With 2 or 4 nodes per layer nothing gets above 80%, on either
  set. They do not have enough pieces to follow a spiral.
- **relu is done by M = 16**, logistic needs M = 64.
- **No overfitting at M = 64.** There are 4417 parameters for 480 training samples, but train
  and test are less than one percentage point apart (0.63 and 0.84).
- **The boundary grows into the spiral.** In `part2_boundary_evolution.png` relu has the spiral
  by epoch 100 (98.54% train). logistic sits on a straight line for hundreds of epochs (53.12%
  at epoch 300) and then gets it (96.25% at epoch 1000).

**One thing that looks odd.** `(16, 16)` with logistic gets 76.67% in Table 4 after 1500 epochs,
but 96.25% in the figure after only 1000. Both are right. The figure's model is trained with
`partial_fit`, which puts the training samples in a different order from `fit`. A logistic
network spends a long time on a flat stretch of the loss, and when it leaves depends on that
order. The handout says the two need not match.

---

# Part 3: the three answers, in one line each

- **Q1.** One hidden node means one straight line, so H = 1 equals logistic regression
  (87.25 / 88.00). More nodes add bends: corners for leaky ReLU, a smooth curve for tanh.
- **Q2.** `(18, 18)` is best for both activations, not the deepest. Under a fixed budget, more
  depth means narrower layers and a longer way back for the gradient.
- **Q3.** The train-test gap does not grow with width. Small networks underfit; at M = 64 the
  gap is under one percentage point, so no clear overfitting.

The full answers are in `report.pdf`, and [p3-answers.md](p3-answers.md) goes through them
sentence by sentence.

---

# Walking through the notebook

From the repo folder:

```bash
uv sync --group notebook
uv run --group notebook jupyter lab notebooks/demo.ipynb
```

Run all cells once beforehand (about 30 s). What each cell shows and what to say about it:

**Setup cell.** Loads both datasets and prints their shapes.
> "The data comes from the handout's generators. 400 and 100 samples for Two-Moons, 480 and 120
> for Two-Spirals, two features each."

**Gradient check.**
> "For every weight we compare the gradient from backpropagation with the slope we get by
> nudging that weight up and down. The worst mismatch is around 1e-10, far below the 1e-6
> limit. The smallest |z| in the batch is 0.0020, so no hidden node sits on the corner of leaky
> ReLU."

**Table 2.**
> "With one hidden node both activations give 87.25% train and 88.00% test, the same as logistic
> regression, because the boundary can only be a line. With 32 nodes we reach 94 to 95% on the
> test set."

**Loss of the two H = 8 models.**
> "tanh keeps going down to 0.1456. Leaky ReLU flattens near 0.26: it is still on the
> straight-line solution at epoch 2000."

**Decision boundaries at H = 8.**
> "Top row leaky ReLU: straight pieces with corners. Bottom row tanh: a smooth curve. At epoch
> 10 both are rough, by 200 both look like a line, and by 2000 tanh has bent around the moons."

**Table 3 and the 4 x 2 grid.**
> "Same 36 nodes, four arrangements. Two layers of 18 are best for both activations. relu works
> for every arrangement. logistic gets worse beyond two layers: with six layers the boundary is
> nearly a straight line."

**Loss curves of the six-layer network.**
> "relu is below 0.03 within 100 epochs. logistic sits at 0.69 for about 60 epochs and then
> stays near 0.65. Early on, almost no gradient gets through six logistic layers."

**Table 4.**
> "Width 2 and 4 underfit. relu is at 99.17% by width 16. At width 64 both are at 98.33% on the
> test set, and train and test are within one percentage point, so there is no overfitting."

**Boundary evolution.**
> "One `partial_fit` call is one epoch. relu wraps around the spiral by epoch 100. logistic is
> still a straight line at epoch 300 and gets there by epoch 1000."

**If asked to change something.** A single model is one line in a new cell, for example
`part1.ScratchMLP(8, "tanh", lr=0.5).fit(Xm_train, ym_train)`. The constants at the top of each
script (`HIDDEN_SIZES`, `SNAPSHOT_EPOCHS`, `LEAK`, `SETTINGS`, `ARCHITECTURES`, `WIDTHS`,
`EVOLUTION_EPOCHS`) control the tables, but the figures have fixed grids, so the part notes list
what each change needs. Restart the kernel after editing a script.

---

# Likely questions, with short answers

1. **What does one hidden node compute?** `phi(w . x + b)`: a weighted sum of the inputs, then
   the activation.
2. **Why do we need an activation function?** Without it the network is a product of matrices,
   which is one matrix: a linear model, whatever the depth.
3. **Why can H = 1 not beat a linear classifier?** The output depends on x only through
   `w . x + b`, and tanh, leaky ReLU and the sigmoid all only go up. So the prediction flips at
   one value of `w . x + b`, which is a straight line.
4. **Why is `dZ2 = (y_hat - y) / N`?** The derivative of the cross-entropy through the sigmoid
   simplifies to `y_hat - y`. The `1/N` comes from the mean in the loss.
5. **What does the gradient check prove?** That the backward pass matches the loss it claims to
   differentiate. It does not prove the loss itself is the right one.
6. **Why skip a batch with |z| < 0.001?** Leaky ReLU has a corner at 0. A nudge of 0.00001 could
   cross it, and then the measured slope mixes the two sides.
7. **Why is the bias not regularised?** The handout's loss only penalises `W1` and `W2`. The
   bias only shifts the boundary; it does not make it more wiggly.
8. **Why is the leaky ReLU boundary made of straight pieces?** Each node is linear on either
   side of its corner, so the whole network is linear inside each region. tanh is smooth
   everywhere, so its boundary is too.
9. **Why does deeper not help with 36 nodes?** Each layer gets narrower and the gradient has
   further to travel. With logistic it shrinks at every layer.
10. **What is the vanishing gradient?** The logistic slope is at most 0.25. Going back through
    six layers multiplies six of them, at most 0.00024, so early in training the first layers
    get almost no signal.
11. **Is the M = 64 network overfitting?** No. Train and test are 0.63 and 0.84 percentage
    points apart, and one test sample alone is 0.83.
12. **Why does `fit` give 76.67% and `partial_fit` 96.25% for the same network?** They put the
    training samples in a different order, so the two runs take different paths, and the
    logistic network leaves its flat stretch at a different time.

The other files in this folder go deeper, with the derivations, the code and more questions.
