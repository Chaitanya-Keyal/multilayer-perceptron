# Part 1: a two-layer MLP from scratch

## In short

`ScratchMLP` is a network with 2 inputs, one hidden layer of H nodes and one sigmoid output,
written with numpy only. One epoch is a forward pass, the loss, a backward pass and one update of
all four parameter arrays, on all 400 training points at once. Before training, a
finite-difference check confirms the gradients (largest relative error 3.5e-10 for leaky ReLU,
8.0e-11 for tanh). After 2000 epochs, H = 1 gives 87.25% train and 88.00% test for both
activations, exactly a logistic regression, because one hidden node can only draw a straight
line. H = 32 reaches 94.00% (leaky ReLU) and 95.00% (tanh) on the test set. At H = 8 tanh has
bent around the moons (95.50% train) while leaky ReLU is still close to a line (87.50%).

## The network

| Array | Shape | What it is |
|---|---|---|
| `X` | N x 2 | the inputs, one row per point |
| `W1`, `b1` | 2 x H, 1 x H | weights and biases of the hidden layer |
| `Z1`, `A1` | N x H | hidden layer before and after the activation |
| `W2`, `b2` | H x 1, 1 x 1 | weights and bias of the output node |
| `Z2`, `y_hat` | N x 1 | output before and after the sigmoid |

The network has 2H + H + H + 1 = 4H + 1 parameters: 5, 9, 33 and 129 for H = 1, 2, 8, 32.

**Initialisation.** One `np.random.RandomState(42)`. `W1` is drawn first, from a Gaussian with
standard deviation 1/sqrt(2), then `W2` with standard deviation 1/sqrt(H). Both biases start at
zero. Dividing by the square root of the number of inputs to a layer keeps the sums `Z1` and
`Z2` at a sensible size whatever H is.

## One epoch, step by step

### 1. Forward

```
Z1 = X W1 + b1          each hidden node: a weighted sum of x1 and x2, plus its bias
A1 = phi(Z1)            the activation, applied to every entry
Z2 = A1 W2 + b2         the output node: a weighted sum of the hidden nodes
y_hat = sigmoid(Z2)     squashed into (0, 1): the probability of class 1
```

The two activations and their slopes:

| | phi(z) | phi'(z) |
|---|---|---|
| leaky ReLU | z if z > 0, else 0.05 z | 1 if z > 0, else 0.05 |
| tanh | tanh(z) | 1 - tanh(z)^2 |

At exactly z = 0 leaky ReLU takes the 0.05 branch, as the handout writes it.

### 2. Loss

```
L = -(1/N) * sum[ y ln(y_hat) + (1 - y) ln(1 - y_hat) ]  +  lambda/(2N) * (|W1|^2 + |W2|^2)
```

- The first part is the binary cross-entropy. For a point of class 1 it is `-ln(y_hat)`: zero if
  the network is sure and right, large if it is sure and wrong.
- `y_hat` is clipped to [1e-15, 1 - 1e-15] before the logarithm, so that `ln(0)` can never
  happen. The clip is only used for the loss, not for the gradients.
- The second part is the L2 penalty, with lambda = 1e-4. `|W|^2` is the sum of the squared
  entries. The biases are not in it.

### 3. Backward

Each line is the chain rule applied one step further back. `dX` means "the gradient of the loss
with respect to X", with the same shape as X.

```
dZ2 = (y_hat - y) / N
dW2 = A1^T dZ2 + (lambda/N) W2        db2 = sum of dZ2 over the rows
dA1 = dZ2 W2^T
dZ1 = dA1 * phi'(Z1)                  (entry by entry)
dW1 = X^T dZ1 + (lambda/N) W1         db1 = sum of dZ1 over the rows
```

Where each one comes from:

- **dZ2.** For one point, the cross-entropy is `-[y ln(s) + (1 - y) ln(1 - s)]` with
  `s = sigmoid(z)`. The sigmoid's slope is `s(1 - s)`. Differentiating and simplifying leaves
  `s - y`. The `1/N` is the mean in the loss.
- **dW2.** `Z2 = A1 W2 + b2`, so each weight's gradient is the sum over the points of
  (its input) x (the output's gradient): `A1^T dZ2`. The penalty adds `(lambda/N) W2`.
- **db2.** The bias is added to every row, so its gradient is the sum of `dZ2`. It is a sum and
  not a mean because `dZ2` already carries the `1/N`.
- **dA1.** Going back through `Z2 = A1 W2`: each hidden node gets the output's gradient times
  its own weight.
- **dZ1.** Going back through the activation: multiply by its slope at that point.
- **dW1, db1.** The same pattern as `dW2` and `db2`, one layer earlier, with `X` as the input.

### 4. Update

```
W1 = W1 - 0.1 * dW1     and the same for b1, W2, b2
```

All four move at once, using gradients computed from the same forward pass.

## The code

`part1_mlp_scratch.py`. The class follows the four steps line by line.

- **`__init__`** draws `W1` and then `W2`, and stores the four arrays in `self.params` under the
  keys `'W1'`, `'b1'`, `'W2'`, `'b2'`. The parameters exist as soon as the model is created,
  because the gradient check needs them before any training.
- **`_forward(X, params=None)`** returns `Z1`, `A1` and `y_hat`. `params` lets the plots use a
  stored snapshot in place of the current parameters.
- **`loss_and_grads(X, y)`** runs the forward pass, computes the loss, and returns it with a
  dictionary of the four gradients. `y` is reshaped to N x 1 first. Without that,
  `y_hat - y` would be an N x 1 array minus a length-N array, and numpy would quietly turn it
  into an N x N table.
- **`fit(X, y, snapshot_epochs)`** repeats: get loss and gradients, store the loss, update. After
  the update of an epoch listed in `snapshot_epochs` it stores a copy of the parameters. So
  `loss_history_[t - 1]` is the loss at the start of epoch t, and `snapshots[t]` holds the
  parameters after epoch t. `snapshots[2000]` is the trained model.
- **`predict_proba(X)`** returns `y_hat` as a flat array; **`predict(X)`** returns 1 where
  `y_hat >= 0.5`, else 0.
- **`sigmoid`** is written as `e / (1 + e)` for negative z, with `e = exp(-|z|)`. It gives the
  same value as `1 / (1 + exp(-z))`, but `exp` never sees a large positive number, so there is
  no overflow warning for extreme inputs.

## The gradient check

A mistake in backpropagation does not crash anything. The network just trains badly. So the
gradients are checked against a slower method that cannot be wrong in the same way.

- **The idea.** Pick one entry of one parameter. Add eps = 1e-5 to it and compute the loss.
  Subtract eps and compute the loss again. `(L_plus - L_minus) / (2 eps)` is the slope of the
  loss along that entry. Compare with the entry of the analytic gradient.
- **The measure.** `|analytic - numeric| / max(|analytic| + |numeric|, 1e-12)`, the relative
  error. The largest value over all entries must be below 1e-6.
- **The batch.** The first 5 training rows, with a new `ScratchMLP(3)`. So N = 5 in the loss.
- **The corner of leaky ReLU.** If a pre-activation in `Z1` is within 0.001 of zero, the batch
  is skipped and the next five rows are tried. Here the smallest is 0.0020, so rows 0 to 4 are
  used.
- **Why central differences.** Nudging both ways cancels the leading error term. A one-sided
  difference would be accurate to about 1e-5; the two-sided one to about 1e-10.

What the script prints:

| | dW1 | dW2 | db1 | db2 |
|---|---|---|---|---|
| leaky ReLU | 2.031e-10 | 3.535e-10 | 1.176e-10 | 2.345e-12 |
| tanh | 4.663e-11 | 8.002e-11 | 8.879e-11 | 5.912e-12 |

The handout asks for `dW1` and `dW2`; the biases are checked as well. The script stops with an
error if any of them is 1e-6 or more.

## The results

Table 2, with the number of misclassified points worked out from the accuracies:

| Activation | H | Final train loss | Train % | Test % | Wrong (train / test) |
|---|---|---|---|---|---|
| leaky ReLU | 1 | 0.2887 | 87.25 | 88.00 | 51 / 12 |
| leaky ReLU | 2 | 0.2787 | 88.25 | 88.00 | 47 / 12 |
| leaky ReLU | 8 | 0.2607 | 87.50 | 89.00 | 50 / 11 |
| leaky ReLU | 32 | 0.0891 | 98.25 | 94.00 | 7 / 6 |
| tanh | 1 | 0.2925 | 87.25 | 88.00 | 51 / 12 |
| tanh | 2 | 0.2901 | 87.25 | 88.00 | 51 / 12 |
| tanh | 8 | 0.1456 | 95.50 | 94.00 | 18 / 6 |
| tanh | 32 | 0.0986 | 97.25 | 95.00 | 11 / 5 |

- **H = 1 is the linear ceiling.** A logistic regression trained with the same loss and settings
  gets 87.25% / 88.00%, also 51 and 12 mistakes. The handout predicted about 88%.
- **H = 2 does not help yet.** Two nodes could bend the boundary once, but in 2000 epochs they
  have not found a bend that pays off.
- **tanh gets there at H = 8**, leaky ReLU at H = 32.
- **Leaky ReLU at H = 8 is slow, not stuck for good.** Its loss at the start is 1.14, while for
  H = 1, 2 and 32 it is between 0.65 and 0.68: with seed 42 the eight starting weights happen
  to be a poor fit. Experiments outside the submission:
  - trained on, it reaches 97.25% train / 95.00% test at epoch 5000 and 98.50% / 97.00% at
    epoch 10000;
  - with seeds 0 to 4 it reaches 96.75 to 97.50% train within the 2000 epochs.
- **The final train loss** is `loss_history_[-1]`: the loss computed in epoch 2000, before that
  epoch's update. The script also prints the loss after the update. They agree to four decimals
  everywhere except leaky ReLU with H = 8 (0.2607 and 0.2606), because that model's loss is
  still falling.
- **Train against test.** At H = 32 the training accuracy (98.25%, 97.25%) is a few points above
  the test accuracy (94.00%, 95.00%). The data has noise 0.20, so some points of each moon lie
  inside the other and no smooth boundary can get them all.

## The figure

`part1_boundaries.png`: H = 8, leaky ReLU on top and tanh below, after 10, 200 and 2000 epochs.
The shaded regions are the predicted classes, the black line is where `y_hat = 0.5`, and the
dots are the training points.

| Epoch | leaky ReLU train % | tanh train % |
|---|---|---|
| 10 | 61.50 | 80.50 |
| 200 | 87.25 | 88.00 |
| 2000 | 87.50 | 95.50 |

- **Leaky ReLU: straight pieces and corners.** Every hidden node is linear on each side of its
  own line `w . x + b = 0`. Those lines cut the plane into regions, and inside one region the
  whole network is linear. So the boundary is straight inside a region and can only turn where
  it crosses one of the lines.
- **tanh: one smooth curve.** tanh has no corner anywhere, so neither has the boundary.
- **Both pass through a straight line first.** At epoch 200 both are near 88%, the linear
  solution. Bending the boundary comes later, and tanh has done it by epoch 2000.

## Questions and answers

1. **Why one output node and not two?** There are two classes, so one probability is enough:
   `y_hat` for class 1 and `1 - y_hat` for class 0.
2. **Why the sigmoid at the output?** It turns any number into a value between 0 and 1 that can
   be read as a probability, and with cross-entropy the gradient becomes the simple
   `y_hat - y`.
3. **Why cross-entropy and not squared error?** With a sigmoid, squared error has a tiny
   gradient when the network is confidently wrong. Cross-entropy does not.
4. **What if all weights started at zero?** Every hidden node would compute the same thing and
   get the same gradient, so they would stay identical for ever. Random starting weights break
   the tie. Starting the biases at zero is fine.
5. **Why is W1 drawn before W2?** Both come from the same random generator, so the order
   decides which numbers each one gets. The handout fixes the order so that everyone has the
   same starting point.
6. **What is "full-batch"?** Every update uses all 400 training points. Mini-batch methods use
   a few at a time. Full-batch is exact and has no randomness after the initialisation.
7. **What does the L2 penalty do here?** Very little: lambda = 1e-4. It adds
   `(lambda/N) W` to each weight gradient, which pulls the weights slightly towards zero.
8. **Why predict class 1 when `y_hat >= 0.5`?** 0.5 is where the two classes are equally likely.
   The handout puts the tie on the class 1 side.
9. **Why is the training loss not zero at H = 32?** The moons overlap because of the noise, and
   2000 epochs with a learning rate of 0.1 do not fit every point.
10. **How do you know the test set was not used for anything else?** `fit` only sees
    `X_train` and `y_train`. `X_test` appears in one line, the test accuracy. The scaling
    statistics come from the training set inside `common.py`.
11. **Why store snapshots and not retrain for each epoch count?** One run of 2000 epochs passes
    through epochs 10 and 200 anyway. Copying the parameters there costs nothing and gives the
    exact state of the same run.
12. **Why a copy of the parameters?** The update changes the arrays in place. Without `.copy()`
    all three snapshots would point at the same arrays and show the final model.

## Changes that might be asked for

| Change | Where |
|---|---|
| Another H, or other epochs in the figure | `HIDDEN_SIZES`, `SNAPSHOT_EPOCHS` at the top |
| Another slope for leaky ReLU | `LEAK` |
| Learning rate, penalty, epochs, seed | arguments of `ScratchMLP(...)` |
| A new activation, such as plain ReLU | one branch each in `_phi` and `_phi_prime`, and its name in `ACTIVATIONS` |
| Mini-batches | in `fit`, shuffle the row indices each epoch and call `loss_and_grads` on slices |
| A loss curve | plot `model.loss_history_` (the notebook does this for H = 8) |
| A third class | the output would need one node per class and a softmax; that is a different `dZ2` and loss |
