# Part 2: depth, width and activation with scikit-learn

## In short

All of Part 2 uses scikit-learn's `MLPClassifier` on Two-Spirals with the same settings (Adam,
learning rate 0.01, 1500 epochs, seed 42) and two activations, `logistic` and `relu`. Part 2.1
spreads 36 hidden nodes over 1, 2, 3 or 6 layers: two layers of 18 are best for both activations
(97.50% and 99.17% test), and the six-layer logistic network is the worst (66.67%), because its
loss never leaves the 0.65 to 0.70 range. Part 2.2 keeps two layers and widens them: M = 2 and 4
underfit, relu is at 99.17% by M = 16, logistic needs M = 64, and even with 4417 parameters the
train and test accuracies stay within one point. The boundary snapshots of `(16, 16)` show relu
wrapping around the spiral by epoch 100 and logistic only by epoch 1000.

## MLPClassifier and its settings

`hidden_layer_sizes` is a tuple with one number per hidden layer: `(18, 18)` is two hidden
layers of 18 nodes. The output is always one sigmoid node here, since there are two classes.

The settings of the handout's Table 1 are in one dictionary, `SETTINGS`, used for every model:

| Setting | Value | What it means |
|---|---|---|
| `solver` | `'adam'` | The update rule. Adam keeps a running average of each weight's gradient and of its square, and scales each weight's step by them. It usually needs far fewer epochs than plain gradient descent. |
| `learning_rate_init` | 0.01 | The base step size Adam starts from. |
| `alpha` | 1e-4 | The L2 penalty, the same idea as lambda in Part 1. |
| `batch_size` | `'auto'` | min(200, number of samples) = 200. With 480 training points an epoch is 3 updates, on 200, 200 and 80 points. |
| `max_iter` | 1500 | The number of epochs. So 4500 updates in all. |
| `n_iter_no_change` | 1500 | scikit-learn normally stops early once the loss has stopped improving for this many epochs. Setting it to 1500 means it never stops early. |
| `random_state` | 42 | Seeds the starting weights and the shuffling of the training points. |

Because every model runs all 1500 epochs, scikit-learn prints a `ConvergenceWarning` ("maximum
iterations reached"). The script hides that one warning, as the handout says to.

Things `MLPClassifier` does on its own:

- **Starting weights.** Drawn uniformly from a range that depends on the sizes of the two layers
  a weight matrix connects. For `logistic` the range is smaller than for `relu`.
- **Shuffling.** The training points are put in a new order every epoch before being cut into
  mini-batches.
- **The loss.** Cross-entropy plus the L2 term, as in Part 1.

What we read off a trained model `clf`:

| Attribute | What it holds |
|---|---|
| `clf.coefs_` | the weight matrices, one per layer; `clf.coefs_[0]` connects the 2 inputs to the first hidden layer |
| `clf.intercepts_` | the bias vectors |
| `clf.loss_` | the training loss of the last epoch |
| `clf.loss_curve_` | the training loss of every epoch |
| `clf.n_iter_` | the number of epochs run; the script checks that it is 1500 |
| `clf.score(X, y)` | the accuracy on (X, y) |

`clf.loss_` is the average over the last epoch's mini-batches, computed while the weights were
still being updated. It is not recomputed afterwards.

## Part 2.1: 36 nodes, four arrangements

Table 3:

| Architecture | Activation | Train % | Test % | Final loss | Layer-1 norm |
|---|---|---|---|---|---|
| `(36,)` | logistic | 85.83 | 83.33 | 0.3892 | 19.4912 |
| `(36,)` | relu | 99.17 | 97.50 | 0.0331 | 16.4026 |
| `(18, 18)` | logistic | 99.17 | 97.50 | 0.0400 | 16.7455 |
| `(18, 18)` | relu | 98.96 | 99.17 | 0.0163 | 7.2583 |
| `(12, 12, 12)` | logistic | 74.17 | 75.83 | 0.5045 | 9.0309 |
| `(12, 12, 12)` | relu | 98.96 | 97.50 | 0.0165 | 4.3625 |
| `(6, 6, 6, 6, 6, 6)` | logistic | 64.17 | 66.67 | 0.6531 | 13.4539 |
| `(6, 6, 6, 6, 6, 6)` | relu | 98.96 | 98.33 | 0.0184 | 2.4838 |

One test point is 1/120 = 0.83%, and one training point is 0.21%.

**Same nodes, different number of parameters.** The four arrangements all have 36 hidden nodes,
but the weights between them differ: 145, 415, 361 and 235 parameters. A single wide layer has
the fewest, because its 36 nodes only connect to 2 inputs and 1 output.

**Reading the table.**
- **relu works whatever the arrangement.** Test accuracy is 97.50 to 99.17%, a spread of two
  test points. `(18, 18)` is the best with one mistake in 120.
- **logistic depends heavily on it.** `(18, 18)` matches relu's range (97.50%). One layer of 36
  is at 83.33%, three layers at 75.83% and six layers at 66.67%.
- **The best split is not the deepest**, for either activation.

**Why six logistic layers fail.** During backpropagation the gradient is multiplied, at every
layer, by the slope of the activation there.
- The slope of the logistic function is `s(1 - s)`, which is at most 0.25 (at s = 0.5). Six
  layers multiply six such numbers: at most 0.25^6 = 0.00024. So the first layers get almost no
  signal and hardly change. This is the vanishing gradient.
- The slope of relu is exactly 1 wherever the node is active, so the signal arrives undiminished
  along active paths.
- `part2_loss_curves.png` shows the result. The logistic loss sits at 0.693 (ln 2, the value for
  "always predict 0.5") for the first 60 or so epochs, drops to about 0.655, and stays there:
  its lowest value in 1500 epochs is 0.6529. The relu loss is below 0.03 by epoch 100 and ends
  at 0.0184.
- In `part2_depth_vs_width.png` the six-layer logistic boundary is almost a straight line.

**Why narrow layers are a problem in themselves.** Each layer can only pass on what fits in its
nodes. With 6 nodes per layer, everything the network has worked out about a point has to be
squeezed into 6 numbers, six times in a row. A wider layer has more pieces to build the spiral's
turns from.

**Why one and three logistic layers are also weak (83.33% and 75.83%).** These two are slow,
not stuck. Their loss sits near 0.66 for hundreds of epochs and is then still falling when
training ends: for `(36,)` it is 0.58 at epoch 1000 and 0.39 at epoch 1500, for `(12, 12, 12)`
0.62 and 0.50. The six-layer network is different: its loss is flat to the end.

**The layer-1 norm.** `np.linalg.norm(clf.coefs_[0], 'fro')` is the square root of the sum of
the squared entries of the first weight matrix: one number for how large the first layer's
weights have become.
- It partly just reflects the size of the matrix: `(36,)` has 2 x 36 entries, the six-layer
  network 2 x 6.
- For relu it falls steadily with depth: 16.40, 7.26, 4.36, 2.48.
- For logistic it is larger at every depth (19.49, 16.75, 9.03, 13.45). One reason: a logistic
  node needs large weights to make its S-curve steep enough to act as a sharp edge.

**An experiment outside the submission:** the same eight models with `random_state` 0 to 4.

| Architecture | logistic test %, five seeds | relu test %, five seeds |
|---|---|---|
| `(36,)` | 50.83 to 82.50 | 95.00 to 98.33 |
| `(18, 18)` | 97.50 every time | 98.33 to 99.17 |
| `(12, 12, 12)` | 75.00 to 98.33 | 99.17 every time |
| `(6, 6, 6, 6, 6, 6)` | 48.33 to 66.67 | 85.83 to 99.17 |

So the six-layer logistic failure is not a quirk of seed 42, and `(18, 18)` is the one
arrangement that is reliable for both activations. Among the relu models, the order of the top
three is within noise.

## Part 2.2: the width sweep

Two hidden layers of width M. The number of parameters:

```
inputs to layer 1:   2M weights + M biases
layer 1 to layer 2:  M^2 weights + M biases
layer 2 to output:   M weights + 1 bias
P(M) = M^2 + 5M + 1  ->  15, 37, 337, 4417
```

The script counts the entries of `coefs_` and `intercepts_` and checks them against this.

Table 4, with the gap (train minus test, in points):

| M | P | logistic train / test | gap | relu train / test | gap |
|---|---|---|---|---|---|
| 2 | 15 | 62.92 / 64.17 | -1.25 | 66.88 / 60.83 | +6.05 |
| 4 | 37 | 61.25 / 58.33 | +2.92 | 76.04 / 79.17 | -3.13 |
| 16 | 337 | 76.67 / 75.00 | +1.67 | 99.17 / 99.17 | 0.00 |
| 64 | 4417 | 98.96 / 98.33 | +0.63 | 99.17 / 98.33 | +0.84 |

**Reading the table.**
- **M = 2 and 4 underfit.** Nothing is above 80% on either set. The gaps there are large and of
  both signs, which is what noise looks like when the model has not learnt the shape: the test
  set is only 120 points.
- **relu generalises best at M = 16**: 99.17% on both sets, 119 of 120 test points.
- **logistic generalises best at M = 64**: 98.33%. At M = 16 it is only at 76.67%, with its loss
  still falling at epoch 1500 (0.56 at epoch 1000, 0.49 at epoch 1500). It is not out of
  capacity; it has not finished training.
- **No overfitting at M = 64.** 4417 parameters for 480 training points, about nine each. Yet
  the gaps are +0.63 and +0.84, about one test point. relu goes from 119 to 118 correct test
  points between M = 16 and M = 64, which is one point.
- **Why so little overfitting.** The spirals have little noise (0.05), so a boundary that fits
  the training points also fits the test points, which lie on the same two arms. There is also
  the small L2 penalty.

## Part 2.2: the boundary as it trains

`MLPClassifier.fit` gives no access to the model in the middle of training. `partial_fit` runs
exactly one epoch per call, so calling it 1000 times in a loop lets us look at the model after
any epoch. `classes=[0, 1]` tells it which labels exist, since in general one call might not see
them all.

Training accuracy at the five snapshots (also in the figure's titles):

| Epoch | 1 | 25 | 100 | 300 | 1000 |
|---|---|---|---|---|---|
| logistic | 50.42 | 62.29 | 61.04 | 53.12 | 96.25 |
| relu | 50.83 | 71.46 | 98.54 | 98.96 | 98.96 |

- **relu** starts with a few straight cuts, has a rough spiral by epoch 25 and a clean one by
  epoch 100.
- **logistic** draws a single straight line for hundreds of epochs. Its accuracy even drops
  (53.12% at epoch 300) before the boundary curls into the spiral by epoch 1000.

**Why this does not match Table 4.** For `(16, 16)` with logistic, Table 4 says 76.67% after
1500 epochs and the figure says 96.25% after 1000. Both are right.
- `fit` keeps one list of the training points and reshuffles it every epoch. `partial_fit`
  starts each call from the original order and shuffles that once. The random numbers are the
  same, but the resulting orders differ from the second epoch on, so the two runs follow
  different paths.
- For relu this makes no visible difference (98.96% against 99.17%).
- A logistic network spends a long time on a flat stretch of the loss, and when it gets off
  depends on small things like the order of the mini-batches.
- A side effect of `partial_fit`: `clf.n_iter_` stays at 1, because each call counts its own
  single epoch.

## Questions and answers

1. **What is Adam, in one line?** Gradient descent with a separate, self-adjusting step size for
   every weight, based on that weight's recent gradients.
2. **Why is Part 2 mini-batch when Part 1 was full-batch?** It is scikit-learn's default for
   Adam: batches of at most 200. The handout keeps the default.
3. **How many updates does a model get?** 3 per epoch for 1500 epochs: 4500.
4. **Why set `n_iter_no_change` to 1500?** So that no model stops early and all are trained for
   the same number of epochs, which makes them comparable.
5. **Is it a problem that the models "did not converge"?** The warning only says that the epoch
   limit was reached. For this comparison that is intended.
6. **Why is one layer of 36 logistic nodes worse than two of 18?** A second layer can combine the
   first layer's curves into more complicated shapes. One layer has to build every turn of the
   spiral by adding up 36 separate S-shaped steps, and with these settings that fit is still
   under way at epoch 1500.
7. **Why does depth hurt logistic but not relu?** The gradient is multiplied by the activation's
   slope at each layer: at most 0.25 for logistic, 1 for an active relu node.
8. **Does relu have its own problem?** Yes. A relu node whose input is negative for every point
   outputs 0 and gets no gradient, so it never recovers. In a layer of 6 nodes, losing one is a
   large share of the layer. With other seeds the six-layer relu network varied more than the
   other relu models (85.83 to 99.17% test).
9. **Why are some test accuracies above the training accuracy?** The test set is small. One test
   point is 0.83%, so a difference of one or two points either way is chance.
10. **Is 99.17% the best possible?** It is 119 of 120 test points. The data has a little noise,
    so the last point may simply sit on the wrong arm.
11. **What would overfitting look like in Table 4?** Training accuracy near 100% with test
    accuracy clearly lower, and the gap growing with M. We see neither.
12. **Why do the boundaries of relu have corners here too?** For the same reason as leaky ReLU
    in Part 1: the network is linear inside each region where no node switches on or off.

## Changes that might be asked for

| Change | Where |
|---|---|
| Another solver setting (learning rate, alpha, epochs, seed) | the `SETTINGS` dictionary |
| Other architectures or widths | `ARCHITECTURES`, `WIDTHS` |
| Another activation, such as `'tanh'` | `ACTIVATIONS` (the figures have one column or row per activation, so their grids would need the new size) |
| Other snapshot epochs | `EVOLUTION_EPOCHS` |
| Loss curves of another architecture | `loss_curve_figure` takes the last entry of `ARCHITECTURES`; change `arch` there |
| Test accuracy at the snapshots | in `boundary_evolution`, also store `clf.score(X_test, y_test)` |
