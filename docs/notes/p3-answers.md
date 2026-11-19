# Part 3: the three answers, and what is behind each sentence

## In short

The report answers three questions in at most 120 words each. Every answer has to point at our
own tables or figures. This file repeats each answer's claims one by one, says where the number
comes from and gives the reasoning in more words than the report has room for. The answers
themselves are in `report.pdf`, and `tests/test_report.py` checks their comparisons against
`results/results.json`.

## Q1: why H = 1 cannot beat a linear classifier, what changes for H >= 2, and the two shapes

**"The output depends on x only through z = w1 . x + b1."**
With one hidden node, `y_hat = sigmoid(w2 * phi(z) + b2)`. The sample x enters only through the
single number z.

**"phi and the sigmoid are monotonic, so y_hat >= 0.5 on one side of a threshold on z."**
Monotonic means the function only goes up as its input goes up (or only down). tanh, leaky ReLU
and the sigmoid all only go up. So as z grows, `y_hat` moves in one direction only (up if w2 is
positive, down if negative), and it can cross 0.5 at most once. The points where z equals that
crossing value form the line `w1 . x + b1 = c`. A straight line.

**"Both H = 1 rows of Table 2 reach 87.25% train and 88.00% test, the same as our logistic
regression."**
Table 2, and the line the script prints under it. A logistic regression is a straight-line
classifier trained with the same loss, and the H = 1 networks arrive at the same accuracy.

**"With H >= 2 several nodes add up and the boundary can bend: test accuracy is still 88.00% at
H = 2 but 94.00% and 95.00% at H = 32."**
Each hidden node has its own line. The output adds up their contributions, so the place where
the sum crosses 0.5 no longer has to be straight. Being able to bend is not the same as having
learnt to: in our run two nodes do no better than a line, and the gain shows at H = 8 for tanh
and at H = 32 for both.

**"At H = 8 leaky ReLU gives straight segments with sharp corners, still almost a line (87.50%
train); tanh gives one smooth curve (95.50% train)."**
`part1_boundaries.png`, right-hand column. Leaky ReLU is linear on each side of its corner, so
the network is linear inside each region between the nodes' lines, and the boundary can only
turn where it crosses one. tanh is smooth, so its boundary is. The 87.50% is a slow start with
this seed (see [p1-scratch-mlp.md](p1-scratch-mlp.md)), which is why the answer says "still".

## Q2: which split of 36 nodes is best, and why depth does not help by itself

**"For both activations the two-layer split (18, 18) is best: 99.17% test for relu and 97.50%
for logistic."**
Table 3, test column. For relu the next best is 98.33%; for logistic it is 83.33%.

**"The deepest split is second for relu (98.33%) and worst for logistic (66.67%)."**
Table 3 again. So the answer to "is it the deepest one?" is no, for both. For relu the
difference between first and second is one test sample.

**"With a fixed budget, more depth means narrower layers, so each layer has fewer pieces to
shape the boundary with."**
36 nodes in six layers is 6 per layer. Each relu node contributes one straight cut and each
logistic node one smooth step, so a layer of 6 has fewer of them to combine than a layer of 18.

**"The gradient also has to pass back through more layers, and the logistic slope is at most
0.25, so early in training very little reaches the first layers."**
Backpropagation multiplies by the activation's slope at every layer. For the logistic function
that is `s(1 - s)`, largest at s = 0.5 where it is 0.25. Six of those multiply to at most
0.00024. "Early in training" matters: later on Adam does move the first layer's weights (see
[p2-depth-width.md](p2-depth-width.md)), but the loss stays near 0.65 and the boundary stays
close to a straight line.

**"In Figure 3 the logistic loss never gets far below 0.69 (lowest 0.6529, final 0.6531), while
relu ends at 0.0184."**
`part2_loss_curves.png`, the "Lowest value of loss_curve_" lines the script prints, and the
Final Loss column of Table 3. A loss of 0.693 is what a model gets by always answering 0.5, so
0.65 is barely better than knowing nothing.

## Q3: the train-test gap as the width grows

**"As M goes from 2 to 64, P(M) grows from 15 to 4417, but the train-test gap does not grow
with it."**
`P(M) = M^2 + 5M + 1` gives 15, 37, 337, 4417. If more parameters meant more overfitting, the
gap would widen from left to right in Table 4. It does not.

**"For logistic it is -1.25, +2.92, +1.67 and +0.63 points at M = 2, 4, 16, 64; for relu +6.05,
-3.13, +0.00 and +0.84."**
Gap = train accuracy minus test accuracy, taken between the two printed values. The script
prints it as the last column of Table 4. A negative gap means the test accuracy is the higher
one.

**"The three largest are at M <= 4, where the networks underfit (every accuracy is below
80%)."**
+6.05, -3.13 and +2.92. A model cannot truly be better on unseen samples, so gaps of both signs
on a 120-sample test set are chance. These networks are too small to learn the shape.

**"At M = 64 both gaps are about one test sample (0.83)."**
One test sample is 1/120 = 0.83 percentage points.

**"logistic generalises best at M = 64 (98.33% test) and relu at M = 16 (99.17%)."**
The highest test accuracy in each row of Table 4.

**"So M = 64 shows no clear overfitting, despite about nine parameters per training sample."**
4417 / 480 = 9.2. Overfitting would show as a training accuracy well above the test accuracy.
relu does lose one test sample between M = 16 and M = 64 (119 to 118 correct), which is too
small a change to call overfitting.

## If a number changes

The results quoted in the answers are not typed into the report. `report/report.tex` reads them
from `results/numbers.tex`, which `python -m tools.run_all` writes. The comparisons around them
("best", "second", "the three largest") are checked by `tests/test_report.py`, which fails if a
rerun makes one of them untrue. The same file checks that each answer stays within 120 words,
counted on the text of the built PDF.
