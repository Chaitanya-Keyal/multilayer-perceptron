# Part 3: the three answers, and what is behind each sentence

## In short

The report answers three questions in at most 120 words each. Every answer has to point at our
own tables or figures. This file repeats each answer's claims one by one, says where the number
comes from and gives the reasoning in more words than the report has room for. The answers
themselves are in `report.pdf`, and `tests/test_report.py` checks the statements against
`results/results.json`.

## Q1: why H = 1 cannot beat a linear classifier, what changes for H >= 2, and the two shapes

**"The output depends on x only through z = w1 . x + b1."**
With one hidden node, `y_hat = sigmoid(w2 * phi(z) + b2)`. The point x enters only through the
single number z.

**"phi and the sigmoid are monotonic, so y_hat >= 0.5 on one side of a threshold on z."**
Monotonic means the function only goes up as its input goes up (or only down). tanh, leaky ReLU
and the sigmoid all only go up. So as z grows, `y_hat` moves in one direction only (up if w2 is
positive, down if negative), and it can cross 0.5 at most once. The points where z equals that
crossing value form the line `w1 . x + b1 = c`. A straight line.

**"Both H = 1 rows of Table 2 reach 87.25% train and 88.00% test, exactly our logistic
regression."**
Table 2, and the line the script prints under it. A logistic regression is the best a straight
line can do with this loss, and the H = 1 networks arrive at the same accuracy.

**"With H >= 2 several such units add up and the boundary can bend."**
Each hidden node has its own line. The output adds up their contributions, so the place where
the sum crosses 0.5 no longer has to be straight. In our run H = 2 is still at 88.00% test; the
gain shows up at H = 8 for tanh and H = 32 for leaky ReLU.

**"Leaky ReLU gives straight segments with sharp corners, still almost a line after 2000 epochs
(87.50% train); tanh gives one smooth curve around both moons (95.50% train)."**
`part1_boundaries.png`, right-hand column. Leaky ReLU is linear on each side of its corner, so
the network is linear inside each region between the nodes' lines, and the boundary can only
turn where it crosses one. tanh is smooth, so its boundary is. The 87.50% is a slow start with
this seed (see [p1-scratch-mlp.md](p1-scratch-mlp.md)), which is why the answer says "still".

## Q2: which split of 36 nodes is best, and why depth does not help by itself

**"For both activations the two-layer split (18, 18) is best: 99.17% test for relu and 97.50%
for logistic."**
Table 3, test column. For relu the next best is 98.33%; for logistic it is 83.33%.

**"The deepest split is second for relu (98.33%) and worst for logistic (66.67%)."**
Table 3 again. So the answer to "is it the deepest one?" is no, for both.

**"With a fixed budget, more depth means narrower layers, and everything the network knows
about a point has to fit through each 6-node layer."**
36 nodes in six layers is 6 per layer. The output of a layer is all the next layer gets. If six
numbers cannot describe where a point sits on the spiral, later layers cannot recover it.

**"The gradient also passes back through more layers, and the logistic derivative is at most
0.25, so after six layers very little reaches the first ones."**
Backpropagation multiplies by the activation's slope at every layer. For the logistic function
that is `s(1 - s)`, largest at s = 0.5 where it is 0.25. Six of those multiply to at most
0.00024.

**"In Figure 3 its loss never drops below 0.6529 (final 0.6531), while relu ends at 0.0184."**
`part2_loss_curves.png` and the Final Loss column of Table 3. A loss of 0.693 is what a model
gets by always answering 0.5, so 0.65 is barely better than knowing nothing.

## Q3: the train-test gap as the width grows

**"As M goes from 2 to 64, P(M) grows from 15 to 4417, but the train-test gap does not grow
with it."**
`P(M) = M^2 + 5M + 1` gives 15, 37, 337, 4417. If more parameters meant more overfitting, the
gap would widen from left to right in Table 4. It does not.

**"It is largest, and changes sign, for the two small networks (relu: +6.05 points at M = 2,
-3.13 at M = 4), which underfit: every accuracy there is below 80%."**
Gap = train accuracy minus test accuracy, from the printed values. A negative gap means the test
accuracy is the higher one. A model cannot truly be better on unseen points, so gaps of both
signs on a 120-point test set are chance. These networks are simply too small.

**"At M = 64 the gap is only +0.63 (logistic) and +0.84 (relu), about one test point (0.83)."**
One test point is 1/120 = 0.83 points.

**"logistic generalises best at M = 64 (98.33% test) and relu at M = 16 (99.17%)."**
The highest test accuracy in each row of Table 4.

**"So there is no clear overfitting at M = 64, despite about nine parameters per training
point."**
4417 / 480 = 9.2. Overfitting would show as a training accuracy well above the test accuracy.
relu does lose one test point between M = 16 and M = 64 (119 to 118 correct), which is too small
a change to call overfitting.

## If a number changes

The numbers in the answers are not typed into the report. `report/report.tex` reads them from
`results/numbers.tex`, which `python -m tools.run_all` writes. The words around them ("best",
"second", "does not grow") are checked by `tests/test_report.py`, which fails if a rerun makes
one of them untrue. The same file checks that each answer stays within 120 words, counted on the
text of the built PDF.
