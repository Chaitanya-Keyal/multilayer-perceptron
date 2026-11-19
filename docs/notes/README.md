# Study notes

The ideas behind each part, how the code implements them, and what our numbers show. Not part of
the submission. Read them in this order:

- [guide.md](guide.md): the whole assignment in plain language, a walk through the notebook and
  twelve likely questions; the first file to read
- [assignment.md](assignment.md): what is asked, the rules that apply everywhere, and where each
  item lives in the code and the output
- [p1-scratch-mlp.md](p1-scratch-mlp.md): the network of Part 1, backpropagation step by step,
  the gradient check, and Table 2
- [p2-depth-width.md](p2-depth-width.md): `MLPClassifier` and its settings, 36 nodes arranged
  four ways, the width sweep, and the boundary as it trains
- [p3-answers.md](p3-answers.md): the three answers of the report, sentence by sentence

Every file except this one and guide.md opens with an In short. The Part 1 and Part 2 files
then go through the idea, our results and what they mean, questions with answers, and the
changes most likely to be asked for. The Part 3 file goes through the three answers sentence by
sentence. Numbers are quoted from `results/` or read off the four figures and the loss curves.
Anything marked as an experiment was run separately and is not part of the submission.
