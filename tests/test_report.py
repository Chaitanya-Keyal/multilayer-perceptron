"""The report against the recorded results: the statements its answers make in words, the word
limit, and the note under Table 2."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "results" / "results.json").read_text())
TABLE2 = {(a, H): row for a, H, *row in RESULTS["part1"]["rows"]}  # loss, train, test, loss after
TABLE3 = {(tuple(arch), a): row for arch, a, *row in RESULTS["part2"]["table3"]}
TABLE4 = {(M, a): row for M, a, *row in RESULTS["part2"]["table4"]}  # P, train, test, gap
A1, A2, A3, A4 = (36,), (18, 18), (12, 12, 12), (6, 6, 6, 6, 6, 6)


def test_q1_one_hidden_node_is_a_linear_classifier():
    linear_train, linear_test = RESULTS["part1"]["linear"]
    for activation in ("leaky_relu", "tanh"):
        _, train, test, _ = TABLE2[activation, 1]
        assert (train, test) == (linear_train, linear_test)
        assert TABLE2[activation, 2][2] == linear_test  # no gain yet at H = 2
        assert TABLE2[activation, 32][2] > linear_test + 0.05
    # at H = 8 leaky ReLU is still near the linear level and tanh is well past it
    assert TABLE2["leaky_relu", 8][1] - linear_train < 0.01
    assert TABLE2["tanh", 8][1] - linear_train > 0.05


def test_q2_two_layers_are_best_and_six_are_not():
    for activation in ("logistic", "relu"):
        test = {arch: TABLE3[arch, activation][1] for arch in (A1, A2, A3, A4)}
        assert max(test, key=test.get) == A2 and test[A2] > max(test[A1], test[A3], test[A4])
    relu = sorted((TABLE3[arch, "relu"][1] for arch in (A1, A2, A3, A4)), reverse=True)
    assert TABLE3[A4, "relu"][1] == relu[1] > relu[2]  # second
    assert TABLE3[A4, "logistic"][1] == min(
        TABLE3[arch, "logistic"][1] for arch in (A1, A2, A3, A4)
    )
    lowest = RESULTS["part2"]["lowest_loss"]
    assert 0.65 < lowest["logistic"] < 0.69 and lowest["relu"] < 0.02


def test_q3_gap_does_not_grow_with_width():
    gap = {key: row[3] for key, row in TABLE4.items()}
    largest = sorted(gap, key=lambda key: abs(gap[key]), reverse=True)[:3]
    assert all(M <= 4 for M, _ in largest)
    assert {gap[key] > 0 for key in largest} == {True, False}  # of both signs
    assert all(abs(gap[64, a]) < 1 for a in ("logistic", "relu"))
    small = [acc for M in (2, 4) for a in ("logistic", "relu") for acc in TABLE4[M, a][1:3]]
    assert max(small) < 0.80
    best = {
        a: max((2, 4, 16, 64), key=lambda M, a=a: TABLE4[M, a][2]) for a in ("logistic", "relu")
    }
    assert best == {"logistic": 64, "relu": 16}
    assert TABLE4[16, "relu"][2] > TABLE4[64, "relu"][2]  # a strict best, not a tie
    assert TABLE4[64, "logistic"][0] == 4417


def test_table_2_note_names_the_only_row_that_differs():
    differs = [
        key for key, (loss, _, _, after) in TABLE2.items() if f"{loss:.4f}" != f"{after:.4f}"
    ]
    assert differs == [("leaky_relu", 8)]


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="needs pdftotext")
def test_answers_stay_within_120_words():
    command = ["pdftotext", "-layout", ROOT / "report.pdf", "-"]
    text = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    assert "??" not in text  # a number missing from results/numbers.tex
    answers = text[text.index("Part 3: Short Analysis") : text.index("Figure 1:")]
    parts = re.split(r"\bQ[123]\.", answers)[1:]
    assert len(parts) == 3
    for answer in parts:
        assert 60 < len(answer.split()) <= 120
        assert re.search(r"Table [234]|Figure [1-4]", answer)  # each answer points at our results
