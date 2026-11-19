"""The handout's generators: copied exactly, and giving the splits the handout describes."""

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from common import generate_moons, generate_spirals

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("generate", "n_train", "n_test"), [(generate_moons, 400, 100), (generate_spirals, 480, 120)]
)
def test_split_sizes_and_scaling(generate, n_train, n_test):
    X_train, y_train, X_test, y_test = generate()
    assert X_train.shape == (n_train, 2) and y_train.shape == (n_train,)
    assert X_test.shape == (n_test, 2) and y_test.shape == (n_test,)
    assert set(np.unique(y_train)) == {0, 1}
    # standardised with the training statistics, so only the training set is exactly centred
    assert np.allclose(X_train.mean(axis=0), 0) and np.allclose(X_train.std(axis=0), 1)
    assert not np.allclose(X_test.mean(axis=0), 0)


@pytest.mark.parametrize("generate", [generate_moons, generate_spirals])
def test_same_arrays_every_call(generate):
    for a, b in zip(generate(), generate(), strict=True):
        assert np.array_equal(a, b)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="needs pdftotext")
def test_common_is_the_handouts_code():
    """common.py against the code printed in the handout, ignoring whitespace (the PDF text has
    lost its indentation)."""
    command = ["pdftotext", "-layout", ROOT / "assignment.pdf", "-"]
    pdf = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    code = pdf[pdf.index("import numpy as np") : pdf.index("No test-set leakage")]
    # the code runs over a page break: drop the page number and the next page's header
    code = re.sub(r"\n\s*1\s*\n.*?Assignment 2\s*\n", "\n", code, flags=re.S)

    def squash(text):
        return re.sub(r"\s+", "", text).replace("•", "")

    assert squash(code) == squash((ROOT / "common.py").read_text())
