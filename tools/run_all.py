r"""Runs both parts as their scripts do and keeps what they print and return.

    uv run python -m tools.run_all          both parts, about 30 s
    uv run python -m tools.run_all part1    one of them; the other comes from results.json

results/partN.txt     what `python partN_*.py` prints, byte for byte (the zip check diffs these)
results/results.json  every number the parts return
results/numbers.tex   the numbers report/report.tex quotes, as \resdef{key}{value}

The four figures are written to the repo root, where the scripts themselves write them.
"""

import contextlib
import io
import json
import os
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import sklearn

import part1_mlp_scratch
import part2_sklearn_exploration

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARTS = {"part1": part1_mlp_scratch, "part2": part2_sklearn_exploration}
SHORT = {"leaky_relu": "lrelu", "tanh": "tanh"}


class Tee(io.StringIO):
    """Keeps everything printed and still shows it on the terminal."""

    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


def run(name):
    out = Tee()
    with contextlib.redirect_stdout(out):
        numbers = PARTS[name].main()
    (RESULTS / f"{name}.txt").write_text(out.getvalue())
    return numbers


def plain(x):
    """numpy scalars and tuples as JSON values."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x.item() if isinstance(x, np.generic) else x


def pct(x):
    return f"{100 * x:.2f}"


def sci(x):
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def macros(r):
    m = {}
    p1, p2 = r["part1"], r["part2"]
    for activation, check in p1["gradient_check"].items():
        m[f"gc-{SHORT[activation]}"] = sci(check["worst"])
        m[f"gc-{SHORT[activation]}-rows"] = f"{check['start']}--{check['start'] + 4}"
    for activation, H, loss, train, test, loss_after in p1["rows"]:
        key = f"t2-{SHORT[activation]}-{H}"
        m[f"{key}-loss"], m[f"{key}-after"] = f"{loss:.4f}", f"{loss_after:.4f}"
        m[f"{key}-train"], m[f"{key}-test"] = pct(train), pct(test)
    m["linear-train"], m["linear-test"] = (pct(a) for a in p1["linear"])

    for arch, activation, train, test, loss, norm in p2["table3"]:
        key = f"t3-{len(arch)}-{activation}"  # the number of hidden layers names the architecture
        m[f"{key}-train"], m[f"{key}-test"] = pct(train), pct(test)
        m[f"{key}-loss"], m[f"{key}-norm"] = f"{loss:.4f}", f"{norm:.4f}"
    for M, activation, P, train, test, gap in p2["table4"]:
        key = f"t4-{M}-{activation}"
        m[f"{key}-train"], m[f"{key}-test"] = pct(train), pct(test)
        m[f"{key}-gap"] = f"{gap:+.2f}".replace("-", r"\textminus{}")
        m[f"t4-{M}-P"] = str(P)
    for activation, loss in p2["lowest_loss"].items():
        m[f"t3-6-{activation}-minloss"] = f"{loss:.4f}"
    for activation, accs in p2["evolution"].items():
        for epoch, acc in zip(part2_sklearn_exploration.EVOLUTION_EPOCHS, accs, strict=True):
            m[f"evo-{activation}-{epoch}"] = pct(acc)

    m["python"] = platform.python_version()
    m["numpy"], m["sklearn"], m["matplotlib"] = (
        np.__version__,
        sklearn.__version__,
        matplotlib.__version__,
    )
    return m


def main():
    os.chdir(ROOT)  # the scripts save their figures in the current folder
    RESULTS.mkdir(exist_ok=True)
    names = sys.argv[1:] or list(PARTS)
    if unknown := set(names) - set(PARTS):
        raise SystemExit(f"unknown part(s): {', '.join(sorted(unknown))}")
    path = RESULTS / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        results[name] = plain(run(name))
    path.write_text(json.dumps(results, indent=1) + "\n")
    lines = ["% Written by python -m tools.run_all from results/results.json; do not edit."]
    lines += [rf"\resdef{{{k}}}{{{v}}}" for k, v in macros(results).items()]
    (RESULTS / "numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
