"""Builds the submission zip and checks it the way a grader will use it.

    uv run python -m tools.package          dist/2023B4A70727H_A2.zip
    uv run python -m tools.package ID       the same files under another student ID

The zip holds exactly the nine files the handout names, flat at its root, dated by the last
commit that changed one of them. It is then checked: report.pdf is what report/ builds from
results/numbers.tex now; common.py and Part 1 import only numpy, matplotlib and the standard
library (Part 2 also scikit-learn); and, unzipped into an empty folder with the figures removed,
each script runs with one plain command, prints exactly what results/partN.txt recorded and
writes the same figures again, byte for byte.
"""

import ast
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from tools import build_pdf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["part1_mlp_scratch.py", "part2_sklearn_exploration.py"]
FIGURES = [
    "part1_boundaries.png",
    "part2_depth_vs_width.png",
    "part2_loss_curves.png",
    "part2_boundary_evolution.png",
]
FILES = ["common.py", *SCRIPTS, *FIGURES, "README.txt", "report.pdf"]
ALLOWED = {"numpy", "matplotlib", "common"} | set(sys.stdlib_module_names)
ALLOWED_PART2 = ALLOWED | {"sklearn"}


def stamp():
    """The date of the last commit that changed one of the files, so that the same commit always
    gives the same zip, byte for byte."""
    command = ["git", "log", "-1", "--format=%cd", "--date=format:%Y %m %d %H %M %S", "--", *FILES]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if run.returncode != 0 or not run.stdout.strip():
        return (1980, 1, 1, 0, 0, 0)  # no git history: the earliest date a zip can hold
    return tuple(int(v) for v in run.stdout.split())


def build(student_id):
    path = ROOT / "dist" / f"{student_id}_A2.zip"
    path.parent.mkdir(exist_ok=True)
    date = stamp()
    with zipfile.ZipFile(path, "w") as zf:
        for name in FILES:
            info = zipfile.ZipInfo(name, date_time=date)
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            zf.writestr(info, (ROOT / name).read_bytes())
    return path


def check_pdf():
    """Rebuild the PDF in an empty folder and compare it, byte for byte, with the one that ships
    (the build has no date or random ID in it)."""
    if shutil.which("latexmk") is None:
        print("  report.pdf: NOT CHECKED, latexmk is not installed")
        return []
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("report/report.tex", "results/numbers.tex", *FIGURES):
            (Path(tmp) / name).parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / name, Path(tmp) / name)
        try:
            fresh = build_pdf.build(Path(tmp) / "report").read_bytes()
        except build_pdf.BuildError as error:
            print("  report.pdf: FAILED")
            return [f"report/ does not build (run python -m tools.build_pdf for the log):\n{error}"]
    same = fresh == (ROOT / "report.pdf").read_bytes()
    print(f"  report.pdf: {'ok' if same else 'FAILED'}")
    return [] if same else ["report.pdf is out of date: run python -m tools.build_pdf"]


def check_imports(folder):
    problems = []
    for name in ("common.py", *SCRIPTS):
        allowed = ALLOWED_PART2 if name.startswith("part2") else ALLOWED
        for node in ast.walk(ast.parse((folder / name).read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            problems += [f"{name} imports {m}" for m in modules if m.split(".")[0] not in allowed]
    for name in ("common.py", *SCRIPTS, "README.txt"):
        if "/home/" in (folder / name).read_text():
            problems.append(f"{name} has an absolute path")
    return problems


def check_runs(folder):
    """Run each script the way a grader would and compare with the recorded output and figures."""
    for name in FIGURES:
        (folder / name).unlink()
    problems = []
    for i, script in enumerate(SCRIPTS, start=1):
        run = subprocess.run([sys.executable, script], cwd=folder, capture_output=True, text=True)
        figures = [name for name in FIGURES if name.startswith(f"part{i}")]
        problem = None
        if run.returncode != 0:
            problem = f"{script} failed:\n{run.stderr}"
        elif run.stdout != (ROOT / "results" / f"part{i}.txt").read_text():
            problem = f"{script} prints something other than results/part{i}.txt"
        elif run.stderr:
            problem = f"{script} wrote to stderr:\n{run.stderr}"
        elif any(not (folder / name).exists() for name in figures):
            problem = f"{script} did not write all of {', '.join(figures)}"
        elif any((folder / name).read_bytes() != (ROOT / name).read_bytes() for name in figures):
            problem = f"{script} writes figures that differ from the ones in the zip"
        print(f"  {script}: {'ok' if problem is None else 'FAILED'}")
        problems += [problem] if problem else []
    return problems


def main():
    ids = sys.argv[1:] or ["2023B4A70727H"]
    problems = check_pdf()
    for student_id in ids:
        path = build(student_id)
        print(path.relative_to(ROOT))
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as zf:
            assert sorted(zf.namelist()) == sorted(FILES)
            zf.extractall(tmp)
            problems += check_imports(Path(tmp)) + check_runs(Path(tmp))
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    if problems:
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
