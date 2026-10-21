"""Small helpers shared by all notebooks.

run("script.py", args...)  runs one step of the study (a script in src/) with the SAME Python that runs
                           this notebook, prints its output live, and stops with a clear message if it fails.
show("results/...csv")     shows a result table.
figure("Fig2_lab_vs_field") shows a paper figure inline.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # the project folder (contains src/, data/, results/)
SRC = ROOT / "src"


def run(script, *args):
    """Run src/<script> with arguments, streaming its output into the notebook."""
    cmd = [sys.executable, "-u", str(SRC / script), *map(str, args)]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               TQDM_MININTERVAL="20",   # progress bar: one update every 20 s, keeps the notebook readable
               TQDM_NCOLS="90")
    print("> python src/" + script + (" " + " ".join(map(str, args)) if args else ""))
    print("  started", time.strftime("%H:%M:%S"), flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:          # the notebook "Interrupt" button
        p.terminate()
        print("\n  Stopped by you. Re-running this cell continues where it stopped (finished work is kept).")
        return
    code = p.wait()
    mins = (time.time() - t0) / 60
    if code != 0:
        raise RuntimeError(f"src/{script} failed (exit code {code}) after {mins:.1f} min. "
                           "Read the last lines above; copy them to Claude if you need help.")
    print(f"  finished {time.strftime('%H:%M:%S')}  ({mins:.1f} min)")


def show(relpath, n=60):
    """Display a CSV from the project folder as a table."""
    import pandas as pd
    from IPython.display import display
    p = ROOT / relpath
    if not p.exists():
        print("Not found yet:", relpath, "- run the step above first.")
        return None
    df = pd.read_csv(p)
    with pd.option_context("display.max_rows", n, "display.max_columns", 30, "display.width", 200,
                           "display.precision", 1):
        display(df.head(n))


def figure(name, width=900):
    """Display figures/final/<name>.png inline."""
    from IPython.display import Image, display
    p = ROOT / "figures" / "final" / f"{name}.png"
    if not p.exists():
        print("Not found yet:", p.relative_to(ROOT), "- run make_figures first.")
        return
    display(Image(filename=str(p), width=width))


def where():
    print("Project folder :", ROOT)
    print("Python used    :", sys.executable)
    ok = (ROOT / "src" / "fm.py").exists()
    print("src/ found     :", ok)
    if "plantfm" not in sys.executable:
        print("\nWARNING: this notebook is not using the 'plantfm' environment.\n"
              "Click the kernel name at the top right of VS Code -> Python Environments -> plantfm.")
