"""Salience recipe sweep — train N configs, export+eval each, pick the best.

The recipe (not the technique) is the cheapest salience lever (SetFit stays
competitive with full fine-tuning even at thousands of examples). This sweeps
the two unambiguous SetFit axes — max_steps (contrastive exposure) and
batch_size — holding the corpus constant (pre-adversarial train_sentences.jsonl)
so each result is interpretable.

Per config, sequentially (retrains clobber shared artifacts, so never parallel):
  1. train_salience_setfit.py <cfg args> --extra-corpus ...   (writes package head)
  2. export_setfit_onnx.py                                     (writes onnx body)
  3. stash body+head -> models/sweep/<name>/  (so the winner is re-installable)
  4. model_eval.py --tag sweep-<name>  (COGNIKERNEL_V2_BODY_DIR -> the fresh body)
  5. record acc / macro_f1 / deployment_view / held-out acc

At the end prints a comparison table and the winner. NOTE: with n=368 eval and
several configs, prefer a config on a SMOOTH trend over an isolated spike (a
single best number can be sweep-noise); the held-out 6-way acc from training is
an independent second signal.

Usage: uv run python scripts/sweep_salience.py [--configs s3000_b16,s2000_b32]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "models" / "sweep"
EVAL_DIR = ROOT / "research" / "model_eval"
EXPORT_ONNX = ROOT / "models" / "salience_setfit" / "onnx"
PKG_HEAD = ROOT / "src" / "cognikernel" / "extraction" / "heads" / "salience_v2.npz"
TRAIN_DEPS = ["--with", "setfit", "--with", "transformers<4.50"]
EXPORT_DEPS = TRAIN_DEPS + ["--with", "onnx", "--with", "onnxruntime"]

_BASE = ["--extra-corpus", "research/train_corpus/train_sentences.jsonl"]
_ADV = ["--extra-corpus", "research/train_corpus/train_sentences_adv.jsonl"]
_HUM = ["--extra-corpus", "research/train_corpus/train_sentences_hum.jsonl"]
_BOOST = ["--extra-corpus", "research/train_corpus/train_sentences_boost.jsonl"]

# name -> FULL train args (recipe + corpus). Recipe configs vary one axis at a
# time on the base corpus; `adv*` configs hold the recipe and swap in the
# adversarial corpus so the ONLY variable is the meta-framing training data.
ALL_CONFIGS = {
    "s800_b16":  ["--max-steps", "800", "--batch", "16", *_BASE],
    "s2000_b16": ["--max-steps", "2000", "--batch", "16", *_BASE],
    "s3000_b16": ["--max-steps", "3000", "--batch", "16", *_BASE],
    "s2000_b32": ["--max-steps", "2000", "--batch", "32", *_BASE],
    "s4000_b16": ["--max-steps", "4000", "--batch", "16", *_BASE],
    # adversarial-data experiment at the better-balanced 800-step recipe.
    "adv800":    ["--max-steps", "800", "--batch", "16", *_ADV],
    # humanized-data experiment (Spike A1) at the same 800-step recipe.
    "hum800":    ["--max-steps", "800", "--batch", "16", *_HUM],
    # P0-driven: thin-class boost (THREAD/ABANDONED/SOFT) — targets the biggest
    # measured model-vs-ceiling gap (THREAD recall 0.08 vs 0.96).
    "boost800":  ["--max-steps", "800", "--batch", "16", *_BOOST],
}


def _run(cmd: list[str], log: Path) -> str:
    """Run a uv command, STREAMING output live to a log (so a long job is
    observable and a mid-run kill still leaves partial progress on disk),
    then return the captured text for parsing."""
    lines: list[str] = []
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            lines.append(line)
        proc.wait()
    return "".join(lines)


def train(name: str, args: list[str], log: Path) -> float | None:
    # `args` already carries its own --extra-corpus (per-config), so nothing is
    # appended here — this is what lets adv* configs use a different corpus.
    cmd = ["uv", "run", *TRAIN_DEPS, "python", "scripts/train_salience_setfit.py", *args]
    out = _run(cmd, log)
    m = re.search(r"held-out 6-way acc:\s*([\d.]+)%", out)
    if "saved SetFit model" not in out:
        print(f"    !! train did not complete for {name}")
        return None
    return float(m.group(1)) / 100 if m else None


def export(name: str, log: Path) -> bool:
    cmd = ["uv", "run", *EXPORT_DEPS, "python", "scripts/export_setfit_onnx.py"]
    out = _run(cmd, log)
    if "PASS" not in out:
        print(f"    !! export parity did NOT pass for {name}")
        return False
    return True


def stash(name: str) -> Path:
    dst = SWEEP_DIR / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("body.onnx", "tokenizer.json"):
        src = EXPORT_ONNX / f
        if src.exists():
            shutil.copy2(src, dst / f)
    shutil.copy2(PKG_HEAD, dst / "salience_v2.npz")
    return dst


def evaluate(name: str, body_dir: Path, log: Path) -> dict | None:
    import os
    env = os.environ.copy()
    env["COGNIKERNEL_V2_BODY_DIR"] = str(body_dir)
    cmd = ["uv", "run", "--extra", "embedding", "python", "scripts/model_eval.py",
           "--tag", f"sweep-{name}"]
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env)
        f.write((proc.stdout or "") + (proc.stderr or ""))
    res = sorted(EVAL_DIR.glob(f"results_sweep-{name}_*.json"))
    if not res:
        return None
    return json.loads(res[-1].read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(ALL_CONFIGS),
                    help="comma-separated config names from ALL_CONFIGS")
    args = ap.parse_args()
    names = [c.strip() for c in args.configs.split(",") if c.strip() in ALL_CONFIGS]
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    logdir = SWEEP_DIR / "_logs"
    logdir.mkdir(exist_ok=True)

    rows = []
    for name in names:
        t0 = time.time()
        print(f"\n=== {name}  ({ALL_CONFIGS[name]}) ===", flush=True)
        held = train(name, ALL_CONFIGS[name], logdir / f"{name}.train.log")
        if held is None:
            print(f"    skipped (train failed) — see {logdir / f'{name}.train.log'}")
            continue
        if not export(name, logdir / f"{name}.export.log"):
            continue
        body = stash(name)
        rep = evaluate(name, body, logdir / f"{name}.eval.log")
        if rep is None:
            print("    eval produced no result")
            continue
        v2 = rep["salience_v2"]
        dv = v2["deployment_view"]
        rows.append({
            "name": name, "held_out": held, "acc": v2["accuracy"],
            "macro_f1": v2["macro_f1"], "capture_recall": dv["capture_recall"],
            "false_capture": dv["false_capture_rate"],
            "typing_acc": dv["typing_accuracy_of_captured"],
            "mins": round((time.time() - t0) / 60, 1),
        })
        r = rows[-1]
        print(f"    held={r['held_out']} acc={r['acc']} macroF1={r['macro_f1']} "
              f"cap_recall={r['capture_recall']} false_cap={r['false_capture']} "
              f"({r['mins']}m)", flush=True)

    print("\n" + "=" * 92)
    print(f"{'config':14} {'held6way':>9} {'acc':>6} {'macroF1':>8} "
          f"{'cap_rec':>8} {'false_cap':>10} {'typ_acc':>8}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: -x["macro_f1"]):
        print(f"{r['name']:14} {r['held_out']:>9} {r['acc']:>6} {r['macro_f1']:>8} "
              f"{r['capture_recall']:>8} {r['false_capture']:>10} {r['typing_acc']:>8}")
    if rows:
        best = max(rows, key=lambda x: x["macro_f1"])
        print(f"\nbest by macro_f1: {best['name']}  "
              f"(stashed at models/sweep/{best['name']}/ — install-heads-able)")
    (SWEEP_DIR / "sweep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
