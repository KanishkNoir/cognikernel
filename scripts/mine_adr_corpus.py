"""Mine real Architecture Decision Records into training data.

ADRs are the highest-quality public source for CogniKernel's ontology: real
decisions with rationale, considered-and-rejected alternatives, constraints —
and (MADR/Nygard status fields) explicit SUPERSEDED-BY links, i.e. ground-truth
supersession pairs across many domains.

Curated public repos (shallow/sparse clones into a scratch dir), parsed into:
  research/train_corpus/mined_adr_sentences.jsonl  {text,label,register,source,license,confidence}
  research/train_corpus/mined_adr_pairs.jsonl      {new_text,old_text,label,kind,source,license}

Labeling is deliberately conservative (section structure + cue rules, each item
tagged strong/weak confidence); an LLM-assisted refinement pass can follow.
This corpus is TRAINING material — the frozen eval (research/model_eval) comes
from our own stores and never overlaps these sources.

Usage: uv run python scripts/mine_adr_corpus.py [--workdir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("research/train_corpus")

# (repo_url, license, sparse_dirs or None for full depth-1 clone)
REPOS = [
    ("https://github.com/joelparkerhenderson/architecture-decision-record", "MIT", None),
    ("https://github.com/adr/madr", "MIT", None),
    ("https://github.com/arachne-framework/architecture-decisions", "unspecified-public", None),
    ("https://github.com/npryce/adr-tools", "GPL-3.0", None),
    ("https://github.com/backstage/backstage", "Apache-2.0", ["docs"]),
]

_ADR_DIR_RE = re.compile(r"(adr|decision|architecture[-_]decisions?)", re.I)
_ADR_FILE_RE = re.compile(r"^(\d{3,4}[-_].+|adr[-_]?\d+.*|.*decision.*)\.md$", re.I)

_SECTION_RE = re.compile(r"^#{1,4}\s+(.*)$")
_STATUS_SUPERSEDED_RE = re.compile(r"superseded\s+by\s+\[?([^\]\n]+)\]?", re.I)
_CHOSEN_RE = re.compile(r"^chosen option:?\s*(.*)", re.I)
_DECISION_CUE = re.compile(
    r"\b(we (will|shall|choose|chose|use|adopt|standardi[sz]e)|is chosen|"
    r"we decided|the decision is)\b", re.I)
_REJECT_CUE = re.compile(
    r"\b(instead of|rather than|we will not|we won'?t|rejected|ruled out|"
    r"not chosen|decided against|do not use)\b", re.I)
_CONSTRAINT_CUE = re.compile(r"\b(must|must not|never|always|shall not|required to)\b", re.I)

_MIN_LEN, _MAX_LEN = 40, 300


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=600)


def clone(url: str, dest: Path, sparse_dirs: list[str] | None) -> bool:
    try:
        if sparse_dirs:
            _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", url, str(dest)])
            _run(["git", "sparse-checkout", "set", *sparse_dirs], cwd=dest)
            _run(["git", "checkout"], cwd=dest)
        else:
            _run(["git", "clone", "--depth", "1", url, str(dest)])
        return True
    except Exception as exc:
        print(f"  clone failed: {url} ({exc})")
        return False


def find_adr_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.md"):
        rel = p.relative_to(root)
        parts_ok = any(_ADR_DIR_RE.search(part) for part in rel.parts[:-1])
        if parts_ok or _ADR_FILE_RE.match(p.name):
            if p.stat().st_size < 100_000:
                out.append(p)
    return out


def sentences(block: str):
    """Crude sentence split adequate for declarative ADR prose."""
    block = re.sub(r"\s+", " ", block).strip()
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z`\"'\(])", block):
        s = s.strip(" -*•")
        if _MIN_LEN <= len(s) <= _MAX_LEN and not s.startswith("|"):
            yield s


def parse_adr(text: str) -> dict:
    """Split an ADR into {title, status_raw, sections{name_lower: text}}."""
    lines = text.splitlines()
    title = ""
    sections: dict[str, list[str]] = {}
    current = "preamble"
    for ln in lines:
        m = _SECTION_RE.match(ln)
        if m:
            head = m.group(1).strip().lower()
            if not title:
                title = m.group(1).strip()
                current = "preamble"
                continue
            current = head
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(ln)
    sec = {k: "\n".join(v).strip() for k, v in sections.items()}
    status = ""
    for k, v in sec.items():
        if k.startswith("status"):
            status = v
    return {"title": title, "status": status, "sections": sec}


def decision_line(adr: dict) -> str:
    """The single sentence that best states the ADR's decision."""
    for key, body in adr["sections"].items():
        if key.startswith(("decision outcome", "decision", "outcome")):
            for s in sentences(body):
                m = _CHOSEN_RE.match(s)
                if m and len(m.group(1)) >= _MIN_LEN:
                    return m.group(1)
                if _DECISION_CUE.search(s) or _CHOSEN_RE.match(s):
                    return s
            for s in sentences(body):
                return s  # first plausible sentence
    return adr["title"] if _MIN_LEN <= len(adr["title"]) <= _MAX_LEN else ""


def mine_repo(root: Path, repo: str, license_: str, items: list, pairs: list) -> tuple[int, int]:
    adrs: dict[str, dict] = {}
    files = find_adr_files(root)
    for f in files:
        try:
            parsed = parse_adr(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        parsed["path"] = str(f.relative_to(root)).replace("\\", "/")
        adrs[f.stem.lower()] = parsed

    n_items = n_pairs = 0

    def add(text, label, register, confidence, path):
        nonlocal n_items
        items.append({
            "id": hashlib.sha256(text.lower().encode()).hexdigest()[:12],
            "text": text, "label": label, "register": register,
            "source": f"adr:{repo}:{path}", "license": license_,
            "confidence": confidence,
        })
        n_items += 1

    for stem, adr in adrs.items():
        path = adr["path"]
        for key, body in adr["sections"].items():
            if key.startswith(("decision outcome", "decision", "outcome")):
                for s in sentences(body):
                    if _CHOSEN_RE.match(s) or _DECISION_CUE.search(s):
                        add(s, "DECISION", "plain", "strong", path)
                    elif _REJECT_CUE.search(s):
                        add(s, "APPROACH_ABANDONED_DO_NOT_RETRY", "plain", "weak", path)
                    elif _CONSTRAINT_CUE.search(s):
                        add(s, "CONSTRAINT_HARD", "plain", "weak", path)
            elif key.startswith(("context", "problem", "consequences")):
                for s in sentences(body):
                    if _REJECT_CUE.search(s):
                        add(s, "APPROACH_ABANDONED_DO_NOT_RETRY", "plain", "weak", path)
                    elif _CONSTRAINT_CUE.search(s):
                        add(s, "CONSTRAINT_HARD", "plain", "weak", path)
            elif key.startswith(("considered options", "options")):
                # option lists are fragments; skip — the pros/cons prose is
                # rationale (explanation register), mined as NOISE examples.
                for s in sentences(body):
                    if not _REJECT_CUE.search(s) and not _DECISION_CUE.search(s):
                        add(s, "NOISE", "explanation", "weak", path)

        # Supersession pairs from the status field.
        m = _STATUS_SUPERSEDED_RE.search(adr["status"] or "")
        if m:
            ref = m.group(1).lower()
            new_adr = None
            for stem2, cand in adrs.items():
                if stem2 != stem and (stem2 in ref or ref in stem2
                                      or cand["title"].lower() in ref or ref in cand["title"].lower()):
                    new_adr = cand
                    break
            old_line, new_line = decision_line(adr), decision_line(new_adr) if new_adr else ""
            if old_line and new_line and old_line != new_line:
                pairs.append({
                    "new_text": new_line, "old_text": old_line, "label": 1,
                    "kind": "adr_superseded", "source": f"adr:{repo}:{path}",
                    "license": license_,
                })
                n_pairs += 1

    # Same-repo hard negatives: decision lines of distinct, non-linked ADRs.
    decision_lines = [(s, decision_line(a)) for s, a in adrs.items()]
    decision_lines = [(s, d) for s, d in decision_lines if d]
    for i in range(0, max(0, len(decision_lines) - 1), 2):
        (s1, d1), (s2, d2) = decision_lines[i], decision_lines[i + 1]
        if d1 != d2:
            pairs.append({
                "new_text": d1, "old_text": d2, "label": 0,
                "kind": "same_repo_live", "source": f"adr:{repo}",
                "license": license_,
            })
            n_pairs += 1
    return n_items, n_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="ck_adr_mine_"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    pairs: list[dict] = []
    for url, license_, sparse in REPOS:
        repo = url.rsplit("/", 1)[-1]
        dest = work / repo
        print(f"mining {repo} …")
        if not dest.exists() and not clone(url, dest, sparse):
            continue
        n_i, n_p = mine_repo(dest, repo, license_, items, pairs)
        print(f"  {n_i} sentences, {n_p} pairs")

    # in-file dedup by text
    seen: set[str] = set()
    uniq = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        uniq.append(it)

    with open(OUT_DIR / "mined_adr_sentences.jsonl", "w", encoding="utf-8") as f:
        for it in uniq:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "mined_adr_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"\ntotal: {len(uniq)} sentences  {len(pairs)} pairs")
    print("  labels:", dict(Counter(i["label"] for i in uniq)))
    print("  confidence:", dict(Counter(i["confidence"] for i in uniq)))
    print("  pair kinds:", dict(Counter(p["kind"] for p in pairs)))


if __name__ == "__main__":
    main()
