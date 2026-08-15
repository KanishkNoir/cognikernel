# v0.1.2 — smarter file ranking, and a silent gap for MultiEdit users closed

**No single catastrophic bug this time** — instead, a batch of measured fixes
to how CogniKernel decides which files make it into the skeleton (the
summary Claude reads instead of opening every file), plus one real gap:
editing files with `MultiEdit` wasn't updating CogniKernel's understanding
of them at all. Worth upgrading either way.

```
pip install --upgrade cognikernel
```

---

## Better file rankings, measured rather than assumed

CogniKernel builds a "skeleton" of your project — a compact summary of
classes, functions, and signatures — so Claude doesn't have to read every
file to know what's in it. When a project has more files than fit in the
summary's budget, CogniKernel has to choose which ones matter most. This
release fixes three ways that choice was going wrong, found by actually
measuring outcomes across 17 real projects rather than guessing:

**Project size was skewing the rankings.** Part of how a file's importance
was scored — roughly, "how central is this file to the rest of your
code" — behaved differently depending on how many files were in the
project. The same file, with the same role in the codebase, could score
very differently just because the project was bigger or smaller. That's
now fixed: a file's importance is measured consistently no matter the
project's size.

**Files were getting stripped down before they were dropped.** When too
much needed to fit, CogniKernel used to trim detail from *every* file
first — cutting classes and methods down to almost nothing — and only
then start removing whole files if it still didn't fit. That meant you
could end up with thin, half-useful summaries of files you'd never touch,
while a file you actually needed still got cut. It now removes the least
useful whole files first, and only trims detail as a last resort.

**Test files were crowding out real source code.** Test files naturally
contain lots of small functions — one per test case — which made the old
scoring think they were more "important" than they really are. In this
project's own skeleton, a test file was ranking *above every actual source
file*. Tests are still included (you do work on them), just no longer
allowed to push real application code out of the picture by default.

**Combined effect**, measured on 17 real projects: when predicting which
files someone would touch next, the skeleton now includes the right file
45.5% of the time, up from 37.5%.

**A fourth fix, found afterward:** CogniKernel gives a ranking boost to
files that have been "recently active." That boost used to be all-or-nothing
— any file mentioned twice or more got the identical boost regardless of
whether that activity was an hour ago or several work sessions back, and a
file mentioned only once got no boost at all, even if that one mention was
from the session that had just ended. We found a real case where this
caused a file someone had *just edited* to be dropped from the skeleton in
favor of an older file that simply happened to be brought up more times,
further in the past. The boost is now a sliding scale, weighing both how
recently and how often a file came up — verified to fix that exact case on
the real project it was found on.

---

## `MultiEdit` edits weren't updating your project's skeleton

`MultiEdit` — the tool Claude Code uses for most multi-part file edits — was
never connected to CogniKernel's refresh step. Editing a file with
`MultiEdit` left its skeleton entry exactly as it was before the edit,
silently, with no indication anything was out of date. It's now treated
identically to a normal edit.

---

## The `skeleton` tool wasn't giving full detail when it said it would

CogniKernel has a dedicated `skeleton` tool for pulling one file's complete,
unabridged detail — it's specifically what CogniKernel tells you to reach
for after declining to let you re-read a file it's already summarized for
you. A separate internal limit meant that even through this "full detail"
path, a class with 18 methods still only showed 5 of them. It now genuinely
returns everything for the file you ask about. Separately, wherever a
listing elsewhere *is* still trimmed for space, it now says so explicitly
(e.g. `+3 more public symbols not shown`) instead of silently looking
complete.

---

## New: the groundwork for better activity tracking

CogniKernel now records every real file edit, per work session, in its own
table. This release doesn't use that data for anything yet — it's being
collected so a future release can rank "recently active" files by genuine
edit history instead of today's mention-based heuristic.

---

## Notes

- **Schema migrates automatically** the first time you open an existing
  project after upgrading. Nothing needs rebuilding by hand.
- **Adaptive, per-project token budgets are designed but not shipped** in
  this release.

**Full changelog:** [`CHANGELOG.md`](CHANGELOG.md) ·
[`v0.1.1...v0.1.2`](https://github.com/KanishkNoir/cognikernel/compare/v0.1.1...v0.1.2)
