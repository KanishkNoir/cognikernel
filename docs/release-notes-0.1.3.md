# v0.1.3 — the right work item, less chatter kept as memory, and memory that explains itself

**If you installed 0.1.2 fresh, upgrade now:** the memory tools may be missing
entirely (details below). Beyond that fix, this release makes the session block
more trustworthy: it shows the work item you actually left open, keeps less of
Claude's own narration as "decisions", and gives you commands to ask memory why
it holds what it holds.

```
pip install --upgrade "cognikernel[embedding]"
```

---

## Fresh installs had no memory tools

CogniKernel's MCP server is built on the `mcp` library, and 0.1.2 accepted any
version of it. `mcp` 2.0 (released 2026-07-28) removed a module the server
needs, so a new install pulled in 2.x and the server stopped at startup. The
session-context block still arrived, but `recall`, `find_related`, `skeleton` and
`get_session_state` were missing, and Claude Code showed the server as
"failed".

0.1.3 requires `mcp` below 2. If you stay on 0.1.2, run
`pip install "mcp<2"` to get the tools back. The install check in CI now starts
the real server and requires it to list its tools, so this can't slip through
again.

---

## The session block shows the work item you left open

The "Working on" line is meant to answer "what were we in the middle of?". Several
things made it wrong:

- **Narration piled up.** Every "now running the tests" or "now writing the
  store" line Claude wrote became its own open work item.
- **Instructions outranked deferred work.** An ordinary request like "add the
  response schema" could beat the thing you'd explicitly parked for later.
- **A pointer could erase what it pointed at.** "This is the active work item for
  the next session." could replace the sentence that actually named the work.
- **The work item could be crowded out** of the block by a large set of decisions.

All four are fixed:
- Later narration retires earlier narration.
- An explicit "next session we'll…" handoff survives and ranks first.
- Instructions no longer outrank deferred work.
- Pointers no longer delete their subject.
- The work item's space is reserved before anything else is packed.

When the work item was opened in an earlier session, the block now says which
one: `Working on: build the replay command (S1 · 09-12)`. The fixes were checked
against every work-item event from four real benchmark projects.

---

## Less of Claude's own chatter remembered as decisions

- **Step narration** such as "Now let's run the full test suite." or "Let me check
  the frontend structure first." now ranks below real decisions, including lines
  already in your memory. A sentence that gives a reason ("…instead, because it
  matches production") still counts as a decision.
- **Answer and report lines** ("Summary: …", "Verified: 13/13 tests pass",
  "ANSWER: …") are no longer rescued as settings. Settings written the same way
  ("Max attempts: 2") still are.
- **Claude narrating CogniKernel itself** ("CogniKernel's Stop hook will persist
  this") now ranks far below project facts. It was marked before but never
  actually moved. The check was narrowed first, so a project that discusses
  CogniKernel in its own design isn't affected.

---

## Memory follows you into subdirectories

If a session changed into a subdirectory (`cd packages/core`), everything it
captured from then on went to a separate, empty memory. On one real project, 7
of 8 captures went there. Captures from a subdirectory now go to the memory that
already exists for the enclosing git repository. A memory is never created
automatically at the repository root, so unrelated projects inside a larger
repository stay separate.

---

## New: ask memory why

- **`cognikernel why <project> <claim id or words>`** shows:
  - where a claim came from: the session, the sentence and who said it
  - what the quality check noted
  - why it ranks where it does, factor by factor
  - what it replaced or was replaced by, and when

  Replaced claims can be looked up too.
- **`cognikernel show <project> --as-of <date or commit>`** lists what memory
  believed at a past time. Where older memory never recorded when something was
  replaced, it says so instead of guessing.
- **`cognikernel explain-recall <project> <query>`** shows why `recall` and the
  per-prompt push found what they did, and why they didn't find something. Add
  `--claim #id` to ask about one claim.
- **`cognikernel doctor`** now reports how many extra round trips CogniKernel's
  own tools added to your sessions.

---

## Cheaper sessions by default

New projects start in **advisory** mode. The old default refused Claude's first
read of every file in the skeleton, on the idea that signatures would often be
enough. On the benchmark, 89% of those refusals were followed straight away by
the same read, so each one cost an extra round trip and saved nothing.
- The rule that works stays on for everyone: re-reading a file already read in the
  same session is still refused.
- Existing projects keep their setting. Set `hook_policy = "strict"` in
  `.cognikernel/config.toml` to opt back in.

---

## Corrected numbers

Claude Code repeats a response's token usage on every transcript line that
response produces. Both `cognikernel telemetry` and the published benchmark added
every line, overstating tokens by 1.9× to 3.0×, and by different amounts on
different projects.

Both now count once per response:
- **Telemetry:** rows recorded the old way are labelled and kept out of
  `doctor`'s figures. Re-run `cognikernel telemetry <project>` to recount sessions
  whose transcripts still exist.
- **Benchmark:** the corrected results are in `docs/benchmark.md`. Against no
  memory at all, CogniKernel costs 23% more on the Relay project, from round trips
  its own tools add. The advisory default above removes part of that.

---

## Notes

- **Schema migrates automatically** (v20 → v22) the first time you open a project
  after upgrading. It adds belief-history and round-trip telemetry columns, and no
  rows are rewritten.
- **A `tool_guidance = "lean"` setting** is available for experiments. It keeps the
  tools but stops asking Claude to check memory before every read. The default is
  unchanged.
- **`cognikernel doctor` now points at the right folder** when the fine-tuned models
  aren't installed: `~/.cognikernel/models/…`, where `install-heads` puts them. It
  used to name a path inside the Python install.

**Full changelog:** [`CHANGELOG.md`](../CHANGELOG.md) ·
[`v0.1.2...v0.1.3`](https://github.com/KanishkNoir/cognikernel/compare/v0.1.2...v0.1.3)
