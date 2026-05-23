"""
Parse real session JSONL transcripts and count tool calls.
Distinguishes 'orientation' reads (before first Write/Edit) from 'implementation' reads.
Also extracts token usage per turn.
"""
import json, sys
from pathlib import Path
from collections import defaultdict

EXPLORE_TOOLS = {"Read", "Glob", "Grep"}
WRITE_TOOLS   = {"Write", "Edit", "NotebookEdit"}
MCP_TOOLS     = {"mcp__cognikernel__get_session_state"}

def analyse_session(jsonl_path: str, label: str) -> dict:
    path = Path(jsonl_path)
    if not path.exists():
        return {"label": label, "missing": True}

    tool_calls  = []
    token_turns = []  # (input_tokens, cache_read, cache_creation, output_tokens)

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            rtype = rec.get("type", "")

            # assistant turns contain tool_use blocks
            if rtype == "assistant":
                msg = rec.get("message", {})
                usage = msg.get("usage", {})
                if usage:
                    token_turns.append({
                        "input":    usage.get("input_tokens", 0),
                        "cache_r":  usage.get("cache_read_input_tokens", 0),
                        "cache_c":  usage.get("cache_creation_input_tokens", 0),
                        "output":   usage.get("output_tokens", 0),
                    })
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "")
                        inp  = block.get("input", {})
                        summary = (
                            inp.get("file_path") or
                            inp.get("pattern")   or
                            inp.get("query", "")[:60] or
                            inp.get("path", "")
                        )
                        tool_calls.append({
                            "tool":    name,
                            "summary": str(summary)[:70],
                        })

    # Find index of first write call
    first_write = next(
        (i for i, t in enumerate(tool_calls) if t["tool"] in WRITE_TOOLS), None
    )

    orientation = [
        t for i, t in enumerate(tool_calls)
        if t["tool"] in EXPLORE_TOOLS
        and (first_write is None or i < first_write)
    ]
    all_reads  = [t for t in tool_calls if t["tool"] in EXPLORE_TOOLS]
    all_writes = [t for t in tool_calls if t["tool"] in WRITE_TOOLS]
    mcp_calls  = [t for t in tool_calls if t["tool"] in MCP_TOOLS]

    by_tool: dict[str, int] = defaultdict(int)
    for t in tool_calls:
        by_tool[t["tool"]] += 1

    # Token summary across all turns
    total_input  = sum(t["input"]   for t in token_turns)
    total_cr     = sum(t["cache_r"] for t in token_turns)
    total_cc     = sum(t["cache_c"] for t in token_turns)
    total_output = sum(t["output"]  for t in token_turns)

    return {
        "label":            label,
        "total_tool_calls": len(tool_calls),
        "orientation_reads":orientation,
        "all_reads":        len(all_reads),
        "all_writes":       len(all_writes),
        "mcp_calls":        len(mcp_calls),
        "by_tool":          dict(by_tool),
        "first_write":      first_write,
        "tokens": {
            "input":    total_input,
            "cache_r":  total_cr,
            "cache_c":  total_cc,
            "output":   total_output,
        },
        "turns": len(token_turns),
    }


def print_report(r: dict) -> None:
    if r.get("missing"):
        print("  (file not found)")
        return
    print(f"  Turns            : {r['turns']}")
    print(f"  Total tool calls : {r['total_tool_calls']}")
    print(f"  Read/Glob/Grep   : {r['all_reads']}")
    print(f"  Write/Edit       : {r['all_writes']}")
    print(f"  MCP calls        : {r['mcp_calls']}")
    print(f"  First write at   : tool call #{r['first_write']}")
    orient = r["orientation_reads"]
    print(f"  Orientation reads (before 1st write): {len(orient)}")
    for t in orient:
        print(f"    {t['tool']:6s}  {t['summary']}")
    tok = r["tokens"]
    print(f"  Tokens  input={tok['input']}  cache_read={tok['cache_r']}  "
          f"cache_create={tok['cache_c']}  output={tok['output']}")
    if r["by_tool"]:
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(r["by_tool"].items()))
        print(f"  By tool: {breakdown}")


BASE = r"C:\Users\Admin\.claude\projects"

sessions = [
    # sandbox_baseline — cold, no CogniKernel
    (f"{BASE}\\C--Users-Admin-OneDrive-Desktop-sandbox-baseline\\314bd92a-e9e0-4795-8d05-6c77a4447c1a.jsonl",
     "BASELINE session-A  (cold, no CogniKernel)"),
    (f"{BASE}\\C--Users-Admin-OneDrive-Desktop-sandbox-baseline\\f4a23c33-fb2c-4c1d-a83b-dea46e4cf608.jsonl",
     "BASELINE session-B  (cold, no CogniKernel)"),
    (f"{BASE}\\C--Users-Admin-OneDrive-Desktop-sandbox-baseline\\fb628b95-a9d4-46ce-8a50-c0b9f1cd622b.jsonl",
     "BASELINE session-C  (cold, no CogniKernel)"),

    # sandbox_cognikernel — v1 events (no symbol skeleton yet)
    (f"{BASE}\\C--Users-Admin-OneDrive-Desktop-sandbox-cognikernel\\35019833-ec73-454a-875d-b78f2ab2dce3.jsonl",
     "COGNIKERNEL session-1  (v1, no skeleton)"),
    (f"{BASE}\\C--Users-Admin-OneDrive-Desktop-sandbox-cognikernel\\c2ddb957-ea03-4609-8def-01bc3a34c0c5.jsonl",
     "COGNIKERNEL session-2  (v1, no skeleton)"),
]

results = []
for path, label in sessions:
    print(f"\n[{label}]")
    r = analyse_session(path, label)
    print_report(r)
    results.append(r)

# Summary table
print("\n")
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Session':<45} {'Reads':>5}  {'Writes':>6}  {'Orient':>6}  {'MCP':>4}  {'Input tok':>10}")
print("-" * 80)
for r in results:
    if r.get("missing"):
        continue
    tok = r["tokens"]
    print(f"{r['label']:<45} {r['all_reads']:>5}  {r['all_writes']:>6}  "
          f"{len(r['orientation_reads']):>6}  {r['mcp_calls']:>4}  {tok['input']:>10,}")
