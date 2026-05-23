"""Find what get_session_state returned in Session 2."""
import json
from pathlib import Path

sess2 = r"C:\Users\Admin\.claude\projects\K--CogniKernel-MetaTest-META-C\73470e59-29f5-4f75-b0b0-1755f6eda5a9.jsonl"

lines = Path(sess2).read_text(encoding="utf-8").splitlines()
objects = [json.loads(l) for l in lines if l.strip()]

# Find the get_session_state tool_use and its result
found_tool = False
for i, obj in enumerate(objects):
    t = obj.get("type")
    msg = obj.get("message", {})
    content = msg.get("content", [])
    if not isinstance(content, list):
        content = []

    for block in content:
        if not isinstance(block, dict):
            continue
        # MCP tool call
        if block.get("type") == "tool_use" and "session_state" in block.get("name",""):
            print(f"=== get_session_state call (line {i}) ===")
            print(json.dumps(block.get("input", {}), indent=2))
            found_tool = True
        # Tool result (what the MCP returned)
        if found_tool and block.get("type") == "tool_result":
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                for rb in result_content:
                    if isinstance(rb, dict) and rb.get("type") == "text":
                        print(f"\n=== MCP response ===")
                        print(rb.get("text", "")[:2000])
            else:
                print(f"\n=== MCP response ===")
                print(str(result_content)[:2000])
            found_tool = False

# Also check first assistant text after session start
print("\n=== First assistant text in session 2 ===")
for obj in objects[:10]:
    msg = obj.get("message", {})
    content = msg.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "").strip()
                if t:
                    print(t[:1000])
                    break
