import json

path = r"C:\Users\Admin\.claude\projects\K--CogniKernel-MetaTest-META-A\c184855e-90bb-4e16-92d6-86b62cfca225.jsonl"
lines = open(path, encoding="utf-8").readlines()

seen_block_types = set()
for line in lines:
    obj = json.loads(line)
    t = obj.get("type")
    if t in ("assistant", "user"):
        msg = obj.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    seen_block_types.add(block.get("type", "?"))

print("Content block types found:", seen_block_types)
print()

# Show one example of each block type
shown = set()
for line in lines:
    obj = json.loads(line)
    t = obj.get("type")
    if t in ("assistant", "user"):
        msg = obj.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    bt = block.get("type", "?")
                    if bt not in shown:
                        shown.add(bt)
                        print(f"=== block type: {bt} ===")
                        print(json.dumps(block, indent=2)[:500])
                        print()
