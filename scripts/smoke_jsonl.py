import sys
sys.path.insert(0, r"C:\Users\Admin\OneDrive\Desktop\CogniKernel\src")

from memlora.extraction.jsonl_converter import jsonl_to_transcript

path = r"C:\Users\Admin\.claude\projects\K--CogniKernel-MetaTest-META-A\c184855e-90bb-4e16-92d6-86b62cfca225.jsonl"
raw = open(path, encoding="utf-8").read()
result = jsonl_to_transcript(raw)

lines = result.splitlines()
print(f"Converted: {len(raw)} bytes JSONL -> {len(result.encode())} bytes text")
print(f"Lines: {len(lines)}")
print(f"Sections (## headers): {sum(1 for l in lines if l.startswith('## '))}")
print()
print("--- first 60 lines ---")
print("\n".join(lines[:60]))
