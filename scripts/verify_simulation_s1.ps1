# verify_simulation_s1.ps1 — Verify Session 1 extraction is complete before starting Session 2
#
# Run this after CK arm Session 1 closes and the Stop hook fires.
# Checks that all N1-N6 decisions were extracted and correctly typed.
#
# Usage:
#   powershell -File scripts/verify_simulation_s1.ps1

$CK_PATH = "C:\Users\Admin\OneDrive\Desktop\notesapi_ck"
$MEMLORA = "memlora"
$PYTHON  = "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"

Write-Host "=== Session 1 Extraction Verification ===" -ForegroundColor Cyan
Write-Host "Project: $CK_PATH"
Write-Host ""

# Show full injection state
Write-Host "--- memlora show output ---" -ForegroundColor Yellow
& $MEMLORA show $CK_PATH
Write-Host ""

# Check for specific decisions in DB
$PYTHON_CHECK = @"
import sys, sqlite3
sys.path.insert(0, 'src')
from memlora.config import Config
from memlora.storage.connection import get_db_path, hash_project_path

cfg = Config.load()
pid = hash_project_path('$CK_PATH')
db_path = get_db_path(cfg, pid)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT event_type, payload FROM events WHERE archived=0 ORDER BY id"
).fetchall()

import json

checks = {
    'N1_sqlite':       ('CONSTRAINT_HARD', ['sqlite', 'not postgres', 'local']),
    'N2_int_ids':      ('DECISION',        ['integer', 'int id', 'uuid']),
    'N3_soft_delete':  ('CONSTRAINT_HARD', ['soft delete', 'is_deleted', 'no delete', 'logical']),
    'N4_tags_string':  ('DECISION',        ['tags', 'comma', 'column', 'no table']),
    'N5_no_auth':      ('CONSTRAINT_HARD', ['auth', 'authentication', 'local']),
    'N6_no_celery':    ('APPROACH_ABANDONED', ['celery', 'broker', 'redis']),
}

found = {k: False for k in checks}
for row in rows:
    payload_str = row['payload'].lower()
    event_type  = row['event_type']
    for key, (etype, keywords) in checks.items():
        if found[key]:
            continue
        if etype in event_type and any(kw in payload_str for kw in keywords):
            found[key] = True

print()
print('Decision extraction check:')
all_pass = True
for key, ok in found.items():
    status = 'PASS' if ok else 'FAIL'
    color  = '' if ok else '!'
    print(f'  {color}{status} {key}')
    if not ok:
        all_pass = False

print()
total = conn.execute('SELECT COUNT(*) FROM events WHERE archived=0').fetchone()[0]
print(f'Total active events: {total}')

failures = conn.execute('SELECT COUNT(*) FROM extraction_failures').fetchone()[0]
print(f'Extraction failures: {failures}')
if failures > 0:
    print('  Run: memlora failures $CK_PATH')

print()
if all_pass:
    print('All checks passed — safe to start Session 2.')
else:
    print('Some checks FAILED — investigate before starting Session 2.')
    print('Run: memlora doctor $CK_PATH')
    sys.exit(1)
"@

& $PYTHON -c $PYTHON_CHECK
