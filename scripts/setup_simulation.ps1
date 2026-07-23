# setup_simulation.ps1 — Create and verify clean-slate simulation sandboxes
#
# Run this BEFORE starting either simulation arm.
# Creates two fresh directories, verifies no Claude auto-memory or cognikernel DB
# exists for these paths, and runs cognikernel init for the CK arm.
#
# Usage:
#   powershell -File scripts/setup_simulation.ps1

$ErrorActionPreference = "Stop"

$CK_PATH      = "C:\Users\Admin\OneDrive\Desktop\notesapi_ck"
$VANILLA_PATH = "C:\Users\Admin\OneDrive\Desktop\notesapi_vanilla"
$PYTHON       = "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
$COGNIKERNEL      = "cognikernel"

function Get-ClaudeEncodedPath($projectPath) {
    $resolved = [System.IO.Path]::GetFullPath($projectPath)
    return $resolved -replace "[:\\\/]", "-"
}

function Check-CleanSlate($projectPath, $label) {
    $encoded = Get-ClaudeEncodedPath $projectPath
    $claudeDir = Join-Path $env:USERPROFILE ".claude\projects\$encoded"
    $memoryFile = Join-Path $claudeDir "memory\MEMORY.md"
    $sessionsDir = Join-Path $claudeDir "sessions"

    Write-Host ""
    Write-Host "[$label] Checking clean slate for: $projectPath"
    Write-Host "  Encoded path: $encoded"

    $issues = @()

    if (Test-Path $claudeDir) {
        $issues += "Claude project dir already exists: $claudeDir"
    }
    if (Test-Path $memoryFile) {
        $issues += "Auto-memory file exists (BIAS RISK): $memoryFile"
    }
    if (Test-Path $sessionsDir) {
        $sessionCount = (Get-ChildItem $sessionsDir -Filter "*.jsonl" -ErrorAction SilentlyContinue).Count
        if ($sessionCount -gt 0) {
            $issues += "Prior session transcripts found ($sessionCount files) — BIAS RISK"
        }
    }

    if ($issues.Count -eq 0) {
        Write-Host "  Clean slate: OK" -ForegroundColor Green
    } else {
        Write-Host "  BIAS RISKS DETECTED:" -ForegroundColor Red
        foreach ($issue in $issues) {
            Write-Host "    - $issue" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "  To clear: Remove-Item '$claudeDir' -Recurse -Force" -ForegroundColor Yellow
        return $false
    }
    return $true
}

# ── 1. Verify sandboxes don't already contain prior state ─────────────────────

Write-Host "=== Simulation Clean-Slate Verification ===" -ForegroundColor Cyan

$ck_clean      = Check-CleanSlate $CK_PATH "CK arm"
$vanilla_clean = Check-CleanSlate $VANILLA_PATH "Vanilla arm"

if (-not $ck_clean -or -not $vanilla_clean) {
    Write-Host ""
    Write-Host "ERROR: One or more arms have existing state. Clear them before proceeding." -ForegroundColor Red
    Write-Host "       See above for the Remove-Item commands." -ForegroundColor Red
    exit 1
}

# ── 2. Create directories ─────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== Creating Sandbox Directories ===" -ForegroundColor Cyan

foreach ($path in @($CK_PATH, $VANILLA_PATH)) {
    if (Test-Path $path) {
        $contents = Get-ChildItem $path
        if ($contents.Count -gt 0) {
            Write-Host "ERROR: $path already exists and is non-empty. Remove it first." -ForegroundColor Red
            exit 1
        }
        Write-Host "  $path — already exists (empty, OK)"
    } else {
        New-Item -ItemType Directory -Path $path | Out-Null
        Write-Host "  Created: $path"
    }
}

# Initialize git for both (cognikernel needs a recognizable project root)
foreach ($path in @($CK_PATH, $VANILLA_PATH)) {
    if (-not (Test-Path (Join-Path $path ".git"))) {
        git -C $path init --quiet
        Write-Host "  git init: $path"
    }
}

# ── 3. Initialize cognikernel for CK arm ─────────────────────────────────────────

Write-Host ""
Write-Host "=== Initializing CogniKernel (CK arm) ===" -ForegroundColor Cyan

try {
    & $COGNIKERNEL init $CK_PATH
    Write-Host "  cognikernel init: OK" -ForegroundColor Green
} catch {
    Write-Host "  cognikernel init FAILED: $_" -ForegroundColor Red
    exit 1
}

# Verify DB was created and is empty
$db_path = & $PYTHON -c "
import sys
sys.path.insert(0, 'src')
from cognikernel.config import Config
from cognikernel.storage.connection import get_db_path, hash_project_path
cfg = Config.load()
pid = hash_project_path('$CK_PATH')
print(get_db_path(cfg, pid))
" 2>$null

if ($db_path -and (Test-Path $db_path)) {
    $event_count = & $PYTHON -c "
import sqlite3, sys
conn = sqlite3.connect('$db_path')
count = conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]
print(count)
" 2>$null
    Write-Host "  DB path: $db_path"
    Write-Host "  Events in DB: $event_count (should be 0)"
    if ($event_count -ne "0") {
        Write-Host "  WARNING: DB is not empty!" -ForegroundColor Yellow
    }
} else {
    Write-Host "  (Could not verify DB path — check manually with: cognikernel doctor '$CK_PATH')" -ForegroundColor Yellow
}

# ── 4. Verify doctor passes for CK arm ───────────────────────────────────────

Write-Host ""
Write-Host "=== Running cognikernel doctor (CK arm) ===" -ForegroundColor Cyan
try {
    & $COGNIKERNEL doctor $CK_PATH
} catch {
    Write-Host "  doctor FAILED: $_" -ForegroundColor Red
    exit 1
}

# ── 5. Summary ────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "CK arm:      $CK_PATH"
Write-Host "Vanilla arm: $VANILLA_PATH"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Confirm hooks are registered for CK arm in Claude Code settings"
Write-Host "     (Stop hook, PreToolUse Read hook, PostToolUse Write/Edit hooks)"
Write-Host "  2. Open a Claude Code session in: $CK_PATH"
Write-Host "  3. Run Session 1 prompts from: research\benchmarking\simulation.md"
Write-Host "  4. After Session 1 closes: cognikernel show '$CK_PATH'"
Write-Host "     Verify N1-N6 decisions are present before starting Session 2"
Write-Host ""
Write-Host "IMPORTANT: Run CK arm FIRST (no evaluator hindsight from Vanilla arm)"
