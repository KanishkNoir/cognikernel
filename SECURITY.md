# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** via
[GitHub Security Advisories](https://github.com/KanishkNoir/cognikernel/security/advisories/new)
— do not open a public issue for security reports.

You can expect an acknowledgement within a few days. Please include a minimal
reproduction and the version/commit you tested against.

## Scope notes

CogniKernel runs locally and stores project memory in SQLite under
`~/.cognikernel` (or `COGNIKERNEL_DIR`). Reports of particular interest:

- Anything that lets one project's hooks or MCP tools read or write another
  project's memory store without configuration to do so.
- Injection of attacker-controlled content into the session context block that
  survives sanitization (`src/cognikernel/extraction/sanitize.py`).
- Escapes from the fail-open contract: a hook failure that can corrupt state
  rather than degrade legibly.

There is no bug bounty; credit is given in release notes unless you prefer
otherwise.
