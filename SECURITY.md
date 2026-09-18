# Security Policy

## Supported versions

Only the latest released version receives security fixes.

| Version | Supported |
| --- | --- |
| latest | :white_check_mark: |
| older | :x: |

## Reporting a vulnerability

- Do not open a public issue or PR for a security bug.
- Open a private GitHub Security Advisory ("Report a vulnerability" under the
  repo's Security tab), or email the maintainer directly.
- Include the failing input, the affected version, and (if you have one) a
  reproducing test case — same as any other bug, but private.

## What's in scope

The harness itself: `cascade`, `reliability.assertions`, `metrics`,
`routing`, `fingerprint`, and the `memory` package (vault, warm, cold,
mcp_server). Vulnerabilities in upstream dependencies (chromadb, mcp,
libsql-experimental, boto3, streamlit) should go to those projects, though
flagging them here too is welcome.

## Known hardening and its limits

**SSRF guard on evidence URLs** (`memory.vault._url_is_safe` /
`_verify_evidence_url`). `save_memory`'s evidence-URL verification makes an
outbound HTTP request on the caller's behalf — an unguarded version of that
is a classic SSRF sink, since the URL can come from any MCP client. The guard:

- Rejects non-HTTP(S) schemes and any hostname resolving to a private,
  loopback, link-local, multicast, reserved, or unspecified address.
- Re-validates every redirect hop before following it (redirects are walked
  manually, not via `requests`' `allow_redirects=True`) — a URL that passes
  the initial check cannot use a redirect to reach an internal host that
  wasn't itself checked.

**Known residual gap:** the safety check and the actual request each do
their own DNS resolution, moments apart. A DNS-rebinding attacker controlling
the resolved hostname could in principle answer the safety check and the
real request differently. This isn't closed (it would require pinning the
validated IP and connecting to it directly rather than by hostname). Judged
acceptable for the current threat model — evidence URLs come from your own
local or MCP-connected clients, not arbitrary internet input — but is a
real limitation, not an oversight, and would need addressing before this
guard is trusted in a context where evidence URLs can come from an untrusted
party over the network.

**FTS5 query sanitization** (`memory.vault._sanitize_fts_query`). Free-text
search queries are wrapped as quoted FTS5 terms so operator characters
(`"`, `-`, `NEAR`, `*`, …) can never raise `sqlite3.OperationalError` or be
used to construct an unintended query.

**MCP server trust model.** `memory.mcp_server` runs over stdio with no
authentication layer of its own — appropriate for its intended use (a local
process spawned by a trusted MCP client on the same machine), not for
exposing the vault over a network transport. If you wire this server up to
an HTTP/SSE transport, use MCP's own authorization extensions (issuer
validation, Client ID Metadata Documents, or Enterprise-Managed
Authorization) rather than assuming stdio's implicit local trust still
holds.

## Security-relevant defaults

- Local SQLite databases (metrics, memory vault) are unencrypted files on
  disk, readable by any local process running as the same user. This
  project makes no confidentiality claim beyond normal filesystem
  permissions.
- The `[cold]` extra's example configuration (MinIO defaults, etc.) in the
  README is for local development only — never reuse `minioadmin`/
  `minioadmin`-style credentials against a real deployment.
