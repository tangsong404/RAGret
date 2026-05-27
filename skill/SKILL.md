---
name: ragret
description: >-
  Use RAGret for any semantic search, retrieval, or knowledge-base query — especially
  when the user asks to search "my data", "the docs", "the wiki", "knowledge base",
  "previous reports", "internal notes", or anything that sounds like it lives in a
  private or internal knowledge store (not the open web). Also triggers when the user
  explicitly says "RAGret". This is the skill to use for RETRIEVAL-AUGMENTED tasks
  where you need to find information from a known internal source. Do NOT use for
  web search (use WebSearch) or local file search (use Grep/Glob).
---

# RAGret

RAGret is an open-source, self-hosted semantic retrieval service. It indexes documents and provides a JSON API for searching them. Think of it as your private search engine for internal knowledge bases.

For more info: [github.com/SugarSong404/RAGret](https://github.com/SugarSong404/RAGret.git).

★ Insight ──────────────────────────────────────────
RAGret fits between web search and local file search:
- **Web search** → public, up-to-date, any topic
- **RAGret** → private/internal knowledge, semantically indexed, with provenance
- **Local grep** → raw text search in files you can see
────────────────────────────────────────────────────

## Quick start

Once the user confirms they have a RAGret instance:

1. **Verify environment** — check `$env:RAGRET_API_KEY` and `$env:BASE_URL` (or ask user to set them)
2. **List indexes** — see what knowledge bases are available
3. **Search** — query the right index with a natural-language question
4. **Cite** — always run the two-hop citation pipeline to give answers with source provenance

## Setup

### Configuration

Two environment variables are required. Ask the user to set these in their terminal **before** you start (never ask for raw secrets in chat):

| Variable | Purpose | Example |
|---|---|---|
| `RAGRET_API_KEY` | API authentication | `sk-...` |
| `BASE_URL` | RAGret server address | `http://127.0.0.1:8765` or `https://ragret.example.com` |

### Verify connection

```powershell
# Check variables are set
if (-not $env:RAGRET_API_KEY) { "Missing RAGRET_API_KEY" }
if (-not $env:BASE_URL) { "Missing BASE_URL" }
```

If `BASE_URL` is not provided, default to `http://127.0.0.1:8765`.

## Usage

### 1. List available indexes

Shows all knowledge bases your API key can access:

```powershell
curl.exe -sS -H "X-API-Key: $env:RAGRET_API_KEY" "$env:BASE_URL/api/subscribe-indexes"
```

### 2. Search an index

```powershell
curl.exe -sS -G "$env:BASE_URL/api/search/INDEX_NAME" `
  -H "X-API-Key: $env:RAGRET_API_KEY" `
  --data-urlencode "query=your natural language question"`
  --data-urlencode "k=5"
```

Optional parameters:

| Parameter | Default | Description |
|---|---|---|
| `k` | 10 | Number of results to recall |
| `score_threshold` | 0.3 | Vector similarity threshold |
| `rerank_top_n` | 5 | Number of results after reranking |
| `format` | json | `json` or `text` |

#### JSON response format

Each entry in the `result` array contains:

```json
{
  "content": "…chunk content…",
  "source": "docs/report.pdf",
  "chunk_index": 2,
  "vector_score": 0.82,
  "relevance_score": 0.91,
  "rrf_score": 0.015,
  "dense_rank": 3,
  "bm25_rank": 1,
  "parent_url": "/api/kb/mykb/parents/docs/report.pdf.txt",
  "line_start": 240,
  "line_end": 258
}
```

New fields for citation:

| Field | Meaning |
|---|---|
| `parent_url` | URL to the parent document (UTF-8 .txt) — derive as `/api/kb/{kb}/parents/{source}.txt` |
| `line_start` | Start line in parent document (1-based) |
| `line_end` | End line range in parent document (inclusive) |

`parent_url` is derived from `source`: `/api/kb/{kb}/parents/{source}.txt` (path is URL-encoded). When `RAGRET_PUBLIC_HOST` is configured, `parent_url` is a full URL; otherwise it's a relative path — prepend `BASE_URL` to construct the full URL: `${env:BASE_URL}${parent_url}`.

**Note:** Older indexes may not have `line_start` / `line_end` — those fields will be `null`. `parent_url` may still be available as long as `source` is present.

#### Text response format (`format=text`)

```text
passage 1:
Some chunk content here...

    source: docs/report.pdf
    parent_url: https://.../api/kb/mykb/parents/docs/report.pdf.txt
    lines: 240-258
---
passage 2:
...
```

The text format also includes `source:`, `parent_url:`, and `lines:` metadata after each passage.

### 3. Two-hop citation pipeline (MANDATORY)

Every search **must** follow this fixed two-hop pipeline. This is not optional — do it on every single search call.

**Why this matters:** Chunks alone lack context. The two-hop pipeline lets you retrieve the surrounding document context around each matching chunk, producing more accurate, citable answers.

#### Step-by-step

**Hop 1 — Search:**
1. Call Search API to get `content` + `parent_url` + `line_start` / `line_end`
2. Extract `parent_url` from each result (if null, skip this result's second hop — the chunk is all you have)

**Hop 2 — Download + context (for each result with a valid `parent_url`):**
1. Download the parent document to a local temporary directory:
   ```powershell
   # Construct full URL
   $parentUrlFull = "${env:BASE_URL}$parent_url"
   
   # Download to local temp
   $tmpDir = "$env:TEMP\ragret-parents"
   New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
   
   $tmpFile = Join-Path $tmpDir "parent_$([System.IO.Path]::GetRandomFileName())"
   curl.exe -sS -H "X-API-Key: $env:RAGRET_API_KEY" $parentUrlFull -o $tmpFile
   ```
2. Read the relevant line range using Get-Content or Grep:
   ```powershell
   # Read specific lines  (if line_start/line_end are available and not null)
   $context = Get-Content $tmpFile | Select-Object -Skip ($line_start - 1) -First ($line_end - $line_start + 1)
   ```
3. Use the expanded context along with the original chunk to construct your answer
4. Optionally keep the temp file for reference; you may clean up with `Remove-Item $tmpDir -Recurse -Force` if desired

**Present results with citations** — include `source` (file name) and line numbers so the user knows exactly where the information came from. Example citation format:

> The refund window is 30 days from purchase (source: `docs/report.pdf`, lines 240-258).

## Parent document endpoint

`GET /api/kb/{name}/parents/{path}` — returns `text/plain`

- Used by the two-hop pipeline to fetch the full parent document
- Path is URL-encoded (e.g., `docs/report.pdf` → `docs/report.pdf.txt`; the `.txt` extension is always appended)
- Authentication: same as Search API (X-API-Key, Bearer, or cookie)
- This endpoint is **not** meant to be called directly by users — the skill uses it as part of the two-hop pipeline

## Error handling

| Symptom | Likely cause | What to do |
|---|---|---|
| `curl: (6) Could not resolve host` | Wrong `BASE_URL` | Ask the user to verify the URL |
| HTTP 401/403 | Missing/invalid `RAGRET_API_KEY` | Ask the user to set the key in their env (not in chat) |
| HTTP 404 on search | Wrong index name | List indexes first to find the correct name |
| HTTP 404 on parent_url | KB name mismatch in parent URL | Verify the index name matches the KB name; the parent_url is derived automatically |
| Empty result set | No matching documents | Try rephrasing the query |
| Connection refused | RAGret not running | Ask the user to start their RAGret instance |

## Full example

This example shows the complete two-hop workflow.

```powershell
# ===== Hop 1: Search =====
curl.exe -sS -G "${env:BASE_URL}/api/search/product_docs" `
  -H "X-API-Key: ${env:RAGRET_API_KEY}" `
  --data-urlencode "query=How do we handle refunds within 30 days?" `
  --data-urlencode "k=3"

# Response includes parent_url, line_start, line_end for each result.
# Suppose result 0 has:
#   parent_url: /api/kb/product_docs/parents/docs/report.pdf.txt
#   line_start: 240, line_end: 258

# ===== Hop 2: Download parent document context =====
$tmpDir = "$env:TEMP\ragret-parents"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$tmpFile = Join-Path $tmpDir "report_section.txt"
curl.exe -sS -H "X-API-Key: ${env:RAGRET_API_KEY}" `
  "${env:BASE_URL}/api/kb/product_docs/parents/docs/report.pdf.txt" `
  -o $tmpFile

# Read the line range
Get-Content $tmpFile | Select-Object -Skip 239 -First 19

# ===== Answer with citation =====
# "The refund policy states a 30-day window from purchase date.
#  Source: docs/report.pdf, lines 240-258."
```

## Scripts

For a streamlined experience, use the bundled helper script:

- `scripts/ragret.ps1` — PowerShell wrapper for listing indexes, searching, and fetching parent document context

Commands:
```
.\ragret.ps1 list                                    # List indexes
.\ragret.ps1 search <index> -Query "..."             # Search
.\ragret.ps1 search <index> -Query "..." -Expand     # Search + two-hop citation
.\ragret.ps1 parent -ParentUrl <url> -Lines "240-258" # Fetch parent doc context
```

Read the script's header for usage: `Get-Content "$PSScriptRoot/scripts/ragret.ps1"`

## Rules

- **Never** ask for raw API keys in chat or use them as plain-text arguments
- Always verify environment variables exist before making API calls
- Default `BASE_URL` to `http://127.0.0.1:8765` if the user doesn't provide one
- **Always** run the two-hop citation pipeline after every search — it is not optional
- When `line_start` / `line_end` are null (old indexes), skip the second hop for that result
