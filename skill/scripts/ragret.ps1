<#
.SYNOPSIS
  RAGret API helper — list indexes, search knowledge bases, and fetch parent document context.

.DESCRIPTION
  A thin wrapper around the RAGret HTTP API. Requires two environment variables:
    RAGRET_API_KEY  - API key for authentication
    BASE_URL        - RAGret server URL (default: http://127.0.0.1:8765)

.PARAMETER Command
  "list" to list indexes, "search" to search an index, or "parent" to fetch a parent document.

.PARAMETER Index
  The index name to search (required for "search" command).

.PARAMETER Query
  The search query text (required for "search" command).

.PARAMETER K
  Number of results to recall (search only; default: 10).

.PARAMETER Format
  Output format: "json" (default) or "text".

.PARAMETER Expand
  When set, automatically runs the two-hop citation pipeline after search:
  downloads each result's parent document and reads the relevant line range.

.PARAMETER ParentUrl
  The parent document URL path (for "parent" command).

.PARAMETER Lines
  Line range to display, e.g. "240-258" (for "parent" command).

.EXAMPLE
  .\ragret.ps1 list
  .\ragret.ps1 search product_docs -Query "refund policy" -Format text -K 3
  .\ragret.ps1 search product_docs -Query "refund policy" -Expand
  .\ragret.ps1 parent -ParentUrl "/api/kb/product_docs/parents/docs/report.pdf.txt" -Lines "240-258"
#>

param(
  [Parameter(Position = 0, Mandatory)]
  [ValidateSet("list", "search", "parent")]
  [string]$Command,

  [Parameter(Position = 1)]
  [string]$Index,

  [Parameter()]
  [string]$Query,

  [Parameter()]
  [int]$K = 10,

  [Parameter()]
  [ValidateSet("json", "text")]
  [string]$Format = "json",

  [Parameter()]
  [switch]$Expand,

  [Parameter()]
  [string]$ParentUrl,

  [Parameter()]
  [string]$Lines
)

# --- Configuration ---
$ErrorActionPreference = "Stop"

$baseUrl = if ($env:BASE_URL) { $env:BASE_URL.TrimEnd('/') } else { "http://127.0.0.1:8765" }
$apiKey = $env:RAGRET_API_KEY

if (-not $apiKey) {
  Write-Error "RAGRET_API_KEY is not set. Set it in your environment: `$env:RAGRET_API_KEY = 'sk-...'"
  exit 1
}

$headers = @{ "X-API-Key" = $apiKey }

# --- Helper: fetch and display parent document context ---
function Expand-ParentContext {
  param([string]$ParentUrl, [int]$LineStart, [int]$LineEnd)

  $tmpDir = "$env:TEMP\ragret-parents"
  New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

  # Derive a safe filename from the parent URL
  $safeName = [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($ParentUrl)).TrimEnd('=').Replace('/', '_')
  $tmpFile = Join-Path $tmpDir "$safeName.txt"

  $fullUrl = "${baseUrl}${ParentUrl}"
  Write-Host "`nFetching parent document: $fullUrl" -ForegroundColor Cyan

  curl.exe -sS -H "X-API-Key: $apiKey" $fullUrl -o $tmpFile
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  if ($LineStart -and $LineEnd) {
    $lineCount = $LineEnd - $LineStart + 1
    Write-Host "Context (lines $LineStart-$LineEnd):`n" -ForegroundColor Yellow
    Get-Content $tmpFile | Select-Object -Skip ($LineStart - 1) -First $lineCount
  } else {
    Write-Host "Full document saved to: $tmpFile" -ForegroundColor Green
  }

  return $tmpFile
}

# --- Commands ---
switch ($Command) {
  "list" {
    $url = "$baseUrl/api/subscribe-indexes"
    Write-Host "Fetching indexes from $url ..." -ForegroundColor Cyan
    $result = curl.exe -sS -H "X-API-Key: $apiKey" $url
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $result | ConvertFrom-Json | ConvertTo-Json -Depth 3
  }

  "search" {
    if (-not $Index) { Write-Error "Index name is required for 'search' command."; exit 1 }
    if (-not $Query) { Write-Error "Query text is required for 'search' command."; exit 1 }

    $url = "$baseUrl/api/search/$Index"
    Write-Host "Searching index '$Index'..." -ForegroundColor Cyan

    $params = @("--data-urlencode", "query=$Query", "--data-urlencode", "k=$K")
    if ($Format -eq "text") { $params += "--data-urlencode", "format=text" }

    $result = curl.exe -sS -G $url -H "X-API-Key: $apiKey" @params
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($Format -eq "text") {
      $result
    } else {
      $parsed = $result | ConvertFrom-Json
      $parsed | ConvertTo-Json -Depth 3
    }

    # ---- Two-hop citation pipeline (when -Expand is set) ----
    if ($Expand -and $Format -eq "json") {
      $parsed = if ($Format -eq "text") { $null } else { $result | ConvertFrom-Json }

      if ($parsed -and $parsed.result) {
        for ($i = 0; $i -lt $parsed.result.Count; $i++) {
          $item = $parsed.result[$i]
          $pu = $item.parent_url
          $ls = $item.line_start
          $le = $item.line_end

          if ([string]::IsNullOrEmpty($pu)) {
            Write-Host "`n[Skipping hop 2 for result $i — no parent_url]" -ForegroundColor DarkGray
            continue
          }

          # Parse line numbers (null-safe)
          $lsNum = if ($null -ne $ls) { [int]$ls } else { $null }
          $leNum = if ($null -ne $le) { [int]$le } else { $null }

          if ($lsNum -and $leNum) {
            Expand-ParentContext -ParentUrl $pu -LineStart $lsNum -LineEnd $leNum
          } else {
            Expand-ParentContext -ParentUrl $pu
          }
        }
      }
    }
  }

  "parent" {
    if (-not $ParentUrl) { Write-Error "ParentUrl is required for 'parent' command."; exit 1 }

    if ($Lines -match '^(\d+)-(\d+)$') {
      Expand-ParentContext -ParentUrl $ParentUrl -LineStart [int]$matches[1] -LineEnd [int]$matches[2]
    } else {
      Expand-ParentContext -ParentUrl $ParentUrl
    }
  }
}
