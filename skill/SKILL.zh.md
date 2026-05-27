---
name: ragret
description: >-
  用于任何语义搜索、检索或知识库查询场景 —— 尤其是当用户需要搜索"我的数据"、"文档"、"Wiki"、"知识库"、
  "之前的报告"、"内部笔记"，或任何听起来像是存储在私有/内部知识库（非开放网络）的内容时触发。
  当用户明确提到"RAGret"时同样触发。本技能用于检索增强型任务，需要从已知的内部来源查找信息。
  不要用于网络搜索（使用 WebSearch）或本地文件搜索（使用 Grep/Glob）。
---

# RAGret

RAGret 是一个开源、自托管的语义检索服务。它负责索引文档，并通过 JSON API 提供搜索能力。可以把它理解为你私有的内部知识库搜索引擎。

更多信息：[github.com/SugarSong404/RAGret](https://github.com/SugarSong404/RAGret.git)。

★ 小贴士 ──────────────────────────────────────────
RAGret 处于网络搜索和本地文件搜索之间：
- **网络搜索** → 公开的、最新的、任意主题
- **RAGret** → 私有/内部知识库，语义索引，带有来源追溯
- **本地 grep** → 对可见文件的原始文本搜索
────────────────────────────────────────────────────

## 快速开始

用户确认他们有 RAGret 实例后：

1. **检查环境** — 确认 `$env:RAGRET_API_KEY` 和 `$env:BASE_URL`（或让用户设置）
2. **列出索引** — 查看可用的知识库
3. **搜索** — 用自然语言问题查询正确的索引
4. **引用溯源** — 每次搜索后必须执行两跳引用管线，确保答案可追溯

## 设置

### 配置

需要两个环境变量。请让用户在终端中**先设置好**再开始（绝不要在聊天中索要明文密钥）：

| 变量 | 用途 | 示例 |
|---|---|---|
| `RAGRET_API_KEY` | API 认证 | `sk-...` |
| `BASE_URL` | RAGret 服务器地址 | `http://127.0.0.1:8765` 或 `https://ragret.example.com` |

### 验证连接

```powershell
# 检查变量是否已设置
if (-not $env:RAGRET_API_KEY) { "缺少 RAGRET_API_KEY" }
if (-not $env:BASE_URL) { "缺少 BASE_URL" }
```

如果用户未提供 `BASE_URL`，默认使用 `http://127.0.0.1:8765`。

## 使用

### 1. 列出可用索引

查看当前 API Key 可以访问的所有知识库：

```powershell
curl.exe -sS -H "X-API-Key: $env:RAGRET_API_KEY" "$env:BASE_URL/api/subscribe-indexes"
```

### 2. 搜索索引

```powershell
curl.exe -sS -G "$env:BASE_URL/api/search/索引名称" `
  -H "X-API-Key: $env:RAGRET_API_KEY" `
  --data-urlencode "query=你的自然语言问题"`
  --data-urlencode "k=5"
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `k` | 10 | 召回数量 |
| `score_threshold` | 0.3 | 向量相似度阈值 |
| `rerank_top_n` | 5 | 重排后返回条数 |
| `format` | json | `json` 或 `text` |

#### JSON 响应格式

`result` 数组中每个条目包含以下字段：

```json
{
  "content": "…chunk 正文…",
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

新增引用字段：

| 字段 | 含义 |
|---|---|
| `parent_url` | 父文档的 GET 地址（UTF-8 .txt）— 推导规则：`/api/kb/{kb}/parents/{source}.txt` |
| `line_start` | 该 chunk 在父文档中的起始行（1-based） |
| `line_end` | 该 chunk 在父文档中的结束行（含） |

`parent_url` 由 `source` 推导：`/api/kb/{kb}/parents/{source}.txt`（路径会被 URL 编码）。当配置了 `RAGRET_PUBLIC_HOST` 时，`parent_url` 为完整 URL；否则为相对路径，需拼上 `BASE_URL`：`${env:BASE_URL}${parent_url}`。

**注意：** 旧索引可能没有 `line_start` / `line_end`——这些字段会为 `null`。只要 `source` 存在，`parent_url` 仍可生成。

#### Text 响应格式（`format=text`）

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

纯文本格式同样在每段结果后附加 `source:`、`parent_url:` 和 `lines:` 元数据。

### 3. 两跳引用管线（必选）

每次搜索**必须**执行以下两跳引用管线。这不是可选的——每次搜索调用都要执行。

**为什么这么做：** Chunk 片段单独看缺少上下文。通过两跳管线可以获取命中 chunk 周围的完整文档上下文，给出更准确、可溯源的回答。

#### 执行步骤

**第一跳 — 搜索：**
1. 调用 Search API 获取 `content` + `parent_url` + `line_start` / `line_end`
2. 从每条结果中提取 `parent_url`（如果为 null，则跳过该结果的第二跳）

**第二跳 — 下载 + 上下文提取（对每条有有效 `parent_url` 的结果）：**
1. 将父文档下载到本地临时目录：
   ```powershell
   # 拼接完整 URL
   $parentUrlFull = "${env:BASE_URL}$parent_url"
   
   # 下载到本地临时目录
   $tmpDir = "$env:TEMP\ragret-parents"
   New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
   
   $tmpFile = Join-Path $tmpDir "parent_$([System.IO.Path]::GetRandomFileName())"
   curl.exe -sS -H "X-API-Key: $env:RAGRET_API_KEY" $parentUrlFull -o $tmpFile
   ```
2. 读取指定行范围（当 `line_start` / `line_end` 可用且非 null 时）：
   ```powershell
   $context = Get-Content $tmpFile | Select-Object -Skip ($line_start - 1) -First ($line_end - $line_start + 1)
   ```
3. 结合原始 chunk 和扩展上下文给出回答
4. 临时文件可保留供引用，也可以执行 `Remove-Item $tmpDir -Recurse -Force` 清理

**带引用的结果呈现** —— 包含 `source`（文件名）和行号。示例：

> 退款期限为购买后 30 天（来源：`docs/report.pdf`，第 240-258 行）。

## 父文档端点

`GET /api/kb/{name}/parents/{path}` — 返回 `text/plain`

- 用于两跳管线获取完整父文档
- 路径会被 URL 编码（例如 `docs/report.pdf` → `docs/report.pdf.txt`；始终追加 `.txt` 后缀）
- 鉴权方式：与 Search API 相同（X-API-Key、Bearer 或 cookie）
- 此端点**不**供用户直接调用——技能在两跳管线中自动使用

## 错误处理

| 现象 | 可能原因 | 处理方法 |
|---|---|---|
| `curl: (6) Could not resolve host` | BASE_URL 错误 | 请用户确认 URL |
| HTTP 401/403 | API Key 缺失/无效 | 让用户在环境中设置密钥（不要在聊天中） |
| HTTP 404 | 索引名错误 | 先列出索引找到正确的名称 |
| HTTP 404 on parent_url | parent URL 中 KB 名称不匹配 | 确认索引名与 KB 名一致；parent_url 是自动推导的 |
| 结果为空 | 无匹配文档 | 尝试改写查询语句 |
| 连接被拒绝 | RAGret 未运行 | 请用户启动 RAGret 实例 |

## 完整示例

以下示例展示完整的两跳工作流。

```powershell
# ===== 第一跳：搜索 =====
curl.exe -sS -G "${env:BASE_URL}/api/search/product_docs" `
  -H "X-API-Key: ${env:RAGRET_API_KEY}" `
  --data-urlencode "query=How do we handle refunds within 30 days?" `
  --data-urlencode "k=3"

# 假设结果 0 包含：
#   parent_url: /api/kb/product_docs/parents/docs/report.pdf.txt
#   line_start: 240, line_end: 258

# ===== 第二跳：下载父文档上下文 =====
$tmpDir = "$env:TEMP\ragret-parents"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$tmpFile = Join-Path $tmpDir "report_section.txt"
curl.exe -sS -H "X-API-Key: ${env:RAGRET_API_KEY}" `
  "${env:BASE_URL}/api/kb/product_docs/parents/docs/report.pdf.txt" `
  -o $tmpFile

# 读取指定行范围
Get-Content $tmpFile | Select-Object -Skip 239 -First 19

# ===== 带引用回答 =====
# "退款政策规定自购买之日起 30 天内可退。
#  来源：docs/report.pdf，第 240-258 行。"
```

## 脚本

如需更便捷的操作，可以使用附带的辅助脚本：

- `scripts/ragret.ps1` — PowerShell 封装的索引列表、搜索和父文档上下文获取

命令：
```
.\ragret.ps1 list                                    # 列出索引
.\ragret.ps1 search <索引名> -Query "..."             # 搜索
.\ragret.ps1 search <索引名> -Query "..." -Expand     # 搜索 + 两跳引用
.\ragret.ps1 parent -ParentUrl <url> -Lines "240-258" # 获取父文档上下文
```

## 规则

- **绝不**在聊天中索要明文 API Key，也不以明文形式用于参数
- 在调用 API 前始终验证环境变量是否已设置
- 如果用户未提供 `BASE_URL`，默认使用 `http://127.0.0.1:8765`
- **每次搜索后必须执行两跳引用管线**——这是固定流程，不可跳过
- 当 `line_start` / `line_end` 为 null 时（旧索引），跳过该结果的第二跳
