<h1 align="center" style="font-size: 2.75em; font-weight: 700; border-bottom: none;">RAGret</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/SugarSong404/RAGret?style=flat-square" alt="License" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/docker-CUDA-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker (CUDA)" /></a>
  <a href="https://github.com/SugarSong404/RAGret"><img src="https://img.shields.io/github/stars/SugarSong404/RAGret?style=flat-square&logo=github" alt="GitHub stars" /></a>
  <a href="https://github.com/SugarSong404/RAGret/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs welcome" /></a>
</p>

<p align="center">中文文档 · <a href="README.zh-CN.md">README.zh-CN.md</a></p>

## What is RAGret?

`RAGret` (not “regret” 😂) is a self-hosted RAG web app aimed at small teams (about 15–30 people) and low-cost servers (GPU memory ≤ 8 GB).

With `RAGret`, team members can publish knowledge bases to a shared hub, subscribe to others’ bases, and query them afterward via HTTP GET.

### Highlights

- Creators can set visibility scopes and control access flexibly.
- **Agent-friendly**: the app exposes **API keys** and a **SKILL.md** so agents can plug into team workflows quickly.
- **Quick Q&A**: a built-in chat UI (LangGraph agent) that lists your knowledge bases and searches them with natural language—configure an OpenAI-compatible LLM via `.env`.
- Ingest via **tar upload** or **GitLab / GitHub webhooks**, so it fits common doc storage habits. Support for Feishu and similar online docs is planned.
- Many formats: **PDF, Word (docx), Excel (xlsx), Markdown (md), Email (eml), TXT, CSV, web links (html)**.
- Bilingual UI (Chinese / English), light / dark themes, and brand tweaks via YAML (e.g. favicon, page title).

### Stack

Indexing and retrieval use **BCE embedding + SQLite + BCE reranking**, backed by:

- [BCEmbedding (GitHub)](https://github.com/netease-youdao/BCEmbedding)
- [Models on Hugging Face](https://huggingface.co/maidalun1020) (`bce-embedding-base_v1`, `bce-reranker-base_v1`)

## Quick start

Pick **one** GPU path: **CUDA** or **Intel XPU**. Pick **one** runtime: **local Python** or **Docker**.

**General notes:**

- Use only one GPU stack and one runtime per environment.
- **Hugging Face mirror (optional):** if downloads are slow or blocked, set **`HF_ENDPOINT`** before running **`warmup_hf_models.py`** or **`docker build`** (see below).

```bash
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
```

### Environment setup

#### Local Python

1. **Python 3.10+** (tested on 3.12). Create a venv or conda env.
2. **Install PyTorch for your GPU (pick one):**
   - **NVIDIA CUDA:** follow **[Start Locally](https://pytorch.org/get-started/locally/)**, or e.g.  
     `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
   - **Intel XPU:** follow **[Get started with Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)**. After installing the [Intel GPU drivers](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html), e.g.  
     `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu`
3. **App deps:** `pip install -r requirements.txt`
4. **Models (once, before indexing/search):** from the **repo root**, online:

   ```bash
   python warmup_hf_models.py
   ```

   Weights land in **`./models`**. You can also download BCE weights manually into **`./models`**.

5. **Verify GPU:**
   - CUDA: `python -c "import torch; print(torch.cuda.is_available())"` → `True`
   - XPU: `python -c "import torch; print(torch.xpu.is_available())"` → `True`

   On **Intel XPU**, only **embedding** uses the GPU; upstream **BCEmbedding** **rerank** does not support **XPU**.

---

#### Docker (CUDA only)

This repo’s Docker image targets **CUDA** only (`Dockerfile`). For Intel XPU, use **local Python** above.

Build (warmup bakes weights into **`/opt/hf`** in the image):

```bash
docker build -t ragret .
# China mirror; disable proxy when using it
docker build -t ragret --build-arg HF_ENDPOINT=https://hf-mirror.com .
```

Run with **`--gpus all`** (or `'--gpus "device=0"'`).

```bash
docker run --name ragret -it --gpus all -p 8765:8765 ragret
```

You can also download the model to the host machine and start the container quickly by skipping warmup as follows:

```bash
docker build -t ragret --build-arg RAGRET_SKIP_WARMUP=1 .
docker run --name ragret -it --gpus all -p 8765:8765 -v /path/on/host/models:/opt/hf ragret
```

### Configuration (`.env`)

Copy the template and edit values at the **repo root** (`.env` is gitignored):

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Example:

```env
RAGRET_HOST=127.0.0.1
RAGRET_PORT=8765

# Quick Q&A agent (OpenAI-compatible API)
RAGRET_LLM_BASE_URL=https://your-api.example.com/v1
RAGRET_LLM_MODEL=your-model-name
RAGRET_LLM_API_KEY=your-api-key
```

All settings use the `RAGRET_` prefix. CLI flags such as `--host` or `--llm-model` override `.env` when provided.

Without LLM settings, Quick Q&A falls back to **direct index search** (no LangGraph agent).

### Start the server

1. **Build the web UI** (output goes to `ragret/static/`):

   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

2. **Run the API** from the repo root:

   ```bash
   python -m ragret serve
   # or: python ragret.py serve
   ```

   Open `http://127.0.0.1:8765/` (or your `RAGRET_HOST` / `RAGRET_PORT`). The home page is **Quick Q&A**; use the sidebar for **Knowledge plaza**, tasks, and account settings.

   Optional overrides:

   ```bash
   python -m ragret serve --host 0.0.0.0 --port 8765
   ```

### Run tests (optional)

```bash
pip install -r requirements.txt
pytest tests/ -q
```

## User guide

### Quick Q&A

After sign-in, the home page is **Quick Q&A**: ask questions in natural language. The agent can list knowledge bases you own or subscribe to and call retrieval tools against them.

- Configure **`RAGRET_LLM_*`** in `.env` (see above) for full agent mode.
- Without LLM config, answers come from a simple multi-KB index search.
- Chat history on this page is **not kept** after refresh.

Use **Knowledge plaza** in the sidebar (`/plaza`) to browse and subscribe to bases.

### Preferences and credentials

Open **Account**:

![account](assets/screenshot_account.png)

1. Change avatar, theme, and language.
2. Create up to **3 API keys** to query knowledge bases you own or subscribe to.
3. For GitHub or GitLab webhooks, paste a **PAT** in the right fields. For safety, scope the PAT to **read-only repo** access.

### Create a knowledge base

Open **Add knowledge base**:

![add](assets/screenshot_add.png)

1. **Required:** name and description so agents can pick the right base.
2. Optional: README-style description file and cover image.
3. Set visibility. **Locked** defaults to **creator-only**; you can add members after creation.
4. Choose type. **Tar upload** is straightforward; **webhook** is shown below.

![webhook](assets/screenshot_webhook.png)

The first build pulls from the repo, so **repo URL** and **branch** are required. Copy **Webhook URL** and **Secret Token** into your repo’s webhook settings, then click build.

### Tasks in progress

On modest hardware, chunking and indexing are queued. Each build click or webhook run registers a task.

Open **Task list** to see queued and running jobs:

![task](assets/screenshot_task.png)

Cancel tasks when needed.

![Tdetail](assets/screenshot_Tdetail.png)

### Manage your knowledge bases

Open **My knowledge bases**, then select one to manage.

Notes:

1. For **webhook** bases, if you rename the base, update the webhook URL in the repo.
2. **Rebuilds are incremental** for all types. To add files via **tar**, upload an archive of the **full document set** you want indexed.
3. Webhook bases can be pulled from the repo manually on this page.

![rebuild](assets/screenshot_rebuild.png)

4. Use the search box at the bottom to try retrieval against the base.

### Using knowledge bases

1. Subscribe in **Knowledge plaza** (sidebar).
2. Copy an API key from **Account**.
3. Set **`RAGRET_API_KEY`**.

**HTTP API examples** (`BASE` = e.g. `http://127.0.0.1:8765`):

```bash
# List subscribed indexes (API key)
curl -sS -H "X-API-Key: $RAGRET_API_KEY" "$BASE/api/subscribe-indexes"

# Search (API key)
curl -sS -G "$BASE/api/search/INDEX_NAME" -H "X-API-Key: $RAGRET_API_KEY" --data-urlencode "query=…"

# Quick Q&A (signed-in session cookie or Bearer token from /api/auth/login)
curl -sS -X POST "$BASE/api/quick-qa" -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -d '{"question":"What is in my docs?","lang":"en"}'
```

**Agents:** download `SKILL.md` from the UI (**SKILL.md** in the sidebar) and import it into Claude Code, Cursor, OpenClaw, or other agent tools.

## Roadmap

1. More formats: tables, PPT, images.
2. Sync with Feishu and similar online docs.
3. Distributed deployment for higher concurrency and larger teams.
4. Stability fixes across the stack.
