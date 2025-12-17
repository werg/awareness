# Awareness: Decoupled Contextual Memory for LLMs

A research scaffold for **repository-scale aware reasoning** through decoupled context encoding.

See [PLAN.md](PLAN.md) for the full architectural specification.

## Core Idea

**Awareness** decouples reasoning (Decoder) from context (Encoder). Instead of forcing raw tokens into the prompt, we project a mutable corpus into a latent Key/Value store. The decoder attends to this store via Cross-Attention, enabling repository-scale awareness with constant-time inference cost per token.

## Architecture (from PLAN.md)

1. **Context Encoder** ($E_\theta$) - Bidirectional transformer that maps documents to (K, V) tensor pairs
2. **Latent Memory Store** ($\mathcal{M}$) - Persistent tensor database: `{doc_id -> (K, V)}`
3. **Reasoning Decoder** ($D_\phi$) - Decoder-only LLM with Gated Cross-Attention (GCA) in upper 1/3 of layers

## Status

This is a **stub codebase** defining interfaces and structure. Implementation details are intentionally left abstract pending:

- Model selection experiments
- Dataset construction (RepoStack, distilled traces)
- Training methodology refinement

## Structure

```
src/awareness/
├── config.py           # Configuration dataclasses
├── memory.py           # Latent Memory Store interface
├── models/
│   ├── encoder.py      # Context Encoder interface
│   └── decoder.py      # Reasoning Decoder + GCA interface
├── training/
│   └── trainer.py      # Training methodology outline
└── data/
    ├── crawler/        # GitHub repository crawler
    └── processor/      # Commit extraction and processing

scripts/
├── train.py            # Training entry point (stub)
├── eval.py             # Evaluation entry point (stub)
├── crawl_repos.py      # Repository crawler script
└── process_repos.py    # Commit processor script
```

## Installation

### Set up a virtual environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install the package

```bash
# Core package
pip install -e .

# With data crawling dependencies
pip install -e ".[data]"
```

## Data Collection

The training process requires git commit history from popular repositories. The `data` module provides tools to crawl GitHub and extract commits as training data.

### Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
# Edit .env with your GitHub token(s)
```

**Environment Variables:**

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Yes | GitHub personal access token (see below) |
| `GITHUB_TOKENS` | No | Multiple tokens, comma-separated (for higher throughput) |
| `AWARENESS_DB_PATH` | No | SQLite database path (default: `./data/crawl_state.db`) |
| `AWARENESS_REPOS_PATH` | No | Directory for cloned repos (default: `./data/repos`) |
| `AWARENESS_OUTPUT_PATH` | No | Directory for JSONL output (default: `./data/training`) |

**GitHub Token Setup:**

You need a GitHub Personal Access Token. Either type works:

**Option A: Fine-grained token (recommended)**
1. Go to [github.com/settings/tokens?type=beta](https://github.com/settings/tokens?type=beta)
2. Click "Generate new token"
3. Set expiration (e.g., 90 days)
4. Under "Repository access", select "Public Repositories (read-only)"
5. No additional permissions needed
6. Click "Generate token"

**Option B: Classic token**
1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Select scope: `public_repo` (only needed for public repo access)
4. Click "Generate token"

**Multiple tokens:** For higher throughput, create multiple tokens and set them comma-separated:
```bash
GITHUB_TOKENS=ghp_token1,ghp_token2,ghp_token3
```
Note: The Search API is rate-limited per-IP (not per-token), so multiple tokens primarily help with the Core API.

### Quick Start

```bash
# 1. Set up configuration
cp .env.example .env
# Edit .env with your GitHub token(s)

# 2. Discover repositories via GitHub API
python -m awareness.data.cli discover --target 100000

# 3. Download repositories
python -m awareness.data.cli download --concurrency 10

# 4. Process commits into training data
python -m awareness.data.cli process

# 5. Check progress
python -m awareness.data.cli stats
```

### Adaptive Discovery

The crawler uses **adaptive subdivision** to minimize API calls:

1. **Start broad**: Query `stars:>=50 language:python` (single API call)
2. **Check total_count**: GitHub returns how many repos match
3. **Subdivide only if needed**: If `total_count > 1000`, split into star ranges

This is much more efficient than iterating through all star ranges upfront. For languages with few repos (e.g., Gleam, Coq), a single API call retrieves everything. Only popular languages like Python/JavaScript require subdivision.

### Tiered Language Discovery

The crawler uses a tiered approach to discover high-quality repositories across different language ecosystems:

| Tier | Languages | Min Stars | Rationale |
|------|-----------|-----------|-----------|
| **Mainstream** | Python, JavaScript, TypeScript, Java, Go, Rust, C++, C, Ruby, PHP, C#, Swift, Kotlin, Scala, Shell | 50+ | Large ecosystems, many repos |
| **Niche** | Haskell, OCaml, F#, Elixir, Erlang, Clojure, Elm, PureScript, Common Lisp, Scheme, Racket, Agda, Idris, Coq, Lean, R, Julia, MATLAB, Nim, D, Ada, Fortran, Assembly, Prolog, Lua, Perl, Tcl, Forth, Smalltalk, Crystal, V | 20+ | Smaller but high-quality ecosystems |
| **Emerging** | Zig, Gleam, Mojo, Pony, Roc, Unison, Koka, Vale, Odin, Carbon | 10+ | New languages, growing ecosystems |

This enables discovering ~100K+ repositories while capturing high-quality code from niche and emerging language communities.

### Rate Limiting

The crawler handles GitHub's rate limits automatically:

- **Core API**: 5,000 requests/hour per token (multiple tokens supported)
- **Search API**: 30 requests/minute (2.5s delay enforced between searches)
- **Secondary limits**: Automatic detection and exponential backoff (60s → 120s → 300s)

Use multiple tokens for higher throughput:
```bash
export GITHUB_TOKENS=token1,token2,token3
```

### Resumable & Idempotent

The crawler is designed for long-running operations that may be interrupted:

**What gets tracked:**
- Each `(tier, language, star_threshold)` query is tracked in the `discovery_progress` table
- When you restart, already-completed queries are skipped automatically
- Fully-completed languages are skipped entirely (no API calls)

**Greedy storage:**
- All discovered repos are stored immediately to the database
- The `target_count` only controls when to stop making *new* queries
- If interrupted mid-query, all results from that query are already saved

**Idempotent updates:**
- Re-discovering the same repo updates its metadata (stars, size, etc.)
- Existing download/processing state is preserved
- No duplicate entries are created

```bash
# Resume an interrupted discovery (default behavior)
python -m awareness.data.cli discover --target 100000

# Start fresh (clear all progress tracking)
python -m awareness.data.cli discover --target 100000 --fresh

# Check what's been completed
python -m awareness.data.cli stats
```

**Progress display on startup:**
```
Resuming discovery: 47 queries already complete
  mainstream: 35 queries, 42,156 repos
  niche: 12 queries, 3,241 repos
```

### Periodic Refresh

Re-run discovery to find new repositories and update metadata:

```bash
# Refresh: re-discover all queries, update star counts
python -m awareness.data.cli refresh

# Reprocess: re-extract commits with updated filters
python -m awareness.data.cli reprocess --all
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `discover` | Discover repositories via GitHub Search API |
| `download` | Clone discovered repositories |
| `process` | Extract commits into JSONL training data |
| `stats` | Show database statistics |
| `refresh` | Re-discover to find new repos and update metadata |
| `reprocess` | Reset processed repos for re-extraction |
| `full` | Run complete pipeline (discover → download → process) |

### Command Options

**discover**
```bash
python -m awareness.data.cli discover \
  --target 100000 \          # Target repo count
  --min-stars 50 \           # Min stars for mainstream (niche: 20, emerging: 10)
  --include-niche \          # Include niche languages (default: true)
  --include-emerging \       # Include emerging languages (default: true)
  --fresh                    # Clear progress and start fresh
```

**download**
```bash
python -m awareness.data.cli download \
  --output ./data/repos \    # Output directory
  --concurrency 10 \         # Parallel clones
  --randomize                # Random order for balanced language coverage
```

By default, downloads are ordered by stars (highest first). Use `--randomize` to select repos randomly, which gives you a more balanced language distribution early in the download process.

**process**
```bash
python -m awareness.data.cli process \
  --output ./data/training \ # Output directory for JSONL
  --workers 4                # Parallel workers
```

### Output Format

Commits are extracted as JSONL with one record per line:

```json
{
  "repo": "facebook/react",
  "sha": "abc123",
  "parent_sha": "def456",
  "message": "Fix memory leak in useEffect",
  "language": "javascript",
  "files_changed": ["src/hooks/useEffect.js"],
  "additions": 15,
  "deletions": 8,
  "diff": "diff --git a/..."
}
```

### Estimated Resources

For 100K repositories:
- **Discovery**: ~2-3 hours (with search API throttling)
- **Download**: ~4-5 days (10 concurrent clones)
- **Storage**: ~5TB raw clones, ~500GB-1TB processed JSONL
