# Purple Agent — AgentX Sprint 1 (OfficeQA)

An A2A-protocol agent for the OfficeQA benchmark in the AgentX-AgentBeats Sprint 1 competition. It answers U.S. Treasury Bulletin questions by retrieving bundled Treasury source documents, reasoning over the most relevant table- and section-level context, and returning exact-match-friendly final answers.

This repository is also configured for lightweight cloud deployment targets such as Railway.

## How It Works

1. **Source retrieval**: The agent bundles 697 Treasury Bulletin source documents (`.txt` files spanning 1939–2025). When a question arrives, it first uses date references (years, months, fiscal years) to narrow candidate files, then re-ranks them lexically using sampled previews from the head/middle/tail of each bulletin and pulls only locally relevant snippets instead of sending whole documents. If the question has no explicit date, it falls back to lightweight lexical retrieval over cached source previews.

2. **LLM reasoning**: Selected source documents are injected as context into the configured LLM provider. The current benchmark-safe default is Kimi (`kimi-k2.5`) with OfficeQA-specific settings that keep tools and numeric audit off unless explicitly enabled.

3. **Post-processing**: Answers pass through format adaptation, lightweight bare-answer extraction, optional compute verification, and truncation to meet scoring requirements.

4. **Selective upgrades from `officeqa-agent-v2`**: `purple_agent` stays the main submission version, but now absorbs several proven ideas incrementally:
   - layered retrieval with follow-up bulletin search
   - optional rollout voting / ensembling
   - optional Python code execution for precise calculations

## Capabilities

| Capability | Status | Notes |
|---|---|---|
| OfficeQA (Treasury Bulletin questions) | Implemented | Main submission path; source retrieval + LLM reasoning |
| Task routing scaffolding | Implemented | Current competition mode is OfficeQA-specialized |
| Multi-provider LLM (Anthropic, OpenAI, Groq, Nebius, Kimi, DeepInfra) | Implemented | Kimi is the current benchmark-safe default; others are optional |
| Tool-augmented reasoning (calculator, JSON, search, aggregate, Python) | Implemented | OfficeQA honors `OFFICEQA_USE_TOOLS`; default is off |
| Format adapter / output normalization | Implemented | Covers XML, JSON, CSV, and finance-specific patterns |
| Rollout voting / answer ensembling | Optional | Disabled by default (`NUM_ROLLOUTS=1`) |
| Numeric audit | Optional | Disabled by default for OfficeQA benchmark safety |
| Web search (Anthropic only) | Implemented, disabled | `ENABLE_WEB_SEARCH=false` by default |

## Quick Start

### 1. Set up environment
```bash
cp sample.env .env
# Edit .env — set KIMI_API_KEY (or switch to another provider section)
```

### 2. Run locally
```bash
docker compose up --build
```

### 3. Verify
```bash
curl http://localhost:9009/.well-known/agent-card.json
```

## Deploy on Railway

This project is Railway-ready via the included `Dockerfile` and `railway.toml`.

1. Create a new Railway project from this repository.
2. Add the required environment variable:
   - `KIMI_API_KEY`
3. Optional overrides:
   - `LLM_PROVIDER=kimi`
   - `KIMI_MODEL=kimi-k2.5`
   - `MAX_SOURCE_FILES=2`
   - `MAX_LLM_CALLS=4`
   - `MAX_TOKENS=6000`
   - `LLM_MAX_CONCURRENCY=2`
4. Railway will inject `PORT` automatically. The server now respects that port and uses `RAILWAY_PUBLIC_DOMAIN` to build a correct public agent-card URL.
5. After deploy, verify:

```bash
curl https://YOUR_DOMAIN.up.railway.app/.well-known/agent-card.json
```

## Project Structure

```
├── src/
│   ├── agent.py        # Core OfficeQA pipeline: retrieval, prompting, provider calls
│   ├── executor.py     # Thin A2A task bridge that delegates to Agent
│   ├── retrieval.py    # Treasury file ranking, table-aware snippet extraction, source retrieval
│   ├── postprocess.py  # Bare-answer cleaning, format adaptation, exact-match helpers
│   ├── providers.py    # Multi-provider clients, budgets, retries, rate-limit safeguards
│   ├── messenger.py    # Lightweight A2A client helper for remote agent calls
│   └── server.py       # A2A server, OfficeQA agent card, Railway landing page
├── tests/              # Offline and integration tests
├── treasury_data/    # 697 Treasury Bulletin source documents (bundled in Docker image)
├── Dockerfile
├── amber-manifest.json5  # AgentBeats platform manifest
├── pyproject.toml
├── uv.lock
├── sample.env
└── scenario.toml     # Leaderboard submission config
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `kimi` | `anthropic`, `openai`, `groq`, `nebius`, `kimi`, or `deepinfra` |
| `KIMI_API_KEY` | — | API key for Moonshot / Kimi (current default provider) |
| `KIMI_MODEL` | `kimi-k2.5` | Model to use |
| `NEBIUS_API_KEY` | — | API key for Nebius / DeepSeek fallback |
| `NEBIUS_MODEL` | `deepseek-ai/DeepSeek-R1-0528` | Nebius model |
| `ENABLE_TOOLS` | `false` | Global tool toggle |
| `OFFICEQA_USE_TOOLS` | `false` | OfficeQA-specific tool override; must also have `ENABLE_TOOLS=true` |
| `ENABLE_NUMERIC_AUDIT` | `false` | Disable extra LLM audit during benchmark runs |
| `ENABLE_WEB_SEARCH` | `false` | Enable web search (Anthropic only) |
| `TREASURY_DATA_DIR` | `/home/agent/treasury_data` in Docker; auto-detected locally | Directory for Treasury source files |
| `MAX_SOURCE_FILES` | `2` | Maximum number of Treasury source files to inspect per question |
| `SOURCE_FOLLOWUP_YEARS` | `6` | Search later bulletins when a question references a target year |
| `SOURCE_PREVIEW_CHARS` | `8000` | Preview size for lexical reranking |
| `SOURCE_MAX_CHARS` | `12000` | Final cap on Treasury context characters injected into the model |
| `MAX_LLM_CALLS` | `4` | Max LLM calls per question (shared across routing, audit, main loop) |
| `MAX_API_RETRIES` | `1` | Per-call retry cap for upstream provider errors |
| `LLM_MAX_CONCURRENCY` | `2` | Process-level cap on concurrent upstream LLM requests |
| `MAX_TOKENS` | `6000` | Upper bound for generated completion tokens in benchmark-safe mode |
| `NUM_ROLLOUTS` | `1` | Number of answer rollouts before optional voting |
| `VOTE_TEMPERATURE` | `0.4` | Temperature for extra rollouts when voting is enabled |
| `CODE_TIMEOUT` | `30` | Timeout in seconds for the Python calculation tool |

## Submitting to AgentBeats Leaderboard

1. Push code to GitHub — CI auto-builds `ghcr.io/yonghongzhang-io/purple-agent-officeqa:latest`
2. Fork [officeqa-agentbeats-leaderboard](https://github.com/RDI-Foundation/officeqa-agentbeats-leaderboard)
3. Edit `scenario.toml` with your agent ID and image
4. Run the GitHub Actions workflow, then submit results PR
