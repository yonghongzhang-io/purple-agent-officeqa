# Purple Agent — AgentX Sprint 1 (OfficeQA)

An A2A-protocol agent for the OfficeQA benchmark in the AgentX-AgentBeats Sprint 1 competition. Answers U.S. Treasury Bulletin questions by reasoning over bundled source documents using LLM.

## How It Works

1. **Source retrieval**: The agent bundles 697 Treasury Bulletin source documents (`.txt` files spanning 1939–2025). When a question arrives, it extracts date references (years, months, fiscal years) to select relevant source files.

2. **LLM reasoning**: Selected source documents are injected as context into DeepSeek-R1 (via Nebius), which reasons over the data to produce an answer. The agent uses task-type-aware system prompts optimized for numerical precision and exact-match scoring.

3. **Post-processing**: Answers pass through format adaptation, compute verification (unit/sign consistency), hedge detection, and truncation to meet scoring requirements.

## Capabilities

| Capability | Status | Notes |
|---|---|---|
| OfficeQA (Treasury Bulletin questions) | Implemented | Source retrieval + LLM reasoning |
| Task routing (officeqa/crm/bwim/financial/general) | Implemented | Heuristic + optional LLM classifier fallback |
| Multi-provider LLM (Anthropic, OpenAI, Groq, Nebius, DeepInfra) | Implemented | Only Nebius/DeepSeek-R1 tested in production |
| Tool-augmented reasoning (calculator, JSON, search, aggregate) | Implemented | Available for multi-step calculations |
| Format adapter / output normalization | Implemented | Covers XML, JSON, CSV, and finance-specific patterns |
| Numeric audit | Implemented | LLM-based audit for calculation answers, shares MAX_LLM_CALLS budget |
| Web search (Anthropic only) | Implemented, disabled | `ENABLE_WEB_SEARCH=false` by default |

## Quick Start

### 1. Set up environment
```bash
cp sample.env .env
# Edit .env — set NEBIUS_API_KEY (or another provider key)
```

### 2. Run locally
```bash
docker compose up --build
```

### 3. Verify
```bash
curl http://localhost:9009/.well-known/agent-card.json
```

## Project Structure

```
├── src/
│   ├── executor.py   # Core logic: source retrieval, task routing, LLM reasoning, tool loop
│   └── server.py     # A2A server: agent card, endpoint setup
├── treasury_data/    # 697 Treasury Bulletin source documents (bundled in Docker image)
├── Dockerfile
├── amber-manifest.json5  # AgentBeats platform manifest
├── sample.env
└── scenario.toml     # Leaderboard submission config
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `nebius` | `anthropic`, `openai`, `groq`, `nebius`, or `deepinfra` |
| `NEBIUS_API_KEY` | — | API key for Nebius (current default provider) |
| `NEBIUS_MODEL` | `deepseek-ai/DeepSeek-R1-0528` | Model to use |
| `MAX_LLM_CALLS` | `4` | Max LLM calls per question (shared across routing, audit, main loop) |
| `ENABLE_TOOLS` | `true` | Enable built-in tools |
| `ENABLE_WEB_SEARCH` | `false` | Enable web search (Anthropic only) |
| `TREASURY_DATA_DIR` | `/data/treasury` | Directory for Treasury source files |

## Submitting to AgentBeats Leaderboard

1. Push code to GitHub — CI auto-builds `ghcr.io/yonghongzhang-io/purple-agent-officeqa:latest`
2. Fork [officeqa-agentbeats-leaderboard](https://github.com/RDI-Foundation/officeqa-agentbeats-leaderboard)
3. Edit `scenario.toml` with your agent ID and image
4. Run the GitHub Actions workflow, then submit results PR
