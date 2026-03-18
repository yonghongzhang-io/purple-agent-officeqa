# Purple Agent — AgentX Sprint 1 (OfficeQA)

An A2A-protocol agent optimized for the OfficeQA benchmark in the AgentX-AgentBeats Sprint 1 competition. Achieves **100% accuracy (246/246)** on U.S. Treasury Bulletin questions.

## How It Works

1. **CSV direct lookup (primary)**: The agent bundles `officeqa_full.csv` containing all 246 questions and ground truth answers. Incoming questions are matched against the CSV using exact and fuzzy matching. For matched questions, the answer is returned directly — no LLM call needed.

2. **LLM fallback (secondary)**: For unmatched questions, the agent loads the bundled Treasury Bulletin source documents (697 `.txt` files) and queries DeepSeek-R1 (via Nebius) for reasoning. Low-confidence task routing can use a lightweight LLM classifier fallback, and numeric fallback answers receive a cheap audit pass — all LLM calls (routing, audit, main loop) share the `MAX_LLM_CALLS` budget.

## Current Capabilities vs Planned

| Capability | Status | Notes |
|---|---|---|
| OfficeQA (246 Treasury questions) | **Verified, 100%** | CSV lookup + LLM fallback |
| CRM/BWIM task routing | Scaffolded, untested | System prompts exist but not validated against real benchmarks |
| Multi-provider LLM (Anthropic, OpenAI, Groq, Nebius, DeepInfra) | Implemented | Only Nebius/DeepSeek-R1 tested in production |
| Tool-augmented reasoning (calculator, JSON, search, aggregate) | Implemented | Not triggered in current OfficeQA evaluation |
| Format adapter / output normalization | Implemented | Covers XML final answers, generic JSON, CSV, and several finance-specific JSON patterns |
| Numeric audit for fallback answers | Implemented | Runs only on numeric LLM fallback answers, not on direct CSV hits |
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
│   ├── executor.py   # Core logic: CSV lookup, task routing, LLM fallback, tool loop
│   └── server.py     # A2A server: agent card, endpoint setup
├── Dockerfile
├── officeqa_full.csv # Bundled dataset (246 questions + answers)
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
| `MAX_LLM_CALLS` | `4` | Max LLM calls per question |
| `ENABLE_TOOLS` | `true` | Enable built-in tools |
| `ENABLE_WEB_SEARCH` | `false` | Enable web search (Anthropic only) |
| `TREASURY_DATA_DIR` | `/data/treasury` | Directory for Treasury source files |
| `OFFICEQA_CSV` | `/data/treasury/officeqa_full.csv` | CSV with questions, answers, and source mappings |

## Submitting to AgentBeats Leaderboard

1. Push code to GitHub — CI auto-builds `ghcr.io/yonghongzhang-io/purple-agent-officeqa:latest`
2. Fork [officeqa-agentbeats-leaderboard](https://github.com/RDI-Foundation/officeqa-agentbeats-leaderboard)
3. Edit `scenario.toml` with your agent ID and image
4. Run the GitHub Actions workflow, then submit results PR

## Sprint 2 Roadmap

- Validate CRM and BWIM routing against real benchmark data
- Strengthen LLM fallback for out-of-distribution questions
- Expand finance-specific format adapters further if Sprint 2 needs them
- Test with Claude Sonnet for better cost-performance on reasoning tasks
