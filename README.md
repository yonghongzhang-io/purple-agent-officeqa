# Purple Agent — AgentX Sprint 1

A high-performance A2A-protocol agent for the AgentX-AgentBeats Sprint 1 competition (BWIM + CRMArena + OfficeQA).

## Features

- **Task-type routing**: Auto-detects BWIM, CRMArena, financial, and general tasks; uses specialized system prompts for each
- **Multi-provider LLM**: Supports Anthropic (Claude), OpenAI (GPT-4o), Groq, Nebius, DeepInfra
- **Tool-augmented reasoning**: Calculator (with math functions), JSON formatter, data search/filter, aggregation
- **Web search** (Anthropic/OpenAI): Enable for document retrieval tasks
- **Multi-turn conversation**: Maintains context across follow-up messages
- **Output validation**: Ensures `<FINAL_ANSWER>` tags are always present

## Quick Start

### 1. Set up environment
```bash
cp sample.env .env
# Edit .env — set ANTHROPIC_API_KEY (recommended) or another provider
```

### 2. Run locally
```bash
docker compose up --build
```

### 3. Verify
```bash
curl http://localhost:9009/.well-known/agent-card.json
```

### 4. Test
```bash
curl -X POST http://localhost:9009/task \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-001",
    "message": {
      "messageId": "msg-001",
      "role": "user",
      "parts": [{"kind": "text", "text": "Create a JSON object with fields: name=Alice, age=30, city=NYC"}]
    }
  }'
```

## Project Structure

```
purple_agent/
├── src/
│   ├── executor.py   # Agent logic: task routing, LLM calls, tool loop
│   └── server.py     # A2A server: agent card, endpoint setup
├── Dockerfile
├── docker-compose.yml
├── sample.env        # Copy to .env and fill in keys
└── scenario.toml     # Leaderboard submission config
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, `groq`, `nebius`, or `deepinfra` |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key (recommended for best accuracy) |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model to use |
| `ANTHROPIC_MAX_TOKENS` | `16000` | Max output tokens |
| `ENABLE_TOOLS` | `true` | Enable built-in tools (calculator, JSON, search, aggregate) |
| `ENABLE_WEB_SEARCH` | `false` | Enable web search for document retrieval |
| `MAX_LLM_CALLS` | `4` | Max LLM calls per response (competition limit: 4) |

## Submitting to AgentBeats Leaderboard

1. Build and push your Docker image:
   ```bash
   docker build -t ghcr.io/YOUR_GITHUB_USERNAME/purple-agent:latest .
   docker push ghcr.io/YOUR_GITHUB_USERNAME/purple-agent:latest
   ```

2. Update `scenario.toml` with your `agentbeats_id` and image path.

3. Register your image on [agentbeats.dev](https://agentbeats.dev).

4. Fill in the [Sprint 1 submission form](https://docs.google.com/forms/d/e/1FAIpQLSflVhb-qlsCJp5zMTutgW9cOc5Ywfn_EtPDyngqiqFVMPXgLQ/viewform).
