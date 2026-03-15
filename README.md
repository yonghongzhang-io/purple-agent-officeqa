# Purple Agent — AgentX Sprint 1

A high-performance A2A-protocol agent for the AgentX-AgentBeats Sprint 1 competition (BWIM + CRMArena).

## Quick Start

### 1. Set up environment
```bash
cp sample.env .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

### 2. Run locally
```bash
docker compose up --build
```

### 3. Verify the agent is running
```bash
curl http://localhost:9009/.well-known/agent-card.json
```

### 4. Test with a manual request
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
│   ├── executor.py   # Agent logic: LLM calls, tool loop, conversation history
│   └── server.py     # A2A server: agent card, endpoint setup
├── Dockerfile
├── docker-compose.yml
├── sample.env        # Copy to .env and fill in keys
└── scenario.toml     # Leaderboard submission config
```

## Submitting to AgentBeats Leaderboard

1. Build and push your Docker image:
   ```bash
   docker build -t ghcr.io/YOUR_GITHUB_USERNAME/purple-agent:latest .
   docker push ghcr.io/YOUR_GITHUB_USERNAME/purple-agent:latest
   ```

2. Update `scenario.toml` with your `agentbeats_id` and image path.

3. Register your image on [agentbeats.dev](https://agentbeats.dev).

4. Fill in the [Sprint 1 submission form](https://docs.google.com/forms/d/e/1FAIpQLSflVhb-qlsCJp5zMTutgW9cOc5Ywfn_EtPDyngqiqFVMPXgLQ/viewform).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model to use |
| `ANTHROPIC_MAX_TOKENS` | `8000` | Max output tokens |
| `ENABLE_TOOLS` | `true` | Enable built-in tools (calculator, JSON formatter) |
| `MAX_LLM_CALLS` | `4` | Max LLM calls per response (competition limit: 4) |
