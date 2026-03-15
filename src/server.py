import argparse
import logging
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()  # Auto-load .env from project root

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Purple Agent — AgentX Sprint 1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 9009)))
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    skills = [
        AgentSkill(
            id="instruction_following",
            name="Natural Language Instruction Following",
            description=(
                "Precisely follows natural language instructions to build, transform, "
                "or produce requested outputs. Handles BWIM (Build What I Mean) tasks."
            ),
            tags=["bwim", "instruction-following", "nlp", "task-completion"],
            examples=[
                "Create a JSON object with the fields: name, age, city",
                "Rewrite this paragraph in a formal tone",
                "Convert this table to CSV format",
            ],
        ),
        AgentSkill(
            id="crm_operations",
            name="CRM Data Operations",
            description=(
                "Manipulates, queries, and reasons about CRM data including "
                "contacts, accounts, opportunities, and sales pipelines."
            ),
            tags=["crm", "data", "salesforce", "pipeline", "json"],
            examples=[
                "Find all open opportunities over $50,000",
                "Update the account status to Closed Won",
                "Summarize the sales pipeline by stage",
            ],
        ),
        AgentSkill(
            id="reasoning_and_calculation",
            name="Reasoning and Calculation",
            description=(
                "Performs multi-step reasoning, arithmetic, and data transformations "
                "with full precision."
            ),
            tags=["reasoning", "math", "calculation", "analysis"],
            examples=[
                "Calculate the total revenue across all closed deals",
                "What percentage of leads converted to opportunities?",
            ],
        ),
    ]

    agent_card = AgentCard(
        name="Purple Agent — Sprint 1",
        description=(
            "High-performance purple agent for AgentX Sprint 1. "
            "Specializes in BWIM instruction-following and CRMArena tasks. "
            "Uses chain-of-thought reasoning with tool-augmented LLM."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        skills=skills,
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
    )

    task_store = InMemoryTaskStore()
    executor = Executor()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse
    from starlette.routing import Mount, Route

    async def homepage(request: Request):
        return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
  <title>Purple Agent — AgentX Sprint 1</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; background: #0d0d0d; color: #e0e0e0; }
    h1 { color: #a855f7; }
    .badge { display: inline-block; background: #22c55e; color: #000; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; }
    pre { background: #1a1a1a; padding: 16px; border-radius: 8px; overflow-x: auto; border: 1px solid #333; }
    .label { color: #a855f7; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Purple Agent</h1>
  <p><span class="badge">ONLINE</span> &nbsp; AgentX Competition — Sprint 1</p>
  <hr style="border-color:#333">
  <p><span class="label">Model:</span> Llama 3.3 70B (via Groq)</p>
  <p><span class="label">Protocol:</span> A2A (Agent-to-Agent) JSON-RPC 2.0</p>
  <p><span class="label">Skills:</span> Instruction Following · CRM Operations · Reasoning & Calculation</p>
  <hr style="border-color:#333">
  <p><span class="label">Example request:</span></p>
  <pre>curl -X POST https://aiagentcompetition-phase2-production.up.railway.app/ \\
  -H "Content-Type: application/json" \\
  -d '{"jsonrpc":"2.0","id":1,"method":"message/send",
       "params":{"message":{"messageId":"1","role":"user",
       "parts":[{"kind":"text","text":"What is 25% of 480?"}]}}}'</pre>
  <p><a href="/.well-known/agent-card.json" style="color:#a855f7">View Agent Card →</a></p>
</body>
</html>
""")

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build()

    app = Starlette(routes=[
        Route("/", homepage, methods=["GET"]),
        Mount("/", app=a2a_app),
    ])

    logger.info(f"Purple Agent starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
