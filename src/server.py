import argparse
import logging

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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
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

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info(f"Purple Agent starting on {args.host}:{args.port}")
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
