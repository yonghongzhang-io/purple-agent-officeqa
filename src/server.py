"""Purple Agent — A2A server entry point."""

import argparse
import html
import logging
import os

import uvicorn
from dotenv import load_dotenv
from starlette.responses import HTMLResponse

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _default_card_url(host: str, port: int) -> str:
    explicit = os.environ.get("AGENT_CARD_URL")
    if explicit:
        return explicit

    railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    if railway_domain:
        railway_domain = railway_domain.strip().removeprefix("https://").removeprefix("http://")
        return f"https://{railway_domain}/"

    return f"http://{host}:{port}/"


def _landing_page_html(agent_card: AgentCard) -> str:
    card_url = agent_card.url.rstrip("/") + "/.well-known/agent-card.json"
    skill_cards = "".join(
        f"""
        <article class="skill-card">
          <p class="skill-kicker">{html.escape(skill.id.replace("_", " ").title())}</p>
          <h3>{html.escape(skill.name)}</h3>
          <p>{html.escape(skill.description)}</p>
          <div class="chips">
            {"".join(f'<span>{html.escape(tag)}</span>' for tag in skill.tags[:4])}
          </div>
        </article>
        """
        for skill in agent_card.skills
    )
    sample_prompts = "".join(
        f"<li>{html.escape(example)}</li>"
        for skill in agent_card.skills
        for example in skill.examples[:1]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{agent_card.name}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f1320;
      --panel: rgba(19, 24, 41, 0.86);
      --panel-strong: rgba(14, 18, 33, 0.94);
      --card: rgba(255,255,255,0.04);
      --line: rgba(255,255,255,0.08);
      --text: #f6f1ea;
      --muted: #bcc6da;
      --accent: #d7a54f;
      --accent-2: #7ed7cb;
      --shadow: 0 28px 90px rgba(0,0,0,.38);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Helvetica Neue", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(126,215,203,0.18) 0%, transparent 28%),
        radial-gradient(circle at top right, rgba(215,165,79,0.16) 0%, transparent 24%),
        linear-gradient(180deg, #151a2b 0%, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
      padding: 28px;
    }}
    .shell {{
      width: min(1040px, 100%);
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 30px;
      overflow: hidden;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .hero {{
      padding: 42px 42px 28px;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(135deg, rgba(255,255,255,0.02), transparent 55%),
        radial-gradient(circle at 80% 20%, rgba(126,215,203,0.12), transparent 32%);
    }}
    .eyebrow {{
      display: inline-flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .eyebrow span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.88rem;
      letter-spacing: 0.02em;
    }}
    h1 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      font-size: clamp(2.6rem, 6vw, 4.8rem);
      line-height: 0.96;
      letter-spacing: -0.04em;
      max-width: 10ch;
    }}
    .subtitle {{
      margin: 18px 0 0;
      max-width: 70ch;
      color: var(--muted);
      font-size: 1.12rem;
      line-height: 1.7;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 28px;
    }}
    .button {{
      text-decoration: none;
      color: #0f1320;
      background: linear-gradient(135deg, var(--accent), #f2c978);
      padding: 13px 18px;
      border-radius: 16px;
      font-weight: 600;
      box-shadow: 0 12px 28px rgba(215,165,79,0.24);
    }}
    .button.secondary {{
      color: var(--text);
      background: transparent;
      border: 1px solid var(--line);
      box-shadow: none;
    }}
    code {{
      display: inline-block;
      background: rgba(255,255,255,.07);
      padding: 3px 8px;
      border-radius: 9px;
      color: #d6ece8;
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .content {{
      display: grid;
      gap: 22px;
      padding: 28px 42px 42px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .stat {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .stat-value {{
      font-size: 1.1rem;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 22px;
    }}
    .panel {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      font-size: 1.6rem;
      letter-spacing: -0.03em;
    }}
    .panel p,
    .panel li {{
      color: var(--muted);
      line-height: 1.7;
    }}
    .endpoint {{
      display: grid;
      gap: 14px;
    }}
    .endpoint-card {{
      background: rgba(255,255,255,0.035);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
    }}
    .endpoint-card strong {{
      display: block;
      margin-bottom: 8px;
      font-size: 1rem;
    }}
    .skill-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .skill-card {{
      background: rgba(255,255,255,0.035);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
    }}
    .skill-kicker {{
      margin: 0 0 8px;
      color: var(--accent);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }}
    .skill-card h3 {{
      margin: 0 0 10px;
      font-size: 1.05rem;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }}
    .chips span {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(126,215,203,0.08);
      border: 1px solid rgba(126,215,203,0.16);
      color: #d6ece8;
      font-size: 0.82rem;
    }}
    .prompts {{
      margin: 0;
      padding-left: 18px;
    }}
    .footer {{
      margin-top: 10px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.95rem;
    }}
    @media (max-width: 900px) {{
      .stats,
      .grid,
      .skill-grid {{
        grid-template-columns: 1fr;
      }}
      .hero,
      .content {{
        padding-left: 22px;
        padding-right: 22px;
      }}
      h1 {{
        max-width: none;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">
        <span>A2A-compatible API</span>
        <span>OfficeQA benchmark</span>
        <span>Treasury Bulletin retrieval</span>
      </div>
      <h1>{agent_card.name}</h1>
      <p class="subtitle">{agent_card.description}</p>
      <div class="actions">
        <a class="button" href="{card_url}">Open Agent Card</a>
        <a class="button secondary" href="{agent_card.url}">Service Root</a>
      </div>
    </section>

    <section class="content">
      <div class="stats">
        <div class="stat">
          <div class="stat-label">Protocol</div>
          <div class="stat-value">{agent_card.protocol_version}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Streaming</div>
          <div class="stat-value">{"Enabled" if agent_card.capabilities.streaming else "Disabled"}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Input Mode</div>
          <div class="stat-value">{", ".join(agent_card.default_input_modes)}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Output Mode</div>
          <div class="stat-value">{", ".join(agent_card.default_output_modes)}</div>
        </div>
      </div>

      <div class="grid">
        <section class="panel">
          <h2>What This Agent Does</h2>
          <p>
            This deployment serves a source-grounded OfficeQA agent tuned for
            U.S. Treasury Bulletin questions. It retrieves relevant Treasury
            passages, reasons over them with an LLM, and formats benchmark-ready
            answers with exact-match-friendly post-processing.
          </p>
          <div class="endpoint">
            <div class="endpoint-card">
              <strong>Agent Card Endpoint</strong>
              <code>/.well-known/agent-card.json</code>
              <p>Machine-readable metadata for A2A clients, registries, and benchmark runners.</p>
            </div>
            <div class="endpoint-card">
              <strong>Service Root</strong>
              <code>{agent_card.url}</code>
              <p>Human-friendly landing page for quick inspection and health verification.</p>
            </div>
          </div>
        </section>

        <section class="panel">
          <h2>Example Questions</h2>
          <ul class="prompts">
            {sample_prompts}
          </ul>
          <div class="footer">
            Best used for OfficeQA-style Treasury questions and exact-match benchmark evaluation.
          </div>
        </section>
      </div>

      <section class="panel">
        <h2>Core Skills</h2>
        <div class="skill-grid">
          {skill_cards}
        </div>
      </section>
    </section>
  </main>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Purple Agent — OfficeQA")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 9009)), help="Port to bind the server")
    parser.add_argument("--card-url", type=str, default=None, help="URL to advertise in the agent card")
    args = parser.parse_args()

    card_url = args.card_url or _default_card_url(args.host, args.port)

    skills = [
        AgentSkill(
            id="officeqa_treasury_reasoning",
            name="Treasury Bulletin Question Answering",
            description=(
                "Answers OfficeQA-style questions by retrieving relevant U.S. Treasury "
                "Bulletin source passages and reasoning over them with an LLM."
            ),
            tags=["officeqa", "treasury", "retrieval", "reasoning", "finance"],
            examples=[
                "What were total federal receipts in fiscal year 2020?",
                "How much public debt was outstanding in March 2011?",
                "What was the Treasury balance on a given reporting date?",
            ],
        ),
        AgentSkill(
            id="financial_reasoning",
            name="Financial Reasoning and Calculation",
            description=(
                "Performs financial reasoning, arithmetic, and unit-sensitive "
                "post-processing for numeric benchmark questions."
            ),
            tags=["finance", "calculation", "analysis", "numeric-audit"],
            examples=[
                "Calculate the percentage change between two Treasury totals",
                "Convert a result into the required unit and symbol format",
            ],
        ),
        AgentSkill(
            id="structured_output",
            name="Structured Output Formatting",
            description=(
                "Formats answers into benchmark-friendly XML, JSON, CSV, and other "
                "structured response patterns."
            ),
            tags=["formatting", "xml", "json", "csv", "post-processing"],
            examples=[
                "Return the final answer inside a FINAL_ANSWER tag",
                "Produce a strict JSON risk classification object",
            ],
        ),
    ]

    agent_card = AgentCard(
        name="Purple Agent — OfficeQA",
        description=(
            "A source-grounded OfficeQA agent that answers U.S. Treasury Bulletin "
            "questions using Treasury document retrieval, LLM reasoning, and "
            "benchmark-friendly output normalization."
        ),
        url=card_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build()

    async def landing_page(_request):
        return HTMLResponse(_landing_page_html(agent_card))

    app.add_route("/", landing_page, methods=["GET", "HEAD"])

    logger.info(f"Purple Agent starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
