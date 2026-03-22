# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser --disabled-password --gecos "" agent
USER agent
WORKDIR /home/agent

ENV TREASURY_DATA_DIR=/home/agent/treasury_data

COPY --chown=agent:agent pyproject.toml uv.lock README.md ./
COPY --chown=agent:agent src src
COPY --chown=agent:agent treasury_data treasury_data

RUN uv sync --locked --link-mode=copy

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
