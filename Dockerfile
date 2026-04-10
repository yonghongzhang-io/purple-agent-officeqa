# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install curl + unzip as root for treasury data download
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" agent
USER agent
WORKDIR /home/agent

ENV TREASURY_DATA_DIR=/home/agent/treasury_data

# Download Treasury Bulletin corpus at build time (459MB unzipped, ~100MB zip)
RUN mkdir -p /home/agent/treasury_data && \
    curl -fsSL "https://raw.githubusercontent.com/databricks/officeqa/main/treasury_bulletins_parsed/transformed/treasury_bulletins_transformed.zip" \
        -o /tmp/bulletins.zip && \
    unzip -q /tmp/bulletins.zip -d /home/agent/treasury_data/ && \
    rm /tmp/bulletins.zip

COPY --chown=agent:agent pyproject.toml uv.lock README.md ./
COPY --chown=agent:agent src src

RUN uv sync --locked --link-mode=copy

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
