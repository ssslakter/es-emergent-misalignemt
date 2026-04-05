FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps for uv + possible build steps
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl build-essential git \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a persistent venv inside the image
RUN /root/.local/bin/uv venv /opt/venv

# Ensure tools + python from the venv are available without activation.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

WORKDIR /workspace

# Install Python dependencies once (keep requirements as its own layer for caching).
COPY requirements.txt /workspace/requirements.txt
RUN bash -lc "source /opt/venv/bin/activate && uv pip install -r /workspace/requirements.txt"

