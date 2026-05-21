#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="/workspace"

IMAGE_NAME="es-emergent-misalignemt:local"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found on PATH" >&2
  exit 1
fi

if [ ! -f "${SCRIPT_DIR}/.hftoken" ]; then
  echo "Error: missing ${SCRIPT_DIR}/.hftoken" >&2
  exit 1
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"
fi

docker run --rm --init \
  -v "${SCRIPT_DIR}:${WORK_DIR}" \
  -w "${WORK_DIR}" \
  --gpus all \
  "${IMAGE_NAME}" \
  bash -lc "export HF_TOKEN=\"\$(cat .hftoken)\" && export UV_PROJECT_ENVIRONMENT=/opt/venv && source /opt/venv/bin/activate && exec bash run.sh"
