#!/usr/bin/env bash
set -euo pipefail

INSTALLER_DIR=$(cd "$(dirname "$0")" && pwd)
EXTENSION_ROOT=$(cd "$INSTALLER_DIR/.." && pwd)
FORGE_ROOT=""
DEVICE_ID=""
USE_UV=0
REQUIREMENTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --forge-root)
            FORGE_ROOT=$2
            shift 2
            ;;
        --device-id)
            DEVICE_ID=$2
            shift 2
            ;;
        --uv)
            USE_UV=1
            shift
            ;;
        --requirements)
            REQUIREMENTS=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: install-turing.sh [--forge-root DIR] [--device-id N] [--uv] [--requirements]" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$FORGE_ROOT" ]]; then
    MAYBE=$(cd "$EXTENSION_ROOT/../.." && pwd)
    if [[ -f "$MAYBE/webui.sh" ]]; then
        FORGE_ROOT=$MAYBE
    fi
fi

if [[ -z "$FORGE_ROOT" ]]; then
    echo "Pass --forge-root with the Forge Neo directory. Auto-detect only works when this extension lives in extensions/forge-turing-fp16." >&2
    exit 2
fi

FORGE_ROOT=$(cd "$FORGE_ROOT" && pwd)
VENV_PYTHON="$FORGE_ROOT/venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    if [[ "$USE_UV" -eq 1 ]]; then
        if ! command -v uv >/dev/null 2>&1; then
            echo "uv was requested but was not found on PATH." >&2
            exit 1
        fi
        echo "Creating uv venv at $FORGE_ROOT/venv (no pip module; later installs use uv pip --python)"
        uv venv --python python3 "$FORGE_ROOT/venv"
    else
        if ! command -v python3 >/dev/null 2>&1; then
            echo "python3 was not found on PATH, and venv/bin/python does not exist." >&2
            exit 1
        fi
        echo "Creating venv at $FORGE_ROOT/venv"
        python3 -m venv "$FORGE_ROOT/venv"
    fi
fi

ARGS=("$INSTALLER_DIR/turing_install.py" --forge-root "$FORGE_ROOT")
if [[ "$USE_UV" -eq 1 ]]; then
    ARGS+=(--uv)
fi
if [[ "$REQUIREMENTS" -eq 1 ]]; then
    ARGS+=(--requirements)
fi
if [[ -n "$DEVICE_ID" ]]; then
    ARGS+=(--device-id "$DEVICE_ID")
fi

exec "$VENV_PYTHON" "${ARGS[@]}"
