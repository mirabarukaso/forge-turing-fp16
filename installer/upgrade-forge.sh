#!/usr/bin/env bash
set -euo pipefail

INSTALLER_DIR=$(cd "$(dirname "$0")" && pwd)
EXTENSION_ROOT=$(cd "$INSTALLER_DIR/.." && pwd)
FORGE_ROOT=""
DEVICE_ID=""
USE_UV=0
PULL=0
APPLY=0

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
        --pull)
            PULL=1
            shift
            ;;
        --apply)
            APPLY=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: upgrade-forge.sh [--forge-root DIR] [--device-id N] [--uv] [--pull] [--apply]" >&2
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
    echo "Pass --forge-root. Auto-detect only works when this extension lives in extensions/forge-turing-fp16." >&2
    exit 2
fi

FORGE_ROOT=$(cd "$FORGE_ROOT" && pwd)
VENV_PYTHON="$FORGE_ROOT/venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Missing $VENV_PYTHON. Create the virtualenv with install-turing.sh before upgrading." >&2
    exit 1
fi

if [[ "$PULL" -eq 1 ]]; then
    echo "git pull --ff-only in $FORGE_ROOT"
    git -C "$FORGE_ROOT" pull --ff-only
fi

ARGS=("$INSTALLER_DIR/forge_plan.py" --forge-root "$FORGE_ROOT")
if [[ "$USE_UV" -eq 1 ]]; then
    ARGS+=(--uv)
fi
if [[ "$APPLY" -eq 1 ]]; then
    ARGS+=(--apply)
fi
if [[ -n "$DEVICE_ID" ]]; then
    ARGS+=(--device-id "$DEVICE_ID")
fi

exec "$VENV_PYTHON" "${ARGS[@]}"
