#!/usr/bin/env bash
#
# Publish the ZeroGPU Gradio Space — the one that runs real Gemma 3n.
#
# ZeroGPU is the only GPU a free Hugging Face account can host on, and it is
# Gradio-only: a Docker Space cannot use it. So this Space is a Gradio front end
# over the same coaching brain (`src/aura`), which is vendored in here rather
# than pip-installed so the Space has no dependency on a published package.
#
# Only generation runs on the GPU. The crisis screen, affect estimate, topic
# graph and prompt assembly run in the Space's main process — see app.py.
#
# Usage:
#   deploy/spaces-gpu/push.sh [<username>/<space-name>]
#
# Requires `hf auth login` (or HF_TOKEN). Create the Space first with:
#   hf repos create <user>/<name> --type space --space-sdk gradio \
#       --flavor zero-a10g --public

set -euo pipefail

SPACE="${1:-}"
if [[ -z "$SPACE" || "$SPACE" != */* ]]; then
  echo "usage: $0 <username>/<space-name>" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "→ staging the Space"
cp "$HERE/app.py" "$HERE/requirements.txt" "$HERE/README.md" "$STAGE/"

# Vendor only the torch-free brain. The HTTP layer, the engines and the media
# pipeline belong to the FastAPI service; app.py owns generation here.
mkdir -p "$STAGE/aura"
for module in __init__ affect config logging memory prompts safety schemas session; do
  cp "$REPO_ROOT/src/aura/$module.py" "$STAGE/aura/"
done

# The package docstring advertises a public surface this copy does not ship.
cat > "$STAGE/aura/__init__.py" <<'PY'
"""Aura — the coaching brain, vendored into this Space.

The same code the FastAPI service runs, minus the parts a Gradio Space does not
need (the HTTP layer, the engines, the media pipeline). Nothing here imports
torch, which is why it can run in the Space's main process while the GPU worker
only generates tokens.

Source: https://github.com/Arnavdsp/Gemma-3n-Hackathon
"""

from aura.config import Settings, get_settings

__all__ = ["Settings", "__version__", "get_settings"]
__version__ = "1.0.0"
PY

python3 -m py_compile "$STAGE/app.py"

echo "→ uploading to $SPACE"
hf upload "$SPACE" "$STAGE" . --repo-type space --exclude "**/__pycache__/**"

echo "→ done: https://huggingface.co/spaces/${SPACE}"
echo "  live at https://${SPACE/\//-}.hf.space once the build finishes"
echo "  watch it with: hf spaces logs ${SPACE} --follow"
