#!/usr/bin/env bash
#
# Publish this repository to a Hugging Face Space.
#
# A Space is its own git repo whose README.md must carry Spaces front matter
# (sdk, app_port, …). Rather than put that YAML in the project README — where
# GitHub would render it as noise — this script assembles the Space from the
# tracked files plus deploy/huggingface/README.md.
#
# Usage:
#   deploy/huggingface/push.sh <username>/<space-name>
#
# Requires: git, and either `huggingface-cli login` beforehand or a HF token in
# HF_TOKEN. Create the Space first at https://huggingface.co/new-space with
# "Docker" as the SDK and "Blank" as the template.

set -euo pipefail

SPACE="${1:-}"
if [[ -z "$SPACE" || "$SPACE" != */* ]]; then
  echo "usage: $0 <username>/<space-name>" >&2
  exit 2
fi

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "→ staging tracked files from $REPO_ROOT"
git -C "$REPO_ROOT" archive HEAD | tar -x -C "$STAGE"

# The Space's README is the one with the front matter.
cp "$REPO_ROOT/deploy/huggingface/README.md" "$STAGE/README.md"

# Trim what the runtime does not need. Keeping the Space small keeps builds fast.
rm -rf "$STAGE/notebooks" "$STAGE/docs" "$STAGE/.github" "$STAGE/tests"

if [[ -n "${HF_TOKEN:-}" ]]; then
  REMOTE="https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}"
else
  REMOTE="https://huggingface.co/spaces/${SPACE}"
fi

echo "→ pushing to $SPACE"
cd "$STAGE"
git init -q
git add -A
git -c user.email=deploy@localhost -c user.name=deploy \
    commit -qm "Deploy Aura $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
git push -q --force "$REMOTE" HEAD:main

echo "→ done: https://huggingface.co/spaces/${SPACE}"
echo "  the app will be live at https://${SPACE/\//-}.hf.space once the build finishes"
