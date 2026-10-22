#!/usr/bin/env bash
# Runs INSIDE the container once, after it is created (devcontainer.json
# "postCreateCommand"). Working directory is the workspace folder.
set -euo pipefail

workspace="$PWD"

# --- Git identity: include what initialize.sh exported from the host ---------
if [ -s "$workspace/.devcontainer/.gitconfig.host" ]; then
  git config --global include.path "$workspace/.devcontainer/.gitconfig.host"
fi
if ! git config --get user.email > /dev/null; then
  echo "⚠️  No git identity found. Set one on the host (git config --global" \
       "user.name/user.email) and rebuild, or set it inside the container."
fi

# --- SSH: trust GitHub's published host keys (fetched over verified TLS) -----
mkdir -p ~/.ssh && chmod 700 ~/.ssh
curl -fsSL https://api.github.com/meta \
  | python3 -c 'import json,sys; [print("github.com", k) for k in json.load(sys.stdin)["ssh_keys"]]' \
  >> ~/.ssh/known_hosts

# The forwarded agent socket is root-owned on Docker Desktop; let vscode use it.
if [ -S "${SSH_AUTH_SOCK:-}" ]; then
  sudo chmod 666 "$SSH_AUTH_SOCK"
fi

# --- GitHub CLI: config lives in a named volume so the login survives rebuilds -
sudo mkdir -p ~/.config/gh
sudo chown -R "$(id -u):$(id -g)" ~/.config/gh

# --- Python deps: pinned Poetry (keep in sync with ci.yml), venv in .venv ----
POETRY_VERSION=2.5.1
# The image's Python, not .venv's (remoteEnv puts .venv/bin first on PATH).
/usr/local/bin/python -m pip install --user --upgrade pip "poetry==$POETRY_VERSION"
export PATH="$HOME/.local/bin:$PATH"
poetry install --no-interaction --with dev
