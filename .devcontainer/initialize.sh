#!/usr/bin/env sh
# Runs on the HOST before the container is created (devcontainer.json
# "initializeCommand"). Exports only the host's git identity into a gitignored
# file that post-create.sh includes inside the container. Credential helpers and
# other macOS-specific settings are deliberately not carried over.
set -eu

out="$(dirname "$0")/.gitconfig.host"
: > "$out"

name="$(git config --global --get user.name || true)"
email="$(git config --global --get user.email || true)"

if [ -n "$name" ] || [ -n "$email" ]; then
  echo "[user]" >> "$out"
  [ -n "$name" ] && echo "	name = $name" >> "$out"
  [ -n "$email" ] && echo "	email = $email" >> "$out"
fi
exit 0
