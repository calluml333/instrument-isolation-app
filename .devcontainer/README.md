# Dev Container Setup

The dev container is for **linting, tests, data prep, git and `gh`**, run the
same way CI runs them. Anything that needs the GPU (fine-tuning, running the
app at full speed) must run **natively on the Mac**, because Docker on macOS
cannot access the Apple GPU (see [docs/SPEC.md](../docs/SPEC.md) §4.4).

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) (required
  for SSH agent forwarding, see below)
- Either [VS Code](https://code.visualstudio.com/) with the
  [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers),
  or Node.js for the Dev Containers CLI
- A git identity on the host (`git config --global user.name` / `user.email`)
- Your GitHub SSH key loaded in the host's SSH agent (`ssh-add -l` lists it)

## Option 1: VS Code

Press `Cmd/Ctrl + Shift + P` → **Dev Containers: Reopen in Container**. After
changing anything in `.devcontainer/`, use **Dev Containers: Rebuild
Container**.

## Option 2: Dev Containers CLI (terminal, or Claude Code)

```bash
npx @devcontainers/cli up --workspace-folder .
```

```bash
npx @devcontainers/cli exec --workspace-folder . bash
```

`exec` also runs single commands, e.g. `... exec --workspace-folder . git status`.

## Option 3: Docker Compose

Uses [Dockerfile](Dockerfile), which also installs CPU-only PyTorch into the
image. It doesn't include the git/SSH/`gh` setup below.

```bash
docker compose -f .devcontainer/docker-compose.yml up -d --build
```

```bash
docker compose -f .devcontainer/docker-compose.yml exec dev bash
```

## Git and GitHub inside the container

Set up automatically on container creation by
[initialize.sh](initialize.sh) (runs on the host) and
[post-create.sh](post-create.sh) (runs in the container):

| What | How |
|---|---|
| **Git identity** | Your host's `user.name` / `user.email` are exported to `.devcontainer/.gitconfig.host` (gitignored) and included by the container's git config. Nothing else from the host config is copied. |
| **`git push` over SSH** | Docker Desktop forwards the host's SSH agent to `/ssh-agent`. Your private key never enters the container. GitHub's host keys are added to `known_hosts` from `api.github.com/meta`. |
| **`gh`** | Installed by the `github-cli` feature. Its config lives in the `instrument-isolation-app-gh` Docker volume, so you log in once and it survives rebuilds. |

**One-time step:** log in to `gh` inside the container:

```bash
gh auth login
```

Check that everything works:

```bash
git config user.email && ssh -T git@github.com; gh auth status
```

## What's included

- **Python 3.12** and **Poetry**
- **Git** and the **GitHub CLI**
- **VS Code extensions:** Python, Pylance, debugpy, Ruff (formatter and
  linter), GitLens, Jupyter, markdownlint, Claude Code
- **Project dependencies:** installed with `poetry install --with dev` once
  `pyproject.toml` exists (roadmap Phase 0). Until then the step is skipped.

## Common commands

These match the checks in [CI](../.github/workflows/ci.yml):

```bash
poetry run ruff check .
```

```bash
poetry run ruff format --check .
```

```bash
poetry run pytest
```

```bash
npx markdownlint-cli2 "**/*.md"
```

## Troubleshooting

**`git push` fails with "Permission denied (publickey)":**

- On the host, check that your key is loaded: `ssh-add -l`. If not:
  `ssh-add ~/.ssh/id_ed25519`.
- The agent forwarding mount is specific to Docker Desktop for Mac. Other
  Docker setups need a different `source` for the `/ssh-agent` mount in
  [devcontainer.json](devcontainer.json).

**Commits have the wrong author:** set your identity on the host, then
rebuild the container.

**Container build fails:**

- Ensure Docker has enough disk space.
- Try `docker system prune` to clean up.

## Resources

- [Dev Containers documentation](https://containers.dev/)
- [Dev Containers CLI](https://github.com/devcontainers/cli)
- [Docker Desktop SSH agent forwarding](https://docs.docker.com/desktop/features/networking/#ssh-agent-forwarding)
