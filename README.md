# instrument-isolation-app

App to perform source separation (isolating drums, vocals, bass, etc.) from a mixed track

> **Portfolio project, not for sale.** This is a non-commercial portfolio
> piece built on pretrained separation models. It is not sold or distributed
> commercially.

See [docs/SPEC.md](docs/SPEC.md) for the full project specification, and
[docs/models.md](docs/models.md) for the models used and their licences.

## Status

Phase 1 (baseline), part 1a: `isolate separate` produces six stems with
`htdemucs_6s`. Measured results come with part 1b. See the roadmap in
[SPEC §7](docs/SPEC.md#7-phased-roadmap) and the phase specs in
[docs/specs/](docs/specs/).

## Setup

Everything runs in the **dev container** (Docker). Nothing is installed on
the host apart from what starts the container:

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Either [VS Code](https://code.visualstudio.com/) with the Dev Containers
  extension, or Node.js for the [Dev Containers CLI](https://github.com/devcontainers/cli)

Docker on macOS can't use the Apple GPU, so separation runs on CPU
([SPEC §4.4](docs/SPEC.md#44-compute)).

Start the container (or in VS Code: **Dev Containers: Reopen in Container**):

```bash
npx @devcontainers/cli up --workspace-folder .
```

Creating it installs Python 3.12, Poetry and the project (with dev tools) into
`.venv`, which is on the container's `PATH`. Check it works:

```bash
npx @devcontainers/cli exec --workspace-folder . isolate --version
```

Git, SSH and `gh` setup, and troubleshooting, are in
[.devcontainer/README.md](.devcontainer/README.md).

## Usage

Run these inside the container. Put your audio somewhere in the repo folder
(it's mounted into the container), e.g. a gitignored `audio/` folder.

Separate a song into `drums`, `bass`, `other`, `vocals`, `guitar` and
`piano`:

```bash
isolate separate audio/song.wav --out stems
```

This writes `stems/song/<stem>.wav`: 32-bit float WAVs with the same length,
sample rate and channel count as the input, so they line up exactly when
dropped into Reaper or Logic at the same start point. By default the stems
sum exactly to the input; anything the model missed goes into `other`.

Useful options (`isolate separate --help` lists them all):

| Option | What it does |
|---|---|
| `--preset fast` | No time-shift averaging: faster, slightly lower quality (default `best`) |
| `--stems vocals,drums` | Only write these stems |
| `--format flac` | 24-bit FLAC instead of WAV |
| `--bit-depth 24` | 24-bit PCM WAV instead of 32-bit float |
| `--no-residual` | Don't add the leftover signal to `other` |

The model's weights (55 MB) download on first use into
`~/.cache/isolate/models/`, which is your Mac's `~/.cache` mounted into the
container, so they survive rebuilds. To fetch them ahead of time, or see
what's available:

```bash
isolate models download htdemucs_6s
```

```bash
isolate models list
```

The weights are for research use only; see [docs/models.md](docs/models.md).

## Checks

The same checks CI runs. Run each inside the container, either from a shell in
it (`npx @devcontainers/cli exec --workspace-folder . bash`) or by prefixing
the command with `npx @devcontainers/cli exec --workspace-folder .`:

```bash
ruff check .
```

```bash
ruff format --check .
```

```bash
mypy
```

```bash
pytest
```

```bash
poetry check --lock
```

```bash
npx markdownlint-cli2 "**/*.md"
```

Tests never touch the network (pytest-socket blocks it), so they never
download model weights.

## Layout

```text
src/isolate/
  core/      separation engine (Phase 1)
  models/    model backends and registry (Phase 1)
  eval/      evaluation harness (Phase 1)
  app/       web app (Phase 2)
  cli.py     the `isolate` command
tests/
docs/        SPEC.md, phase specs, model licences
```

## Contributing

Work is spec-driven: each phase gets a spec in [docs/specs/](docs/specs/)
before any code. Conventions are in [CLAUDE.md](CLAUDE.md).
