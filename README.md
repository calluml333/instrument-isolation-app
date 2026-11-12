# instrument-isolation-app

App to perform source separation (isolating drums, vocals, bass, etc.) from a mixed track

> **Portfolio project, not for sale.** This is a non-commercial portfolio
> piece built on pretrained separation models. It is not sold or distributed
> commercially.

See [docs/SPEC.md](docs/SPEC.md) for the full project specification, and
[docs/models.md](docs/models.md) for the models used and their licences.

## Status

Phase 1 (baseline): `isolate separate` produces six stems with
`htdemucs_6s`, and `isolate eval` and `isolate bench` measure it (see Results
below). Next is the local web app (Phase 2). See the roadmap in
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

**Long songs:** the whole song is held in memory for now, so peak memory
grows by about 0.44 GB per minute of audio. Up to about 6 minutes stays
under 4 GB. Much longer files can exceed the memory Docker Desktop gives the
container (Settings → Resources); constant-memory streaming is planned for
Phase 4.

## Results

`htdemucs_6s` on the MUSDB18 7-second sample set (50 test tracks), median
SDR in dB, with timings for a 3-minute song on CPU in the dev container (M4
Pro). Details and method: [docs/results/phase-1-baseline.md](docs/results/phase-1-baseline.md).

| Preset | `drums` | `bass` | `other` | `vocals` | 3-minute song | Peak memory |
|---|---|---|---|---|---|---|
| `fast` | 9.23 | 8.48 | 5.30 | 8.67 | 45 s | 2.4 GB |
| `best` | 9.21 | 8.58 | 5.45 | 8.83 | 85 s | 2.5 GB |

`other` includes the model's `guitar` and `piano`, which MUSDB18 can't score
separately.

To score a model or time it yourself:

```bash
isolate eval --preset fast --out results.md
```

```bash
isolate bench --preset fast --duration 180
```

`isolate eval` downloads the MUSDB18 7-second sample set (147 MB, research
use only) into `~/.cache/isolate/datasets/` on first use.

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
