# Phase 0 — Foundations

| | |
|---|---|
| **Status** | Implemented (approved 2026-10-22 with changes: Docker only, see P0-D12) |
| **Phase** | 0 (docs/SPEC.md §7) |
| **Owner** | Callum Little |
| **Last updated** | 2026-10-22 |
| **Branch** | `feature/cl-Phase0Foundations` |

## 1. Goal

Set up everything later phases build on: an installable Python 3.12 package
with an `isolate` command, a Poetry-managed environment inside the dev
container, CI that lints, tests and builds the dev container, and a recorded
licence check for the pretrained models. Everything runs in Docker; the host
only needs Docker and a way to start the container.

**Exit criteria (SPEC §7):** `pytest` green in CI; `isolate --version` works
in the dev container.

## 2. Scope

### In scope

| Item | Traces to |
|---|---|
| Package skeleton `src/isolate/` (core, models, eval, app, cli) with an `isolate` console script | SPEC §5.1, §7 Phase 0 |
| Python `>=3.12,<3.13` everywhere | SPEC §5, §11 (2026-10-12) |
| Poetry project: `pyproject.toml`, dev group, Ruff and pytest config, committed `poetry.lock` | SPEC §7 Phase 0 |
| CI: Python job green and cached; dev container build job | SPEC §7 Phase 0, §9 |
| Every check (Ruff, pytest, markdownlint, git, `gh`) runs inside the dev container | SPEC §4.4, §11 (2026-10-22, Docker only) |
| Licence check for `htdemucs_6s`, `htdemucs`, `htdemucs_ft` weights, recorded in `docs/models.md` | SPEC §4.1, §10 |
| Tests that run with no network access | SPEC §9 |
| Contributor docs: README, CLAUDE.md, the dev container README, this spec and the spec template | SPEC §7 (process) |

### Out of scope

- Any separation functionality, backends, registry, engine or `isolate`
  subcommands (`separate`, `app`, `eval`, `models`): Phase 1.
- Runtime dependencies `torch`, `torchaudio`, `demucs`, `soundfile`: Phase 1
  (see P0-D5).
- Coverage gate and type checking: Phase 1 (SPEC §7, decided 2026-10-22).
- `configs/` and `spaces/` directories from §5.1: created when they get
  content (Phase 1 and Phase 5).
- Forwarding the app's port 7860 from the container: Phase 2, when the app
  exists.
- Licence check for RoFormer checkpoints: Phase 7, when one is considered.
- Dependabot for Python dependencies, and the drift between
  `.devcontainer/Dockerfile` / `docker-compose.yml` and `devcontainer.json`
  (risk R6).

## 3. Requirements and acceptance criteria

| ID | Requirement | Acceptance (how it is verified) | Traces to |
|---|---|---|---|
| P0-R1 | The package `isolate` lives in `src/isolate/` with subpackages `core`, `models`, `eval`, `app` and a module `cli`. Each imports without error and has a one-line docstring saying what it will hold. No Phase 1 functionality. | `tests/test_package.py::test_module_imports` (parametrised over all five) passes. | SPEC §5.1 |
| P0-R2 | `isolate.__version__` equals the installed distribution's version. The version is defined once, in `pyproject.toml`. | `tests/test_package.py::test_version_matches_metadata` passes. | SPEC §7 Phase 0 |
| P0-R3 | An `isolate` console script exists. `isolate --version` prints `isolate <version>` to stdout and exits 0. `python -m isolate --version` behaves the same. `isolate` with no arguments prints usage and exits 0. An unknown option exits 2 with an error on stderr. | `tests/test_cli.py`: in-process tests of `main()`, plus subprocess tests that run the installed `isolate` script and `python -m isolate`. | SPEC §3 W1, §5.1, §7 Phase 0 |
| P0-R4 | The package declares `requires-python = ">=3.12,<3.13"`. | `tests/test_package.py::test_requires_python` reads it from installed metadata. Poetry refuses to install under any other Python. | SPEC §5, §11 |
| P0-R5 | `pyproject.toml` uses PEP 621 `[project]` metadata with the `poetry-core` build backend. It has no runtime dependencies. A `dev` group contains at least ruff, pytest, pytest-cov and pytest-socket. `poetry.lock` is committed and matches `pyproject.toml`. | CI step `poetry check --lock` passes. `poetry show --only main` lists nothing. | SPEC §7 Phase 0 |
| P0-R6 | Ruff is configured in `pyproject.toml`: line length 88, target `py312`, lint rules `E, W, F, I, B, UP, SIM, RUF, PT`, `src = ["src", "tests"]`. The whole repo passes lint and format checks. | CI steps `ruff check .` and `ruff format --check .` pass. | SPEC §9 |
| P0-R7 | pytest is configured in `pyproject.toml`: `testpaths = ["tests"]`, `--strict-markers --strict-config -ra`, coverage of `isolate` reported (term-missing, no threshold), and network sockets disabled for every test (pytest-socket, Unix sockets allowed). No test downloads model weights, and no weights are committed. | `tests/test_network.py::test_network_is_blocked` passes (a TCP connect raises `SocketBlockedError`). `pytest` in CI exits 0. | SPEC §9, §5.3 R8 |
| P0-R8 | The `python` CI job runs on every PR and every push to `main`. It installs a pinned Poetry version, caches dependencies keyed on `poetry.lock`, and runs `poetry check --lock`, Ruff lint, Ruff format check and pytest. The "detect pyproject" step is removed. | Job green on the Phase 0 PR. Warm-cache run time recorded in §7 (target: under 2 minutes). | SPEC §7 Phase 0, §9 |
| P0-R9 | A dev container CI workflow builds the container from `.devcontainer/devcontainer.json` with the Dev Containers CLI, then runs `isolate --version` and `pytest` inside it. It runs on PRs and pushes to `main` that change `.devcontainer/**`, `pyproject.toml`, `poetry.lock` or the workflow file itself. | Workflow green on the Phase 0 PR (which changes `pyproject.toml`). | SPEC §7 Phase 0, §4.4 |
| P0-R10 | In the dev container, `isolate --version` works without `poetry run` (the project venv's `bin` is on `PATH`). The container installs the same pinned Poetry version as CI. | `npx @devcontainers/cli exec --workspace-folder . isolate --version` prints `isolate 0.1.0`; also covered by P0-R9. | SPEC §7 Phase 0 exit criteria |
| P0-R11 | Every local check runs inside the dev container: Ruff, pytest, markdownlint, git and `gh`. The host needs only Docker Desktop plus VS Code or Node (for the Dev Containers CLI). Nothing else is installed on the host. | `npx @devcontainers/cli exec --workspace-folder . npx markdownlint-cli2 "**/*.md"` runs and passes (needs Node in the container). README's setup section lists only those host prerequisites. | SPEC §4.4, §11 (2026-10-22) |
| P0-R12 | `docs/models.md` records, for `htdemucs_6s`, `htdemucs` and `htdemucs_ft`: code licence and weights licence (separately), with source links, plus each model's checkpoint files and download location. SPEC §4.1's licence column and §10's licence risk are updated to match. | File exists. Markdown lint and link check pass in the `docs` CI job. `tests/test_docs.py::test_models_doc_covers_phase_models` checks all three model names are present. | SPEC §4.1, §10 |
| P0-R13 | README.md explains setup (dev container only) and how to run every CI check locally in the container. `CLAUDE.md` states the working conventions: spec-driven process, branch naming `feature/cl-<Name>`, commit style, how to run checks, everything (including git and `gh`) runs in the dev container, no native host installs. | Markdown lint and link check pass. Reviewed at the PR. | SPEC §4.4, §7 |

## 4. Design decisions

| ID | Decision | Alternatives considered | Reason |
|---|---|---|---|
| P0-D1 | PEP 621 `[project]` table with the `poetry-core` backend (Poetry 2.x). | Legacy `[tool.poetry]` metadata; Hatch or uv as the project tool. | Standard metadata any tool can read, while keeping Poetry as the project tool. |
| P0-D2 | Distribution name `instrument-isolation-app`; import package and command `isolate`. | Distribution named `isolate`. | `isolate` is already taken on PyPI (fal-ai), so a distinct distribution name avoids confusion in metadata and lock files. |
| P0-D3 | Version is static in `pyproject.toml` (`0.1.0`) and read at runtime with `importlib.metadata`. | Hard-coded `__version__` in `__init__.py`; a dynamic-versioning plugin driven by git tags. | One source of truth and no plugin. Can move to tag-driven versions at Phase 5 (CI deploy on tags). |
| P0-D4 | CLI built on `argparse`. | Typer; Click. | Zero runtime dependencies in Phase 0, and enough for subcommands. Revisit in Phase 1 if the subcommands get awkward. |
| P0-D5 | No runtime dependencies in Phase 0; `torch` and `demucs` arrive in Phase 1. | Add them now to prove the heavy install works in the container. | Nothing imports them yet, and they would add several minutes and ~1 GB to every CI run. Phase 1 adds them with CPU wheels. |
| P0-D6 | Ruff line length 88 with a curated rule set (`E, W, F, I, B, UP, SIM, RUF, PT`). | Length 100/120; `select = ["ALL"]` with ignores. | Ruff's default width; the set catches real bugs without noise. |
| P0-D7 | Network blocked in every test with pytest-socket (`--disable-socket --allow-unix-socket`). | Rely on convention; run CI without network. | Enforces "no network in tests" mechanically. Phase 1's real-weights jobs opt out per test with a marker. |
| P0-D8 | Coverage reported, but no gate; no type checker. Both move to Phase 1. | A 90% gate and mypy from day one. | The package is near-empty, so a gate means nothing yet (owner decision, 2026-10-22). |
| P0-D9 | CI installs a pinned Poetry with `pipx`, then `actions/setup-python` with `cache: poetry`. | Manual `actions/cache` on `.venv`; third-party `install-poetry` action. | First-party actions, cache keyed on `poetry.lock`, fewest moving parts. |
| P0-D10 | Dev container check is its own workflow (`devcontainer.yml`) with a native `paths:` filter. It runs `npx @devcontainers/cli up`, then `exec` for the checks. | A job in `ci.yml` behind `dorny/paths-filter`; `devcontainers/ci` action. | Native path filtering with no third-party action, and the same CLI developers use locally. |
| P0-D11 | Add `${containerWorkspaceFolder}/.venv/bin` to `PATH` in `devcontainer.json` `remoteEnv`. | Always type `poetry run isolate`. | Meets the exit criterion literally and matches how CI's tools are invoked. |
| P0-D12 | **Docker only** (owner decision, 2026-10-22): no Python, venv or other tooling on the host. GPU (MPS) is given up; all inference is CPU. SPEC §4.4, §6.3, N1 and §11 updated. | Native host Python 3.12 via uv for MPS work (the draft plan). | A reproducible, host-independent setup matters more than speed for a learning/portfolio piece. |
| P0-D13 | Poetry pinned to `2.5.1` (current latest) in `ci.yml` and `post-create.sh`. | Unpinned latest everywhere. | The same Poetry everywhere means the same lock-file behaviour. Dependabot does not track it, so bump by hand. |
| P0-D14 | Add the `ghcr.io/devcontainers/features/node:1` feature so `markdownlint-cli2` runs in the container. | Run markdownlint on the host; a Python Markdown linter (pymarkdown). | Required by Docker-only (P0-D12), and keeps the same linter and config CI uses. |

## 5. Task breakdown

| # | Task | Requirements |
|---|---|---|
| 1 | Phase spec, spec template, `docs/models.md`; SPEC v0.4 (Docker only, licence findings, Phase 1 gains coverage gate and type checking). | P0-R12, P0-R11 |
| 2 | `pyproject.toml` (metadata, dev group, Ruff, pytest config), `poetry.lock`, `src/isolate/` skeleton with docstrings and `__version__`, `tests/test_package.py`. | P0-R1, P0-R2, P0-R4, P0-R5, P0-R6, P0-R7 |
| 3 | `isolate.cli.main` (argparse, `--version`, usage), `__main__.py`, console script, `tests/test_cli.py`. | P0-R3 |
| 4 | Network guard test `tests/test_network.py` and docs test `tests/test_docs.py`. | P0-R7, P0-R12 |
| 5 | Dev container: `.venv/bin` on `PATH`, pinned Poetry in `post-create.sh`, Node feature; rebuild and verify `isolate --version` and markdownlint in the container. | P0-R10, P0-R11 |
| 6 | CI `python` job: remove detect step, pin Poetry, cache, `poetry check --lock`. | P0-R8 |
| 7 | `devcontainer.yml` workflow. | P0-R9 |
| 8 | README, `CLAUDE.md` and `.devcontainer/README.md` (Docker only). | P0-R11, P0-R13 |
| 9 | Traceability table (§7), push, open the PR. | All |

## 6. Risks and open questions

### Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Demucs weights are "for research/scientific purposes only" per the maintainer, and not MIT (see `docs/models.md`). | Hosting them in the public demo is outside the stated terms. | **Accepted** by the owner for this non-commercial learning/portfolio piece (SPEC §11). Weights are never committed or redistributed. |
| R2 | `devcontainer.json` bind-mounts `/run/host-services/ssh-auth.sock` and `~/.cache`, which don't exist on a Linux CI runner. | `devcontainer up` fails in CI. | Create both mount sources on the runner before `up` (the runner has passwordless `sudo`). `post-create.sh` already skips the socket `chmod` when it isn't a socket. |
| R3 | `post-create.sh` fetches `api.github.com/meta` and installs deps, so the dev container workflow depends on the network and takes a few minutes. | Slow or flaky job. | Path filter keeps it off most PRs. Retry once on failure if it proves flaky. |
| R4 | Import name `isolate` clashes with fal-ai's `isolate` package if both are installed in one environment. | Confusing import errors. | Distinct distribution name (P0-D2); the project never installs fal's package. |
| R5 | CPU-only inference (P0-D12). | Local separation is slower than on MPS; the old N1 target (≤ 60 s) is likely out of reach. | N1 target re-set from the Phase 1 measurement; speed work (SPEC §4.3) now benefits local use too. |
| R6 | `.devcontainer/Dockerfile` and `docker-compose.yml` are not used by `devcontainer.json`, install unpinned Poetry and CPU torch, and will drift. | Option 3 in the container README behaves differently. | Out of scope here. Proposed follow-up: delete them or make `devcontainer.json` build from the Dockerfile. |
| R7 | The local app (Phase 2) runs in the container, so it can't open the host browser, "show in Finder" (A7) is impossible, and binding to `127.0.0.1` inside the container would make it unreachable. | W2 changes slightly. | SPEC §6.3 and A7 updated: print the URL, publish the port on the host's loopback only, export to a bind-mounted host folder. Designed in Phase 2. |

### Open questions

None. Resolved on 2026-10-22: licence accepted as-is (R1); Docker only
(P0-D12); coverage gate and type checking deferred to Phase 1 (P0-D8);
SPEC v0.3 merged to `main` before branching.

## 7. Traceability

| ID | Implementation | Verification | Status |
|---|---|---|---|
| P0-R1 | `src/isolate/{core,models,eval,app}/__init__.py`, `src/isolate/cli.py` | `tests/test_package.py::test_module_imports` | Met |
| P0-R2 | `src/isolate/__init__.py` (`importlib.metadata`), `pyproject.toml` `version` | `tests/test_package.py::test_version_matches_metadata` | Met |
| P0-R3 | `src/isolate/cli.py`, `src/isolate/__main__.py`, `[project.scripts]` | `tests/test_cli.py` (in-process and subprocess) | Met |
| P0-R4 | `pyproject.toml` `requires-python` | `tests/test_package.py::test_requires_python` | Met |
| P0-R5 | `pyproject.toml`, `poetry.lock`, `poetry.toml` | CI `poetry check --lock`; `poetry show --only main` is empty | Met |
| P0-R6 | `pyproject.toml` `[tool.ruff]` | CI Ruff lint and format steps | Met |
| P0-R7 | `pyproject.toml` `[tool.pytest.ini_options]` | `tests/test_network.py::test_network_is_blocked`; CI pytest | Met |
| P0-R8 | `.github/workflows/ci.yml` `python` job | PR #8: `Python (lint, format, tests)` green in 20 s on a cold cache ([run](https://github.com/calluml333/instrument-isolation-app/actions/runs/39802038623)) | Met |
| P0-R9 | `.github/workflows/devcontainer.yml` | PR #8: `Build and test in dev container` green in 1 m 14 s ([run](https://github.com/calluml333/instrument-isolation-app/actions/runs/39802038615)) | Met |
| P0-R10 | `.devcontainer/devcontainer.json` `remoteEnv.PATH`, `.devcontainer/post-create.sh` | `npx @devcontainers/cli exec --workspace-folder . isolate --version` → `isolate 0.1.0` (local); P0-R9 workflow | Met |
| P0-R11 | Node feature in `devcontainer.json`; README setup | markdownlint-cli2 run in the container: 0 issues; no host installs | Met |
| P0-R12 | `docs/models.md`, SPEC §4.1, §10 | `tests/test_docs.py`; CI `docs` job (lint + links) | Met |
| P0-R13 | `README.md`, `CLAUDE.md`, `.devcontainer/README.md` | CI `docs` job; review on the PR | Met |

### Divergences from the approved draft

- `poetry.toml` added (in-project venv for everyone), because the container
  had put the venv under the bind-mounted `~/.cache`.
- `filterwarnings = ["error"]` added to the pytest config.
- `post-create.sh` calls `/usr/local/bin/python` explicitly: with `.venv/bin`
  first on `PATH`, `pip install --user` would otherwise run in the venv and
  fail.
