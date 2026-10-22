# CLAUDE.md

Working conventions for this repo. Project details live in
[docs/SPEC.md](docs/SPEC.md), which is the source of truth.

## Spec-driven process

1. Each phase in SPEC §7 gets a spec in `docs/specs/phase-N-<slug>.md`,
   copied from [docs/specs/_template.md](docs/specs/_template.md): scope
   traced to SPEC sections, requirements with stable IDs (`PN-R1`, …),
   acceptance criteria, design decisions, task breakdown, risks.
2. Stop after writing the phase spec and get the owner's approval before any
   implementation code.
3. Implement task by task, with tests alongside, roughly one commit per task.
4. If implementation shows a spec is wrong, update the phase spec (and
   `docs/SPEC.md` for project-level decisions) **in the same commit**, and
   say so. Never diverge silently.
5. Finish with the traceability table at the end of the phase spec, then a PR.

## Everything runs in Docker

Nothing is installed or run natively on the host: no host Python, venvs or
tools. Every command, including git and `gh`, runs in the dev container:

```bash
npx @devcontainers/cli exec --workspace-folder . <cmd>
```

Start it first with `npx @devcontainers/cli up --workspace-folder .`. The
container is CPU-only (Docker on macOS can't use the Apple GPU), so all
inference runs on CPU.

If git reports a "nonexistent object" right after writing one, retry: it is
a brief file-sync delay on the bind mount, not corruption.

## Checks

Run all of these in the container before pushing (they match CI):

```bash
poetry check --lock && ruff check . && ruff format --check . && pytest && npx markdownlint-cli2 "**/*.md"
```

- Tests must not use the network; pytest-socket blocks it. Never download
  or commit model weights.
- Warnings fail the test suite.

## Git

- Branch from up-to-date `main`: `feature/cl-<Name>` (PascalCase name, e.g.
  `feature/cl-Phase0Foundations`); docs-only work may use `docs/cl-<Name>`.
- Commit messages: `type: Capitalised summary` with type `feat`, `fix`,
  `test`, `docs`, `build`, `ci` or `refactor` (optional scope, e.g.
  `build(devcontainer): …`). The body says why. Reference the requirement
  IDs a commit covers on a trailer line: `Requirements: P0-R1, P0-R3`.
- PRs are opened with `gh` from inside the container, link the phase spec,
  list the requirements met, note anything deferred, and get labels.

## Tooling

- Python 3.12 only (`>=3.12,<3.13`). Poetry 2.5.1, pinned in
  `.github/workflows/ci.yml` and `.devcontainer/post-create.sh`; bump both
  together.
- The venv is in-project (`.venv`, set by `poetry.toml`) and on the
  container's `PATH`, so `isolate`, `ruff` and `pytest` work without
  `poetry run`.
- Ruff: line length 88; rules in `pyproject.toml`.
