# Phase N — {Name}

<!--
Copy this file to docs/specs/phase-N-{slug}.md and fill it in before writing
any implementation code. docs/SPEC.md is the source of truth: every in-scope
item traces to a section of it. If implementation shows this spec (or
docs/SPEC.md) is wrong, update it in the same commit as the code.
-->

| | |
|---|---|
| **Status** | Draft / Approved / Implemented |
| **Phase** | N (docs/SPEC.md §7) |
| **Owner** | Callum Little |
| **Last updated** | YYYY-MM-DD |
| **Branch** | `feature/cl-{Name}` |

## 1. Goal

One paragraph: what this phase delivers and its exit criteria, quoted from
docs/SPEC.md §7.

## 2. Scope

### In scope

| Item | Traces to |
|---|---|
| … | SPEC §x.y |

### Out of scope

- … (and which phase picks it up)

## 3. Requirements and acceptance criteria

IDs are stable: never renumber or reuse one. A dropped requirement stays in
the table with status **Dropped** and a reason.

| ID | Requirement | Acceptance (how it is verified) | Traces to |
|---|---|---|---|
| PN-R1 | Specific, testable statement. | Test name, CI job, or command plus expected output. | SPEC §x.y |

## 4. Design decisions

| ID | Decision | Alternatives considered | Reason |
|---|---|---|---|
| PN-D1 | … | … | One line. |

## 5. Task breakdown

Ordered. Each task is roughly one commit, and its commit message references
the requirement IDs it covers.

| # | Task | Requirements |
|---|---|---|
| 1 | … | PN-R1 |

## 6. Risks and open questions

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| … | … | … |

### Open questions

1. …

## 7. Traceability

Filled in at the end of the phase.

| ID | Implementation | Verification | Status |
|---|---|---|---|
| PN-R1 | `path/to/file.py` | `tests/test_x.py::test_y`, CI job | Met / Partly met / Deferred |
