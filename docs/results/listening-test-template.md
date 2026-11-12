# Listening test: {models compared}

Copy this file to `docs/results/phase-N-listening-{topic}.md` and fill it in.
MUSDB18 has no guitar or piano references, so those stems are judged by ear
(SPEC §8). Use the same tracks every time, so runs stay comparable.

| | |
|---|---|
| **Date** | YYYY-MM-DD |
| **Listener** | |
| **Playback** | Headphones / monitors (model), level |
| **Models and presets** | e.g. `htdemucs_6s` `best`, `ensemble` `best` |
| **isolate version** | output of `isolate --version` |

## Tracks

About 10 tracks across genres, each CC-licensed or self-owned (never
committed if not redistributable). Note which stems each track actually has.

| # | Track | Genre | Licence / source | Has guitar | Has piano | Section used |
|---|---|---|---|---|---|---|
| 1 | | rock | | yes / no | yes / no | 0:30–1:00 |
| 2 | | pop | | | | |
| 3 | | jazz | | | | |
| 4 | | electronic | | | | |
| 5 | | acoustic | | | | |

## Scale

Every criterion is scored 1–5, where 5 is best.

| Criterion | 1 | 5 |
|---|---|---|
| **Bleed** | Other instruments clearly audible | None audible |
| **Phasiness** | Hollow, swirly or "underwater" | Natural |
| **Transient smearing** | Attacks soft or blurred | Sharp, as in the mix |
| **Overall** | Unusable | Could be used in a mix as-is |

## Scores

One row per track, model and stem. Score `guitar` and `piano` for every
track that has them; `vocals`, `drums` and `bass` are optional here (they
have numeric scores), but worth noting when something stands out.

| # | Model / preset | Stem | Bleed | Phasiness | Transients | Overall | Notes |
|---|---|---|---|---|---|---|---|
| 1 | | guitar | | | | | |
| 1 | | piano | | | | | |

## Summary

Mean `Overall` per model and stem, and which model each stem should come
from (SPEC M3).

| Stem | Model A | Model B | Pick | Why |
|---|---|---|---|---|
| `guitar` | | | | |
| `piano` | | | | |
