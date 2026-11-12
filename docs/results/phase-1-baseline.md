# Phase 1 baseline: `htdemucs_6s`

The starting point every later model, preset and optimisation is compared
against (SPEC §8). Measured on 2026-11-12 in the dev container on the M4 Pro,
on CPU (Docker can't use the Apple GPU, SPEC §4.4).

## Quality: MUSDB18 7-second samples, test split (50 tracks)

Median per stem, in dB (higher is better). Full reports, with per-track
scores in JSON, are in [phase-1/](phase-1/).

| Stem | SDR, `fast` | SDR, `best` | SI-SDR, `fast` | SI-SDR, `best` |
|---|---|---|---|---|
| `drums` | 9.23 | 9.21 | 8.69 | 8.75 |
| `bass` | 8.48 | 8.58 | 7.51 | 7.73 |
| `other` | 5.30 | 5.45 | 3.75 | 3.74 |
| `vocals` | 8.67 | 8.83 | 7.99 | 8.11 |

- **`other` includes the model's `guitar` and `piano`.** MUSDB18 has no
  guitar or piano references, so those two stems get a listening test in
  Phase 3 instead ([template](listening-test-template.md)).
- **These numbers are only comparable with other runs on this sample set.**
  The clips are short and compressed, so the scores aren't comparable with
  published MUSDB18-HQ results.
- **`best` (2 time shifts) gains 0.0–0.2 dB over `fast`** but nearly doubles
  the time (below). Phase 3 decides whether that's worth it (SPEC M3).

## Speed and memory

`isolate bench` on a synthetic stereo song at 44.1 kHz. Separation time
excludes the model load (0.2 s, weights already cached).

| Song length | Preset | Separation | Real-time factor | Peak memory (RSS) |
|---|---|---|---|---|
| 3 min | `fast` | 45.4 s | 0.25 | 2.43 GB |
| 3 min | `best` | 85.0 s | 0.47 | 2.55 GB |
| 10 min | `fast` | 125.6 s | 0.21 | 5.54 GB |
| 20 min | `fast` | killed by the OOM killer at the container's 7.8 GB limit | – | > 7.8 GB |

- **N1 (local speed):** a 3-minute song takes 45 s with `fast`. The proposed
  target is **≤ 60 s** (the measurement plus about 30% headroom), with a
  stretch target of **≤ 30 s** for the Phase 4 optimisation work.
- **N3 (memory ≤ 4 GB, any length): met for songs up to about 6½ minutes,
  not beyond.** Peak memory grows by about 0.44 GB per minute of audio, from
  about 1.1 GB, because Phase 1 holds the whole song and every stem in memory
  (phase spec P1-D8). A 20-minute song needs about 10 GB. Constant-memory
  streaming moves to Phase 4.

## Environment

| | |
|---|---|
| **Machine** | Apple M4 Pro, Docker Desktop: 12 CPUs, 7.8 GB memory for the container |
| **Torch threads** | 12 |
| **Versions** | Python 3.12.15, isolate 0.1.0, torch 2.15.0+cpu, demucs 4.1.0, numpy 2.5.4, museval 0.4.1 |

## Reproducing

Inside the dev container:

```bash
isolate eval --preset fast --out docs/results/phase-1/eval-htdemucs_6s-fast.md
```

```bash
isolate eval --preset best --out docs/results/phase-1/eval-htdemucs_6s-best.md
```

```bash
isolate bench --preset fast --duration 180 --out docs/results/phase-1/bench-htdemucs_6s-fast-180s.json
```

Repeat the benchmark with `--preset best`, and with `--duration 600` or
`1200` for the memory figures. Each `isolate bench` run is its own process,
so its peak memory is the measurement.
