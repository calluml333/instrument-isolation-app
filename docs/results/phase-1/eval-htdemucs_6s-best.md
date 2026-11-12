# Evaluation: `htdemucs_6s`, preset `best`

| | |
|---|---|
| **Model** | `htdemucs_6s` |
| **Preset** | `best` (overlap 0.25, shifts 2, seed 0) |
| **Dataset** | MUSDB18 7-second samples, test split (50 tracks) |
| **Date** | 2026-11-12 |
| **Wall time** | 367 s |
| **Versions** | python 3.12.15, isolate 0.1.0, torch 2.15.0+cpu, demucs 4.1.0, numpy 2.5.4, museval 0.4.1 |

| Stem | SDR (dB, median) | SI-SDR (dB, median) | Tracks scored |
|---|---|---|---|
| `drums` | 9.21 | 8.75 | 50 |
| `bass` | 8.58 | 7.73 | 50 |
| `other` | 5.45 | 3.74 | 50 |
| `vocals` | 8.83 | 8.11 | 50 |

SDR is BSSEval v4 (museval, 1 s windows); SI-SDR is over the whole clip. Both are medians over tracks. The model's `guitar` and `piano` are added to `other` before scoring, because MUSDB18 has no separate guitar or piano reference. These scores are only comparable with other runs on this sample set, not with published MUSDB18-HQ results (SPEC §8).
