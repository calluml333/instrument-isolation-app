# Evaluation: `htdemucs_6s`, preset `fast`

| | |
|---|---|
| **Model** | `htdemucs_6s` |
| **Preset** | `fast` (overlap 0.25, shifts 0, seed 0) |
| **Dataset** | MUSDB18 7-second samples, test split (50 tracks) |
| **Date** | 2026-11-12 |
| **Wall time** | 330 s |
| **Versions** | python 3.12.15, isolate 0.1.0, torch 2.15.0+cpu, demucs 4.1.0, numpy 2.5.4, museval 0.4.1 |

| Stem | SDR (dB, median) | SI-SDR (dB, median) | Tracks scored |
|---|---|---|---|
| `drums` | 9.23 | 8.69 | 50 |
| `bass` | 8.48 | 7.51 | 50 |
| `other` | 5.30 | 3.75 | 50 |
| `vocals` | 8.67 | 7.99 | 50 |

SDR is BSSEval v4 (museval, 1 s windows); SI-SDR is over the whole clip. Both are medians over tracks. The model's `guitar` and `piano` are added to `other` before scoring, because MUSDB18 has no separate guitar or piano reference. These scores are only comparable with other runs on this sample set, not with published MUSDB18-HQ results (SPEC §8).
