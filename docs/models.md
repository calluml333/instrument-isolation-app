# Pretrained models and licences

Licence record for every pretrained model the project uses (SPEC §4.1).
Code and weights are licensed separately, so each is listed on its own.
Checked on 2026-10-22.

> **Not legal advice.** This records what the upstream sources say. The
> owner decides what the project does with it.

## Summary

| Model | Stems | Code licence | Weights licence | Weights can be… |
|---|---|---|---|---|
| `htdemucs_6s` | drums, bass, other, vocals, guitar, piano | MIT | **Research / scientific use only** (maintainer statement; no formal licence file) | Used locally for development and evaluation. Hosting in a public demo is unclear (see below). |
| `htdemucs` | drums, bass, other, vocals | MIT | Same as above | Same as above |
| `htdemucs_ft` | drums, bass, other, vocals | MIT | Same as above | Same as above |

## Code licence: MIT

- The `demucs` package is MIT-licensed:
  [LICENSE](https://github.com/facebookresearch/demucs/blob/main/LICENSE)
  (copyright Meta Platforms, Inc. and affiliates) and the
  [README's licence section](https://github.com/facebookresearch/demucs#license).
- The PyPI package declares `License: MIT`
  ([pypi.org/project/demucs](https://pypi.org/project/demucs/)).
- The original repository is no longer maintained. The author continues it at
  [adefossez/demucs](https://github.com/adefossez/demucs), also MIT.

## Weights licence: research use only

No licence file in the repository covers the weights. The README's licence
section only mentions the code. The only explicit statements come from the
author (Alexandre Défossez, `adefossez`) in GitHub issues:

| Source | What it says (paraphrased, with short quotes) |
|---|---|
| [facebookresearch/demucs#327, comment of 2022-05-23](https://github.com/facebookresearch/demucs/issues/327#issuecomment-1134828611) (issue: "License of pre-trained models") | The weights are "not covered by the MIT license" and are "provided only for scientific purposes". |
| [facebookresearch/demucs#267, comment of 2022-01-04](https://github.com/facebookresearch/demucs/issues/267#issuecomment-1004721776) | The MusDB dataset licence means the weights are mostly usable for research; a real app is outside those terms and not something Meta can officially support. |
| [facebookresearch/demucs#508, comment of 2023-08-02](https://github.com/facebookresearch/demucs/issues/508#issuecomment-1663033356) (issue: "Include Models in commercial products?") | Weights are provided for research purposes. |

**Why:** the models were trained on MUSDB18-HQ plus 800 extra songs (see the
[README](https://github.com/facebookresearch/demucs#readme)). MUSDB18 is
released for academic use only, and its tracks carry non-commercial
Creative Commons licences
([sigsep MUSDB18 page](https://sigsep.github.io/datasets/musdb.html)).

**Conflicting third-party labels:** some re-uploads of the weights on Hugging
Face are tagged `license: mit` (for example
[adefossez/HTDemucs](https://huggingface.co/adefossez/HTDemucs)). A
[discussion on that repo](https://huggingface.co/adefossez/HTDemucs/discussions/1)
asks whether the tag is intentional, citing the issues above, and has no
answer from the author. This project treats the author's explicit statements
as authoritative, not the tags.

## What this means for the project

| Use | Assessment |
|---|---|
| Downloading and running the weights locally for development, evaluation and the local app (Phases 1–4) | Fits "research / scientific purposes" for a non-commercial portfolio project. |
| Committing or redistributing the weights | Not done: weights are always fetched from the upstream URL at runtime (SPEC §5.3 R8). |
| Commercial use | Out of scope for the project anyway (SPEC §2 non-goals). |
| Running them in the **public Hugging Face Spaces demo** (Phase 5) | A public demo isn't obviously "scientific purposes". Accepted by the owner for this non-commercial learning/portfolio piece (SPEC §11, 2026-10-22). |

## Checkpoints

### What the project downloads

`isolate` downloads **safetensors** copies of the weights from the author's
Hugging Face repos, pinned to a commit, and checks each file's full SHA-256
(from the Hugging Face file metadata) before using it (SPEC R8, Phase 1
P1-D3). Unlike the original `.th` files, which are Python pickles, safetensors
can't run code when loaded. The pins live in the model registry
(`src/isolate/models/registry.py`).

| Model | Hugging Face repo | Pinned commit | File | SHA-256 | Size |
|---|---|---|---|---|---|
| `htdemucs_6s` | [adefossez/HTDemucs-6s](https://huggingface.co/adefossez/HTDemucs-6s) | `3c5ee475be622df764938de97e4281a7b07ffa58` | `5c90dfd2.safetensors` | `d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411` | 55 MB |
| `htdemucs` | [adefossez/HTDemucs](https://huggingface.co/adefossez/HTDemucs) | `cbc8a9b1a87023b7fd74e7b3412e6321c0eab003` | `955717e8.safetensors` | `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd` | 84 MB |
| `htdemucs_ft` | [adefossez/HTDemucs-ft](https://huggingface.co/adefossez/HTDemucs-ft) | `d74ac89c3a1e874fc78f152555cf4d8533f06cd4` | `f7e0c4bc`, `d12395a8`, `92cfc3b6`, `04573f0d` `.safetensors` | see the repo's file list | 4 × 84 MB |

Only `htdemucs_6s` is in the registry so far (Phase 1). The other two are
added in Phase 3.

### Original checkpoints

Meta's original `.th` checkpoints are hosted at
`https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/`, as listed in
the package's
[`remote/files.txt`](https://github.com/facebookresearch/demucs/blob/main/demucs/remote/files.txt).
The hash suffix in each file name is the start of the file's SHA-256, which
`demucs` checks on download (via `torch.hub` with `check_hash=True`). They
have the same signatures as the safetensors files above.

| Model | Bag definition | Checkpoint files | Size |
|---|---|---|---|
| `htdemucs_6s` | [`htdemucs_6s.yaml`](https://github.com/facebookresearch/demucs/blob/main/demucs/remote/htdemucs_6s.yaml) | `5c90dfd2-34c22ccb.th` | 55 MB |
| `htdemucs` | [`htdemucs.yaml`](https://github.com/facebookresearch/demucs/blob/main/demucs/remote/htdemucs.yaml) | `955717e8-8726e21a.th` | 84 MB |
| `htdemucs_ft` | [`htdemucs_ft.yaml`](https://github.com/facebookresearch/demucs/blob/main/demucs/remote/htdemucs_ft.yaml) | `f7e0c4bc-ba3fe64a.th`, `d12395a8-e57c48e6.th`, `92cfc3b6-ef3bcb9c.th`, `04573f0d-f3cf25b2.th` (one specialised model per stem) | 4 × 84 MB |
