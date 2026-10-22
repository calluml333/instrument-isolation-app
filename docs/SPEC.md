# Instrument Isolation App — Project Specification

| | |
|---|---|
| **Status** | Draft v0.4 |
| **Owner** | Callum Little |
| **Last updated** | 2026-10-22 |
| **Targets** | Local web app (runs in Docker), public demo on Hugging Face Spaces, Python CLI |
| **Python** | 3.12 |
| **Runtime** | Docker only: nothing runs natively on the host (§4.4) |
| **Models** | Pretrained only (no training or fine-tuning) |

> **Portfolio project, not for sale.** This project is non-commercial. It is
> built as a portfolio piece and will not be sold or distributed commercially.
> The evaluation data (MUSDB18 samples) is licensed for non-commercial research
> use only.

---

## 1. Summary

A music source-separation ("stem splitter") tool that takes a stereo mix and
produces one audio file per instrument group:

| Stem | Contents |
|---|---|
| `drums` | Kit, percussion |
| `bass` | Bass guitar, synth bass |
| `guitar` | Electric and acoustic guitars |
| `piano` | Acoustic and electric piano |
| `vocals` | Lead and backing vocals |
| `other` | Residual: everything not captured above (synths, organ, strings, brass, FX…) |

The `other` stem is deliberate. Anything that isn't one of the five named
targets goes there instead of being forced into one of them, and it lets the
stems sum back to the original mix (see §5.4).

**Why `piano` and not `keys`:** the project uses pretrained models only (§4),
and the best available 6-stem model separates *piano* specifically. Synths and
organ mostly end up in `other`. The stem is named for what it actually
contains. Widening it to all keyboards would need fine-tuning (Appendix B).

The project is **Python end to end** and has two parts that share one core
library:

1. **Core library and CLI (`isolate`):** the separation engine. It wraps
   pretrained models behind one interface, combines them (§4.2), and handles
   chunking, resampling and mixture consistency.
2. **Web app:** a browser UI over the core library. It runs in two modes:
   - **Local:** on the user's own machine (in Docker, CPU), full-length songs,
     nothing leaves the machine. This is the "real" app.
   - **Demo:** hosted publicly on Hugging Face Spaces, with limits (short
     clips, CPU only). This is the portfolio showcase.

### Why pretrained models only

Fine-tuning needs the multitrack datasets (MUSDB18-HQ ~30 GB, MoisesDB far
larger) stored locally, and that storage isn't available. Pretrained models
already give useful quality for all six stems. The ML depth of the project
comes from **inference engineering** instead: combining models, comparing them
on a fixed evaluation set, and making them fast enough for a CPU-only demo
(§4). The fine-tuning plan is kept in Appendix B in case storage becomes
available later.

### Why an app and not a DAW plugin

The original idea was a VST3/AU plugin for Reaper and Logic. It was dropped
from v1 because separation **cannot run in real time** (see Appendix A), so a
plugin would work the same way an app does: take the audio, process it
offline, hand stems back. The only thing a plugin adds is ARA2 (reading the
clip without bouncing), which is also its hardest part. Building it means C++,
JUCE, ONNX export and notarized installers, all outside the Python focus of
the project. An app gives the same DAW workflow (bounce → separate → drag
stems in) and a public demo link that anyone can try. The plugin stays as a
stretch goal (Appendix A).

---

## 2. Goals and non-goals

### Goals

- G1. Separate a stereo file (WAV/AIFF/FLAC/MP3, 44.1/48 kHz) into the six stems
  above.
- G2. A **local web app** that makes it easy to get stems into Reaper or Logic:
  drop a file in, preview the stems, export them to a folder ready to drag
  into a DAW.
- G3. A **public demo** anyone can try from a link, without installing
  anything.
- G4. Get the best quality available **from pretrained models**, by combining
  them and choosing between them based on measured results.
- G5. Fast enough to be pleasant: see the speed targets in §6.5.
- G6. **Local mode is fully offline** after the one-off model download. No
  audio leaves the machine.

### Non-goals (v1)

- Training or fine-tuning any model (Appendix B).
- Real-time, zero-latency separation of live input.
- A DAW plugin (VST3/AU). Deferred to stretch, see Appendix A.
- A hosted service for full-length songs. GPU hosting costs money, and storing
  users' (likely copyrighted) uploads is a liability. The public demo is
  deliberately limited (§6.4).
- Splitting stems further, e.g. lead vs backing vocals or kick vs snare.
- Commercial distribution. This is a portfolio project and will not be sold.

---

## 3. Users and core workflows

**Primary user:** a producer or musician working in Reaper or Logic who wants
stems from a bounced track.

**Secondary user:** someone viewing the portfolio (e.g. a recruiter) who wants
to hear what it does in under a minute.

### W1: CLI (developer / batch)

```bash
isolate separate song.wav --out stems/ --stems drums,bass,guitar,piano,vocals,other
```

Writes `stems/song/<stem>.wav`, the same length and sample rate as the input.
`--model` picks a model or ensemble (§4.2); the default is the best one
measured in §8.

### W2: Local app (main workflow)

1. The user runs `isolate app` in the dev container and opens
   `http://127.0.0.1:7860` in their browser.
2. They drag in a song (or a bounce from their DAW).
3. They pick a quality preset (**Fast** / **Best**) and click **Separate**.
   Progress is shown, and they can cancel.
4. Each stem appears with its own player, and can be downloaded on its own.
5. They click **Export**, which writes all stems to a folder (default
   `~/Music/Stems/<song>/`) or downloads them as a ZIP.
6. They drag the stems from Finder into Reaper or Logic. Every stem has the
   same length and sample rate as the source, so placing them all at the same
   start point lines them up exactly.

### W3: Public demo

1. The visitor opens the Spaces link.
2. They upload a song (or pick a built-in example clip), choose a 30-second
   section, and click **Separate**.
3. They listen to and download the stems. Uploads and results are deleted
   automatically after the session (§6.4).

---

## 4. Models

### 4.1 Candidate pretrained models

| Model | Stems | License (code / weights) | Notes |
|---|---|---|---|
| **HTDemucs-6s** (Meta, `demucs` v4) | drums, bass, vocals, guitar, piano, other | MIT / research use only (`docs/models.md`) | The only widely available model with guitar and piano stems. Its piano stem is the weakest. |
| **HTDemucs-ft** | drums, bass, vocals, other | MIT / research use only (`docs/models.md`) | A bag of 4 fine-tuned models, one per stem. Best Demucs quality on the 4 core stems, but about 4× slower. |
| HTDemucs (4s) | drums, bass, vocals, other | MIT / research use only (`docs/models.md`) | Faster than `-ft`, slightly lower quality. |
| BS-RoFormer / Mel-RoFormer | Usually vocals or 4s (community checkpoints) | Code MIT; **weights vary** | State of the art on vocals. Many checkpoints have unclear licenses, so only use ones with a clear license. |

**License check (Phase 0):** confirm the license of each set of pretrained
weights before it's used or redistributed, and record it in
`docs/models.md`.

**Default for v1: `htdemucs_6s`.** It is the only model that produces all six
stems on its own, so it's the baseline (Phase 1) and the fallback.

### 4.2 Model backends and ensembles

- M1. **Backend interface.** Every model sits behind one interface in
  `isolate.core`: it takes a stereo waveform at 44.1 kHz and returns a dict of
  named stems at the same length. It also reports which stems it supports, its
  native sample rate and segment length. The engine, CLI and app only use this
  interface, so models can be added and compared without changing anything
  else.
- M2. **Ensemble backend.** A backend that runs several models and takes each
  stem from the model that does it best. The planned ensemble:
  - `drums`, `bass`, `vocals` from **HTDemucs-ft**
  - `guitar`, `piano` from **HTDemucs-6s**
  - `other` = mix − (sum of the five above), so the stems always sum exactly
    to the input
- M3. **Selection is measured, not assumed.** Which model supplies each stem,
  and whether the ensemble becomes the default, is decided by the evaluation in
  §8. Speed counts too: the ensemble only becomes the **Best** preset if the
  quality gain is worth the extra time. **Fast** stays on a single model.
- M4. **Adding a model** means one new backend class plus an entry in a model
  registry (name, weights URL, checksum, license, supported stems).

### 4.3 Speed optimisation

The public demo runs on a free CPU tier (§6.4), so inference speed matters.
Options, measured in the performance phase:

- **Dynamic INT8 quantisation** of the transformer layers (PyTorch).
- **Export** to ONNX Runtime or Core ML, with the STFT/iSTFT kept outside the
  exported graph (HTDemucs's complex spectrogram branch doesn't export
  cleanly). Requires a parity test against PyTorch (max abs error < 1e-3,
  SDR delta < 0.05 dB).
- **Fewer shifts / smaller overlap** for the Fast preset.

Each option is kept only if its quality cost (SDR change, measured in §8) and
its speed gain are both reported, and the trade-off is worth it.

### 4.4 Compute

The dev machine is an **Apple M4 Pro**. **Everything runs in Docker** (the
dev container): tests, linting, git and `gh`, evaluations and the local app.
Nothing is installed or run natively on the host apart from Docker itself and
the tools that start the container (VS Code or the Dev Containers CLI).

⚠️ **Docker on macOS cannot access the Apple GPU**, so all inference runs on
**CPU**: locally in the container, in CI and on the demo. This is a deliberate
trade of speed for a reproducible, host-independent setup (decided
2026-10-22). It makes the speed work in §4.3 matter for the local app too, not
only the demo. The device selection in R6 still supports `mps`/`cuda` so a
native or GPU setup can be added later without code changes.

---

## 5. Core library — requirements

Python version: **3.12** for everything (devcontainer, CI and the
Spaces deployment). It is new enough for current PyTorch and old enough that
`demucs`, `torchaudio`, `museval` and `gradio` all have wheels for it.

### 5.1 Repository layout

```text
src/isolate/
  core/          # separation engine: chunking, overlap-add, resampling,
                 #   stem writing; progress + cancel hooks
  models/        # backend interface, registry, demucs/ensemble backends
  eval/          # evaluation harness: datasets, metrics, reports
  app/           # web app: UI, job runner, local/demo mode config
  cli.py         # `isolate` command: separate, app, eval, models
configs/         # app/*.yaml, eval/*.yaml
tests/
docs/
spaces/          # Hugging Face Spaces entry point + README metadata
```

The app and CLI only talk to `isolate.core` and `isolate.models`. This keeps
the deployed demo small and lets the UI be swapped out later without touching
the engine.

### 5.2 Inference requirements

- R1. Accept mono or stereo input at any common sample rate. Resample to
  44.1 kHz internally, and resample the outputs back to the input rate.
- R2. Process long files in **overlapping chunks** (e.g. 7.8 s with 25%
  overlap and a cross-fade window), so memory use doesn't depend on song
  length.
- R3. Optional **shift trick / test-time augmentation**: average over N random
  time shifts. Quality vs speed, exposed as the Fast/Best preset.
- R4. Outputs are sample-aligned with the input and have the same length.
- R5. Write 24-bit or 32-bit float WAV. Optionally FLAC.
- R6. Device selection: `auto` (default: `mps` → `cuda` → `cpu`), or forced.
- R7. The engine API takes a **progress callback** (fraction done) and a
  **cancel token**, checked between chunks, so the UI can show progress and
  stop a job cleanly.

### 5.3 Model management

- R8. Weights are downloaded on first use from the URL in the model registry
  (M4), verified against a checksum, and cached in
  `~/.cache/isolate/models/`. The dev container bind-mounts the host's
  `~/.cache`, so downloads survive container rebuilds. No weights are
  committed to the repo.
- R9. `isolate models list` shows the available models and ensembles, their
  stems, license and whether they're downloaded. `isolate models download
  <name>` fetches one ahead of time (e.g. for offline use or the demo build).

### 5.4 Mixture consistency

- R10. **Residual projection**: after inference, put the error
  `mix − Σstems` into `other` (or spread it across the stems), so the stems sum
  exactly to the input. Producers expect this, because muting nothing should
  give back the original mix. On by default in the app. The ensemble (M2) does
  this by construction.

---

## 6. Web app — architecture and requirements

### 6.1 Architecture

```text
 Browser UI ──► App layer (Gradio) ──► Job runner ──► isolate.core ──► model backend
                     │                 (1 worker,         │
                     │                  queue, cancel)    ▼
                     └──────────────── Job storage (temp dir per job,
                                        content-hash cache, TTL cleanup)
```

- **UI framework: Gradio** for v1. It handles uploads, audio players,
  progress bars and request queueing out of the box, and it is the native
  format for Hugging Face Spaces, so one codebase serves both modes.
- **Job runner:** a single worker processes one separation at a time (the
  model uses most of the machine). Other requests wait in Gradio's queue with
  their position shown.
- **Storage:** each job gets a temp directory. Results are cached by a hash
  of the input audio plus the preset and model, so separating the same file
  twice is instant.
- **Mode** is selected by config (`ISOLATE_MODE=local|demo`). The two modes
  run the same code with different limits (§6.4).

**Alternative considered:** FastAPI backend + custom frontend (e.g. Svelte
with the Web Audio API). This gives full control over the UI, e.g. a
synchronised multitrack mixer with mute/solo. It is more work, so it is kept
for later if Gradio's components become limiting (§11).

### 6.2 Functional requirements

- A1. Upload by drag-and-drop or file picker. WAV, AIFF, FLAC and MP3.
  Unsupported or corrupt files get a clear error message.
- A2. Quality preset: **Fast** (single model, no shifts) / **Best** (the
  best measured model or ensemble, with shifts).
- A3. Choose which stems to output (default: all six).
- A4. Shows progress and the time remaining, and the job can be cancelled.
- A5. One audio player and one download button per stem, plus the original
  mix for A/B comparison.
- A6. **Download all** as a ZIP. Files are named `<song> - <stem>.wav`, so
  they sort together and are clear once dropped into a DAW.
- A7. Local mode only: **Export to folder**, written to a host folder that is
  bind-mounted into the container (default host path
  `~/Music/Stems/<song>/`). The UI shows the host path. A "show in Finder"
  button isn't possible from inside Docker (§4.4).
- A8. Shows which model and preset produced the stems.
- A9. Built-in example clips (licensed for reuse, e.g. CC-licensed
  recordings), so demo visitors can try it without their own file.

### 6.3 Local mode

- Launched with `isolate app` inside the dev container. It prints the URL;
  the container can't open the host's browser.
- **Reachable from the host's `127.0.0.1` only.** Inside the container the
  server has to listen on `0.0.0.0` for port forwarding to work, so the
  restriction is enforced by publishing port 7860 on the host's loopback
  interface only, never on all interfaces.
- No file-length limit, beyond a warning for very long files (> 20 min).
- Runs on CPU (§4.4).
- Cache is stored in `~/.cache/isolate/`, with a size cap
  (default 5 GB, oldest removed first).

### 6.4 Demo mode (Hugging Face Spaces)

- Runs on the **free CPU tier** (2 vCPU, 16 GB RAM).
- Limits, all configurable:
  - Max upload size 20 MB.
  - Processes a **30-second section** chosen by the user (default: from
    0:30). The rest of the file is discarded immediately.
  - Fast preset only, using the fastest acceptable model from §4.3.
  - One job at a time, with a queue length cap.
- **Privacy and copyright:**
  - Uploads and results are deleted at the end of the session, or after
    1 hour at most. Nothing is stored long-term or logged.
  - The page shows a short notice: upload only audio you have the right to
    use. The demo is a non-commercial portfolio piece.
- Deployed with Spaces' Gradio SDK, pinned to Python 3.12 in the Space's
  README metadata. Model weights are fetched at build time (R9). Deployed
  automatically by CI on tagged releases.
- Free Spaces go to sleep when idle. The repo README warns that the first
  load can take about a minute.

### 6.5 Non-functional requirements

- N1. **Local speed:** a 3-minute stereo song with the Fast preset, on CPU in
  the dev container on the M4 Pro. The target is set from the Phase 1
  baseline measurement: the original ≤ 60 s target assumed the GPU (MPS),
  which Docker can't use (§4.4).
- N2. **Demo speed:** a 30-second clip separates in **≤ 90 s** on the Spaces
  free CPU tier.
- N3. **Memory:** peak RAM ≤ 4 GB during separation, whatever the song length
  (chunking, R2).
- N4. **Robustness:** errors in decoding or inference show in the UI as a
  readable message. They never crash the server or leave temp files behind.
- N5. **Privacy:** local mode makes no network requests apart from the
  one-off model downloads (R8). Demo mode follows §6.4.
- N6. **Accessibility basics:** all controls usable by keyboard, with labels
  on every control.

---

## 7. Phased roadmap

The app comes early on purpose. The baseline model is good enough to build and
demo the whole product, and the model work then improves quality inside a
product that already works.

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **0. Foundations** | Python 3.12 package skeleton, Poetry deps, CI (lint + tests + devcontainer build), model license check (`docs/models.md`) | `pytest` green in CI; `isolate --version` works in the dev container |
| **1. Baseline** | Backend interface + registry (M1, M4); `htdemucs_6s` backend; `isolate.core` engine; `isolate separate` CLI; evaluation harness on MUSDB18 samples (§8); coverage gate and type checking in CI | Baseline results table (§8); N1 speed measured and its target set |
| **2. Local app v1** | Gradio app in local mode: A1–A8 | Full W2 workflow works, ending with stems in both Reaper and Logic |
| **3. Model comparison** | `htdemucs`, `htdemucs_ft` and ensemble backends (M2); comparison report; default chosen per M3 | Ensemble beats `htdemucs_6s` alone on the core stems, or the report shows why it doesn't; Fast/Best presets set from the results |
| **4. Performance** | Quantisation and export experiments (§4.3); benchmark report | N2 met on CPU with a quality cost that's reported and accepted |
| **5. Public demo** | Demo mode (§6.4) on Hugging Face Spaces; example clips; CI deploy; README with link, results and screenshots | Link works from a fresh browser; N2 met on Spaces; uploads confirmed deleted |
| **6. Polish** | Synchronised stem mixer (mute/solo/gain); waveform view | Mixer playback stays in sync over a full song |
| **7. Stretch** | RoFormer backend (if a clearly licensed checkpoint exists); Reaper ReaScript integration; in-browser inference (WebGPU); fine-tuning (Appendix B); DAW plugin (Appendix A) | — |

---

## 8. Evaluation

Used to compare models and settings, not to report absolute scores.

- **Data: the MUSDB18 7-second sample set.** The `musdb` Python package
  downloads it on demand: 7-second excerpts of all 150 MUSDB18 tracks, a few
  hundred MB at most. It is stored in the user cache, not in the repo.
  - It has drums, bass, vocals and other stems only. **Guitar and piano can't
    be scored numerically** with it.
  - Its audio is compressed and short, so its scores are **only comparable with
    each other**, not with published MUSDB18-HQ results.
  - Licensed for non-commercial research use. Used for evaluation only, never
    redistributed.
- **Metrics:** per-stem **SDR** (museval / BSSEval v4, median over tracks) and
  **SI-SDR**, on the official test split (50 tracks).
- **Guitar and piano:** a **listening test** on a fixed set of ~10
  CC-licensed or self-owned tracks across genres (rock, pop, jazz, electronic,
  acoustic). Note artefacts (bleed, phasiness, transient smearing) for each
  model, in a consistent template.
- **Performance:** wall-clock time and peak memory for each model and preset,
  in the dev container on the M4 Pro (CPU), and on the Spaces CPU tier.
- **Reports** go in `docs/results/<phase>-<topic>.md` (e.g.
  `docs/results/phase-3-model-comparison.md`). The latest summary table is
  also shown in the README.

---

## 9. Testing strategy

| Layer | Tests |
|---|---|
| Core unit | Chunk/overlap-add round-trip (identity backend → output equals input), resampling, I/O, cancel token stops between chunks, residual projection makes stems sum to the mix |
| Backends | Each backend against the interface contract (stems, lengths, sample rate), using a tiny stand-in model; the ensemble picks the right stem from each source |
| Core integration | Separation on a 10 s fixture produces 6 stems of the right length |
| App | Handlers tested directly with pytest: demo limits enforced (size, clip length, preset), temp files cleaned on success, error and cancel; the local app's port is published on the host's `127.0.0.1` only |
| End-to-end | `gradio_client` smoke test against a running app (upload fixture → 6 stems back), run in CI on CPU |
| Model regression | Evaluation on a few fixed MUSDB18 sample clips in CI (CPU), failing if SDR drops past a threshold |
| Export | Parity test for any exported model (§4.3) |
| Manual | Release checklist: W2 in Reaper and Logic; W3 on the deployed Space from a fresh browser |

CI uses a tiny stand-in model for most tests so they run fast on CPU. Only the
regression and end-to-end jobs load real weights, and those are cached between
runs.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `piano` stem quality is weak in `htdemucs_6s` | Disappointing results for keyboard-heavy music | Name the stem honestly; document the limitation; listening tests; fine-tuning stays possible later (Appendix B) |
| Synths and organ land in `other` | Users expecting a "keys" stem | Clear stem descriptions in the UI and README |
| No numeric scores for guitar and piano | Model choices for those stems rest on listening tests | Fixed listening-test set and template; same tracks every time |
| Ensemble is too slow | Best preset feels sluggish | Selection rule M3; Fast preset stays single-model |
| Spaces free CPU is too slow | Demo feels broken | 30 s clip limit; Fast preset; optimisation phase (§4.3); queue position shown; examples cached in advance |
| Spaces sleeps when idle | Slow first load for visitors | README notice; cached examples load instantly once awake |
| Users upload copyrighted music to the demo | Legal/ToS exposure | Clip-only processing, no storage, automatic deletion, clear notice |
| Pretrained weights are research-use only (`docs/models.md`) | Hosting them in the demo is outside the stated terms | Accepted for a non-commercial learning/portfolio piece (§11); never redistributed or committed |
| CPU-only inference in Docker (§4.4) | The local app is slower than it would be on the GPU | Speed work in §4.3 applies locally too; N1 target set from measurement |
| Dragging from the browser into a DAW is clunky | Worse W2 workflow | Export to a bind-mounted host folder (A7) |
| Gradio limits the UI (e.g. synced mixer) | Phase 6 harder | Custom Gradio component, or move to FastAPI + custom frontend; the core library is already separate |

---

## 11. Decisions and open questions

### Decided

- **2026-10-12 — Commercial use:** none. This is a portfolio piece and will not
  be sold.
- **2026-10-12 — Python version:** 3.12.
- **2026-10-12 — Product form:** a web app (local + hosted demo) instead of a
  DAW plugin. The plugin is deferred to stretch (Appendix A).
- **2026-10-22 — No training:** pretrained models only, because there's no
  storage for the training datasets. Fine-tuning moves to Appendix B.
- **2026-10-22 — Stem names:** `piano` instead of `keys`, to match what the
  pretrained model separates.
- **2026-10-22 — Evaluation:** MUSDB18 7-second samples for numeric scores,
  listening tests for guitar and piano.
- **2026-10-22 — Model work in scope:** swappable backends, an ensemble, and
  speed optimisation.
- **2026-10-22 — Docker only:** everything (tests, tools, evaluations, the
  local app) runs in the dev container; nothing runs natively on the host.
  Inference is therefore CPU-only (§4.4). This drops the native macOS wrapper
  idea (formerly Phase 6 and open question 2).
- **2026-10-22 — Model licences:** the Demucs weights are research-use only
  (`docs/models.md`). Accepted as-is, because this is a non-commercial
  learning/portfolio piece.

### Open

1. **UI framework beyond v1:** stay on Gradio (with custom components) or
   move to FastAPI + a custom frontend for the Phase 6 mixer? Decide after
   Phase 2, once Gradio's limits are clearer.
2. **Spaces hardware:** stay on the free CPU tier, or apply for a community
   GPU grant if the demo gets traffic?

---

## 12. References

- Rouard, Massa, Défossez, *Hybrid Transformers for Music Source Separation*
  (HTDemucs), ICASSP 2023 — `github.com/facebookresearch/demucs`
- Lu et al., *Music Source Separation with Band-Split RoPE Transformer*
  (BS-RoFormer), 2023
- Rafii et al., *MUSDB18*, 2017 — `sigsep.github.io/datasets/musdb.html`;
  `musdb` Python package — `github.com/sigsep/sigsep-mus-db`
- Gradio documentation — `gradio.app`; Hugging Face Spaces — `huggingface.co/docs/hub/spaces`

---

## Appendix A. DAW plugin (deferred, stretch)

Kept here so the analysis isn't lost if the plugin is picked up later.

### A.1 Why it can't be a "normal" real-time effect

Separation models need **seconds of context** (HTDemucs uses ~7.8 s windows)
and take far longer to compute than a 64–512 sample audio callback allows.
Running them in `processBlock()` would cause dropouts, or would need
multi-second latency compensation. A plugin therefore has to work
**offline inside the DAW**: get the audio, separate it on a background thread,
then play back or export the results. That is what the app already does,
which is why the app came first.

### A.2 Integration modes

| Mode | How audio gets in | How stems get out | Reaper (VST3) | Logic (AU) |
|---|---|---|---|---|
| **ARA2** | Plugin reads the clip directly from the host | Plugin plays back the selected stem; drag-out or export to files | ✅ | ✅ (Logic 10.7+) |
| **Capture** | User plays the region once; plugin records it | Playback in sync with the host transport; export | ✅ | ✅ |
| **Multi-out** | ARA or capture | 6 stereo output buses, one per stem | ✅ | ⚠️ Logic only allows multi-out for *instrument* AUs, not effects |

### A.3 Stack, if built

- **JUCE 8** (C++20, CMake): one codebase for VST3, AU and Standalone, with
  ARA support.
- **ONNX Runtime** (C++) with the CoreML execution provider, or a direct
  Core ML conversion, reusing the export work from §4.3.
- STFT/iSTFT implemented in C++ (vDSP), with a parity test against PyTorch.
- Validation with `pluginval`, `auval -v` and the VST3 validator, plus a
  signed and notarized installer (needs an Apple Developer ID).

### A.4 Cheaper step first

A **Reaper ReaScript** (Lua) action that renders the selected item, calls
`isolate separate`, and imports the stems as new tracks. This gives one-click
DAW integration for Reaper with no C++ (roadmap Phase 7).

---

## Appendix B. Fine-tuning (deferred, needs dataset storage)

Kept for if storage for the datasets becomes available (an external drive of
~200 GB+ would cover both). This is how `piano` would become a full `keys`
stem.

- **Data:** MoisesDB (240 tracks, has guitar, piano, keys and synth stems) as
  the main fine-tuning set; MUSDB18-HQ as the benchmark; Slakh2100 (CC BY 4.0,
  synthetic) as extra guitar/keys data. MUSDB18-HQ and MoisesDB are
  non-commercial only. A config file maps each dataset's instrument labels to
  our stems. Splits are by song, never by segment.
- **Strategy, from cheapest to most expensive:** freeze the encoders and
  transformer and train only the guitar/keys decoders → also unfreeze the last
  few transformer layers at a 10× lower learning rate → LoRA adapters on the
  attention projections.
- **Training tricks:** 4–6 s segments, small batches with gradient
  accumulation, mixed precision, on-the-fly remixing augmentation (stems from
  different songs, random gain, pitch/tempo shift, channel swap, polarity
  flip), L1 + multi-resolution STFT loss, early stopping on validation SDR.
- **Compute:** Kaggle or Colab free GPUs, since the dev container is
  CPU-only (§4.4); checkpoints resumable so free-tier limits don't lose work.
- **Exit criterion:** `keys` and `guitar` SDR improve by ≥ 1.0 dB over the
  pretrained baseline, with no core stem dropping by more than 0.2 dB.
