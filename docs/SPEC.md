# Instrument Isolation App — Project Specification

| | |
|---|---|
| **Status** | Draft v0.2 |
| **Owner** | Callum Little |
| **Last updated** | 2026-10-12 |
| **Targets** | Local web app (macOS first), public demo on Hugging Face Spaces, Python CLI |
| **Python** | 3.12 |

> **Portfolio project, not for sale.** This project is non-commercial. It is
> built as a portfolio piece and will not be sold or distributed commercially.
> This matters in practice: the training datasets (MUSDB18-HQ, MoisesDB) are
> licensed for non-commercial research use only, so neither the fine-tuned
> model nor anything that ships it may be used commercially.

---

## 1. Summary

A music source-separation ("stem splitter") tool that takes a stereo mix and
produces one audio file per instrument group:

| Stem | Contents |
|---|---|
| `drums` | Kit, percussion |
| `bass` | Bass guitar, synth bass |
| `guitar` | Electric and acoustic guitars |
| `keys` | All keyboard and synth parts: piano, electric piano, organ, synth pads, synth leads |
| `vocals` | Lead and backing vocals |
| `other` | Residual: everything not captured above (strings, brass, FX…) |

The `other` stem is a deliberate addition to the five requested stems. Without
it, anything that isn't one of the five targets gets forced into one of them
(e.g. strings bleeding into `keys`). It also lets the stems sum back to the
original mix (see §5.4).

The project is **Python end to end** and has three parts that share one core
library:

1. **ML pipeline:** data prep, fine-tuning, evaluation.
2. **Core library and CLI (`isolate`):** the separation engine.
3. **Web app:** a browser UI over the core library. It runs in two modes:
   - **Local:** on the user's own machine, full-length songs, GPU-accelerated,
     nothing leaves the machine. This is the "real" app.
   - **Demo:** hosted publicly on Hugging Face Spaces, with limits (short
     clips, CPU only). This is the portfolio showcase.

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
- G4. Train **without a local high-end GPU**: start from pretrained weights and
  use transfer learning, with only light compute.
- G5. Quality that is useful for practice, remixing and sampling. Target
  numbers are in §8.
- G6. **Local mode is fully offline.** No audio leaves the machine.

### Non-goals (v1)

- Real-time, zero-latency separation of live input.
- A DAW plugin (VST3/AU). Deferred to stretch, see Appendix A.
- A hosted service for full-length songs. GPU hosting costs money, and storing
  users' (likely copyrighted) uploads is a liability. The public demo is
  deliberately limited (§6.4).
- Splitting stems further, e.g. lead vs backing vocals or kick vs snare.
- Training a model from scratch.
- Commercial distribution. This is a portfolio project and will not be sold.
  The dataset licenses rule it out anyway (§4.3).

---

## 3. Users and core workflows

**Primary user:** a producer or musician working in Reaper or Logic who wants
stems from a bounced track.

**Secondary user:** someone viewing the portfolio (e.g. a recruiter) who wants
to hear what it does in under a minute.

### W1: CLI (developer / batch)

```bash
isolate separate song.wav --out stems/ --stems drums,bass,guitar,keys,vocals,other
```

Writes `stems/song/<stem>.wav`, the same length and sample rate as the input.

### W2: Local app (main workflow)

1. The user runs `isolate app`. A browser tab opens at `http://127.0.0.1:7860`.
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

## 4. Modelling approach

### 4.1 Base model choice

Use **transfer learning from a pretrained separator**. Candidates:

| Model | Stems out of the box | License (code / weights) | Notes |
|---|---|---|---|
| **HTDemucs-6s** (Meta, `demucs` v4) | drums, bass, vocals, guitar, piano, other | MIT / MIT | Already 6 stems, close to our list. The piano stem is known to be weak. Hybrid time + spectrogram transformer. |
| HTDemucs / HTDemucs-ft (4s) | drums, bass, vocals, other | MIT / MIT | Strongest Demucs model on the 4 stems. Would need new heads for guitar and keys. |
| BS-RoFormer / Mel-RoFormer | Usually vocals or 4s (community checkpoints) | Code MIT; **weights vary** | State of the art on vocals. Heavier, and many checkpoints have unclear licenses. |
| Open-Unmix (UMX-L) | 4s | MIT | Simple and light, but noticeably lower quality. |

**Decision: start from `htdemucs_6s`.** It already outputs almost exactly the
stems we need, so the app works before any training. Fine-tuning then becomes
about improving `keys` and `guitar`, not adding stems from scratch. We map
`piano` → `keys` and widen the training targets for that stem to include
synths and organ.

Keep **BS-RoFormer** as a stretch option, either as a stronger vocals/drums
model or as a teacher for distillation, if the licenses allow it.

### 4.2 Transfer-learning strategy (low compute)

Order of attempts, from cheapest to most expensive:

1. **Zero-shot baseline.** Run `htdemucs_6s` as it is and measure (§8). This
   gives the floor and must be done before any training.
2. **Head/decoder fine-tune.** Freeze both encoders and the cross-domain
   transformer. Train only the decoders for `guitar` and `keys` (and
   optionally `other`).
3. **Partial unfreeze.** Also unfreeze the last N transformer layers with a
   10× lower learning rate.
4. **Adapters / LoRA** on the transformer's attention projections, if full
   partial-unfreezing overfits or doesn't fit in memory.

Training tricks that keep compute low:

- Short segments (4–6 s), batch size 4–8, gradient accumulation.
- Mixed precision (bf16 on MPS/CUDA where it is stable).
- **On-the-fly remixing augmentation:** build new mixtures each step by
  combining stems from *different* songs, plus random gain, pitch/tempo shift,
  channel swap and polarity flip. This multiplies the effective size of a
  small dataset.
- Loss: L1 on the waveform plus multi-resolution STFT loss. The SDR-style
  objective is used only for model selection.
- Early stopping on validation SDR per stem.

### 4.3 Data

| Dataset | Stems | Size | License | Use |
|---|---|---|---|---|
| **MoisesDB** | 11 top-level stems incl. guitar, piano, keys/synth, bass, drums, vocals | 240 tracks | Non-commercial research | **Main fine-tuning set** for guitar and keys |
| MUSDB18-HQ | drums, bass, vocals, other | 150 tracks | Non-commercial research | Standard benchmark, regression check on the 4 core stems |
| Slakh2100 | MIDI-rendered, incl. guitar, piano, bass, drums (no vocals) | 2100 tracks | CC BY 4.0 | Extra guitar/keys data. Synthetic, so use for pre-fine-tuning or mix it in at a low ratio |
| MedleyDB 2.0 | Multitrack, many instruments | ~200 tracks | CC BY-NC-SA | Optional extra data or held-out eval |

**Stem mapping.** A config file (`configs/stem_map.yaml`) maps each dataset's
instrument labels to our six stems. For example, MoisesDB `piano`,
`keyboards`, `organ` and `synth` (pads and leads) all map to `keys`, and
`bowed strings`/`wind` map to `other`. All datasets go through this one mapping layer.

**Licensing consequence.** MUSDB18 and MoisesDB are non-commercial. A model
fine-tuned on them is fine for this portfolio project but **cannot be sold**.
That is acceptable because the project is non-commercial (see the note at the
top). If the fine-tuned weights are published (e.g. on the Hugging Face Hub
for the demo), they go out under a non-commercial license (CC BY-NC 4.0) with
a model card that lists the training data. If the project ever went
commercial, the model would need retraining on cleared data, such as Slakh or
licensed multitracks.

**Splits.** Split by song, never by segment. Keep the official MUSDB18 test
set untouched for reporting.

### 4.4 Compute plan

The dev machine is an **Apple M4 Pro**, which is good enough for fine-tuning
(heads, partial unfreeze, LoRA) on PyTorch's **MPS** backend.

| Option | Cost | Use for |
|---|---|---|
| Local M4 Pro (MPS) | Free | Baseline, debugging, small fine-tunes |
| Kaggle Notebooks (T4×2 / P100, ~30 GPU-h/week) | Free | Longer fine-tunes |
| Google Colab (T4) | Free / ~£10 per month | Ad-hoc runs |
| Rented GPU (RunPod, Lambda, vast.ai; A10/A100/4090 at ~$0.3–1.5/h) | Pay per use | One or two final runs, if needed |

⚠️ **Docker on macOS cannot access the Apple GPU.** The devcontainer is
CPU-only and fits linting, tests and data prep. MPS training, and running the
local app at full speed, have to happen **natively** (Poetry venv on the
host). Cloud runs use the same code with `--device cuda`.

Expected budget: fine-tuning should take tens of GPU-hours at most. Keep
training resumable (checkpoint every N steps) so free-tier session limits
don't lose work.

---

## 5. Core library and ML pipeline — requirements

Python version: **3.12** for everything (host venv, devcontainer, CI and the
Spaces deployment). It is new enough for current PyTorch and old enough that
`demucs`, `torchaudio`, `museval` and `gradio` all have wheels for it.

### 5.1 Repository layout

```text
src/isolate/
  core/          # separation engine: model loading, chunking, overlap-add,
                 #   resampling, stem writing; progress + cancel hooks
  data/          # dataset loaders, stem mapping, augmentation
  models/        # htdemucs wrapper, fine-tune heads, LoRA adapters
  app/           # web app: UI, job runner, local/demo mode config
  train.py       # training entry point (config-driven)
  evaluate.py    # SDR / SI-SDR evaluation on held-out sets
  cli.py         # `isolate` command: separate, app, evaluate, train
configs/         # stem_map.yaml, train/*.yaml, app/*.yaml
scripts/         # dataset download/prep helpers
spaces/          # Hugging Face Spaces entry point + README metadata
tests/
docs/
```

The app and CLI only talk to `isolate.core`. They never import training code.
This keeps the deployed demo small and lets the UI be swapped out later
without touching the engine.

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

### 5.3 Training requirements

- R8. Config-driven (which layers to freeze, LR per group, segment length,
  augmentation).
- R9. Checkpoints can be resumed. Log to TensorBoard or W&B.
- R10. Run validation SDR per stem every N steps and keep the best checkpoint
  per metric.
- R11. Results are reproducible from a seed plus the config.

### 5.4 Mixture consistency

- R12. Optional **residual projection**: after inference, spread the error
  `mix − Σstems` across the stems (or put it all into `other`), so the stems
  sum exactly to the input. Producers expect this, because muting nothing
  should give back the original mix. On by default in the app.

### 5.5 Model packaging

- R13. Fine-tuned weights are published to a **Hugging Face Hub model repo**
  with a model card (training data, non-commercial license, eval table).
  The app downloads them on first run, checks the checksum, and caches them.
  The pretrained `htdemucs_6s` weights remain available as a fallback model.
- R14. Optional dynamic INT8 quantization (`torch.ao.quantization`) for the
  CPU-only demo, with the SDR cost measured and reported.

---

## 6. Web app — architecture and requirements

### 6.1 Architecture

```text
 Browser UI ──► App layer (Gradio) ──► Job runner ──► isolate.core ──► model
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
  of the input audio plus the preset and model version, so separating the
  same file twice is instant.
- **Mode** is selected by config (`ISOLATE_MODE=local|demo`). The two modes
  run the same code with different limits (§6.4).

**Alternative considered:** FastAPI backend + custom frontend (e.g. Svelte
with the Web Audio API). This gives full control over the UI, e.g. a
synchronised multitrack mixer with mute/solo. It is more work, so it is kept
for v1.x if Gradio's components become limiting (§11).

### 6.2 Functional requirements

- A1. Upload by drag-and-drop or file picker. WAV, AIFF, FLAC and MP3.
  Unsupported or corrupt files get a clear error message.
- A2. Quality preset: **Fast** (shifts=0) / **Best** (shifts=2).
- A3. Choose which stems to output (default: all six).
- A4. Shows progress and the time remaining, and the job can be cancelled.
- A5. One audio player and one download button per stem, plus the original
  mix for A/B comparison.
- A6. **Download all** as a ZIP. Files are named `<song> - <stem>.wav`, so
  they sort together and are clear once dropped into a DAW.
- A7. Local mode only: **Export to folder** (default
  `~/Music/Stems/<song>/`) and a button to show that folder in Finder.
- A8. Shows which model and preset produced the stems (for reproducibility
  and the eval write-ups).
- A9. Built-in example clips (licensed for reuse, e.g. from Slakh or
  CC-licensed recordings), so demo visitors can try it without their own file.

### 6.3 Local mode

- Launched with `isolate app`. Opens the browser automatically.
- **Binds to `127.0.0.1` only** by default, so it isn't reachable from the
  network. `--host` is needed to override it.
- No file-length limit, beyond a warning for very long files (> 20 min).
- Uses MPS on Apple Silicon automatically.
- Cache is stored in `~/Library/Caches/isolate/`, with a size cap
  (default 5 GB, oldest removed first).

### 6.4 Demo mode (Hugging Face Spaces)

- Runs on the **free CPU tier** (2 vCPU, 16 GB RAM).
- Limits, all configurable:
  - Max upload size 20 MB.
  - Processes a **30-second section** chosen by the user (default: from
    0:30). The rest of the file is discarded immediately.
  - Fast preset only. INT8 model if R14 shows the quality cost is small.
  - One job at a time, with a queue length cap.
- **Privacy and copyright:**
  - Uploads and results are deleted at the end of the session, or after
    1 hour at most. Nothing is stored long-term or logged.
  - The page shows a short notice: upload only audio you have the right to
    use. The demo is a non-commercial portfolio piece.
- Deployed with Spaces' Gradio SDK, pinned to Python 3.12 in the Space's
  README metadata. Deployed automatically by CI on tagged releases.
- Free Spaces go to sleep when idle. The repo README warns that the first
  load can take about a minute.

### 6.5 Non-functional requirements

- N1. **Local speed:** a 3-minute stereo song separates in **≤ 60 s** on the
  M4 Pro (MPS) with the Fast preset. Stretch: ≤ 30 s.
- N2. **Demo speed:** a 30-second clip separates in **≤ 90 s** on the Spaces
  free CPU tier.
- N3. **Memory:** peak RAM ≤ 4 GB during separation, whatever the song length
  (chunking, R2).
- N4. **Robustness:** errors in decoding or inference show in the UI as a
  readable message. They never crash the server or leave temp files behind.
- N5. **Privacy:** local mode makes no network requests apart from the
  one-off model download (R13). Demo mode follows §6.4.
- N6. **Accessibility basics:** all controls usable by keyboard, with labels
  on every control.

---

## 7. Phased roadmap

The app comes **before** fine-tuning on purpose. The pretrained model is good
enough to build and demo the whole product early. Fine-tuning then improves
the quality inside a product that already works.

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **0. Foundations** | Python 3.12 package skeleton, Poetry deps, CI (lint + tests), dataset download scripts | `pytest` green in CI; MUSDB18-HQ and MoisesDB prepared locally |
| **1. Baseline** | `isolate.core` engine with pretrained `htdemucs_6s`; `isolate separate` CLI; evaluation harness | Baseline SDR table (§8) for all 6 stems; N1 speed measured |
| **2. Local app v1** | Gradio app in local mode: A1–A8 | Full W2 workflow works, ending with stems in both Reaper and Logic |
| **3. Fine-tune** | Stem mapping, augmentation, head/partial fine-tuning on MoisesDB (+Slakh); weights + model card on the HF Hub | `keys` and `guitar` SDR improve by **≥ 1.0 dB** over the baseline, and no core stem drops by more than 0.2 dB |
| **4. Public demo** | Demo mode (§6.4) on Hugging Face Spaces; example clips; CI deploy; README with link, results and screenshots | Link works from a fresh browser; N2 met; uploads confirmed deleted |
| **5. Polish** | Synchronised stem mixer (mute/solo/gain); waveform view; optional native macOS wrapper (pywebview) for drag-out into the DAW | Mixer playback stays in sync over a full song |
| **6. Stretch** | Reaper ReaScript integration; ONNX export + in-browser inference (WebGPU); RoFormer models / distillation; DAW plugin (Appendix A) | — |

---

## 8. Evaluation

- **Metrics:** per-stem **SDR** (museval / BSSEval v4, median over tracks, the
  MUSDB convention) and **SI-SDR**. Report both.
- **Test sets:**
  - MUSDB18-HQ test (50 tracks), for drums, bass, vocals and other only.
  - A held-out MoisesDB split (~30 tracks, never used in training), for all
    six stems.
- **Reference numbers (published, for context only):** HTDemucs scores
  roughly 9 dB average SDR on MUSDB18-HQ. The `htdemucs_6s` guitar stem is
  usable, and its piano stem is weak, with noticeable bleed. Our own Phase 1
  measurements replace these.
- **Listening tests:** a fixed set of ~10 tracks across genres (rock, pop,
  jazz, electronic, acoustic). Note artefacts (phasiness, bleed, transient
  smearing) for each release.
- **Performance:** track wall-clock time and peak memory per model version,
  on the M4 Pro (local) and on the Spaces CPU tier (demo).

Results go in `docs/results/<version>.md` so progress can be compared over
time. The latest table is also shown in the README and the model card.

---

## 9. Testing strategy

| Layer | Tests |
|---|---|
| Core unit | Stem mapping, chunk/overlap-add round-trip (identity model → output equals input), resampling, I/O, cancel token stops between chunks |
| Core integration | Separation on a 10 s fixture produces 6 stems of the right length; mixture consistency holds when R12 is enabled |
| App | Handlers tested directly with pytest: demo limits enforced (size, clip length, preset), temp files cleaned on success, error and cancel; local mode binds to `127.0.0.1` |
| End-to-end | `gradio_client` smoke test against a running app (upload fixture → 6 stems back), run in CI on CPU |
| Model regression | Evaluation on a small fixed subset in CI (CPU, a few clips), failing if SDR drops past a threshold |
| Manual | Release checklist: W2 in Reaper and Logic; W3 on the deployed Space from a fresh browser |

CI uses a tiny stand-in model for most tests so they run fast on CPU. Only the
regression and end-to-end jobs load the real weights.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `keys` is ill-defined (piano vs synth vs pads overlap with `other` and `guitar`) | Low keys SDR, confusing results | Clear stem map; widen the target; keep the `other` residual; judge by ear as well as by SDR |
| Spaces free CPU is too slow | Demo feels broken | 30 s clip limit; Fast preset; INT8 model (R14); queue position shown; examples cached in advance |
| Spaces sleeps when idle | Slow first load for visitors | README notice; cached examples load instantly once awake; ZeroGPU or a paid tier only if really needed |
| Users upload copyrighted music to the demo | Legal/ToS exposure | Clip-only processing, no storage, automatic deletion, clear notice |
| Dragging from the browser into a DAW is clunky | Worse W2 workflow | Export to folder + Finder button (A7); pywebview wrapper in Phase 5 |
| Free-tier GPU limits | Slow fine-tuning | Fine-tune small parts of the model; MPS locally; resumable checkpoints |
| Dataset licenses are non-commercial | Can't sell it | Accepted: this is a non-commercial portfolio project; weights published as CC BY-NC |
| Gradio limits the UI (e.g. synced mixer) | Phase 5 harder | Custom Gradio component, or move to FastAPI + custom frontend; the core library is already separate |

---

## 11. Decisions and open questions

### Decided (2026-10-12)

- **`keys` stem:** includes everything keyboard- or synth-based: piano,
  electric piano, organ, synth pads and synth leads.
- **Commercial use:** none. This is a portfolio piece and will not be sold.
  Non-commercial datasets are therefore acceptable.
- **Python version:** 3.12.
- **Product form:** a web app (local + hosted demo) instead of a DAW plugin.
  The plugin is deferred to stretch (Appendix A).

### Open

1. **UI framework beyond v1:** stay on Gradio (with custom components) or
   move to FastAPI + a custom frontend for the Phase 5 mixer? Decide after
   Phase 2, once Gradio's limits are clearer.
2. **Native wrapper:** is a pywebview macOS app worth it for drag-out into the
   DAW, or is Export to folder enough? Decide after using W2 for real.
3. **Spaces hardware:** stay on the free CPU tier, or apply for a community
   GPU grant if the demo gets traffic?

---

## 12. References

- Rouard, Massa, Défossez, *Hybrid Transformers for Music Source Separation*
  (HTDemucs), ICASSP 2023 — `github.com/facebookresearch/demucs`
- Lu et al., *Music Source Separation with Band-Split RoPE Transformer*
  (BS-RoFormer), 2023
- Pereira et al., *MoisesDB: A Dataset for Source Separation Beyond 4-Stems*,
  ISMIR 2023
- Rafii et al., *MUSDB18-HQ*, 2019
- Manilow et al., *Slakh2100*, WASPAA 2019
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
  Core ML conversion. Benchmark both.
- ONNX export (opset ≥ 17) with **STFT/iSTFT kept outside the graph**, because
  HTDemucs's complex spectrogram branch doesn't export cleanly. STFT/iSTFT
  are implemented in C++ (vDSP), with a PyTorch-vs-ONNX parity test (max abs
  error < 1e-3, SDR delta < 0.05 dB).
- Validation with `pluginval`, `auval -v` and the VST3 validator, plus a
  signed and notarized installer (needs an Apple Developer ID).

### A.4 Cheaper step first

A **Reaper ReaScript** (Lua) action that renders the selected item, calls
`isolate separate`, and imports the stems as new tracks. This gives one-click
DAW integration for Reaper with no C++ (roadmap Phase 6).
