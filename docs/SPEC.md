# Instrument Isolation App — Project Specification

| | |
|---|---|
| **Status** | Draft v0.1 |
| **Owner** | Callum Little |
| **Last updated** | 2026-10-12 |
| **Targets** | VST3 (Reaper), AU (Logic Pro), plus a standalone Python CLI |

---

## 1. Summary

A music source-separation ("stem splitter") tool that takes a stereo mix and
produces one audio file per instrument group:

| Stem | Contents |
|---|---|
| `drums` | Kit, percussion |
| `bass` | Bass guitar, synth bass |
| `guitar` | Electric and acoustic guitars |
| `keys` | Piano, electric piano, organ, synth pads/leads |
| `vocals` | Lead and backing vocals |
| `other` | Residual: everything not captured above (strings, brass, FX…) |

The `other` stem is a deliberate addition to the five requested stems. Without
it, anything that isn't one of the five targets gets forced into one of them
(e.g. strings bleeding into `keys`). It also lets the stems sum back to the
original mix (see §5.4).

The project has two halves that share one model:

1. **ML pipeline (Python, this repo):** data prep, fine-tuning, evaluation,
   export, and a reference CLI.
2. **Plugin (C++/JUCE):** a VST3/AU plugin that runs the exported model
   on-device.

---

## 2. Goals and non-goals

### Goals
- G1. Separate a stereo file (WAV/AIFF/FLAC/MP3, 44.1/48 kHz) into the six stems
  above.
- G2. Work inside **Reaper (VST3)** and **Logic Pro (AU)** on macOS (Apple
  Silicon first).
- G3. Train **without a local high-end GPU**: start from pretrained weights and
  use transfer learning, with only light compute.
- G4. Quality that is useful for practice, remixing and sampling. Target
  numbers are in §8.
- G5. Fully offline. No audio leaves the machine.

### Non-goals (v1)
- Real-time, zero-latency separation of live input (see §6.1 for why).
- Windows/Linux plugin builds. JUCE makes them possible later, but they won't
  be tested in v1.
- Splitting stems further, e.g. lead vs backing vocals or kick vs snare.
- Training a model from scratch.
- Commercial distribution. The dataset licenses rule this out for now (§4.3).

---

## 3. Users and core workflows

**Primary user:** a producer or musician working in Reaper or Logic who wants
stems from a bounced track.

### W1: CLI (developer / batch)
```
isolate separate song.wav --out stems/ --stems drums,bass,guitar,keys,vocals,other
```
Writes `stems/song/<stem>.wav`, the same length and sample rate as the input.

### W2: Plugin in the DAW
1. The user puts the plugin on a track or audio item.
2. The plugin gets the audio, either directly from the clip through **ARA2**
   or by recording it in a "capture" pass.
3. The user clicks **Separate**. A progress bar shows while separation runs in
   a background thread.
4. A stem list appears, with mute/solo per stem and a preview.
5. The user either:
   - drags each stem onto a new DAW track, or
   - clicks **Export** to write the stems as WAVs next to the project, or
   - in multi-out mode (Reaper only, see §6.2), routes each stem to its own
     output bus.

---

## 4. Modelling approach

### 4.1 Base model choice

Use **transfer learning from a pretrained separator**. Candidates:

| Model | Stems out of the box | License (code / weights) | Notes |
|---|---|---|---|
| **HTDemucs-6s** (Meta, `demucs` v4) | drums, bass, vocals, guitar, piano, other | MIT / MIT | Already 6 stems, close to our list. The piano stem is known to be weak. Hybrid time + spectrogram transformer, ~27M params per model. |
| HTDemucs / HTDemucs-ft (4s) | drums, bass, vocals, other | MIT / MIT | Strongest Demucs model on the 4 stems. Would need new heads for guitar and keys. |
| BS-RoFormer / Mel-RoFormer | Usually vocals or 4s (community checkpoints) | Code MIT; **weights vary** | State of the art on vocals. Heavier, and many checkpoints have unclear licenses. |
| Open-Unmix (UMX-L) | 4s | MIT | Simple and easy to export, but noticeably lower quality. |

**Decision: start from `htdemucs_6s`.** It already outputs almost exactly the
stems we need, so v0 works without any training. Fine-tuning then becomes about
improving `keys` and `guitar`, not adding stems from scratch. We map `piano` →
`keys` and widen the training targets for that stem to include synths and
organ.

Keep **BS-RoFormer** as a Phase 4 option, either as a stronger vocals/drums
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
instrument labels to our six stems. For example, MoisesDB `piano` + `keyboards`
+ `synth (pads/leads)` → `keys`, and `bowed strings`/`wind` → `other`. All
datasets go through this one mapping layer.

**Licensing consequence.** MUSDB18 and MoisesDB are non-commercial. A model
fine-tuned on them is fine for a portfolio or personal project, but **cannot be
sold**. Going commercial would need cleared data, such as Slakh or licensed
multitracks. This is tracked in Open Questions.

**Splits.** Split by song, never by segment. Keep the official MUSDB18 test
set untouched for reporting.

### 4.4 Compute plan

The dev machine is an **Apple M4 Pro**, which is good enough for fine-tuning
(heads, partial unfreeze, LoRA) on PyTorch's **MPS** backend.

| Option | Cost | Use for |
|---|---|---|
| Local M4 Pro (MPS) | Free | Phases 0–2, debugging, small fine-tunes |
| Kaggle Notebooks (T4×2 / P100, ~30 GPU-h/week) | Free | Longer fine-tunes |
| Google Colab (T4) | Free / ~£10 per month | Ad-hoc runs |
| Rented GPU (RunPod, Lambda, vast.ai; A10/A100/4090 at ~$0.3–1.5/h) | Pay per use | One or two final runs, if needed |

⚠️ **Docker on macOS cannot access the Apple GPU.** The devcontainer is
CPU-only and fits linting, tests and data prep. MPS training has to run
**natively** (Poetry venv on the host). Cloud runs use the same code with
`--device cuda`.

Expected budget: Phase 2 fine-tuning should take tens of GPU-hours at most.
Keep training resumable (checkpoint every N steps) so free-tier session limits
don't lose work.

---

## 5. ML pipeline (Python) — functional requirements

### 5.1 Repository layout
```
src/isolate/
  data/          # dataset loaders, stem mapping, augmentation
  models/        # htdemucs wrapper, fine-tune heads, LoRA adapters
  train.py       # training entry point (Hydra/argparse config)
  evaluate.py    # SDR / SI-SDR evaluation on held-out sets
  separate.py    # inference: chunking, overlap-add, stem writing
  export/        # ONNX export + parity checks
  cli.py         # `isolate` command
configs/         # stem_map.yaml, train/*.yaml
scripts/         # dataset download/prep helpers
tests/
plugin/          # C++/JUCE plugin (separate CMake project)
docs/
```

### 5.2 Inference requirements
- R1. Accept mono or stereo input at any common sample rate. Resample to
  44.1 kHz internally, and resample the outputs back to the input rate.
- R2. Process long files in **overlapping chunks** (e.g. 7.8 s with 25%
  overlap and a cross-fade window), so memory use doesn't depend on song
  length.
- R3. Optional **shift trick / test-time augmentation**: average over N random
  time shifts. Quality vs speed, user-configurable.
- R4. Outputs are sample-aligned with the input and have the same length.
- R5. Write 24-bit or 32-bit float WAV. Optionally FLAC.
- R6. Device selection: `cpu`, `mps`, `cuda`.

### 5.3 Training requirements
- R7. Config-driven (which layers to freeze, LR per group, segment length,
  augmentation).
- R8. Checkpoints can be resumed. Log to TensorBoard or W&B.
- R9. Run validation SDR per stem every N steps and keep the best checkpoint
  per metric.
- R10. Results are reproducible from a seed plus the config.

### 5.4 Mixture consistency
- R11. Optional **residual projection**: after inference, spread the error
  `mix − Σstems` across the stems (or put it all into `other`), so the stems
  sum exactly to the input. Producers expect this, because muting nothing
  should give back the original mix.

### 5.5 Export
- R12. Export the model to **ONNX** (opset ≥ 17) for C++ inference.
- R13. Keep **STFT/iSTFT outside the ONNX graph**. HTDemucs's complex
  spectrogram branch doesn't export cleanly. Export only the network core,
  and implement STFT/iSTFT in C++ (JUCE FFT, or vDSP on macOS).
- R14. A **parity test** compares Python PyTorch output with ONNX Runtime
  output on fixed clips. Pass if max abs error < 1e-3 and SDR delta
  < 0.05 dB.
- R15. Optional FP16 or INT8 (dynamic) quantization, with the SDR cost
  reported for each.

---

## 6. Plugin — architecture and requirements

### 6.1 Why it can't be a "normal" real-time effect

Separation models need **seconds of context** (HTDemucs uses ~7.8 s windows)
and take far longer than a 64–512 sample audio callback to compute. Running
them in `processBlock()` would cause dropouts, or would need multi-second
latency compensation. So the plugin works **offline-in-DAW**: it gets the
audio, separates it on a background thread, then plays back or exports the
results.

### 6.2 Integration modes

| Mode | How audio gets in | How stems get out | Reaper (VST3) | Logic (AU) |
|---|---|---|---|---|
| **A. ARA2** (preferred) | Plugin reads the clip directly from the host | Plugin plays back the selected stem; drag-out or export to files | ✅ | ✅ (Logic 10.7+) |
| **B. Capture** (fallback) | User plays the region once; plugin records it | Playback in sync with the host transport; export | ✅ | ✅ |
| **C. Multi-out** | ARA or capture | 6 stereo output buses, one per stem | ✅ | ⚠️ Logic only allows multi-out for *instrument* AUs, not effects. Possible as an instrument AU with a side-chain input, but awkward. |

**Recommendation:** v1 ships **Mode B (capture) + export** first, because it
is the simplest and works everywhere. **Mode A (ARA2)** is the v1.x target and
gives the "Melodyne-style" UX. Mode C is Reaper-only polish.

**Quick win for Reaper:** a ReaScript (Lua) action that renders the selected
item, calls the Python CLI, and imports the stems as new tracks. This gives a
working end-to-end DAW workflow in Phase 1, well before any C++ exists.

### 6.3 Tech stack
- **JUCE 8** (C++20, CMake). One codebase builds VST3, AU and Standalone.
  The ARA SDK is integrated through JUCE's `juce_audio_plugin_client` ARA
  support.
- **Inference runtime: ONNX Runtime** (C++ API) with the **CoreML execution
  provider** on macOS, so it can use the Neural Engine or GPU, falling back to
  the CPU provider.
  - Alternative: convert directly to Core ML with `coremltools`. Possibly
    faster on Apple Silicon, but macOS-only. Benchmark both in Phase 3.
- STFT/iSTFT: Apple **Accelerate/vDSP**, behind a small interface so other
  platforms can be added later.
- Model weights ship next to the plugin bundle
  (`Contents/Resources/model.onnx`, roughly 80–170 MB depending on precision).
  If embedding them causes problems, download on first run with a checksum.

### 6.4 Plugin functional requirements
- P1. The plugin loads in Reaper (VST3) and Logic (AU) and passes `auval`
  validation and the Steinberg VST3 validator.
- P2. Separation runs on a worker thread. The audio thread never blocks or
  allocates.
- P3. Shows progress and can be cancelled. The UI stays responsive.
- P4. Per-stem mute/solo/gain for preview, and export to WAV (choice of
  folder; default is the project folder).
- P5. Separated results are cached per audio source, using a content hash, so
  reopening a project doesn't redo the work. The cache location is set by the
  user and the cache size is capped.
- P6. Quality preset: **Fast** (no shifts, FP16) / **Best** (shifts=2, FP32).
- P7. Plugin state (which stem is selected, gains, cache reference) is saved
  with the DAW project.
- P8. Handles host sample rates 44.1–96 kHz and any buffer size.

### 6.5 Non-functional requirements
- N1. **Speed:** a 3-minute stereo song separates in **≤ 60 s** on an M-series
  Mac in Fast mode (stretch goal: ≤ 30 s).
- N2. **Memory:** peak RAM ≤ 2 GB during separation.
- N3. **No crashes in the host:** inference errors show up in the UI and never
  throw across the plugin boundary.
- N4. **Distribution:** a signed and notarized `.pkg` installer for macOS
  (needs an Apple Developer ID).

---

## 7. Phased roadmap

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **0. Foundations** | Python package skeleton, Poetry deps, CI (lint + tests), dataset download scripts | `pytest` green in CI; MUSDB18-HQ and MoisesDB prepared locally |
| **1. Baseline** | `isolate separate` CLI using pretrained `htdemucs_6s`; evaluation harness; Reaper ReaScript integration | Baseline SDR table (§8) for all 6 stems; end-to-end stems in Reaper |
| **2. Fine-tune** | Stem mapping, augmentation, head/partial fine-tuning on MoisesDB (+Slakh) | `keys` and `guitar` SDR improve by **≥ 1.0 dB** over the baseline, and no core stem drops by more than 0.2 dB |
| **3. Export** | ONNX export, C++ STFT, parity tests, ORT vs Core ML benchmark | Parity test passes; N1 speed target met in a C++ test harness |
| **4. Plugin v1** | JUCE plugin with Capture mode, preview and export; VST3 + AU | P1–P8 pass; manual test in Reaper and Logic |
| **5. Plugin v1.x** | ARA2 integration; multi-out for Reaper; installer | Separation from the clip without a capture pass in both DAWs |
| **6. Stretch** | RoFormer-based vocals/drums, distillation to a smaller model, Windows build | — |

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
- **Performance:** track wall-clock time and peak memory per model version on
  the reference machine (M4 Pro).

Results go in `docs/results/<version>.md` so progress can be compared over
time.

---

## 9. Testing strategy

| Layer | Tests |
|---|---|
| Python unit | Stem mapping, chunk/overlap-add round-trip (identity model → output equals input), resampling, I/O |
| Python integration | Separation on a 10 s fixture produces 6 stems of the right length; mixture consistency holds when R11 is enabled |
| Model regression | Evaluation on a small fixed subset in CI (CPU, a few clips), failing if SDR drops past a threshold |
| Export | ONNX parity (R14) |
| C++ | STFT/iSTFT round-trip, chunking, parity with Python outputs on fixtures (GoogleTest or Catch2) |
| Plugin | `pluginval` (strictness 5+), `auval -v`, VST3 validator, manual DAW test checklist |

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `keys` is ill-defined (piano vs synth vs pads overlap with `other` and `guitar`) | Low keys SDR, confusing results | Clear stem map; widen the target; keep the `other` residual; judge by ear as well as by SDR |
| HTDemucs ONNX export is fragile (complex STFT, dynamic shapes) | Phase 3 slips | Keep STFT outside the graph (R13); fixed segment length; fall back to Core ML or LibTorch |
| Free-tier GPU limits | Slow iteration | Fine-tune small parts of the model; MPS locally; resumable checkpoints |
| Dataset licenses are non-commercial | Can't sell it | Accept for a portfolio project; record a commercial path in Open Questions |
| ARA2 integration is complex | v1.x slips | Ship Capture mode first; ARA is additive |
| Logic doesn't support multi-out for effects | Mode C limited to Reaper | Use export/drag-out in Logic |
| Large model file inside the plugin bundle | Installer size, load time | FP16 weights; load lazily on first Separate |

---

## 11. Open questions

1. **Stem definition:** should `keys` include synth leads, or only
   piano/organ/pads? This decides the stem map.
2. **Commercial ambitions:** is this strictly a portfolio project? If not, the
   choice of datasets changes.
3. **Python version:** the `feature/cl-DevContainerConfig` branch bumps the
   devcontainer to Python 3.14, but `main` is on 3.11. Check that
   `demucs`, `torchaudio`, `museval` and `onnx` all have wheels for 3.14
   before standardising. 3.11 or 3.12 is the safer choice for the ML stack.
4. **Windows support:** worth planning for in v1 (ORT DirectML EP), or leave
   it as stretch?
5. **Inference runtime:** ONNX Runtime vs Core ML. Decide from the Phase 3
   benchmarks.

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
- JUCE ARA documentation — `juce.com`; Celemony ARA SDK — `github.com/Celemony/ARA_SDK`
- ONNX Runtime CoreML Execution Provider — `onnxruntime.ai`
