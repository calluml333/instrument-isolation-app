# Phase 1 — Baseline

| | |
|---|---|
| **Status** | Implemented: 1a (PR 11), 1b (PR 12) |
| **Phase** | 1 (docs/SPEC.md §7) |
| **Owner** | Callum Little |
| **Last updated** | 2026-11-12 |
| **Branch** | `feature/cl-Phase1aEngine` (PR 1a), then `feature/cl-Phase1bEval` (PR 1b) |

## 1. Goal

Make `isolate separate song.wav` produce six real stems from `htdemucs_6s`, on
CPU in the dev container, through the engine every later phase reuses. Then
measure it: a baseline quality table on the MUSDB18 7-second samples, and
speed and memory on a 3-minute song.

The phase ships as **two PRs**, so each is a reviewable size:

- **1a, engine:** backend interface, model registry and weights cache,
  `htdemucs_6s` backend, `isolate.core` engine, `isolate separate` and
  `isolate models`, type checking and the coverage gate.
- **1b, evaluation:** evaluation harness, `isolate eval`, benchmark, baseline
  results, the N1 target, and the CI regression check.

**Exit criteria (SPEC §7):** baseline results table (§8); N1 speed measured
and its target set.

## 2. Scope

### In scope

| Item | Traces to |
|---|---|
| Backend interface and model registry | SPEC M1, M4 |
| `htdemucs_6s` backend | SPEC §4.1, §7 Phase 1 |
| Engine: resampling, chunking with overlap-add, shifts, sample alignment, output writing, device selection, progress and cancel | SPEC R1–R7 |
| Weights download, checksum and cache; `isolate models list` / `download` | SPEC R8, R9 |
| Residual projection | SPEC R10 |
| `isolate separate` | SPEC §3 W1 |
| Evaluation harness: MUSDB18 7-second test split, SDR and SI-SDR, reports; listening-test template | SPEC §8 |
| Speed and memory benchmark; set the N1 target | SPEC N1, N3, §7 |
| Model regression check in CI | SPEC §9 |
| Coverage gate and type checking | SPEC §7 Phase 1 (moved from Phase 0) |

### Out of scope

- `htdemucs`, `htdemucs_ft` and ensemble backends, and the final Fast/Best
  choice: Phase 3. Phase 1 defines provisional presets for `htdemucs_6s` only.
- `isolate app` and anything Gradio: Phase 2.
- Streaming (constant-memory) processing for very long files, quantisation
  and export: Phase 4, if the measurement in P1-R21 shows it's needed.
- Running the listening test itself: the template ships in 1b, and the test
  is run in Phase 3, where it decides between models.
- `--model` values other than `htdemucs_6s`, and MP3 output.

## 3. Requirements and acceptance criteria

Tests marked **[weights]** need the real model weights. They are skipped by
default and run in their own CI job (P1-R15). Every other test runs offline.

### 1a — engine

| ID | Requirement | Acceptance (how it is verified) | Traces to |
|---|---|---|---|
| P1-R1 | **Backend interface.** `isolate.models.base.Backend` (a `Protocol`) exposes `name`, `stems` (ordered names), `sample_rate` (44100), `segment_seconds`, `normalize_input`, and `separate(chunk) -> dict[str, ndarray]`. `normalize_input` *(added in task 4)* asks the engine to standardise the whole mix (zero mean, unit variance) before chunking and undo it on the outputs, as Demucs expects; per-chunk normalisation inside a backend would change the level from chunk to chunk. `chunk` is float32 `(2, n)` at the backend rate with `n` ≤ one segment. It returns one `(2, n)` array per stem, for every name in `stems`. | A reusable contract test, `tests/contract.py::check_backend`, runs against the test stand-in and (**[weights]**) `htdemucs_6s`. | SPEC M1, §5.1 |
| P1-R2 | **Registry.** `isolate.models.registry` maps a model name to a `ModelSpec`: description, stems, licence and its source link, weight files (URL, SHA-256, size), and a factory that builds the backend. It contains `htdemucs_6s`. Unknown names raise `UnknownModelError`, listing the valid names. | `tests/test_registry.py`. | SPEC M4 |
| P1-R3 | **Weights cache.** Weights are downloaded on first use from the registry URL into `~/.cache/isolate/models/<model>/`, overridable with `ISOLATE_CACHE_DIR`. The download goes to a temp file, is checked against SHA-256, and only then renamed into place. A checksum mismatch deletes the file and raises `ChecksumError`. Files already present and verified are never downloaded again. | `tests/test_weights.py` uses `file://` URLs, so no sockets are needed: first fetch, cache hit with no second fetch, checksum mismatch, interrupted download leaves no file. | SPEC R8 |
| P1-R4 | **`htdemucs_6s` backend.** Loads the pinned safetensors weights (P1-D3) and returns `drums, bass, other, vocals, guitar, piano` for a chunk, by running demucs's `apply_model` on that chunk with no internal splitting or shifts. | **[weights]** contract test (P1-R1) and `test_htdemucs_6s_separates_fixture`. | SPEC §4.1 |
| P1-R5 | **Audio input.** `isolate.core.audio.load` reads WAV, AIFF, FLAC and MP3, mono or stereo, at any sample rate. It returns float32 `(channels, n)` and the rate. Unsupported, corrupt or empty files raise `AudioLoadError`, naming the file and the reason. Inputs with more than 2 channels are rejected. | `tests/test_audio.py` covers each format (written to `tmp_path` at test time), a corrupt file, a text file and an empty file. | SPEC R1, A1 |
| P1-R6 | **Resampling and channels.** Input is resampled to the backend rate with `soxr` (high quality), and mono is duplicated to stereo. Stems are resampled back to the input rate and channel count (mono input gives mono stems). Every stem has exactly the input's length. | `tests/test_engine.py`: 22.05 / 44.1 / 48 / 96 kHz, mono and stereo, through the identity stand-in. Lengths and channels match the input; at 44.1 kHz the round trip is within 1e-6 *(not bit-exact as drafted: cross-fading divides by the summed window weights, so float rounding remains)*, and at other rates the RMS error is below −60 dBFS. | SPEC R1, R4 |
| P1-R7 | **Chunked overlap-add.** The engine splits the input into backend-length segments with configurable overlap (default 25%), cross-fades with a triangular window normalised by the summed window weights, and trims padding. | `test_overlap_add_identity_round_trip`: with the identity stand-in, output equals input (max abs error < 1e-6) for lengths of 1 sample, exactly one segment, and 3.5 segments. | SPEC R2, R4 |
| P1-R8 | **Shifts.** With `shifts=N > 0`, the input is separated N times with random offsets of up to 0.5 s and the results averaged. The offsets come from a seeded generator (default seed 0), so output is deterministic. | `test_shifts_deterministic`: identical results for the same seed. `test_shifts_identity`: the identity stand-in still round-trips. | SPEC R3 |
| P1-R9 | **Presets.** `fast` = overlap 0.25, shifts 0; `best` = overlap 0.25, shifts 2. Both are provisional, for `htdemucs_6s` only, and are revisited in Phase 3. | `test_presets`. | SPEC R3, A2 |
| P1-R10 | **Device selection.** `auto` picks `mps`, then `cuda`, then `cpu`, whichever is available first. A forced device that isn't available raises `DeviceUnavailableError`. In the container, `auto` resolves to `cpu`. | `tests/test_device.py`, with the availability checks monkeypatched. | SPEC R6 |
| P1-R11 | **Progress and cancel.** `separate()` takes an optional `progress(fraction)` callback and an optional `CancelToken`. Progress is called after every chunk, never decreases, and ends at exactly 1.0. The token is checked between chunks: once cancelled, `separate()` raises `Cancelled` before starting the next chunk, and nothing is written. | `tests/test_engine.py`: progress sequence; cancel after chunk 2 stops after 2 backend calls; no output files exist. | SPEC R7 |
| P1-R12 | **Residual projection.** On by default. After resampling back, `other += input − Σ stems`, so the stems sum to the input (max abs error < 1e-5). It can be turned off. If `other` isn't among the requested stems, the residual still goes into `other` before stems are selected. | `test_residual_projection_sums_to_input` with a stand-in that drops 10% of the signal; `test_residual_off`. | SPEC R10 |
| P1-R13 | **`isolate separate`.** Usage: `isolate separate INPUT... [--out DIR] [--stems LIST] [--model NAME] [--preset fast\|best] [--device auto\|cpu\|mps\|cuda] [--format wav\|flac] [--bit-depth 24\|32f] [--no-residual]`. Defaults: `--out stems`, all stems, `htdemucs_6s`, `best`, `auto`, `wav`, and `32f` for WAV or `24` for FLAC *(refined in task 9: FLAC has no float mode, and `--format flac --bit-depth 32f` is a usage error)*. It writes `<out>/<input stem>/<stem>.<ext>` and shows progress on stderr. Exit codes: 0 success, 1 a readable error with no traceback, 2 usage error, 130 interrupted (with partial output removed). | `tests/test_cli_separate.py` uses the stand-in model (registered via a test fixture) and checks files, lengths, rates, exit codes and error text. **[weights]** one end-to-end run. | SPEC §3 W1, R5 |
| P1-R14 | **`isolate models`.** `list` prints each model's name, stems, licence and whether it's downloaded. `download NAME` fetches and verifies the weights. | `tests/test_cli_models.py` (offline, with `file://` registry entries). | SPEC R9 |
| P1-R15 | **Test stand-ins and the weights CI job.** `tests/fakes.py` provides an identity backend (returns the input as `other` and zeros elsewhere) and a lossy one, both with short segments so tests are fast. The `weights` pytest marker is registered and skipped unless `--run-weights` is passed. Those tests are allowed sockets. A `weights` CI job runs them with `~/.cache/isolate/models` cached, keyed on a hash of the registry. | Default `pytest` skips them; the `weights` CI job is green. | SPEC §9 |
| P1-R16 | **Type checking and coverage gate.** mypy runs in strict mode on `src/` (tests are checked non-strict). Coverage is measured with branches and must be ≥ 90%. Both run in the `python` CI job and inside the container. | CI `python` job runs `mypy` and `pytest --cov-fail-under=90`. | SPEC §7 Phase 1 |
| P1-R17 | **Docs and SPEC.** `docs/models.md` records the safetensors source (Hugging Face repo, pinned revision, SHA-256). SPEC M1 says the interface lives in `isolate.models` (matching §5.1). SPEC R8 names the Hugging Face source. README documents `isolate separate`. | Markdown lint and link check; review. | SPEC §4.1, M1, R8 |

### 1b — evaluation

| ID | Requirement | Acceptance (how it is verified) | Traces to |
|---|---|---|---|
| P1-R18 | **Dataset.** `isolate.eval.musdb` downloads the MUSDB18 7-second sample set (147 MB) on first use into `~/.cache/isolate/datasets/musdb18-7/` with the `musdb` package. It yields the 50 test-split tracks as mixture plus `drums, bass, other, vocals` at 44.1 kHz. `ffmpeg` is installed in the dev container and CI, because `musdb` needs it to decode stems. | A `weights`-marked test (it needs the network) downloads and reads the set in the `eval` CI job. An offline unit test uses a 2-track fake dataset on disk. | SPEC §8 |
| P1-R19 | **Metrics.** Per-stem SDR with museval (BSSEval v4, 1 s windows, median over windows, then median over tracks), and SI-SDR (own implementation, whole clip, median over tracks). Silent reference stems are excluded from that stem's median, and the count of excluded tracks is reported. *(Refined in task 13: museval refuses whole-clip silent references or estimates, so those stems skip the museval call. A silent reference scores null; a silent estimate of a real reference scores −∞, which is also left out of the median and counted as excluded. Leaving stems out doesn't change the others' SDR, because v4 SDR projects only onto the stem's own reference; a test checks this.)* | `tests/test_metrics.py`: SI-SDR of a signal against itself is ≥ 100 dB; against itself scaled ×0.5 also ≥ 100 dB (scale-invariant); known values for signal plus noise at a set SNR (±0.1 dB); SDR at a known SNR (±0.3 dB); silent references and estimates. | SPEC §8 |
| P1-R20 | **`isolate eval`.** `isolate eval --model NAME --preset P [--tracks N] [--track NAME]... [--dataset DIR [--wav]] [--out FILE]` separates the test split and writes a Markdown report (per-stem SDR and SI-SDR, track count, model, preset, versions, date, wall time) plus a JSON file of per-track scores beside it. *(Refined in task 13: MUSDB18 has no guitar or piano reference, so every model stem other than `drums`, `bass` and `vocals` is summed into `other` before scoring. The report says so.)* | `tests/test_eval.py` with the fake dataset and stand-in model. | SPEC §8 |
| P1-R21 | **Benchmark.** `isolate bench --duration 180 --preset P` separates a synthetic 3-minute stereo signal (speed doesn't depend on content). It reports model-load time, separation time, real-time factor and peak RSS, plus the CPU count, torch threads and the container's memory limit. It also runs at 20 minutes to check N3. *(Changed in task 14: a top-level `isolate bench` command run once per measurement, instead of `isolate eval bench` spawning a subprocess. Each run is its own process, so its peak RSS is the measurement, with less machinery.)* | `tests/test_bench.py` (stand-in, 5 s). The numbers themselves go in the results report. | SPEC N1, N3, §8 |
| P1-R22 | **Baseline results.** `docs/results/phase-1-baseline.md`: SDR and SI-SDR tables for `htdemucs_6s` with `fast` and `best` on the 50 test tracks, benchmark results for both presets, and the environment (CPU count, memory, torch and demucs versions). The README shows the summary table. | File exists; numbers reproducible with the commands recorded in it. | SPEC §8, §7 exit criteria |
| P1-R23 | **N1 target.** Based on the measurement, SPEC N1 gets a concrete target for the Fast preset (proposed: the measured time rounded up, plus about 25% headroom, with a stretch target), and N3 records whether 20-minute files stay under 4 GB. The owner approves the numbers in the 1b PR. | SPEC diff in the 1b PR. | SPEC N1, N3, §7 exit criteria |
| P1-R24 | **Regression check.** An `eval` CI job (on PRs touching `src/isolate/{core,models}/**`, `pyproject.toml` or `poetry.lock`, and on `main`) evaluates 3 fixed test tracks with `fast`. It fails if any core stem's median SDR is more than 0.5 dB below the value stored in `tests/data/regression-baseline.json`. The dataset and weights are cached. | CI job green on the 1b PR. | SPEC §9 |
| P1-R25 | **Listening-test template.** `docs/results/listening-test-template.md`: track list fields, and per model and stem: bleed, phasiness, transient smearing and overall rating, each on a 1–5 scale with notes. | Markdown lint; review. | SPEC §8 |

## 4. Design decisions

| ID | Decision | Alternatives considered | Reason |
|---|---|---|---|
| P1-D1 | `demucs==4.1.0` (the original author's maintained fork, published July 2026). | `demucs==4.0.1` (Meta's last release, 2023). | 4.0.1 needs `torchaudio`, whose latest release (2.11) pins torch to 2.11 while torch is at 2.14, and it pulls in training dependencies. 4.1.0 drops `torchaudio`, loads safetensors, and is the version that gets fixes. |
| P1-D2 | torch from PyTorch's CPU wheel index (a Poetry `explicit` source), `torch>=2.15,<3` (2.15 was released after the spec was drafted). The requirement excludes Intel macOS, and numpy's floor is `>=1.26`: demucs caps torch at `<2.3` and numpy at `<2` on Intel macOS only, which otherwise makes the lock unsolvable. The project never runs there. | The default PyPI torch. | On x86-64 Linux (CI), PyPI torch pulls several GB of CUDA libraries that a CPU-only project never uses. The CPU index has cp312 wheels for both x86-64 (CI) and aarch64 (the container on the M4). |
| P1-D3 | Weights from the author's Hugging Face repos as **safetensors** (`adefossez/HTDemucs-6s` at a pinned commit), downloaded by our own code and checked against SHA-256 from the Hugging Face file metadata. Loaded with demucs's `load_safetensors_model`. | Meta's `.th` files from `dl.fbaipublicfiles.com`; demucs's own downloader (`get_model`). | `.th` files are pickles, and loading one can run arbitrary code; safetensors can't. Our own download meets R8 (registry URL, full checksum, our cache directory, offline once cached), which `get_model` doesn't. Same signatures and author as the original weights. |
| P1-D4 | The engine owns chunking, overlap-add and shifts. Backends only ever see one segment. | Let demucs's `apply_model` do the splitting and shifts. | Progress and cancel happen between chunks (R7), the ensemble in Phase 3 needs one chunking scheme across models, and the overlap-add code is then tested once with the identity stand-in. |
| P1-D5 | NumPy float32 `(channels, samples)` at every boundary (I/O, engine, backend interface). torch appears only inside torch-based backends. | torch tensors everywhere. | soundfile and soxr are NumPy-native; the engine and its tests don't depend on torch; non-torch backends (ONNX in Phase 4) fit the same interface. |
| P1-D6 | Audio I/O with `soundfile` (libsndfile, which bundles MP3 decoding); resampling with `soxr`. | torchaudio (its I/O moved to torchcodec, which needs FFmpeg); `sphn` (demucs's new dependency); `julius` for resampling. | Mature, small wheels with no FFmpeg needed at runtime; soxr is fast and high quality. |
| P1-D7 | Default output is 32-bit float WAV; `--bit-depth 24` for PCM. | 24-bit PCM by default. | Float can't clip, and keeps the residual-projected stems summing exactly to the mix. Reaper and Logic both import it directly. |
| P1-D8 | Phase 1 holds the whole song in memory (a 3-minute stereo song is ~60 MB as float32). N3 is measured at 3 and 20 minutes. | Streaming chunk-by-chunk to disk from the start. | Far simpler for now. The `separate()` API (chunk loop, progress, cancel) doesn't change if streaming is added in Phase 4. |
| P1-D9 | Mono input produces mono stems; output channel count, rate and length always match the input. | Always output stereo. | Stems line up with, and sum to, the source in a DAW, whatever it was. |
| P1-D10 | The registry is Python (`dataclass` entries), not YAML. | `configs/models.yaml`. | Type-checked by mypy, holds a factory callable, nothing to parse. |
| P1-D11 | `Backend` is a `typing.Protocol`, enforced by a shared contract test. | An abstract base class. | Backends don't need to inherit anything (e.g. an ONNX backend later); the contract test checks behaviour, which an ABC can't. |
| P1-D12 | mypy (strict on `src/`). | pyright. | Pure Python with no Node download, and it's what the Ruff/pytest toolchain already sits beside. demucs ships `py.typed`; soundfile and soxr get per-module `ignore_missing_imports`. |
| P1-D13 | Evaluation dependencies (`musdb`, `museval`) live in a Poetry group `eval`, installed in the container and CI but not part of the runtime package. `isolate eval` imports them lazily and explains how to install them if missing. | Runtime dependencies. | Keeps the Spaces demo (Phase 5) image small. museval pulls pandas and scipy. |
| P1-D14 | `ffmpeg` installed with `apt-get` in `post-create.sh` and in the CI jobs that need it. | A dev container feature; avoiding it. | `musdb` (via `stempeg`) needs it to decode the `.stem.mp4` files, and only for evaluation. |
| P1-D15 | Real-weights tests sit behind a `weights` marker and `--run-weights` flag, and run in their own CI job with a cache. | Always run them; mock the model. | Keeps the default suite offline and fast (P0-D7), while still proving the real model works on every PR. |
| P1-D16 | Two PRs (1a engine, 1b evaluation), from one phase spec. | One PR; two phase specs. | 1b depends on 1a, and each PR stays reviewable. The exit criteria belong to the phase as a whole. |

## 5. Task breakdown

### PR 1a — `feature/cl-Phase1aEngine`

| # | Task | Requirements |
|---|---|---|
| 1 | This spec; SPEC M1 and R8 wording; `docs/models.md` safetensors source. | P1-R17 |
| 2 | Dependencies: torch (CPU source), demucs 4.1.0, numpy, soundfile, soxr, tqdm. mypy config and dev dependency; branch coverage with `fail_under = 90`; mypy step in the `python` CI job. | P1-R16 |
| 3 | `isolate.core.audio`: load and save (WAV/AIFF/FLAC/MP3 in; WAV 32f/24 and FLAC out), `AudioLoadError`. | P1-R5, P1-R13 |
| 4 | `Backend` protocol, contract test, identity and lossy stand-ins. | P1-R1, P1-R15 |
| 5 | Engine: resampling and channels, chunked overlap-add, shifts, presets, progress and cancel, residual projection. | P1-R6–R9, P1-R11, P1-R12 |
| 6 | Device selection. | P1-R10 |
| 7 | Registry, weights download, checksum and cache. | P1-R2, P1-R3 |
| 8 | `htdemucs_6s` backend; `weights` marker, `--run-weights`, and the `weights` CI job. | P1-R4, P1-R15 |
| 9 | `isolate separate` and `isolate models list` and `download`. | P1-R13, P1-R14 |
| 10 | README usage; traceability for 1a; PR. | P1-R17 |

### PR 1b — `feature/cl-Phase1bEval`

| # | Task | Requirements |
|---|---|---|
| 11 | `eval` Poetry group; `ffmpeg` in `post-create.sh`; dataset loader and fake dataset. | P1-R18 |
| 12 | SDR (museval) and SI-SDR metrics. | P1-R19 |
| 13 | `isolate eval` and the report writer. | P1-R20 |
| 14 | `isolate eval bench`. | P1-R21 |
| 15 | Run the baseline; write `docs/results/phase-1-baseline.md`; README summary; propose the N1 target in SPEC. | P1-R22, P1-R23 |
| 16 | Regression baseline file and the `eval` CI job. | P1-R24 |
| 17 | Listening-test template. | P1-R25 |
| 18 | Traceability for the whole phase; PR. | All |

## 6. Risks and open questions

### Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | The safetensors upload on Hugging Face differs from Meta's original weights. | Baseline isn't comparable with published Demucs numbers. | Same author and signatures. Spot-check once: separate one clip with both and compare (≤ 1e-4 max abs difference). Recorded in `docs/models.md`. |
| R2 | Poetry resolution with the CPU torch source differs between x86-64 (CI) and aarch64 (the container), e.g. the `+cpu` local version. | Lock file works on one platform only. | `poetry check --lock` and installs run on both: the `python` job (x86-64) and the dev container locally (aarch64). Fallback: per-platform markers on the torch dependency. |
| R3 | CPU separation is slow in Docker (7 GB, 12 CPUs allocated to Docker Desktop here). | The Best preset (shifts 2) takes several times longer than Fast; eval over 50 tracks × 2 presets takes a while. | The eval run is a one-off for the report; CI uses 3 tracks, Fast only. Torch thread count is reported in the benchmark. |
| R4 | A 20-minute song needs more than 4 GB in memory (P1-D8). | N3 not met for very long files. | Measured in P1-R21. If it fails, streaming becomes a Phase 4 item and SPEC records it. **Happened:** 4 GB covers about 6½ minutes; a 20-minute song was OOM-killed at 7.8 GB. Streaming is now in SPEC §7 Phase 4, N3 records the status, and the README warns about long songs. |
| R5 | Hugging Face or GitHub downloads are flaky in CI. | Red `weights` / `eval` jobs unrelated to the change. | Caches keyed on the registry and dataset; one retry on download. |
| R6 | museval is old (2023) and pulls pandas/scipy. | Install or compatibility problems on numpy 2. | It's isolated in the `eval` group. Fallback: compute BSSEval v4 SDR with `fast_bss_eval`, documented in the report. |
| R7 | CI time grows with torch in the `python` job (~200 MB CPU wheel). | Slower PR feedback. | Cached `.venv`; P0's 2-minute target is re-measured and recorded. |
| R8 | *(Found in task 2.)* demucs 4.1.0 depends on `sphn`, which has no Linux arm64 wheel, so the container builds it from source (Rust, ~30 s, wheel then cached). Its bundled Opus fails to configure under CMake 4. | A fresh container on the M4 fails `poetry install`. | `CMAKE_POLICY_VERSION_MINIMUM=3.5` in `devcontainer.json` `containerEnv`. `sphn` isn't imported by any demucs module the project uses. CI (x86-64) gets a wheel. |

### Open questions

None blocking. These are judgement calls you may want to override:

1. **Default output format** is 32-bit float WAV (P1-D7), not 24-bit PCM.
2. **Whole song in memory** in Phase 1 (P1-D8); streaming only if the
   20-minute measurement needs it.
3. **Best preset uses shifts 2** (P1-R9), until Phase 3 measures it properly.
4. **The regression threshold** is 0.5 dB on 3 tracks (P1-R24).

## 7. Traceability

Filled in at the end of each PR. 1a CI results are from PR 11: `Python` 57 s
(cold cache, so R7's 2-minute target holds), `Real-weights tests` 1 m 13 s,
`Dev container` 1 m 28 s.

| ID | Implementation | Verification | Status |
|---|---|---|---|
| P1-R1 | `src/isolate/models/base.py` | `tests/contract.py`, `tests/test_backends.py`, `tests/test_demucs_backend.py` (tiny model offline; `htdemucs_6s` with weights) | Met |
| P1-R2 | `src/isolate/models/registry.py` | `tests/test_registry.py` | Met |
| P1-R3 | `src/isolate/models/weights.py`, `src/isolate/models/loader.py` | `tests/test_weights.py` (`file://` URLs) | Met |
| P1-R4 | `src/isolate/models/demucs.py` | `test_htdemucs_6s_meets_the_contract`, `test_htdemucs_6s_separates_fixture` (`weights` CI job) | Met |
| P1-R5 | `src/isolate/core/audio.py` | `tests/test_audio.py` | Met |
| P1-R6 | `src/isolate/core/engine.py` (`_resample`, `_fit_length`, channel handling) | `test_resampling_and_channels_match_input`, `test_mono_input_gives_mono_stems_that_sum_to_it` | Met (acceptance corrected, see P1-R6) |
| P1-R7 | `engine._ChunkRunner._overlap_add` | `test_overlap_add_identity_round_trip`, `test_chunks_never_exceed_one_segment` | Met |
| P1-R8 | `engine._ChunkRunner.run` | `test_shifts_identity_round_trip`, `test_shifts_deterministic`, `test_shifts_average_several_passes` | Met |
| P1-R9 | `engine.PRESETS` | `test_presets` | Met |
| P1-R10 | `src/isolate/core/device.py` | `tests/test_device.py` | Met |
| P1-R11 | `engine.CancelToken`, progress in `_overlap_add` | `test_progress_rises_to_exactly_one`, `test_cancel_stops_between_chunks` | Met |
| P1-R12 | `engine.separate` (residual step) | `test_residual_*` | Met |
| P1-R13 | `src/isolate/cli.py` (`separate`) | `tests/test_cli_separate.py`; `test_end_to_end_with_htdemucs_6s` (`weights`) | Met |
| P1-R14 | `src/isolate/cli.py` (`models`) | `tests/test_cli_models.py` | Met |
| P1-R15 | `tests/fakes.py`, `tests/tiny_demucs.py`, `tests/conftest.py`, `weights` job in `ci.yml` | Default run skips 3 `weights` tests; `Real-weights tests` job green | Met |
| P1-R16 | `pyproject.toml` (mypy, coverage), `Mypy` step in `ci.yml` | `Python` job green; coverage 99% | Met |
| P1-R17 | `docs/models.md`, SPEC M1 and R8, README "Usage" | `Docs` job green | Met |
| P1-R18 | `src/isolate/eval/musdb.py`, `src/isolate/eval/deps.py`, `eval` Poetry group, FFmpeg in `post-create.sh` and `ci.yml` | `tests/test_musdb.py` (fake WAV dataset); real download in the `Eval` workflow | Met |
| P1-R19 | `src/isolate/eval/metrics.py` | `tests/test_metrics.py` | Met (refined, see P1-R19) |
| P1-R20 | `src/isolate/eval/runner.py`, `isolate eval` in `src/isolate/cli.py` | `tests/test_eval.py`; reports in `docs/results/phase-1/` | Met (refined, see P1-R20) |
| P1-R21 | `src/isolate/eval/bench.py`, `isolate bench` in `cli.py` | `tests/test_bench.py`; benchmark JSON in `docs/results/phase-1/` | Met (changed to `isolate bench`) |
| P1-R22 | `docs/results/phase-1-baseline.md`, README "Results" | Commands recorded in the report | Met |
| P1-R23 | SPEC N1 (≤ 60 s, stretch ≤ 30 s), SPEC N3 status, SPEC §7 Phase 4 | Approved by the owner on PR 12 (2026-11-12) | Met |
| P1-R24 | `tests/test_regression.py`, `tests/data/regression-baseline.json`, `.github/workflows/eval.yml` | `SDR regression check` job green on PR 12 (1 m 28 s) | Met |
| P1-R25 | `docs/results/listening-test-template.md` | `Docs` job green | Met |

1b CI results are from PR 12: `Python` 1 m 37 s (now also installing FFmpeg
and the `eval` group, still under R7's 2 minutes), `Real-weights tests` 53 s,
`SDR regression check` 1 m 28 s, `Dev container` 2 m 26 s. The regression
baseline was recorded on aarch64 (the container) and passes on CI's x86-64.

**Exit criteria:** baseline results table ✅ (`docs/results/phase-1-baseline.md`);
N1 measured ✅ (45 s) and target set ✅ (≤ 60 s, stretch ≤ 30 s, approved
2026-11-12).
