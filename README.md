# Qwen3-TTS Streaming

Real-time streaming audio generation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), with a **[Wyoming protocol](https://github.com/rhasspy/wyoming) server** for integrating Qwen3-TTS as a TTS provider in [Home Assistant](https://www.home-assistant.io/) and other Wyoming-compatible systems.

## Wyoming / Home Assistant Integration

This fork adds a Wyoming protocol wrapper (`qwen_tts/wyoming_server.py`) that exposes Qwen3-TTS as a streaming TTS service over TCP. This lets you use it directly as a TTS engine inside Home Assistant (via the Wyoming integration) or any other Wyoming-compatible voice assistant.

### Installation (Wyoming)

In addition to the base installation below, install the Wyoming dependency:

```bash
sudo apt install sox
pip install torch torchaudio flash-attn
pip install -e .
pip install wyoming
```

### Running the Wyoming Server

#### Recommended: with a voice clone file and optimizations

The recommended way to run the server is with a pre-extracted voice clone `.pt` file and `--enable-optimizations`. This gives the best quality and performance.

A voice clone file can be created by running the **Qwen3-TTS web UI locally** - not the HuggingFace demo, but a locally hosted instance. Inside the web UI you can clone a voice from a short audio sample and download the resulting `.pt` file. Passing this file via `--default-voice-pt` skips real-time embedding extraction on every request.

```bash
python -m qwen_tts.wyoming_server \
  --uri tcp://0.0.0.0:10200 \
  --device cuda \
  --language de \
  --default-voice-pt "/path/to/MyVoice.pt" \
  --enable-optimizations
```

#### With a reference audio file (extracted at startup)

If you don't have a `.pt` file yet, you can point directly to a `.wav` reference audio and provide its transcript. The embeddings are then extracted once at startup:

```bash
python -m qwen_tts.wyoming_server \
  --uri tcp://0.0.0.0:10200 \
  --device cuda \
  --language en \
  --default-voice "MyVoice" \
  --default-voice-ref "/path/to/reference.wav" \
  --default-voice-text "Transcript of the reference audio." \
  --enable-optimizations
```

#### Minimal (no voice clone, no optimizations)

```bash
python -m qwen_tts.wyoming_server \
  --uri tcp://0.0.0.0:10200 \
  --device cuda \
  --language en
```

#### All server arguments

| Argument | Default | Description |
|---|---|---|
| `--uri` | `tcp://0.0.0.0:10200` | TCP address to listen on |
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace model ID |
| `--device` | `cuda` / `cpu` | Compute device |
| `--language` | `en` | Default language code |
| `--default-voice` | `default` | Voice name reported to Wyoming clients |
| `--default-voice-pt` | — | Path to pre-extracted voice `.pt` file (recommended) |
| `--default-voice-ref` | — | Path to reference `.wav` for voice cloning |
| `--default-voice-text` | — | Transcript of reference audio |
| `--enable-optimizations` | off | Enable `torch.compile` + CUDA graph optimizations |
| `--emit-every-frames` | `12` | Frames between audio emissions |
| `--decode-window-frames` | `80` | Decoder context window |
| `--first-chunk-emit-every` | `5` | Phase 1 emit interval |
| `--first-chunk-decode-window` | `48` | Phase 1 decode window |
| `--first-chunk-frames` | `48` | Frames before switching to phase 2 |
| `--debug` | off | Verbose logging |

### Performance

- Tested on an RTX 3090 and RTX 5070 Ti.
- End-to-end latency is roughly **2–4 seconds** before the first audio plays in Home Assistant. This is acceptable for casual use, but not quite real-time conversational speed.
- **The very first request after starting the server will take significantly longer** — the model (and optionally `torch.compile`) needs to be fully loaded and warmed up. Subsequent requests are much faster.
- Streaming can occasionally be glitchy: audio may stutter or trip. This is most likely caused by the model not generating tokens fast enough to keep the audio stream smooth.

### Web UI

A Gradio-based web UI is included for testing synthesis locally before connecting to Home Assistant:

```bash
python streaming_ui.py
```

This opens a browser interface where you can type text, select a voice, and hear the output in real time — useful for verifying voice clones and tuning parameters.

---

## Features

From [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming):
- `stream_generate_voice_clone()` - streaming with voice cloning
- `stream_generate_pcm()` - real-time PCM audio streaming
- `torch.compile` + CUDA graphs optimization

Added in this fork:
- **Two-phase streaming** - faster first-chunk latency
- **Multiple EOS token detection** - broader termination coverage for reliable generation stopping. Fixes sped-up audio and runaway generation in streaming
- **Hann window crossfade** - click-free chunk boundaries with proper fade-in/fade-out
- **Repetition penalty for streaming** - prevents token loops that cause looping audio and runaway generation. Defaults to 1.0 (disabled) because streaming generates frame-by-frame with CUDA graph constraints where repetition manifests differently than the non-streaming path (which defaults to 1.05)

Experiments on branch: [wip/experimental](https://github.com/rekuenkdr/Qwen3-TTS-streaming/tree/wip/experimental)
- **`generate_fast()` codebook predictor** - lightweight codebook generation that skips HuggingFace `generate()` overhead for the 31-step autoregressive loop (1.13x faster per-frame)
- **Manual CUDA graph capture for codebook predictor** - captures the entire 31-step codebook loop as a single CUDA graph replay (2.15x faster per-frame, 12.94ms vs 27.88ms baseline)
- **Batch streaming** - generates audio for multiple texts in parallel via `batch_stream_generate_voice_clone()`, with per-item state tracking and independent EOS detection
- **Async CUDA stream decoding** - overlaps AR token generation with speech decoding on a separate CUDA stream (disabled by default, no measurable speedup on single GPU but may show improvements in multi-GPU setups)

## Installation

```bash
sudo apt install sox
pip install torch torchaudio flash-attn
pip install -e .
```

## Usage

```python
import torch
import sounddevice as sd
from qwen_tts import Qwen3TTSModel

# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-Base",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable optimizations (recommended for streaming)
model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    compile_mode="reduce-overhead",
)

# Create voice clone prompt from reference audio
prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="Transcript of the reference audio.",
)

# Stream audio with two-phase settings
for chunk, sr in model.stream_generate_voice_clone(
    text="Hello, this is a streaming TTS demo!",
    language="en",
    voice_clone_prompt=prompt,
    # Phase 2 settings (stable)
    emit_every_frames=12,
    decode_window_frames=80,
    # Phase 1 settings (fast first chunk)
    first_chunk_emit_every=5,
    first_chunk_decode_window=48,
    first_chunk_frames=48,
):
    sd.play(chunk, sr)
    sd.wait()
```

## Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 8 | Emit audio every N frames |
| `decode_window_frames` | 80 | Decoder context window |
| `overlap_samples` | 512 | Crossfade overlap between chunks (0 to disable) |
| `max_frames` | 10000 | Maximum codec frames to generate |
| `first_chunk_emit_every` | 0 | Phase 1 emit interval (0 = disabled) |
| `first_chunk_decode_window` | 48 | Phase 1 decode window |
| `first_chunk_frames` | 48 | Switch to phase 2 after N frames |
| `repetition_penalty` | 1.0 | Penalizes repeated tokens (1.0 = disabled) |
| `repetition_penalty_window` | 100 | Only penalize tokens from the last N steps (0 = unlimited) |

## Two-Phase Streaming

Standard streaming with Qwen's TTS library waits for `emit_every_frames` (e.g., 12) before emitting the first audio. Two-phase uses aggressive settings for the first chunk to improve latency, then switches to stable settings.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 (First N frames)      │  PHASE 2 (Rest of audio)      │
│  - emit_every = 5 (fast)       │  - emit_every = 12 (stable)   │
│  - decode_window = 48          │  - decode_window = 80         │
│  → FAST first chunk            │  → QUALITY for rest           │
└─────────────────────────────────────────────────────────────────┘
```

### Benchmarks

| Test | Method | emit | 1st Chunk | 1st Spdup | Total | Tot Spdup | RTF |
|------|--------|------|-----------|-----------|-------|-----------|-----|
| 2 | Baseline (no opt) | 12 | 570ms | 1.00x | 3.16s | 1.00x | 0.56 |
| 3 | Optimized | 12 | 389ms | 1.47x | 2.37s | 1.34x | 0.37 |
| 4 | Optimized_2 (stable) | 12 | 382ms | 1.49x | 2.27s | 1.39x | 0.36 |
| 5 | **Two-phase (5→12)** | 5→12 | **208ms** | **2.75x** | 2.58s | 1.23x | 0.39 |

User hears audio **362ms earlier** vs baseline, **174ms earlier** vs only optimized.

**First-chunk latency improvement:**
- vs Baseline: **2.75x faster** (570ms → 208ms, saves 362ms)
- vs Optimized: **1.87x faster** (389ms → 208ms, saves 181ms)
- vs Optimized_2: **1.84x faster** (382ms → 208ms, saves 174ms)

## Audio Quality Fixes

Streaming TTS can produce clicks, pops, and artifacts at chunk boundaries. This fork implements several fixes:

### Crossfade Blending

Chunks are blended using a Hann window crossfade to eliminate boundary discontinuities:

```python
# ~21ms at 24kHz, matches RMS check window
# Lower values may cause clicks, set to 0 to disable
DEFAULT_BLEND_SAMPLES = 512

# Hann crossfade
fade_out = 0.5 * (1 + np.cos(np.pi * t))
fade_in = 0.5 * (1 - np.cos(np.pi * t))
blended = prev_tail * fade_out + curr_head * fade_in
```

### Overlap Trimming

Each chunk is processed in this order to prevent audio duplication (echo artifacts):

1. Crossfade current chunk's HEAD with previous chunk's saved TAIL
2. Apply fade-in (first chunk only)
3. Save FULL processed chunk for next iteration's crossfade
4. Trim END of chunk before emission (this region will be replaced by next chunk's crossfade)
5. Yield trimmed chunk

### First/Last Chunk Fades

- **First chunk**: Hann fade-in prevents pop at audio start
- **Final chunk**: Hann fade-out prevents pop at audio end

## Optimization API

### enable_streaming_optimizations()

Call after loading the model to enable torch.compile and CUDA graphs:

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable optimizations (recommended)
model.enable_streaming_optimizations(
    decode_window_frames=80,         # Must match streaming parameter
    use_compile=True,                # torch.compile the decoder
    compile_mode="reduce-overhead",  # Includes CUDA graphs automatically
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decode_window_frames` | 80 | Window size (must match streaming call) |
| `use_compile` | True | Apply torch.compile to decoder |
| `use_cuda_graphs` | True | Capture CUDA graphs for fixed window |
| `compile_mode` | "reduce-overhead" | torch.compile mode |
| `use_fast_codebook` | False | Use fast codebook generation (experimental) |
| `compile_codebook_predictor` | True | Apply torch.compile to codebook predictor |



Based on:
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)
