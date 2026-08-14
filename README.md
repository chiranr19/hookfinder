# hookfinder

[![CI](https://github.com/chiranr19/hookfinder/actions/workflows/ci.yml/badge.svg)](https://github.com/chiranr19/hookfinder/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://github.com/chiranr19/hookfinder)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Find the 30 seconds that hook you.** An explainable, dependency-light Python
package that locates the most memorable segment of a song — the chorus, the
hook, the part you'd use as a preview clip — and tells you *why* it picked it.

Built on [librosa](https://librosa.org). No cloud services, no API keys, no
model downloads. Point it at an audio file and it returns a start/end timestamp
plus a per-signal breakdown.

```python
from hookfinder import find_hook

hook = find_hook("song.mp3")
print(hook)               # Hook(start=57.00s, end=87.00s, duration=30.00s, score=0.91)
print(hook.components)    # {'repetition': 0.98, 'harmonic': 0.71, 'rhythmic': 0.66, ...}
```

```console
$ hookfinder song.mp3
song.mp3
  best hook: 0:57 -> 1:27 (30s, score 0.912)
    repetition=0.98  harmonic=0.71  rhythmic=0.66  energy=0.80  centrality=1.00
```

## Why

The durable, reusable piece of a music app is rarely the app — it's the
pipeline that answers *"which slice of this track do I preview?"* This package
is that pipeline, extracted and made general. A Tamil-film-music app is just
one consumer of it; so is a podcast teaser tool, a DJ crate-digging helper, or
a dataset-builder for MIR research.

## Install

```bash
pip install hookfinder
```

Or from source:

```bash
git clone https://github.com/chiranr19/hookfinder
cd hookfinder
pip install -e .
```

librosa pulls in `soundfile`/`audioread` for decoding. For formats beyond WAV
(mp3, m4a, webm), having [ffmpeg](https://ffmpeg.org/) on your PATH is
recommended.

## How it works

Every candidate window (default 30 s, stepped across the song) is scored on
five signals, each normalized to `[0, 1]`:

| Signal | Measures | Intuition |
| --- | --- | --- |
| **repetition** | recurrence of the section elsewhere in the song | the hook is the part the song keeps returning to |
| **harmonic** | frame-to-frame chroma stability | settled sections beat transitional churn |
| **rhythmic** | percussive onset strength | the groove is present and driving |
| **energy** | fraction of the window above a loudness threshold | sustained, not a quiet passage |
| **centrality** | structural position prior | hooks rarely live in the intro or outro |

The signals are combined with configurable weights (repetition leads by
default), the top window is chosen, and its start is snapped to a nearby beat
for a clean cut. Because the score is a transparent weighted sum, every result
carries the `components` that produced it — no black box.

The heavy lifting (HPSS, chroma, onset, beat tracking, the recurrence matrix)
runs **once per song**; scanning candidates is cheap.

## Usage

### One-liner

```python
from hookfinder import find_hook

hook = find_hook("song.mp3", clip_duration=30)
clip = (hook.start, hook.end)
```

### Tune it

```python
from hookfinder import HookFinder, Weights

finder = HookFinder(
    clip_duration=15,          # shorter teaser
    step=0.5,                  # finer search
    weights=Weights(repetition=0.5, energy=0.3, rhythmic=0.2,
                    harmonic=0.0, centrality=0.0),
    align_to_beat=True,
)
hook = finder.find("song.mp3")
```

### Several candidates

```python
finder = HookFinder()
for cand in finder.find_candidates("song.mp3", top_k=3):
    print(cand.start, cand.score, cand.components)
```

### Reuse features across settings

```python
from hookfinder import extract_from_file, HookFinder

feats = extract_from_file("song.mp3")        # analyze audio once
short = HookFinder(clip_duration=15).find(feats)
long  = HookFinder(clip_duration=45).find(feats)
```

### Command line

```bash
hookfinder song.mp3                     # best hook + breakdown
hookfinder song.mp3 --top 3             # top 3 non-overlapping candidates
hookfinder song.mp3 -d 15 --json        # 15s clip, JSON output
hookfinder song.mp3 --export teaser.wav # write the clip to disk
hookfinder song.mp3 --w-repetition 0.6 --w-energy 0.4
```

## API

- `find_hook(audio, clip_duration=30, sr=None, weights=None, align_to_beat=True, step=1.0) -> Hook`
- `HookFinder(...).find(audio) -> Hook`
- `HookFinder(...).find_candidates(audio, top_k=5) -> list[Hook]`
- `Hook`: `.start`, `.end`, `.duration`, `.score`, `.components`, `.to_dict()`
- `Weights(repetition, harmonic, rhythmic, energy, centrality)` — auto-normalized
- `extract_from_file(path) -> AudioFeatures` / `extract_features(y, sr) -> AudioFeatures`

`audio` may be a file path, a mono NumPy waveform (pass `sr`), or a precomputed
`AudioFeatures`.

## Limitations

- It's a **signal-based heuristic**, not a trained model. It finds the salient,
  repeated, high-energy section — which is usually the hook, but "catchiness"
  is subjective and it won't always agree with you.
- Instrumental or through-composed music (no repeating chorus) leans on the
  other four signals and is inherently harder.
- Beat tracking and recurrence degrade on very short or very noisy audio.
- It does **not** download audio. Bring your own files. (See
  [`examples/`](examples/) for how a downloader would sit *on top* of this.)

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT © Chiranjeev. See [LICENSE](LICENSE).
