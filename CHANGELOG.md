# Changelog

All notable changes to this project are documented here. This project adheres
to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-08-14

Speed, robustness, and honesty about what the score means.

### Added
- `quality` mode on `find_hook`, `HookFinder`, `extract_features`, and
  `extract_from_file`: `"fast"` (new default) and `"high"` (previous behavior).
  `fast` skips harmonic/percussive separation and analyzes the mix directly —
  roughly **13x faster analysis** (~1.5 s vs ~19 s on a 4.5-minute track).
- `--quality {fast,high}` CLI flag; `quality` echoed in `--json` output.
- Robustness test suite: silence, white noise, pure tone, NaN/Inf, empty input,
  int16 PCM, stereo, degenerate configs, and a ~12-minute long-file smoke test.
- README **Performance** section with measured timings.

### Changed
- Beat tracking now reuses the already-computed onset envelope instead of
  re-deriving it from the waveform.
- Candidate scores are smoothed with a 3-window moving average, so the pick is
  a stable plateau rather than a single-window spike.
- Integer-PCM and stereo waveforms are accepted and converted automatically.
- Suppressed librosa's non-actionable decode deprecation warnings.

### Fixed
- Recurrence matrix is capped at 2000 beat segments by coarsening the beat
  grid, bounding memory on hour-scale inputs (previously grew as N²).
- Non-finite (NaN/Inf) and empty audio now raise a clear `ValueError` instead
  of producing garbage scores.
- `find_candidates(top_k=0)` no longer returns an empty list that would make
  `find()` raise `IndexError`.

### Removed
- Unused `_weight_field_names` helper in `scoring`.

## [0.1.0] - 2026-08-14

Initial release.

### Added
- `find_hook()` one-liner and configurable `HookFinder` class.
- Five explainable scoring signals — repetition, harmonic stability, rhythmic
  intensity, sustained energy, and structural centrality — each normalized to
  `[0, 1]` and combined with configurable `Weights`.
- Per-frame recurrence (self-similarity) analysis as the primary hook signal.
- Beat-aligned clip boundaries.
- `find_candidates()` with non-max suppression for top-k non-overlapping clips.
- `hookfinder` CLI (`--json`, `--top`, `--export`, `--duration`, weight flags)
  and `python -m hookfinder`.
- Feature reuse via `extract_from_file` / `extract_features` /`AudioFeatures`.
- Test suite (synthetic audio, no network) and GitHub Actions CI on Python
  3.9–3.12.
