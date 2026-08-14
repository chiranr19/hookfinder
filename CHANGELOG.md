# Changelog

All notable changes to this project are documented here. This project adheres
to [Semantic Versioning](https://semver.org/).

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
