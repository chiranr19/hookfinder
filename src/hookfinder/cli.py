"""Command-line interface: ``hookfinder song.mp3``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .detector import HookFinder
from .scoring import SIGNALS, Weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hookfinder",
        description="Find the ~30 seconds that hook you in an audio file.",
    )
    parser.add_argument("audio", type=Path, help="Path to an audio file (mp3, wav, m4a, ...).")
    parser.add_argument(
        "-d", "--duration", type=float, default=30.0,
        help="Clip length in seconds (default: 30).",
    )
    parser.add_argument(
        "-n", "--top", type=int, default=1,
        help="Show the top N non-overlapping candidates (default: 1).",
    )
    parser.add_argument(
        "--step", type=float, default=1.0,
        help="Seconds between candidate windows; smaller is finer but slower (default: 1.0).",
    )
    parser.add_argument(
        "--no-beat-align", action="store_true",
        help="Do not snap the clip start to a nearby beat.",
    )
    parser.add_argument(
        "-q", "--quality", choices=["fast", "high"], default="fast",
        help="'fast' (default) skips harmonic/percussive separation; "
             "'high' is slower but uses cleaner per-signal sources.",
    )
    parser.add_argument(
        "--export", type=Path, metavar="OUT",
        help="Write the best clip to this audio file (extension sets the format).",
    )
    parser.add_argument("--json", action="store_true", help="Emit results as JSON.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    weights = parser.add_argument_group("signal weights (relative; auto-normalized)")
    for name in SIGNALS:
        weights.add_argument(f"--w-{name}", type=float, default=None, dest=f"w_{name}")
    return parser


def _weights_from_args(args) -> Weights:
    overrides = {name: getattr(args, f"w_{name}") for name in SIGNALS}
    if all(v is None for v in overrides.values()):
        return Weights()
    defaults = Weights()
    return Weights(**{
        name: (overrides[name] if overrides[name] is not None else getattr(defaults, name))
        for name in SIGNALS
    })


def _export_clip(audio_path: Path, out_path: Path, start: float, end: float) -> None:
    import soundfile as sf

    from .features import load_audio

    y, sr = load_audio(audio_path, sr=None)
    clip = y[int(start * sr):int(end * sr)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, sr)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.audio.exists():
        print(f"error: file not found: {args.audio}", file=sys.stderr)
        return 2

    finder = HookFinder(
        clip_duration=args.duration,
        step=args.step,
        weights=_weights_from_args(args),
        align_to_beat=not args.no_beat_align,
        quality=args.quality,
    )

    try:
        hooks = finder.find_candidates(args.audio, top_k=max(1, args.top))
    except Exception as exc:  # pragma: no cover - surfaced to the user
        print(f"error: could not analyze {args.audio.name}: {exc}", file=sys.stderr)
        return 1

    if args.export:
        best = hooks[0]
        _export_clip(args.audio, args.export, best.start, best.end)

    if args.json:
        payload = {
            "file": str(args.audio),
            "clip_duration": args.duration,
            "quality": args.quality,
            "hooks": [h.to_dict() for h in hooks],
        }
        if args.export:
            payload["export"] = str(args.export)
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{args.audio.name}")
    for i, h in enumerate(hooks, 1):
        label = "best hook" if i == 1 else f"candidate {i}"
        print(
            f"  {label}: {_fmt(h.start)} -> {_fmt(h.end)} "
            f"({h.duration:.0f}s, score {h.score:.3f})"
        )
        breakdown = "  ".join(f"{name}={h.components.get(name, 0):.2f}" for name in SIGNALS)
        print(f"    {breakdown}")
    if args.export:
        print(f"  exported best clip -> {args.export}")
    return 0


def _fmt(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
