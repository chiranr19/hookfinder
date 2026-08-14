"""Find hooks for every audio file in a folder and write a CSV.

This is the shape of the original use case that motivated the package: a music
app that needs a preview clip for each track in a catalog. The app is *just a
consumer* of hookfinder — the pipeline lives here.

    python examples/batch_folder.py ./my_songs hooks.csv --duration 30
"""

import argparse
import csv
from pathlib import Path

from hookfinder import HookFinder

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".aac"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder of audio files.")
    parser.add_argument("out_csv", type=Path, help="Where to write results.")
    parser.add_argument("-d", "--duration", type=float, default=30.0)
    args = parser.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"No audio files found in {args.folder}")
        return 1

    finder = HookFinder(clip_duration=args.duration)
    rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        try:
            hook = finder.find(path)
            rows.append({
                "file": path.name,
                "start": round(hook.start, 2),
                "end": round(hook.end, 2),
                "score": round(hook.score, 3),
                **{k: round(v, 3) for k, v in hook.components.items()},
                "status": "ok",
            })
        except Exception as exc:  # keep going; one bad file shouldn't stop the batch
            print(f"    failed: {exc}")
            rows.append({"file": path.name, "status": f"error: {exc}"})

    fieldnames = ["file", "start", "end", "score",
                  "repetition", "harmonic", "rhythmic", "energy", "centrality", "status"]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\nDone: {ok}/{len(files)} analyzed -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
