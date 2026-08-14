"""Find the hook and write the clip out as a WAV.

    python examples/export_clip.py song.mp3 teaser.wav --duration 20
"""

import argparse
from pathlib import Path

import librosa
import soundfile as sf

from hookfinder import find_hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("-d", "--duration", type=float, default=30.0)
    args = parser.parse_args()

    hook = find_hook(str(args.audio), clip_duration=args.duration)
    print(f"Hook: {hook}")

    y, sr = librosa.load(str(args.audio), sr=None, mono=True)
    clip = y[int(hook.start * sr):int(hook.end * sr)]
    sf.write(str(args.out), clip, sr)
    print(f"Wrote {hook.duration:.0f}s clip -> {args.out}")


if __name__ == "__main__":
    main()
