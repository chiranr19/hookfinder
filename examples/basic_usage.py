"""Find the hook in a single song and print the breakdown.

    python examples/basic_usage.py path/to/song.mp3
"""

import sys

from hookfinder import find_hook


def main():
    if len(sys.argv) < 2:
        print("usage: python examples/basic_usage.py <audio-file>")
        return 1

    hook = find_hook(sys.argv[1], clip_duration=30)

    print(hook)
    print("\nWhy this segment (each signal in [0, 1]):")
    for name, value in hook.components.items():
        bar = "#" * int(round(value * 30))
        print(f"  {name:<11} {value:0.2f} {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
