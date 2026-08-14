"""hookfinder — find the 30 seconds that hook you.

Explainable hook/chorus detection for building preview clips, built on librosa.

Quickstart
----------
>>> from hookfinder import find_hook
>>> hook = find_hook("song.mp3")             # doctest: +SKIP
>>> print(hook.start, hook.end, hook.score)  # doctest: +SKIP
>>> hook.components                          # per-signal breakdown  # doctest: +SKIP
"""

from .detector import Hook, HookFinder, find_hook
from .features import AudioFeatures, extract_features, extract_from_file, load_audio
from .scoring import DEFAULT_WEIGHTS, SIGNALS, Weights

__version__ = "0.1.0"

__all__ = [
    "find_hook",
    "HookFinder",
    "Hook",
    "Weights",
    "DEFAULT_WEIGHTS",
    "SIGNALS",
    "AudioFeatures",
    "extract_features",
    "extract_from_file",
    "load_audio",
    "__version__",
]
