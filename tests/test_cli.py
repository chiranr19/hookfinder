import json

import soundfile as sf

from hookfinder.cli import main


def test_cli_human_output(song_wav, capsys):
    rc = main([str(song_wav), "--duration", "15"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "best hook" in out
    assert "repetition=" in out


def test_cli_json_output(song_wav, capsys):
    rc = main([str(song_wav), "--duration", "15", "--top", "2", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["file"] == str(song_wav)
    assert len(payload["hooks"]) >= 1
    h = payload["hooks"][0]
    assert {"start", "end", "duration", "score", "components"} <= set(h)


def test_cli_missing_file(capsys):
    rc = main(["does_not_exist.wav"])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_cli_export(song_wav, tmp_path, capsys):
    out_clip = tmp_path / "clip.wav"
    rc = main([str(song_wav), "--duration", "15", "--export", str(out_clip)])
    assert rc == 0
    assert out_clip.exists()
    y, sr = sf.read(str(out_clip))
    assert len(y) / sr > 5  # a real, non-empty clip


def test_cli_weight_overrides(song_wav, capsys):
    rc = main([str(song_wav), "-d", "15", "--w-repetition", "0.7", "--w-energy", "0.3", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hooks"]
