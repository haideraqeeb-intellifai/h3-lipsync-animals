"""Transcribe the exact 15-second source audio for H3 prompt grounding."""
import json
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".lip-review-deps"))
import faster_whisper
import numpy as np
from scipy.signal import resample_poly

INPUT = ROOT / "inputs" / "podcastflow_mask30"


def main():
    model = faster_whisper.WhisperModel(
        "small.en", device="cpu", compute_type="int8", cpu_threads=4,
        download_root=str(ROOT / "outputs/lip_sync_review/models"), local_files_only=True,
    )
    cases = [json.loads(p.read_text()) for p in sorted(INPUT.glob("[0-9][0-9]-*/source.json"))]
    if len(sys.argv) > 1:
        selected = set(sys.argv[1:])
        cases = [row for row in cases if row["id"] in selected]
    for row in cases:
        folder = INPUT / row["id"]
        target = folder / "transcript.json"
        if target.exists():
            continue
        with wave.open(str(folder / "audio.wav"), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 48000, 720000)
            samples = np.frombuffer(wav.readframes(720000), dtype="<i2").astype(np.float32) / 32768
        samples = resample_poly(samples, 1, 3).astype(np.float32)
        segments, _ = model.transcribe(samples, language="en", beam_size=3,
                                       vad_filter=False, condition_on_previous_text=False)
        segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
        transcript = " ".join(s["text"] for s in segments).strip()
        assert transcript, row["id"]
        target.write_text(json.dumps({"text": transcript, "segments": segments,
                                      "model": "Systran/faster-whisper-small.en",
                                      "purpose": "Approximate source speech for prompt grounding; audio remains authoritative"}, indent=2) + "\n")
        print(f"TRANSCRIBED {row['number']:02d}/30 {row['id']}: {transcript[:90]}", flush=True)


if __name__ == "__main__":
    main()
