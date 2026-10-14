"""Run one 15-second rectangular-mask H3 generation for each PodcastFlow case."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs/podcastflow_mask30"
OUTPUT = ROOT / "outputs/podcastflow_mask30"
PREFIX = "podcastflow_mask30_final_"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_video(report, out):
    path = Path(report["runs"][0]["video"])
    if not path.is_file():
        path = out / "runtime/output/cold_pilot/lion_00001_.mp4"
    return path


def verified(row, out):
    report = json.loads((out / "report.json").read_text())
    folder = INPUT / row["id"]
    case = json.loads((folder / "case.json").read_text())
    assert report["status"] == "complete" and len(report["runs"]) == 1
    assert report["animal_mask_box"] == case["animal_mask_box"]
    assert report["frames"] == 362 and report["fps"] == 24
    assert report["static_plate_mode"] == "repeated_chunk"
    assert report["inputs_sha256"] == case["prepared_inputs_sha256"]
    assert all(sha(folder / name) == value for name, value in case["prepared_inputs_sha256"].items())
    audit = report["runs"][0]["audio_audit"]
    assert audit["output_video_latent_outside_mask_max_error"] == 0
    assert audit["output_audio_latent_max_error"] == 0
    assert report["runs"][0]["counts"]["fixed_audio_forwards"] == 4
    assert recorded_video(report, out).is_file()


def main():
    rows = json.loads((INPUT / "source_cases.json").read_text())["cases"]
    assert len(rows) == 30
    OUTPUT.mkdir(exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for row in rows:
        name = PREFIX + row["id"]
        out = ROOT / "outputs" / name
        if out.exists():
            verified(row, out)
            print(f"EXISTING {row['number']:02d}/30 {row['id']}", flush=True)
            continue
        folder = INPUT / row["id"]
        case = json.loads((folder / "case.json").read_text())
        cmd = [sys.executable, "-u", str(ROOT / "run_pilot.py"),
               "--name", name, "--case", str(folder / "case.json"), "--single",
               "--fast-static-plate", "--animal-box", *map(str, case["animal_mask_box"])]
        print(f"START {row['number']:02d}/30 {row['id']} {case['animal_mask_box']}", flush=True)
        with (OUTPUT / (row["id"] + ".log")).open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Generation failed, retained at {out}; inspect {OUTPUT / (row['id'] + '.log')}")
        verified(row, out)
        print(f"DONE {row['number']:02d}/30 {row['id']}", flush=True)


if __name__ == "__main__":
    main()
