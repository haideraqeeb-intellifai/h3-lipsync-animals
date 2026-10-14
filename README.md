# H3 animal lip-sync: rectangular-mask runs

This repository preserves the PodcastFlow animal REF2VA experiment using a
single rectangular edit mask around each animal. The supplied image anchors
the shot; the first 15 seconds of the matching source-video audio conditions
H3 and is reused as the final soundtrack. Generation uses H3 Turbo v4, four
steps, seed 42, 24 fps, and a fast static-image VAE encode.

**Status (2026-10-14): 15 of 30 prepared cases have completed generations.**
Cases 01-15 are included with their raw H3 output, a 15-second export,
reference image, WAV, prompt, rectangle overlay, sampled frames, workflow,
generation report, and verification data. Case 16 failed locally and is not
presented as a completed result; cases 17-30 have prepared inputs only. The
framing screen marked 14 of 15 completed cases `pass` and red squirrel
`review`. It is not a lip-sync score or a substitute for watching the clips.

Open [the comparison report](podcastflow-animal-30-h3-rectangle-15s-20261014/index.html)
after cloning the repository to browse the 15 completed cases. It links to
the original PodcastFlow videos and uses local H3 videos and other assets.

## Contents

- `inputs/podcastflow_mask30/`: 30 prepared image/audio/prompt cases, source
  links, hashes, and rectangle coordinates.
- `outputs/podcastflow_mask30_final_*/`: the 15 completed native outputs and
  their per-run reports, workflows, audits, and GPU samples.
- `outputs/podcastflow_mask30/`: generation logs and framing audits for those
  15 cases.
- `podcastflow-animal-30-h3-rectangle-15s-20261014/`: self-contained partial
  report with 15-second H.264/AAC exports and comparison assets.
- `run_pilot.py`, `run_podcastflow_mask30.py`, and the other root scripts:
  preparation, generation, framing audit, and report export.
- `ComfyUI/`, `turbo/`, and `support/`: the source snapshot used by the harness.
  The local ComfyUI and Turbo modifications are included, along with their
  third-party licenses. No model parameters are included.

## Reproduce or continue

The included results are inspectable without a GPU. To run new cases, install
the dependencies in `ComfyUI/requirements.txt`, the `turbo/` node, and the
Python packages used by the harness (`torch`, `torchaudio`, `sageattention`,
`pynvml`, `numpy`, `Pillow`, and OpenCV with SIFT). The recorded environment
used Python 3.12, PyTorch 2.11.0+cu130, SageAttention 2.2.0, FFmpeg, and an
RTX 5090. Model files and their exact sources are listed in
[MODEL_REFERENCES.md](MODEL_REFERENCES.md); place them under `models/` in the
specified subdirectories. Weight downloads are large and subject to upstream
terms and access requirements.

Run `python run_podcastflow_mask30.py` to continue the prepared 30-case batch.
It skips and verifies completed cases before generating missing cases. Run
`python audit_podcastflow_mask30.py` to refresh the framing screen, then
`python build_podcastflow_mask30_report.py` to rebuild the scrolling report
from completed, audited cases. The reports record source SHA-256 hashes and
audio/locked-region latent checks. Historical absolute paths in those records
refer to the original run host; the files are organized by case in this repo.

The original comparison inputs come from the
[PodcastFlow 30-video report](https://s3.renderplatform.com/user-assets/preview/agent/reports/podcastflow-animal-30-videos-20261014/index.html).
Source-video URLs are linked rather than copied here. No new license is
asserted for the source media or this experiment's code.
