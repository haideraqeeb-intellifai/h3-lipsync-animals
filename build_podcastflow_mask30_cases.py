"""Freeze H3 cases using the linked image, matched audio, and reviewed rectangles."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs/podcastflow_mask30"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = json.loads((INPUT / "source_cases.json").read_text())["cases"]
    boxes = json.loads((INPUT / "mask_boxes.json").read_text())
    assert len(rows) == len(boxes) == 30
    template = (ROOT / "prompts/podflow_native_gesture.txt").read_text()
    gesture = next(part for part in template.split("\n\n") if part.startswith("An already-visible"))
    template = template.replace(gesture,
        "The body, paws, wings, clothes and accessories stay in their pictured positions. "
        "Only the native mouth or beak makes small speech movements. The head and torso retain "
        "their original scale and position in every frame. The final frame keeps the opening composition.")
    template_path = INPUT / "prompt_template.txt"
    template_path.write_text(template)
    for row in rows:
        folder = INPUT / row["id"]
        box = boxes[row["id"]]
        assert len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
        transcript = json.loads((folder / "transcript.json").read_text())["text"]
        anchor = (f"The linked presenter image shows the single {row['animal'].lower()} speaker exactly as pictured. "
                  "Keep its species-specific face, markings, covering, accessories, posture, "
                  "microphone, chair, tabletop and background at their pictured sizes and positions. "
                  "Preserve the complete frame edges and the original camera viewpoint.")
        prompt = template.format(animal=row["animal"].lower(), duration="15",
                                 visual_anchor=anchor, dialogue=transcript)
        (folder / "prompt.txt").write_text(prompt)
        case = dict(row, animal_mask_box=box, freeze_prompt=True,
                    prompt_template_sha256=sha(template_path),
                    transcript_sha256=sha(folder / "transcript.json"),
                    prepared_inputs_sha256={name: sha(folder / name) for name in
                                            ("reference.jpg", "audio.wav", "prompt.txt")})
        (folder / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        print(f"CASE {row['number']:02d}/30 {row['id']} box={box}", flush=True)


if __name__ == "__main__":
    main()
