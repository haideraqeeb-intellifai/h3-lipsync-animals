"""Estimate subject-scale drift and make visual review strips for H3 outputs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".lip-review-deps"))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from run_podcastflow_mask30 import recorded_video

INPUT = ROOT / "inputs/podcastflow_mask30"
OUTPUT = ROOT / "outputs/podcastflow_mask30"
PREFIX = "podcastflow_mask30_final_"
SAMPLES = (*range(0, 360, 24), 359)
REVIEW_SAMPLES = (0, 72, 168, 264, 359)


def audit(row):
    folder = INPUT / row["id"]
    case = json.loads((folder / "case.json").read_text())
    out = ROOT / "outputs" / (PREFIX + row["id"])
    if not (out / "report.json").exists():
        return None
    report = json.loads((out / "report.json").read_text())
    if report["status"] != "complete":
        return None
    video = recorded_video(report, out)
    ref = cv2.imread(str(folder / "reference.jpg"))
    w, h = row["width"], row["height"]
    ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LANCZOS4)
    x0, y0, x1, y1 = case["animal_mask_box"]
    inset_x = .13 * (x1 - x0)
    inset_y = .10 * (y1 - y0)
    roi = np.zeros((h, w), np.uint8)
    roi[int((y0 + inset_y) * h):int((y1 - inset_y) * h),
        int((x0 + inset_x) * w):int((x1 - inset_x) * w)] = 255
    sift = cv2.SIFT_create()
    base_points, base_desc = sift.detectAndCompute(ref, roi)
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened(), video
    results = []
    review_frames = {}
    for index in SAMPLES:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        assert ok, (video, index)
        if index in REVIEW_SAMPLES:
            review_frames[index] = image
        points, desc = sift.detectAndCompute(image, roi)
        match_count = 0
        inliers = 0
        scale = None
        translation = None
        if base_desc is not None and desc is not None:
            paired = cv2.BFMatcher().knnMatch(base_desc, desc, k=2)
            good = [first for first, second in paired if first.distance < .75 * second.distance]
            match_count = len(good)
            if match_count >= 12:
                source = np.float32([base_points[m.queryIdx].pt for m in good])
                target = np.float32([points[m.trainIdx].pt for m in good])
                matrix, mask = cv2.estimateAffinePartial2D(
                    source, target, method=cv2.RANSAC, ransacReprojThreshold=3)
                if matrix is not None:
                    inliers = int(mask.sum())
                    scale = float(np.linalg.norm(matrix[0, :2]))
                    translation = [float(matrix[0, 2]), float(matrix[1, 2])]
        results.append(dict(frame=index, second=index / 24, matches=match_count,
                            inliers=inliers, subject_scale=scale,
                            translation_px=translation))
    capture.release()
    scales = [item["subject_scale"] for item in results if item["subject_scale"] is not None
              and item["inliers"] >= 15]
    verdict = "review" if len(scales) < len(SAMPLES) - 2 or max(abs(v - 1) for v in scales) > .03 or max(scales) - min(scales) > .04 else "pass"
    data = dict(case_id=row["id"], method="SIFT matches inside central animal rectangle; RANSAC partial affine scale every second plus the final native frame. This is a framing screen, not a lip-sync score.",
                verdict=verdict, scale_range=[min(scales), max(scales)] if scales else None,
                samples=results, mask_box=case["animal_mask_box"])
    review = OUTPUT / "framing" / row["id"]
    review.mkdir(parents=True, exist_ok=True)
    (review / "audit.json").write_text(json.dumps(data, indent=2) + "\n")
    images = [Image.open(folder / "reference.jpg").convert("RGB")] + [Image.fromarray(cv2.cvtColor(review_frames[index], cv2.COLOR_BGR2RGB)) for index in REVIEW_SAMPLES]
    cell = (240, 260) if h < w else (195, 340)
    strip = Image.new("RGB", (cell[0] * len(images), cell[1]), "white")
    draw = ImageDraw.Draw(strip)
    for i, image in enumerate(images):
        image.thumbnail((cell[0] - 8, cell[1] - 28))
        strip.paste(image, (i * cell[0] + (cell[0] - image.width) // 2, 25))
        draw.text((i * cell[0] + 4, 4), "reference" if i == 0 else f"{REVIEW_SAMPLES[i-1]/24:.1f}s", fill="black")
    strip.save(review / "frames.jpg", quality=90)
    print(f"{verdict.upper()} {row['id']} scales={data['scale_range']}", flush=True)
    return data


def main():
    rows = json.loads((INPUT / "source_cases.json").read_text())["cases"]
    OUTPUT.mkdir(exist_ok=True)
    audits = [result for row in rows if (result := audit(row)) is not None]
    (OUTPUT / "framing_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(f"AUDITED {len(audits)}/30 cases", flush=True)


if __name__ == "__main__":
    main()
