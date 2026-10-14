"""Reject nearly flat output frames; this is not a semantic or lip-sync score."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image


def check(video, reference):
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-of', 'json', str(video)]))
    duration = float(metadata['format']['duration'])
    with Image.open(reference) as image:
        ref = np.asarray(image.convert('L').resize((128, 224)), dtype=np.float32)
    samples = []
    for fraction in [.1, .5, .9]:
        time = duration * fraction
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-ss', str(time), '-i', str(video),
                                       '-frames:v', '1', '-vf', 'scale=128:224,format=gray', '-f', 'rawvideo', '-'])
        frame = np.frombuffer(raw, dtype=np.uint8)
        if frame.size != 128 * 224:
            raise RuntimeError('Could not decode visual-check frame')
        ratio = float(frame.std() / max(float(ref.std()), 1))
        dynamic_range = float(np.percentile(frame, 99) - np.percentile(frame, 1))
        samples.append({'time_seconds': time, 'contrast_ratio_to_reference': ratio,
                        'p99_minus_p1': dynamic_range, 'passed': ratio >= .2 and dynamic_range >= 24})
    return {'status': 'passed' if all(s['passed'] for s in samples) else 'failed',
            'method': 'Three decoded grayscale frames. Reject contrast below 20 percent of the reference or a p99-p1 range below 24/255. A near-blank rejection check only, not semantic fidelity or lip-sync validation.',
            'samples': samples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(args.video, args.reference)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
