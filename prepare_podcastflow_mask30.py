"""Prepare the 2026-10-14 PodcastFlow report for 15-second H3 mask runs."""
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import wave

from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://s3.renderplatform.com/user-assets/preview/agent/reports/podcastflow-animal-30-videos-20261014/index.html"
DEST = ROOT / "inputs" / "podcastflow_mask30"


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.field = None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "article":
            self.row = {"display_ratio": attrs["data-ratio"]}
            self.rows.append(self.row)
        if self.row is None:
            return
        if tag == "source":
            self.row["source_video_url"] = attrs["src"]
        elif tag == "a" and "/headshots/" in attrs.get("href", ""):
            self.row["reference_url"] = attrs["href"]
        elif tag == "h2":
            self.field = "animal"
        elif tag == "p" and attrs.get("class") == "name":
            self.field = "presenter"

    def handle_endtag(self, tag):
        if tag in ("h2", "p"):
            self.field = None
        elif tag == "article":
            self.row = None

    def handle_data(self, data):
        if self.row is not None and self.field:
            self.row[self.field] = data.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path):
    if path.exists():
        return
    temp = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    temp.replace(path)


def extract_audio(url, folder):
    audio = folder / "audio.wav"
    if audio.exists():
        return
    prefix = folder / "source_prefix.partial.mp4"
    temp = folder / "audio.partial.wav"
    prefix.unlink(missing_ok=True)
    chunk_size = 4 * 1024 * 1024
    for index in range(32):
        start = index * chunk_size
        end = start + chunk_size - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"].startswith(f"bytes {start}-")
                    block = response.read()
                assert 0 < len(block) <= chunk_size
                with prefix.open("ab") as out:
                    out.write(block)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
        if index < 2:
            continue
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(prefix), "-vn", "-af",
            "atrim=duration=15,asetpts=PTS-STARTPTS", "-ac", "1", "-ar", "48000",
            "-c:a", "pcm_s16le", str(temp),
        ], capture_output=True, text=True)
        if result.returncode == 0 and temp.exists():
            with wave.open(str(temp), "rb") as wav:
                if wav.getnframes() == 15 * 48000:
                    temp.replace(audio)
                    prefix.unlink()
                    return
        if index % 4 == 3:
            print(f"EXTEND {folder.name} source prefix to {prefix.stat().st_size // 1048576} MiB", flush=True)
    raise RuntimeError(f"Could not extract 15 seconds of audio from {url}")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    report = DEST / "source_report.html"
    download(SOURCE_URL, report)
    parser = ReportParser()
    parser.feed(report.read_text())
    assert len(parser.rows) == 30
    assert sum(row["display_ratio"] == "9:16" for row in parser.rows) == 15
    assert sum(row["display_ratio"] == "16:9" for row in parser.rows) == 15
    rows = []
    for number, row in enumerate(parser.rows, 1):
        assert all(row.get(k) for k in ("animal", "presenter", "source_video_url", "reference_url"))
        assert row["source_video_url"].startswith("https://cdn.podcastflow.ai/")
        assert row["reference_url"].startswith("https://cdn.podcastflow.ai/")
        slug = re.sub(r"[^a-z0-9]+", "-", row["animal"].lower()).strip("-")
        case_id = f"{number:02d}-{slug}"
        folder = DEST / case_id
        folder.mkdir(exist_ok=True)
        image = folder / "reference.jpg"
        audio = folder / "audio.wav"
        download(row["reference_url"], image)
        with Image.open(image) as im:
            im.load()
            iw, ih = im.size
        assert (iw < ih) == (row["display_ratio"] == "9:16")
        width, height = (640, 1120) if iw < ih else (1120, 640)
        assert iw * height == ih * width, (case_id, (iw, ih))
        extract_audio(row["source_video_url"], folder)
        with wave.open(str(audio), "rb") as wav:
            assert wav.getframerate() == 48000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
            assert wav.getnframes() == 15 * 48000, (case_id, wav.getnframes())
        row.update(id=case_id, number=number, reference_filename="reference.jpg",
                   reference_size=[iw, ih], width=width, height=height,
                   native_frames=362, duration_seconds=15, seed=42, ref_image_size="match",
                   source_report_url=SOURCE_URL,
                   source_report_sha256=sha(report),
                   reference_sha256=sha(image), audio_sha256=sha(audio))
        rows.append(row)
        (folder / "source.json").write_text(json.dumps(row, indent=2) + "\n")
        print(f"PREPARED {number:02d}/30 {case_id}", flush=True)
    (DEST / "source_cases.json").write_text(json.dumps({"source_url": SOURCE_URL, "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
