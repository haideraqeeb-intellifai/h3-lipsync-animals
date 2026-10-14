"""Export 30 fifteen-second H3 videos and a browsable comparison report."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw

from run_podcastflow_mask30 import INPUT, OUTPUT, PREFIX, ROOT, recorded_video, sha, verified

DEST = ROOT / "podcastflow-animal-30-h3-rectangle-15s-20261014"


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration:stream=codec_type,width,height,nb_frames,r_frame_rate",
                             "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    *map(str, args)], check=True)


def export(row, audit):
    folder = INPUT / row["id"]
    case = json.loads((folder / "case.json").read_text())
    out = ROOT / "outputs" / (PREFIX + row["id"])
    verified(row, out)
    report = json.loads((out / "report.json").read_text())
    native = recorded_video(report, out)
    asset = DEST / "assets" / row["id"]
    asset.mkdir(parents=True, exist_ok=True)
    for name in ("reference.jpg", "audio.wav", "prompt.txt", "transcript.json", "case.json"):
        shutil.copy2(folder / name, asset / name)
    target = asset / "video.mp4"
    if not target.exists():
        temp = asset / "video.partial.mp4"
        ffmpeg("-i", native, "-i", folder / "audio.wav", "-map", "0:v:0", "-map", "1:a:0",
               "-frames:v", "360", "-t", "15", "-c:v", "libx264", "-threads", "4",
               "-crf", "18", "-preset", "fast", "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart", temp)
        temp.replace(target)
    metadata = probe(target)
    video_stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    assert (video_stream["width"], video_stream["height"]) == (row["width"], row["height"])
    assert int(video_stream["nb_frames"]) == 360
    assert abs(float(metadata["format"]["duration"]) - 15) < .05
    with Image.open(folder / "reference.jpg") as image:
        overlay = image.convert("RGB")
    x0, y0, x1, y1 = case["animal_mask_box"]
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((x0 * overlay.width, y0 * overlay.height,
                    x1 * overlay.width, y1 * overlay.height), outline="#3ef070", width=10)
    overlay.save(asset / "mask.jpg", quality=92)
    shutil.copy2(OUTPUT / "framing" / row["id"] / "frames.jpg", asset / "frames.jpg")
    verification = dict(case_id=row["id"], source_report=row["source_report_url"],
                        source_video=row["source_video_url"], linked_reference=row["reference_url"],
                        source_headshot_sha256=case["reference_sha256"],
                        audio_sha256=case["audio_sha256"],
                        prepared_inputs_sha256=case["prepared_inputs_sha256"],
                        mask_box=case["animal_mask_box"],
                        native_frames=362, export_frames=360, export=metadata,
                        audio_latent_max_error=report["runs"][0]["audio_audit"]["output_audio_latent_max_error"],
                        outside_mask_latent_max_error=report["runs"][0]["audio_audit"]["output_video_latent_outside_mask_max_error"],
                        framing_audit=audit,
                        method="First 360 H3 native frames at 24 fps, with the first 15 seconds of source-video audio muxed as the soundtrack; no spatial crop or zoom applied in export.")
    (asset / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    media = {name: f"assets/{row['id']}/{name}" for name in
             ("reference.jpg", "audio.wav", "video.mp4", "mask.jpg", "frames.jpg", "prompt.txt", "transcript.json", "verification.json")}
    return dict(id=row["id"], number=row["number"], animal=row["animal"], presenter=row["presenter"],
                display_ratio=row["display_ratio"], image_ratio="4:7" if row["width"] < row["height"] else "7:4",
                source_video_url=row["source_video_url"], reference_url=row["reference_url"],
                media=media, mask_box=case["animal_mask_box"], framing_audit=audit,
                generation_seconds=report["runs"][0].get("generation_seconds"))


def page(data):
    payload = json.dumps(data).replace("<", "\\u003c")
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PodcastFlow · H3 Animal Lip Sync Tests</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f3fa;color:#252137;font:16px/1.5 system-ui,sans-serif}main{max-width:1580px;margin:auto;padding:40px 24px}header{max-width:1000px}h1{font-size:clamp(32px,4vw,54px);line-height:1.08;margin:12px 0 18px;letter-spacing:-1.5px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:1.4px;font-weight:750;color:#7652bb}p{margin:10px 0}.muted{color:#696275}.stats,.toolbar{display:flex;gap:10px;flex-wrap:wrap}.stats{margin:24px 0}.stat{background:#fff;border:1px solid #e3ddec;border-radius:14px;padding:14px 22px}.stat strong{display:block;font-size:25px}.toolbar{position:sticky;top:0;z-index:2;background:#f5f3faef;padding:12px 0;margin-bottom:12px;align-items:center}input,button{font:inherit;border:1px solid #dcd3eb;border-radius:9px;padding:10px 14px;background:#fff}input{flex:1;min-width:220px}button{cursor:pointer}button.active{background:#6640a6;color:#fff}#count{font-size:14px;color:#6d647c}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(580px,1fr));gap:22px}.card{background:#fff;border:1px solid #e4deec;border-radius:18px;overflow:hidden;box-shadow:0 5px 20px #34224b08}.card[hidden]{display:none}.body{padding:19px}h2{font-size:25px;margin:4px 0}.name{color:#81718f}.compare{display:grid;grid-template-columns:1fr 1fr;gap:10px;background:#141019;padding:10px}.compare figure{margin:0;min-width:0}.compare video{display:block;width:100%;height:315px;object-fit:contain;background:#141019}.compare figcaption{color:#eee;font-size:13px;padding:6px 2px 0}.links{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;font-size:14px}a{color:#6640a6}.meta{color:#6d647c;font-size:14px}.tag{display:inline-block;border-radius:999px;padding:3px 9px;background:#e9f5ed;color:#26734d;font-size:12px;font-weight:700}.tag.review{background:#fff1d8;color:#8a620a}details{border-top:1px solid #eee6f5;margin-top:12px;padding-top:10px;font-size:13px;color:#71667e}summary{cursor:pointer}code{font-size:12px}footer{color:#867992;margin-top:32px;font-size:13px}@media(max-width:650px){.grid{grid-template-columns:1fr}.compare{grid-template-columns:1fr}.compare video{height:290px}}
</style></head><body><main><header><div class="eyebrow">PodcastFlow / H3 rectangular animal masks</div><h1>30 animal lip-sync tests, 15 seconds each.</h1><p class="muted">Each H3 video uses the linked presenter image and the first 15 seconds of the matching PodcastFlow video’s soundtrack. The green rectangle shows the editable area; the surrounding image is anchored. Play the source and H3 clips side by side.</p></header><div class="stats"><div class="stat"><strong>30</strong>H3 videos</div><div class="stat"><strong>15 s</strong>Each output</div><div class="stat"><strong>15 / 15</strong>Portrait / landscape</div><div class="stat"><strong id="passed">—</strong>Framing screen passed</div></div><p class="muted">The presenter images are 4:7 or 7:4; H3 keeps those exact image proportions. The source PodcastFlow videos are displayed as 9:16 or 16:9. The source video may continue past 15 seconds when opened directly; playback here pauses at 15 seconds.</p><div class="toolbar"><input id="search" type="search" placeholder="Search animals or presenters" aria-label="Search animals or presenters"><button class="active" data-filter="all">All</button><button data-filter="9:16">Portrait</button><button data-filter="16:9">Landscape</button><span id="count"></span></div><section id="grid" class="grid"></section><footer>H3 REF2VA Turbo v4, four steps, seed 42, 24 fps. The original 15-second soundtrack is muxed into each generated video. Framing scale is screened with image matches inside the animal rectangle; this is not a calibrated lip-sync score. Review the videos to judge mouth timing and appearance. Source: <a id="source-link">PodcastFlow 30-video report</a>.</footer></main><script id="data" type="application/json">""" + payload + """</script><script>
'use strict';const data=JSON.parse(document.getElementById('data').textContent),grid=document.getElementById('grid');const esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));document.getElementById('source-link').href=data.source_url;document.getElementById('passed').textContent=data.cases.filter(c=>c.framing_audit.verdict==='pass').length;function pauseOther(active){document.querySelectorAll('video').forEach(v=>{if(v!==active)v.pause()})}for(const c of data.cases){const card=document.createElement('article');card.className='card';card.dataset.ratio=c.display_ratio;card.dataset.search=(c.animal+' '+c.presenter).toLowerCase();const a=c.framing_audit;card.innerHTML=`<div class="compare"><figure><video controls preload="none" poster="${esc(c.media['reference.jpg'])}" src="${esc(c.source_video_url)}"></video><figcaption>Original PodcastFlow · first 15 s</figcaption></figure><figure><video controls preload="none" poster="${esc(c.media['reference.jpg'])}" src="${esc(c.media['video.mp4'])}"></video><figcaption>H3 rectangle mask · 15 s</figcaption></figure></div><div class="body"><div class="eyebrow">${String(c.number).padStart(2,'0')} / ${esc(c.display_ratio)}</div><h2>${esc(c.animal)}</h2><div class="name">${esc(c.presenter)}</div><p class="meta">Image ratio ${esc(c.image_ratio)} · mask [${c.mask_box.map(v=>Number(v).toFixed(2)).join(', ')}] · subject scale ${a.scale_range?a.scale_range.map(v=>Number(v).toFixed(3)).join('–'):'unavailable'}</p><span class="tag ${a.verdict==='pass'?'':'review'}">Framing ${esc(a.verdict)}</span><div class="links"><a href="${esc(c.media['video.mp4'])}">H3 video</a><a href="${esc(c.media['reference.jpg'])}">Linked image</a><a href="${esc(c.media['mask.jpg'])}">Mask overlay</a><a href="${esc(c.media['frames.jpg'])}">Sampled frames</a><a href="${esc(c.media['verification.json'])}">Verification</a><a href="${esc(c.source_video_url)}">Original video</a></div><details><summary>Input and prompt</summary><p>Audio comes from the first 15 seconds of the original video. The linked presenter image anchors the H3 output. The exported H3 video contains the first 360 native frames with the original extracted audio.</p><a href="${esc(c.media['audio.wav'])}">15-second WAV</a> · <a href="${esc(c.media['prompt.txt'])}">Prompt</a> · <a href="${esc(c.media['transcript.json'])}">Approximate transcript</a></details></div>`;for(const video of card.querySelectorAll('video')){video.addEventListener('play',()=>pauseOther(video));if(video.src===c.source_video_url)video.addEventListener('timeupdate',()=>{if(video.currentTime>=15)video.pause()})}grid.append(card)}function filter(){const q=document.getElementById('search').value.toLowerCase().trim(),f=document.querySelector('button.active').dataset.filter;let n=0;for(const card of grid.children){card.hidden=!(card.dataset.search.includes(q)&&(f==='all'||card.dataset.ratio===f));if(!card.hidden)n++}document.getElementById('count').textContent=n+' / 30 videos'}document.getElementById('search').addEventListener('input',filter);document.querySelectorAll('button[data-filter]').forEach(b=>b.addEventListener('click',()=>{document.querySelector('button.active').classList.remove('active');b.classList.add('active');filter()}));filter();
</script></body></html>"""
    portrait_count = sum(case["display_ratio"] == "9:16" for case in data["cases"])
    landscape_count = data["case_count"] - portrait_count
    return (document
            .replace("30 animal lip-sync tests, 15 seconds each.",
                     f"{data['case_count']} completed animal lip-sync tests, 15 seconds each.")
            .replace("<strong>30</strong>H3 videos", f"<strong>{data['case_count']}</strong>H3 videos")
            .replace("<strong>15 / 15</strong>Portrait / landscape",
                     f"<strong>{portrait_count} / {landscape_count}</strong>Portrait / landscape")
            .replace("n+' / 30 videos'", "n+' / '+data.case_count+' videos'"))


def main():
    rows = json.loads((INPUT / "source_cases.json").read_text())["cases"]
    audits = {row["case_id"]: row for row in json.loads((OUTPUT / "framing_audits.json").read_text())}
    assert len(rows) == 30 and set(audits) <= {row["id"] for row in rows}
    cases = [export(row, audits[row["id"]]) for row in rows if row["id"] in audits]
    data = dict(created_at=datetime.now(timezone.utc).isoformat(), source_url=rows[0]["source_report_url"],
                case_count=len(cases), planned_case_count=len(rows),
                status="complete" if len(cases) == len(rows) else "partial", cases=cases)
    DEST.mkdir(exist_ok=True)
    (DEST / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    (DEST / "index.html").write_text(page(data))
    (DEST / "report-files.json").write_text(json.dumps(sorted(p.relative_to(DEST).as_posix() for p in DEST.rglob("*") if p.is_file()), indent=2) + "\n")
    print(f"BUILT {len(cases)} cases at {DEST / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
