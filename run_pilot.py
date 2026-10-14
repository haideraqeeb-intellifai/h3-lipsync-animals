"""Fixed-audio reference-image generation with 4-step H3 Turbo."""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import wave

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
COMFY_REVISION = "e377e263049f9338b4d12a3dd417b36ae62948ff"


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def case_image_settings(case, input_dir):
    filename = case.get("reference_filename", "reference.jpg") if case is not None else "lion.jpg"
    if Path(filename).name != filename:
        raise ValueError("Reference filename must be local to the case directory")
    width, height = (case.get("width", 736), case.get("height", 1280)) if case is not None else (736, 1280)
    if any(type(n) is not int or n < 256 or n > 2048 or n % 32 for n in (width, height)):
        raise ValueError("Output dimensions must be 32-aligned integers between 256 and 2048")
    return input_dir / filename, width, height


def graph(prompt, phase, frames, seed=42, ref_image_size="match"):
    w = {}
    def node(name, kind, **inputs):
        w[name] = {"class_type": kind, "inputs": inputs}
        return [name, 0]
    model = node("diffusion", "UNETLoader", unet_name="minimax_h3_ref2va_pruned_int8_convrot.safetensors", weight_dtype="default")
    clip = node("text", "CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default")
    vae = node("video_vae", "VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
    avae = node("audio_vae", "VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    cond = node("conditioning", "LionAudioConditioning", clip=clip, vae=vae, audio_vae=avae, prompt=prompt, frames=frames, ref_image_size=ref_image_size)
    model = node("adapter", "MiniMaxH3TurboLoRA", model=model, lora_name="minimax_h3_turbo_v4_step600_ema.safetensors", strength=1.0, low_vram=False)
    model = node("attention", "PilotSage", model=model)
    noise = node("noise", "RandomNoise", noise_seed=seed)
    guider = node("guider", "BasicGuider", model=model, conditioning=cond)
    sampler = node("sampler", "MiniMaxH3TurboSampler")
    sigmas = node("schedule", "BasicScheduler", model=model, scheduler="simple", steps=4, denoise=1.0)
    samples = node("denoise", "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=["conditioning", 1])
    samples = node("audit", "AuditAudioLatent", samples=samples)
    images = node("decode", "VAEDecode", samples=samples, vae=vae)
    video = node("video", "CreateVideo", images=images, audio=["conditioning", 2], fps=24.0, bit_depth=8)
    node("save", "SaveVideo", video=video, filename_prefix=phase + "/lion", format="mp4", codec="auto")
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=736)
    parser.add_argument("--name", default="lion_30s")
    parser.add_argument("--single", action="store_true", help="Single cold compatibility generation, no steady-state timing claim")
    parser.add_argument("--case", type=Path, help="Prepared case.json; omit for the original lion pilot")
    parser.add_argument("--test-frames", type=int, help="Override case duration for a compatibility test")
    parser.add_argument("--mask-box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Normalized mouth region to regenerate; preserve the rest of the source portrait")
    parser.add_argument("--animal-box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                        help="One normalized rectangle around the entire animal; preserve the outer image")
    parser.add_argument("--animal-mask-image", type=Path,
                        help="Binary image mask around the animal; white areas may move, black areas stay fixed")
    parser.add_argument("--gesture-ellipse", action="append", nargs=4, type=float,
                        metavar=("CX", "CY", "RX", "RY"),
                        help="Additional normalized oval for an already-visible animal appendage")
    parser.add_argument("--fast-static-plate", action="store_true",
                        help="Encode one repeated-image temporal chunk and reuse its latent pattern")
    parser.add_argument("--verify-static-plate", action="store_true",
                        help="On the first run, compare the fast static latent with a full-video VAE encode")
    parser.add_argument("--reserve-vram", type=float, default=20.0,
                        help="GiB reserved from ComfyUI model loading (default: 20)")
    args = parser.parse_args()
    if args.verify_static_plate and not args.fast_static_plate:
        parser.error("--verify-static-plate requires --fast-static-plate")
    if not 0 <= args.reserve_vram < 32:
        parser.error("--reserve-vram must be between 0 and 32 GiB")
    stop_requested = threading.Event()
    def request_stop(signum, frame):
        stop_requested.set()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    case = json.loads(args.case.read_text()) if args.case else None
    seed = case.get("seed", 42) if case else 42
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    ref_image_size = case.get("ref_image_size", "match") if case else "match"
    if ref_image_size not in ("match", "max"):
        raise ValueError("Reference image size must be 'match' or 'max'")
    if sum(x is not None for x in (args.mask_box, args.animal_box, args.animal_mask_image)) > 1:
        raise ValueError("Choose one of a mouth mask, animal rectangle, or animal shape mask")
    animal_mask_image = args.animal_mask_image.resolve() if args.animal_mask_image else None
    if animal_mask_image is not None and not animal_mask_image.is_file():
        raise ValueError(f"Animal mask image not found: {animal_mask_image}")
    full_animal = args.animal_box is not None or animal_mask_image is not None
    mask_box = args.animal_box if full_animal else args.mask_box
    mask_active = mask_box is not None or animal_mask_image is not None
    if mask_box is not None and not (0 <= mask_box[0] < mask_box[2] <= 1 and 0 <= mask_box[1] < mask_box[3] <= 1):
        raise ValueError("Mask rectangle must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    gesture_ellipses = args.gesture_ellipse or []
    if full_animal and gesture_ellipses:
        raise ValueError("The animal rectangle must be the only video mask region")
    if gesture_ellipses and mask_box is None:
        raise ValueError("Gesture ellipses require a source-portrait mouth mask")
    for cx, cy, rx, ry in gesture_ellipses:
        if not (0 < rx and 0 < ry and 0 <= cx - rx < cx + rx <= 1 and 0 <= cy - ry < cy + ry <= 1):
            raise ValueError("Gesture ellipses must fit within the normalized image")
    input_dir = args.case.resolve().parent if case else ROOT / "inputs"
    image_path, width, height = case_image_settings(case, input_dir)
    audio_path = input_dir / ("audio.wav" if case else "lion_30s.wav")
    if case:
        args.frames = case["native_frames"]
    if args.test_frames is not None:
        if args.test_frames < 5 or args.test_frames > 362:
            raise ValueError("Test frame count must be between 5 and 362")
        args.frames = args.test_frames
    with wave.open(str(audio_path), "rb") as source_file:
        input_seconds = min(source_file.getnframes() / source_file.getframerate(), args.frames / 24)
    out = ROOT / "outputs" / args.name
    out.mkdir(exist_ok=False)
    gate = module("pilot_gate", ROOT / "support/gpu_gate.py")
    gate.gate(out / "gpu_gate.json")
    helper = module("benchmark_helpers", ROOT / "support/benchmark.py")
    report = {"status": "initializing", "frames": args.frames, "fps": 24, "width": width, "height": height,
              "steps": 4, "seed": seed, "ref_image_size": ref_image_size, "runs": [], "input_audio_seconds": input_seconds,
              "static_plate_mode": "repeated_chunk" if args.fast_static_plate else "full_video",
              "mouth_mask_box": None if full_animal else mask_box,
              "animal_mask_box": args.animal_box,
              "animal_mask_image": str(animal_mask_image) if animal_mask_image else None,
              "gesture_mask_ellipses": gesture_ellipses,
              "case_id": case["id"] if case else "01-lion",
              "conditioning": "reference image + reference audio + fixed audio latent" +
                              (" + static portrait latent with a dilated animal shape mask" if animal_mask_image else
                               " + static portrait latent with one animal rectangle" if full_animal else
                               " + static portrait latent with mouth and appendage masks" if gesture_ellipses else
                               " + static portrait latent with mouth-only video mask" if mask_box is not None else ""),
              "patch": "ComfyUI H3 per-token video mask + fixed audio; original pinned legacy AV schedule retained"
                       if mask_active else "Whole-audio zero-mask subset of ComfyUI PR 15375; original pinned legacy AV schedule retained",
              "memory_adaptation": f"LoRA residual blocks 4096 rows; token-independent MLP blocks 8192 rows; expandable CUDA allocator; {args.reserve_vram:g} GiB reserved for workspace; weights and strength unchanged",
              "timing_boundary": "workflow execution through native video and supplied audio saved; CUDA synchronized; nested timers overlap",
              "comfy_revision": COMFY_REVISION,
              "source_files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / "visual_gate.py", ROOT / "ComfyUI/comfy/model_base.py", ROOT / "ComfyUI/comfy/ldm/minimax/model.py", ROOT / "turbo/__init__.py"]},
              "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [image_path, audio_path, input_dir / "prompt.txt"] + ([animal_mask_image] if animal_mask_image else [])},
              "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"], text=True),
              "packages": {name: importlib.metadata.version(name) for name in ["torch", "sageattention", "torchaudio"]}}
    save(out / "report.json", report)
    sys.path.insert(0, str(ROOT / "ComfyUI"))
    sys.argv = [sys.argv[0], "--use-pytorch-cross-attention", "--enable-dynamic-vram", "--reserve-vram", str(args.reserve_vram)]
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    import numpy as np
    import torchaudio
    from PIL import Image
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.cli_args import args as comfy_args
    if comfy_args.reserve_vram != args.reserve_vram or not comfy_args.enable_dynamic_vram or not comfy_args.use_pytorch_cross_attention:
        raise RuntimeError("Requested Comfy memory and attention flags were not applied")
    report["verified_cli_settings"] = {"reserve_vram_gib": comfy_args.reserve_vram,
                                       "enable_dynamic_vram": comfy_args.enable_dynamic_vram,
                                       "use_pytorch_cross_attention": comfy_args.use_pytorch_cross_attention}
    comfy_args.models_directory = str(MODELS)
    comfy_args.disable_all_custom_nodes = True
    comfy_args.cache_none = True
    for kind in ["input", "output", "temp", "user"]:
        directory = out / "runtime" / kind
        directory.mkdir(parents=True)
        setattr(comfy_args, kind + "_directory", str(directory))
    start = time.perf_counter()
    import main as comfy_main
    import nodes
    import execution
    import comfy.sd
    import comfy.samplers
    import comfy.nested_tensor
    import comfy.ldm.modules.attention as attention
    import node_helpers
    async def initialize():
        await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
        if not await nodes.load_custom_node(str(ROOT / "turbo"), module_parent="pilot_turbo"):
            raise RuntimeError("Turbo node registration failed")
    asyncio.run(initialize())
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, temporal_shape
    from comfy.ldm.minimax.model import MiniMaxH3Model
    counts = Counter()
    audit = {}
    expected_audio = None
    expected_video = None
    expected_video_mask = None

    class LionAudioConditioning:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "audio_vae": ("VAE",), "prompt": ("STRING",), "frames": ("INT",), "ref_image_size": (["match", "max"],)}}
        RETURN_TYPES = ("CONDITIONING", "LATENT", "AUDIO")
        FUNCTION = "execute"
        CATEGORY = "pilot"
        def execute(self, clip, vae, audio_vae, prompt, frames, ref_image_size):
            nonlocal expected_audio, expected_video, expected_video_mask
            pixels = torch.from_numpy(np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255).unsqueeze(0)
            with wave.open(str(audio_path), "rb") as source_file:
                sr = source_file.getframerate()
                if source_file.getsampwidth() != 2:
                    raise ValueError("Expected original 16-bit PCM")
                wav = np.frombuffer(source_file.readframes(source_file.getnframes()), dtype="<i2").astype(np.float32).reshape(-1, source_file.getnchannels()) / 32768
            wav = wav[:round(input_seconds * sr)]
            source = torch.from_numpy(wav.T.copy()).unsqueeze(0)
            native_frames, _, audio_t = temporal_shape(frames)
            padded = torch.nn.functional.pad(source, (0, max(0, round(native_frames / 24 * sr) - source.shape[-1])))
            # H3 uses 32 kHz stereo. Original 48 kHz mono is retained for the output mux.
            model_audio = torchaudio.functional.resample(padded, sr, 32000)
            if model_audio.shape[1] == 1:
                model_audio = model_audio.repeat(1, 2, 1)
            if model_audio.shape[1] != 2:
                raise ValueError("H3 requires mono or stereo input audio")
            model_audio = torch.nn.functional.pad(model_audio, (0, max(0, audio_t * 800 - model_audio.shape[-1])))
            cond, latent = MiniMaxH3ReferenceToVideo.execute(clip, vae, audio_vae, prompt, width, height, frames,
                ref_image_size=ref_image_size, ref_images={"ref_image_1": pixels},
                ref_audios={"ref_audio_1": {"waveform": model_audio, "sample_rate": 32000}}).result
            refs = cond[0][1]["minimax_refs"]
            encoded = next(r["audio_latent"] for r in refs if r["kind"] == "audio")
            if encoded.shape[-1] < audio_t:
                raise RuntimeError(f"Encoded audio too short: {encoded.shape[-1]} < {audio_t}")
            expected_audio = encoded[..., :audio_t].clone().float().cpu()
            video = latent["samples"].unbind()[0]
            video_mask = torch.ones_like(video)
            if mask_active:
                resized = comfy.utils.common_upscale(pixels.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
                def encode_plate(frame_count):
                    return vae.encode(resized.expand(frame_count, -1, -1, -1).contiguous())
                if args.fast_static_plate:
                    video_vae = vae.first_stage_model
                    chunk_frames = video_vae.clip_length
                    chunk_tokens = video_vae.tokens_chunk_size
                    token_drop = video_vae.token_drop
                    chunks_needed = 1 + (token_drop + chunk_tokens - 1) // chunk_tokens
                    short_video = encode_plate(chunks_needed * chunk_frames)
                    if short_video.shape[2] != chunks_needed * chunk_tokens - token_drop:
                        raise RuntimeError("Unexpected H3 VAE temporal chunk shape")
                    target_t = video.shape[2]
                    video = short_video[:, :, :chunk_tokens].repeat(
                        1, 1, (target_t + chunk_tokens - 1) // chunk_tokens, 1, 1
                    )[:, :, :target_t].contiguous()
                    if args.verify_static_plate and phase in ("warmup", "cold_pilot"):
                        full_video = encode_plate(native_frames)
                        max_error = float((video.float() - full_video.float()).abs().max())
                        audit["static_plate_equivalence"] = {
                            "max_absolute_error": max_error,
                            "exact_match": bool(torch.equal(video, full_video)),
                            "encoded_frames": chunks_needed * chunk_frames,
                            "full_frames": native_frames,
                        }
                        if max_error > 1e-5:
                            raise RuntimeError(f"Static plate latent mismatch: {max_error}")
                else:
                    video = encode_plate(native_frames)
                if video.shape[2:] != latent["samples"].unbind()[0].shape[2:]:
                    raise RuntimeError(f"Static portrait latent shape mismatch: {video.shape}")
                expected_video = video.float().cpu().clone()
                h, w = video.shape[-2:]
                if animal_mask_image is not None:
                    with Image.open(animal_mask_image) as mask_file:
                        if mask_file.size != (pixels.shape[2], pixels.shape[1]):
                            raise ValueError("Animal mask must match the source image dimensions")
                        mask_pixels = mask_file.convert("L").resize((width, height), Image.Resampling.NEAREST)
                    mask_array = (np.asarray(mask_pixels) >= 128).astype(np.float32)
                    latent_mask = torch.nn.functional.adaptive_max_pool2d(
                        torch.from_numpy(mask_array).view(1, 1, height, width), (h, w))
                    # H3 patchifies each 2x2 latent cell as one video token.
                    latent_mask = torch.nn.functional.max_pool2d(latent_mask, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
                    if not bool(latent_mask.any()):
                        raise ValueError("Animal shape mask is empty on the H3 latent grid")
                    video_mask = latent_mask.to(video).view(1, 1, 1, h, w).repeat(1, video.shape[1], video.shape[2], 1, 1)
                    audit["animal_mask_source_fraction"] = float(mask_array.mean())
                    audit["animal_mask_latent_fraction"] = float(latent_mask.mean())
                else:
                    video_mask = torch.zeros_like(video)
                    x0 = max(0, int(mask_box[0] * w) // 2 * 2)
                    y0 = max(0, int(mask_box[1] * h) // 2 * 2)
                    x1 = min(w, (int(mask_box[2] * w + 1) + 1) // 2 * 2)
                    y1 = min(h, (int(mask_box[3] * h + 1) + 1) // 2 * 2)
                    if x1 <= x0 or y1 <= y0:
                        raise ValueError("Mouth mask is empty on the H3 latent grid")
                    video_mask[..., y0:y1, x0:x1] = 1
                for cx, cy, rx, ry in gesture_ellipses:
                    ys = (torch.arange(h, device=video.device) + 0.5) / h
                    xs = (torch.arange(w, device=video.device) + 0.5) / w
                    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                    oval = (((gx - cx) / rx) ** 2 + ((gy - cy) / ry) ** 2 <= 1)
                    oval = oval.reshape(h // 2, 2, w // 2, 2).amax(dim=(1, 3))
                    oval = oval.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
                    video_mask = torch.maximum(video_mask, oval.to(video_mask.dtype).view(1, 1, 1, h, w))
                expected_video_mask = video_mask.float().cpu()
                if animal_mask_image is None:
                    audit["animal_mask_latent_box" if full_animal else "mouth_mask_latent_box"] = [x0, y0, x1, y1]
                audit["gesture_mask_ellipses"] = gesture_ellipses
                audit["video_mask_fraction"] = float(video_mask[0, 0, 0].mean())
            latent["samples"] = comfy.nested_tensor.NestedTensor([video, expected_audio])
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor([video_mask, torch.zeros_like(expected_audio)])
            audit.update({"audio_latent_shape": list(expected_audio.shape), "source_audio_samples": source.shape[-1],
                          "source_sample_rate": sr, "model_sample_rate": 32000, "audio_mask": 0,
                          "video_mask": "animal_shape_image" if animal_mask_image else "animal_rectangle" if full_animal else "mouth_and_appendage" if gesture_ellipses else "mouth_region" if mask_box is not None else 1})
            return cond, latent, {"waveform": padded, "sample_rate": sr}

    class AuditAudioLatent:
        @classmethod
        def INPUT_TYPES(cls): return {"required": {"samples": ("LATENT",)}}
        RETURN_TYPES = ("LATENT",)
        FUNCTION = "execute"
        CATEGORY = "pilot"
        def execute(self, samples):
            audio = samples["samples"].unbind()[1].float().cpu()
            error = float((audio - expected_audio).abs().max())
            audit["output_audio_latent_max_error"] = error
            if error > 1e-5:
                raise RuntimeError(f"Sampler did not preserve its fixed audio output: {error}")
            if expected_video is not None:
                output_video = samples["samples"].unbind()[0].float().cpu()
                outside_error = float(((output_video - expected_video) * (1 - expected_video_mask)).abs().max())
                audit["output_video_latent_outside_mask_max_error"] = outside_error
                if outside_error > 1e-5:
                    raise RuntimeError(f"Sampler changed video outside editable regions: {outside_error}")
            return (samples,)

    class PilotSage:
        @classmethod
        def INPUT_TYPES(cls): return {"required": {"model": ("MODEL",)}}
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "execute"
        CATEGORY = "pilot"
        def execute(self, model):
            clone = model.clone()
            def override(original, *values, **options):
                counts["sage_dispatches"] += 1
                before = counts["sage_cuda_completed"]
                result = attention.attention_sage(*values, **options)
                if counts["sage_cuda_completed"] == before:
                    raise RuntimeError("SageAttention fell back; requested configuration was not retained")
                return result
            clone.model_options["transformer_options"]["optimized_attention_override"] = override
            return (clone,)
    if not attention.SAGE_ATTENTION_IS_AVAILABLE:
        raise RuntimeError("SageAttention unavailable")
    import sageattention.core as sage_core
    original_cuda = sage_core.sageattn_qk_int8_pv_fp8_cuda
    def counted_cuda(*values, **options):
        result = original_cuda(*values, **options)
        counts["sage_cuda_completed"] += 1
        return result
    sage_core.sageattn_qk_int8_pv_fp8_cuda = counted_cuda
    original_forward = MiniMaxH3Model.forward
    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        if stop_requested.is_set():
            torch.cuda.synchronize()
            raise RuntimeError("Generation stopped at a safe model-forward boundary")
        payload = minimax_payload or {}
        if not payload.get("preserve_audio"):
            raise RuntimeError("Fixed audio mask did not reach H3")
        actual = x[1].float().cpu()
        # Comfy casts the model input to the model compute dtype before this boundary.
        expected = expected_audio.to(x[1].dtype).float()
        err = float((actual - expected).abs().max())
        audit.setdefault("forward_audio_checks", []).append({"timestep": float(timestep.flatten()[0]), "max_error_after_compute_cast": err})
        if err > 1e-5:
            raise RuntimeError(f"Audio changed before model forward: {err}")
        counts["fixed_audio_forwards"] += 1
        return original_forward(self, x, timestep, context, transformer_options, minimax_payload=payload, **kwargs)
    MiniMaxH3Model.forward = forward
    for cls in [LionAudioConditioning, AuditAudioLatent, PilotSage]:
        nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    executor = execution.PromptExecutor(helper.Progress(), cache_type=execution.CacheType.NONE, cache_args={"ram": 0, "ram_inactive": 0, "lru": 0})
    meter = helper.Measurements(torch)
    prompt = (input_dir / "prompt.txt").read_text().strip()
    if (gesture_ellipses or full_animal) and not (case and case.get("freeze_prompt", False)):
        original_gesture = ("An already-visible, anatomically native appendage that is relaxed and not bearing weight may make one small, slow conversational flex and settle back beside its resting position. "
                            "The excursion stays smaller than a quarter of that appendage's width, with its tabletop contact point maintained. "
                            "Gestures remain occasional and subordinate to speech, with no waving, reaching or repeated lifting.")
        if full_animal:
            visible_gesture = ("Only appendages already visible on <Subject 1> make several clear conversational gestures during speech. "
                               "Each moves at its own pictured joint, briefly lifts or pivots, then returns to its exact original resting place. "
                               "The movement is noticeable but stays within the animal's original outline and does not shift its head, torso, camera or setting. "
                               "Copy every moving appendage's form directly from <Picture 1> at every frame: preserve its original covering, silhouette, joint structure and digit count. "
                               "Never substitute anatomy from another species, add an appendage, or create human palms, fingers or thumbs.")
        elif len(gesture_ellipses) >= 2:
            visible_gesture = ("Both already-visible, species-native forepaws gesture throughout the speech. "
                               "Across the shot, each whole fur-covered paw makes several clear conversational lifts: rise fully off the tabletop by roughly one paw height, move inward or forward by about one paw width, then return to its original resting spot. "
                               "Alternate the paws during spoken phrases, with one brief moment when both lift together for emphasis; avoid long stretches with both paws motionless. "
                               "The movement comes from the existing forelimb joints while the shoulders, torso and head stay in their pictured positions. "
                               "Keep each paw's original fur, claws, shape, digit count and attachment to its forelimb at every moment; no extra paws, human palms, fingers or thumbs.")
        else:
            visible_gesture = ("During one spoken phrase, one already-visible, non-supporting species-native appendage makes a clearly visible but restrained conversational gesture: "
                               "the whole distal appendage briefly lifts or pivots at its natural joint, then settles back into its exact pictured resting position. "
                               "For a resting paw, the fur-covered paw moves as a whole rather than only its claw tips. "
                               "Keep the motion within half the appendage's visible width, with no waving, reaching, repeated lifting or new appendages.")
        if original_gesture not in prompt:
            raise ValueError("Expected species-native gesture paragraph is missing from the case prompt")
        prompt = prompt.replace(original_gesture, visible_gesture).replace(
            "Supporting limbs remain planted; hidden limbs remain hidden.",
            "Other supporting limbs remain planted; hidden limbs remain hidden.")
        if full_animal:
            species_examples = ("Each appendage keeps its original species-native silhouette, covering, joints and digit count: paws retain fur and claws, hooves remain solid or cloven hooves, folded wings retain feathers, and suckered arms retain their suckers. "
                                "These forms do not become human palms, fingers or thumbs. Naturally occurring primate hands retain their own reference anatomy. ")
            if species_examples not in prompt:
                raise ValueError("Expected species examples are missing from the case prompt")
            prompt = prompt.replace(species_examples, "All visible parts retain exactly the anatomy depicted in <Picture 1>. ")
    if args.frames < 720:
        prompt = prompt.replace("30-second", f"{args.frames / 24:.2f}-second")
    report["prompt"] = prompt
    for kind in {n["class_type"] for n in graph(prompt, "warmup", args.frames, seed, ref_image_size).values()}:
        cls = nodes.NODE_CLASS_MAPPINGS[kind]
        meter.wrap(cls, "execute" if "execute" in cls.__dict__ else cls.FUNCTION, "node/" + kind)
    meter.wrap(comfy.sd.CLIP, "encode_from_tokens_scheduled", "text_encoder")
    meter.wrap(comfy.samplers.KSAMPLER, "sample", "denoising_loop")
    torch.cuda.synchronize()
    report["initialization_seconds"] = time.perf_counter() - start
    phases = ["cold_pilot"] if args.single else ["warmup", "measured"]
    try:
        for phase in phases:
            gate.owned()
            workflow = graph(prompt, phase, args.frames, seed, ref_image_size)
            save(out / (phase + "_workflow.json"), workflow)
            valid, error, outputs, details = asyncio.run(execution.validate_prompt(phase, workflow, None))
            if not valid: raise RuntimeError(f"Invalid workflow {error}: {details}")
            audit.clear()
            counts.clear()
            meter.calls.clear()
            report["status"] = phase
            save(out / "report.json", report)
            stop = threading.Event()
            samples = []
            def monitor():
                while not stop.is_set():
                    try: samples.append(gate.owned())
                    except Exception as exc: samples.append({"error": str(exc)})
                    stop.wait(1)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            print(f"START {phase}", flush=True)
            try:
                executor.execute(workflow, phase, execute_outputs=outputs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                stop.set()
                thread.join()
                save(out / (phase + "_gpu_samples.json"), samples)
                save(out / (phase + "_audio_audit.json"), audit)
            if not executor.success: raise RuntimeError(str(executor.status_messages))
            if any("error" in s for s in samples): raise RuntimeError("GPU ownership check failed; see telemetry")
            if counts["fixed_audio_forwards"] != 4 or counts["sage_cuda_completed"] < 1:
                raise RuntimeError(f"Runtime contract failed: {counts}")
            videos = list((out / "runtime/output" / phase).glob("*.mp4"))
            if len(videos) != 1: raise RuntimeError(f"Expected one output, found {videos}")
            totals = Counter()
            for call in meter.calls: totals[call["component"]] += call["seconds"]
            report["runs"].append({"phase": phase, "generation_seconds": elapsed, "components_seconds": dict(totals),
                "counts": dict(counts), "audio_audit": dict(audit), "video": str(videos[0]),
                "peak_device_mib": max(s["memory_used_bytes"] for s in samples) / 2**20,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
            save(out / "report.json", report)
            print(f"DONE {phase}: {elapsed:.3f}s {videos[0]}", flush=True)
            from visual_gate import check as check_visual_output
            visual = check_visual_output(videos[0], image_path)
            save(out / (phase + "_visual_gate.json"), visual)
            report["runs"][-1]["visual_gate"] = visual
            save(out / "report.json", report)
            if visual["status"] != "passed":
                raise RuntimeError("Saved video failed the near-blank frame check; no further generations will run")
            if stop_requested.is_set():
                raise RuntimeError("Generation stopped after completed output")
            comfy_main.hook_breaker_ac10a0.restore_functions()
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        traceback.print_exc()
        raise
    finally:
        save(out / "report.json", report)
        meter.close()


if __name__ == "__main__":
    main()
