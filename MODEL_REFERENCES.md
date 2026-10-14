# Model references

Model weights and derived tensor files are intentionally absent from GitHub.
Create the listed subdirectories under `models/`, download the exact files
from the pinned sources, and verify their SHA-256 before generation. The
`h3_silu_temb_grid.safetensors` file belongs in `turbo/`, not `models/`.
Follow each publisher's license and access requirements; a Hugging Face login
may be required if upstream access changes.

| Destination | Pinned source | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | [Comfy-Org, revision a988691](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/a98869194787969724c7425d95d0ed73ce9202af/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | 20970379616 | `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779` |
| `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | [Comfy-Org, revision 0bd506d](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/0bd506d2e895983a9663037febda27aa3948cf48/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | 15687142551 | `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6` |
| `models/vae/minimax_h3_video_vae_fp16.safetensors` | [Comfy-Org, revision 0bd506d](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/0bd506d2e895983a9663037febda27aa3948cf48/vae/minimax_h3_video_vae_fp16.safetensors) | 5207808496 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| `models/vae/minimax_h3_audio_vae_fp32.safetensors` | [Comfy-Org, revision 0bd506d](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/0bd506d2e895983a9663037febda27aa3948cf48/vae/minimax_h3_audio_vae_fp32.safetensors) | 605254808 | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| `models/loras/minimax_h3_turbo_v4_step600_ema.safetensors` | [Larryvrh, commit 7b7ac96](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/7b7ac96b0616100db75ea285090210c3ddf37c04/minimax_h3_turbo_v4_step600_ema.safetensors) | 779849816 | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3` |
| `turbo/h3_silu_temb_grid.safetensors` | [Larryvrh, commit a7624b4](https://raw.githubusercontent.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/a7624b4c00626a8ae7e78860769389d706565190/h3_silu_temb_grid.safetensors) | 5510600 | `30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8` |

The H3 base files are distributed under the
[MiniMax-H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
The Turbo LoRA and grid have their own upstream repository terms; consult
their linked sources before use or redistribution.
