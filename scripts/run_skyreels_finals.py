"""SkyReels-V2-DF-1.3B-540P finals row: official sync-mode long-video recipe on the 128 finals prompts.

Official repo code + checkpoint (Tier B). Recipe per README: ar_step=0, base_num_frames=97,
overlap_history=17, addnoise_condition=20, guidance 6.0, shift 8.0, 30 steps, 540P@24fps.
50 s horizon -> num_frames=1201. Resumable (skips existing outputs); shardable across worker processes.
"""
import argparse
import os
import sys

import imageio
import torch

SKYREELS = "./SkyReels-V2"
CKPT = "./checkpoints/SkyReels-V2-DF-1.3B-540P"
PROMPTS = "./self_forcing/wan_cache/finals128/prompts_used.txt"
OUTDIR = "./self_forcing/wan_cache/finals128"
sys.path.insert(0, SKYREELS)
sys.path.insert(0, "./self_forcing")

# no decord wheel on aarch64; the pipeline imports it only for prefix-video extension, which we don't use
import vbench_decord_shim  # noqa: E402

vbench_decord_shim.install()

from skyreels_v2_infer import DiffusionForcingPipeline  # noqa: E402

NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--num_frames", type=int, default=1201)
    ap.add_argument("--smoke", action="store_true", help="1 prompt, 97 frames, timing printout")
    args = ap.parse_args()

    with open(PROMPTS) as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    assert len(prompts) == 128, f"expected 128 prompts, got {len(prompts)}"

    num_frames = 97 if args.smoke else args.num_frames
    indices = [0] if args.smoke else [i for i in range(128) if i % args.num_shards == args.shard]

    pipe = DiffusionForcingPipeline(
        CKPT, dit_path=CKPT, device=torch.device("cuda"),
        weight_dtype=torch.bfloat16, use_usp=False, offload=False,
    )

    for idx in indices:
        tag = "skyr_smoke" if args.smoke else f"skyr_p{idx:03d}"
        out = os.path.join(OUTDIR, f"{tag}.mp4")
        if os.path.exists(out):
            print(f"SKIP {tag}", flush=True)
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.amp.autocast("cuda", dtype=pipe.transformer.dtype), torch.no_grad():
            frames = pipe(
                prompt=prompts[idx],
                negative_prompt=NEGATIVE,
                image=None,
                height=544,
                width=960,
                num_frames=num_frames,
                num_inference_steps=30,
                shift=8.0,
                guidance_scale=6.0,
                generator=torch.Generator(device="cuda").manual_seed(idx),
                overlap_history=17,
                addnoise_condition=20,
                base_num_frames=97,
                ar_step=0,
                causal_block_size=1,
                fps=24,
            )[0]
        end.record()
        torch.cuda.synchronize()
        imageio.mimwrite(out + ".tmp.mp4", frames, fps=24, quality=8,
                         output_params=["-loglevel", "error"])
        os.rename(out + ".tmp.mp4", out)
        print(f"DONE {tag} {end.elapsed_time(start) / 1000:.1f}s", flush=True)

    print(f"SHARD {args.shard}/{args.num_shards} complete", flush=True)


if __name__ == "__main__":
    main()
