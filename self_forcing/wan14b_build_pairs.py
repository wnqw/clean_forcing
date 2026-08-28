"""Pairs builder for the 14B corrector — port of wan_build_pairs_synth.py.

h_clean = wan14b_refs.pt (300 bidi-teacher clips); h_gen = causal K=21 rollout seeded from each
clip's first NCTX=3 latents + same caption, on the ADAPTED base (env ADAPTED_BASE, required for
the scale-up recipe) — optionally with the corrector LoRA active (env CORRECTOR -> DAgger round).
Resumable. Output: env OUT (round-0: wan_cache/wan14b_pairs_synth.pt;
DAgger: wan_cache/wan14b_pairs_dagger.pt). Run from repo root.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import torch
from wan14b_common import CausalDiffusionInferencePipeline14B, load_cfg, write_timing

DEVICE = "cuda"
K, NCTX = 21, 3
OUT = os.environ.get("OUT", "wan_cache/wan14b_pairs_synth.pt")
CLIPS = os.environ.get("CLIPS", "wan_cache/wan14b_refs.pt")
ADAPTED_BASE = os.environ.get("ADAPTED_BASE")
CORRECTOR = os.environ.get("CORRECTOR")


@torch.no_grad()
def main():
    cfg = load_cfg()
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline14B(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    assert ADAPTED_BASE, "scale-up recipe builds pairs on the adapted base (set ADAPTED_BASE)"
    sd = torch.load(ADAPTED_BASE, map_location="cpu")["merged"]
    pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    print(f"adapted base loaded: {ADAPTED_BASE}", flush=True)
    if CORRECTOR:
        from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters
        apply_lora(pipe.generator.model, rank=16)
        lw = torch.load(CORRECTOR, map_location="cpu")["lora"]
        lw = list(lw.values()) if isinstance(lw, dict) else lw
        for p, w in zip(lora_parameters(pipe.generator.model), lw):
            p.data.copy_(w.to(p.device, p.dtype))
        set_lora_scale(pipe.generator.model, 1.0)
        print(f"DAgger round with corrector: {CORRECTOR}", flush=True)

    d = torch.load(CLIPS, map_location="cpu")
    gt_all, caps = d["gt"], d["captions"]
    N = gt_all.shape[0]

    gens, done = [], 0
    if os.path.exists(OUT):
        prev = torch.load(OUT, map_location="cpu")
        gens = list(prev["gen"].unbind(0))
        done = len(gens)
        print(f"resuming from {done}", flush=True)

    t0 = time.time()
    for c in range(done, N):
        gt = gt_all[c:c + 1].to(DEVICE).to(torch.bfloat16)
        torch.manual_seed(c)
        noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        _, gen = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                initial_latent=gt[:, :NCTX], return_latents=True)
        gens.append(gen[0, :K].half().cpu())
        if (c + 1) % 10 == 0 or c == N - 1:
            torch.save({"gt": gt_all[:len(gens)], "gen": torch.stack(gens), "captions": caps[:len(gens)]}, OUT)
            print(f"{c + 1}/{N} pairs saved ({(time.time() - t0) / (c - done + 1):.0f}s/clip)", flush=True)

    tag = "pairs_dagger" if CORRECTOR else "pairs_round0"
    write_timing(tag, time.time() - t0, N - done, {"out": OUT})
    print(f"done: {len(gens)} pairs -> {OUT}")


if __name__ == "__main__":
    main()
