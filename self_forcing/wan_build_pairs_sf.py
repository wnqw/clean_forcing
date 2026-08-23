"""2nd-host pairs: SF-distilled (4-step, x0-space) rollouts vs the same 300 bidi-teacher synth clips.

h_clean = wan_cache/synth_clips.pt (same Wan2.1 VAE latent space as the distilled student);
h_gen = SF-distilled causal K=21 rollout seeded from each clip's first NCTX latents + caption.
Resumable. Output: wan_cache/pairs_sf.pt
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from omegaconf import OmegaConf

from pipeline.causal_inference import CausalInferencePipeline

DEVICE = "cuda"
K, NCTX = 21, 3
OUT = os.environ.get("OUT", "wan_cache/pairs_sf.pt")
CORRECTOR = os.environ.get("CORRECTOR")  # LoRA ckpt -> DAgger-round rollouts
SF_CKPT = "checkpoints/self_forcing_dmd.pt"


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    sd = torch.load(SF_CKPT, map_location="cpu")
    pipe.generator.load_state_dict(sd.get("generator", sd.get("generator_ema")))
    pipe = pipe.to(dtype=torch.bfloat16).cuda()
    if CORRECTOR:
        from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters
        apply_lora(pipe.generator.model, rank=16)
        lw = torch.load(CORRECTOR, map_location="cpu")["lora"]
        lw = list(lw.values()) if isinstance(lw, dict) else lw
        for p, w in zip(lora_parameters(pipe.generator.model), lw):
            p.data.copy_(w.to(p.device, p.dtype))
        set_lora_scale(pipe.generator.model, 1.0)
        print(f"DAgger round: rollouts with corrector {CORRECTOR}", flush=True)

    d = torch.load("wan_cache/synth_clips.pt", map_location="cpu")
    gt_all, caps = d["gt"], d["captions"]
    N = gt_all.shape[0]

    gens, done = [], 0
    if os.path.exists(OUT):
        prev = torch.load(OUT, map_location="cpu")
        gens = list(prev["gen"].unbind(0))
        done = len(gens)
        print(f"resuming from {done}", flush=True)

    for c in range(done, N):
        gt = gt_all[c:c + 1].to(DEVICE).to(torch.bfloat16)
        torch.manual_seed(c)
        noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        _, gen = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                initial_latent=gt[:, :NCTX], return_latents=True)
        gens.append(gen[0, :K].half().cpu())
        if (c + 1) % 20 == 0 or c == N - 1:
            torch.save({"gt": gt_all[:len(gens)], "gen": torch.stack(gens),
                        "captions": caps[:len(gens)]}, OUT)
            print(f"{c + 1}/{N} pairs saved", flush=True)

    print(f"done: {len(gens)} SF pairs -> {OUT}")


if __name__ == "__main__":
    main()
