"""
v2 DAgger round-1 data: paired cache where the drifted histories come from the CORRECTED
rollout (LoRA r_phi active, scale 1) instead of the raw base. Fixes the covariate shift:
r_phi is deployed on corrected-rollout states, so train it on those states too.
Same clips/seeds/protocol as wan_build_pairs.py; output pairs_k48_dagger1.pt.
Training then aggregates round-0 (pairs_k48.pt) + round-1 pools (DAgger aggregation).
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters
from wan_r_target_faithful import load_clip_latent

DEVICE = "cuda"
K = 48
NCTX = 3
NCLIPS = 40
NPIX = (K - 1) * 4 + 1
OUTDIR = "wan_cache"
OUT = os.path.join(OUTDIR, os.environ.get("OUT_NAME", "pairs_k48_dagger1.pt"))
CKPT = os.environ.get("CKPT", "lora_r_phi_k48.pt")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    _ab = os.environ.get("ADAPTED_BASE")
    if _ab:
        _sd = torch.load(_ab, map_location="cpu")["merged"]
        pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in _sd.items()}, strict=False)
        print(f"adapted base loaded: {_ab}", flush=True)
    model = pipe.generator.model
    apply_lora(model, rank=16)
    sd = torch.load(os.path.join(OUTDIR, CKPT), map_location="cpu")["lora"]
    for i, p in enumerate(lora_parameters(model)):
        p.data.copy_(sd[i].to(p.device, p.dtype))
    set_lora_scale(model, 1.0)                      # rollout WITH the current corrector
    print(f"DAgger round-1: corrected rollouts with {CKPT}", flush=True)

    vids = [l.strip() for l in open("data_gt/disney/videos.txt") if l.strip()]
    caps = [l.strip() for l in open("data_gt/disney/prompt.txt") if l.strip()]

    gts, gens, used = [], [], []
    for c in range(min(NCLIPS, len(vids))):
        gt = load_clip_latent(pipe.vae, os.path.join("data_gt/disney", vids[c]), n_pix=NPIX)[:, :K]
        if gt.shape[1] < K:
            print(f"clip {c}: too short ({gt.shape[1]}), skip"); continue
        torch.manual_seed(c)
        noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        _, gen = pipe.inference(noise=noise, text_prompts=[caps[c]],
                                initial_latent=gt[:, :NCTX].to(torch.bfloat16), return_latents=True)
        gen = gen[:, :K]
        gts.append(gt.half().cpu()); gens.append(gen.half().cpu()); used.append(caps[c])
        print(f"clip {c} cached ({len(gts)} total)", flush=True)

    gt_all = torch.cat(gts, 0); gen_all = torch.cat(gens, 0)
    torch.save({"gt": gt_all, "gen": gen_all, "captions": used}, OUT)
    print(f"saved {OUT}: gt {tuple(gt_all.shape)} gen {tuple(gen_all.shape)} | {len(used)} clips")


if __name__ == "__main__":
    main()
