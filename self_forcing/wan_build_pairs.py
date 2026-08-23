"""
Step 2: paired-data cache on Wan for training the corrector.
For each Disney clip: GT latents (clean history) + GT-seeded self-rollout (drifted history).
Caches gt/gen trajectories (like the toy pairs.pt) so training reuses them without re-rolling.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan_r_target_faithful import load_clip_latent

DEVICE = "cuda"
K = 48          # latent frames (rolling KV-cache window=21 lifts the old 21-frame ceiling)
NCTX = 3        # GT frames used to seed each self-rollout
NCLIPS = 40
NPIX = (K - 1) * 4 + 1   # 189 pixel frames -> K latent (Disney clips are 195 pix)
OUTDIR = "wan_cache"
OUT = os.path.join(OUTDIR, "pairs_k48.pt")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
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
