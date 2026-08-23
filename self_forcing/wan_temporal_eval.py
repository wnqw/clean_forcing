"""
VBench-protocol Temporal Quality eval (our implementation of the official formulas/backbones):
  subject_consistency   DINO ViT-B/16:  mean_t (cos(f1,ft)+cos(f_{t-1},ft))/2      (↑)
  background_consistency CLIP ViT-B/32: same formula on CLIP image feats           (↑)
  temporal_flickering   mean abs diff between consecutive frames (0-255 scale)     (↓)
  dynamic_degree        RAFT-large mean flow magnitude (every 4th pair, 256x448)   (guard: must not collapse)
  i2v_seed_consistency  VBench++-style: DINO cos(frame, mean seed-frame feat), late window (↑)
Drift variants: early-window vs late-window subject/background consistency (decay = drift).
Configs: baseline / v1 / v2_both; 5 held-out clips × 3 seeds, K=48. Saves wan_cache/temporal_eval.npz.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
K, NCTX, NHELD, EARLY, LATE = 48, 3, 5, 32, 32
SEEDS = [0, 1, 2]
OUT = "wan_cache"
CONFIGS = {"baseline": None, "v1": "lora_r_phi_k48.pt", "v2_both": "lora_r_phi_v2_both.pt"}
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def feats(model, x01, mean, std, bs=32):     # x01 (F,3,H,W) in [0,1] -> L2-normed feats (F,D)
    x = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
    x = (x - mean.to(x)) / std.to(x)
    out = []
    for i in range(0, x.shape[0], bs):
        f = model(x[i:i + bs])
        out.append(F.normalize(f.float(), dim=-1))
    return torch.cat(out)


def consistency(f):                          # VBench formula over frames
    c_first = (f[1:] @ f[0]).clamp(0)
    c_prev = (f[1:] * f[:-1]).sum(-1).clamp(0)
    return ((c_first + c_prev) / 2).cpu().numpy()   # per-frame t>=1


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    import timm
    # DINO ViT-B/16 via timm (torch.hub's dino repo does `import utils`, shadowed by Self-Forcing's utils/)
    dino = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).to(DEVICE).eval()
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip = clip.visual.to(DEVICE).eval()
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(DEVICE).eval()

    d = torch.load(os.path.join(OUT, "pairs_k48.pt"), map_location="cpu")
    gt, caps = d["gt"], d["captions"]
    held = list(range(gt.shape[0] - NHELD, gt.shape[0]))
    P_CTX = (NCTX - 1) * 4 + 1

    METRICS = ("subj", "subj_late", "bg", "bg_late", "flick", "dyn", "i2v_late")
    res = {m: {cn: [] for cn in CONFIGS} for m in METRICS}
    for cn, ckpt in CONFIGS.items():
        pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
        pipe.corrector = None
        model = pipe.generator.model
        if ckpt is not None:
            apply_lora(model, rank=16)
            sd = torch.load(os.path.join(OUT, ckpt), map_location="cpu")["lora"]
            for i, p in enumerate(lora_parameters(model)):
                p.data.copy_(sd[i].to(p.device, p.dtype))
            set_lora_scale(model, 1.0)
        for c in held:
            seed_lat = gt[c:c + 1, :NCTX].to(DEVICE).to(torch.bfloat16)
            for s in SEEDS:
                torch.manual_seed(1000 * s + c)
                noise = torch.randn(1, K - NCTX, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
                _, lat = pipe.inference(noise=noise, text_prompts=[caps[c]], initial_latent=seed_lat, return_latents=True)
                pix = pipe.vae.decode_to_pixel(lat[:, :K])[0]                 # (F,3,H,W) [-1,1]
                x01 = ((pix.float() + 1) / 2).clamp(0, 1)

                fd = feats(dino, x01, IMNET_MEAN, IMNET_STD)                  # DINO feats
                cs = consistency(fd)
                res["subj"][cn].append(float(cs.mean()))
                res["subj_late"][cn].append(float(cs[-LATE:].mean()))
                fc = feats(clip, x01, CLIP_MEAN, CLIP_STD)                    # CLIP feats
                cb = consistency(fc)
                res["bg"][cn].append(float(cb.mean()))
                res["bg_late"][cn].append(float(cb[-LATE:].mean()))
                res["flick"][cn].append(float((x01[1:] - x01[:-1]).abs().mean().item() * 255))
                seed_ref = fd[:P_CTX].mean(0); seed_ref = F.normalize(seed_ref, dim=-1)
                res["i2v_late"][cn].append(float((fd[-LATE:] @ seed_ref).mean().item()))
                xs = F.interpolate(x01, size=(256, 448), mode="bilinear", align_corners=False) * 2 - 1
                mags = []
                for i in range(0, xs.shape[0] - 4, 4):                        # every 4th pair
                    fl = raft(xs[i:i + 1], xs[i + 4:i + 5])[-1]
                    mags.append(fl.pow(2).sum(1).sqrt().mean().item())
                res["dyn"][cn].append(float(np.mean(mags)))
        print(f"[{cn}] done", flush=True)
        del pipe; torch.cuda.empty_cache()

    np.savez(os.path.join(OUT, "temporal_eval.npz"), **{f"{m}_{cn}": np.array(res[m][cn]) for m in METRICS for cn in CONFIGS})
    print(f"\n===== VBench-protocol temporal quality ({NHELD} clips × {len(SEEDS)} seeds) =====")
    hdr = ("subj_cons ↑", "subj_late ↑", "bg_cons ↑", "bg_late ↑", "flicker ↓", "dyn_deg (guard)", "i2v_seed_late ↑")
    print(f"{'config':9s} | " + " | ".join(f"{h:>15s}" for h in hdr))
    for cn in CONFIGS:
        row = []
        for m in METRICS:
            a = np.array(res[m][cn]); row.append(f"{a.mean():.3f}±{a.std():.3f}")
        print(f"{cn:9s} | " + " | ".join(f"{r:>15s}" for r in row))
    print("\nsaved wan_cache/temporal_eval.npz")


if __name__ == "__main__":
    main()
