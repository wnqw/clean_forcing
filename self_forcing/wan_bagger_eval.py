"""
BAgger-protocol eval: 50s pure T2V (no GT seed) from MovieGenVideoBench prompts (their cited
prompt source), rolling-KV long rollout, their exact Drifting Metrics
    Delta^Imaging / Delta^Aesthetic = score(first 20% of frames) - score(all frames)   (down=better)
plus VBench-protocol dims (subject/bg consistency, flicker, RAFT dynamic degree).
Configs: baseline | df02 (DF noisy-context sigma_test=0.2) | v1 | v2 (Disney-trained ->
this run doubles as the domain-generalization stress test). Saves per-frame arrays
(wan_cache/bagger_eval.npz) + mp4s (wan_cache/videos_t2v/) for the official-VBench x86 pass.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
import pyiqa
import imageio.v2 as imageio
from omegaconf import OmegaConf
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
KLAT = int(os.environ.get("KLAT", 201))     # latent frames -> 801 px ≈ 50s @16fps (n%3==0)
NPROMPTS = int(os.environ.get("NPROMPTS", 16))
STRIDE_F = 2                  # score every 2nd frame for MUSIQ/aes/DINO/CLIP
OUT = "wan_cache"
VID = os.path.join(OUT, "videos_t2v")
CONFIGS = {"baseline": {}, "df02": {"ctx_sigma": 0.2},
           "v1": {"ckpt": "lora_r_phi_k48.pt"}, "v2": {"ckpt": "lora_r_phi_v2_both.pt"}}
IMNET = (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
CLIPN = (torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
         torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))


def feats(model, x01, norm, bs=32):
    x = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
    x = (x - norm[0].to(x)) / norm[1].to(x)
    return torch.cat([F.normalize(model(x[i:i + bs]).float(), dim=-1) for i in range(0, x.shape[0], bs)])


def consistency(f):
    return (((f[1:] @ f[0]).clamp(0) + (f[1:] * f[:-1]).sum(-1).clamp(0)) / 2).cpu().numpy()


@torch.no_grad()
def score_metric(metric, x01, bs=16):
    return torch.cat([metric(x01[i:i + bs]).flatten() for i in range(0, x01.shape[0], bs)]).cpu().numpy()


@torch.no_grad()
def main():
    os.makedirs(VID, exist_ok=True)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/_undistilled_smoke.yaml"))
    torch.set_grad_enabled(False)
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    aes = pyiqa.create_metric("laion_aes", device=DEVICE)
    import timm, open_clip
    dino = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).to(DEVICE).eval()
    clipm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clipm = clipm.visual.to(DEVICE).eval()
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(DEVICE).eval()

    allp = [l.strip() for l in open("prompts/MovieGenVideoBench.txt") if l.strip()]
    prompts = allp[::max(1, len(allp) // NPROMPTS)][:NPROMPTS]     # evenly strided for diversity
    print(f"{len(prompts)} prompts | {KLAT} latent frames (~{(KLAT-1)*4+1} px)", flush=True)

    res = {}
    for cn, cc in CONFIGS.items():
        pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
        pipe.corrector = None
        if "ckpt" in cc:
            model = pipe.generator.model
            apply_lora(model, rank=16)
            sd = torch.load(os.path.join(OUT, cc["ckpt"]), map_location="cpu")["lora"]
            for i, p in enumerate(lora_parameters(model)):
                p.data.copy_(sd[i].to(p.device, p.dtype))
            set_lora_scale(model, 1.0)
        if "ctx_sigma" in cc:
            pipe.context_noise_sigma = cc["ctx_sigma"]
        per = {m: [] for m in ("img", "aes", "subj", "bg", "flick", "dyn")}
        for pi, prompt in enumerate(prompts):
            torch.manual_seed(pi)
            noise = torch.randn(1, KLAT, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
            _, lat = pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True)
            pix = pipe.vae.decode_to_pixel(lat[:, :KLAT])[0]
            x01 = ((pix.float() + 1) / 2).clamp(0, 1)
            xs = x01[::STRIDE_F]
            per["img"].append(score_metric(musiq, xs))
            per["aes"].append(score_metric(aes, xs))
            per["subj"].append(consistency(feats(dino, xs, IMNET)))
            per["bg"].append(consistency(feats(clipm, xs, CLIPN)))
            per["flick"].append(np.array([(x01[1:] - x01[:-1]).abs().mean().item() * 255]))
            xr = F.interpolate(x01[::8], size=(256, 448), mode="bilinear", align_corners=False) * 2 - 1
            mags = [raft(xr[i:i + 1], xr[i + 1:i + 2])[-1].pow(2).sum(1).sqrt().mean().item()
                    for i in range(xr.shape[0] - 1)]
            per["dyn"].append(np.array([np.mean(mags)]))
            fr = (x01.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
            imageio.mimsave(os.path.join(VID, f"{cn}_p{pi:02d}.mp4"), fr, fps=16, quality=7)
            del pix, x01
            print(f"[{cn}] prompt {pi} done", flush=True)
        res[cn] = per
        del pipe; torch.cuda.empty_cache()

    np.savez(os.path.join(OUT, "bagger_eval.npz"),
             **{f"{cn}_{m}_{i}": arr for cn in res for m in res[cn] for i, arr in enumerate(res[cn][m])})

    def delta20(curves):     # BAgger drifting metric: first-20% mean - overall mean (per video, then avg)
        d = [c[:max(1, len(c) // 5)].mean() - c.mean() for c in curves]
        return np.mean(d), np.std(d)

    def gmean(curves):
        v = [c.mean() for c in curves]
        return np.mean(v), np.std(v)

    print("\n===== BAgger-protocol: 50s T2V, MovieGen prompts =====")
    print(f"{'config':9s} | {'Imaging↑':>12s} | {'Aesthetic↑':>12s} | {'Subj↑':>12s} | {'Δ^Imaging↓':>12s} | {'Δ^Aesthetic↓':>12s} | {'dyn':>10s}")
    for cn in res:
        im, ims = gmean(res[cn]["img"]); ae, aes_ = gmean(res[cn]["aes"]); sj, sjs = gmean(res[cn]["subj"])
        di, dis = delta20(res[cn]["img"]); da, das = delta20(res[cn]["aes"])
        dy, dys = gmean(res[cn]["dyn"])
        print(f"{cn:9s} | {im:6.2f}±{ims:4.2f} | {ae:6.3f}±{aes_:4.3f} | {sj:6.3f}±{sjs:4.3f} | "
              f"{di:+6.2f}±{dis:4.2f} | {da:+6.3f}±{das:4.3f} | {dy:5.1f}±{dys:4.1f}")
    print(f"\nsaved wan_cache/bagger_eval.npz + {len(prompts)*len(CONFIGS)} mp4s in {VID}/")


if __name__ == "__main__":
    main()
