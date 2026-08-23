"""2nd-host closed loop: SF-distilled + r_phi_sf (LoRA scale 1) on the 128 finals prompts, 50s.

Tag: sfc. Baseline comparison row = existing sfd videos (same prompts/protocol).
Resumable; batch-4; seed = prompt index.
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import imageio.v2 as imageio
import torch
from omegaconf import OmegaConf

from pipeline.causal_inference import CausalInferencePipeline
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters

DEVICE = "cuda"
KLAT = 201
BATCH = 4
D = "wan_cache/finals128"
TAG = os.environ.get("TAG", "sfc")
LORA = os.environ.get("LORA", "wan_cache/lora_r_phi_sf.pt")
N = int(os.environ.get("N", 128))  # subset: N strided prompts


@torch.no_grad()
def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    # rolling 21-frame KV window (as in the sfd baseline row): dmd config leaves local_attn_size
    # unset -> full attention with a 21-frame cache, which overflows past 21 latents
    cfg.model_kwargs = OmegaConf.create({"timestep_shift": cfg.timestep_shift,
                                         "local_attn_size": 21, "sink_size": 0})
    torch.set_grad_enabled(False)
    prompts = [ln.strip() for ln in open(f"{D}/prompts_used.txt") if ln.strip()]
    assert len(prompts) == 128

    pipe = CausalInferencePipeline(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
    pipe.generator.load_state_dict(sd.get("generator", sd.get("generator_ema")))
    pipe = pipe.to(dtype=torch.bfloat16).cuda()
    model = pipe.generator.model
    apply_lora(model, rank=16)
    lw = torch.load(LORA, map_location="cpu")["lora"]
    lw = list(lw.values()) if isinstance(lw, dict) else lw
    for p, w in zip(lora_parameters(model), lw):
        p.data.copy_(w.to(p.device, p.dtype))
    set_lora_scale(model, 1.0)
    print(f"SF-distilled + corrector {LORA} @ scale 1, tag={TAG}", flush=True)

    pids = list(range(0, 128, 128 // N))[:N]
    for b0 in range(0, len(pids), BATCH):
        idxs = [i for i in pids[b0:b0 + BATCH]
                if not os.path.exists(f"{D}/{TAG}_p{i:03d}.mp4")]
        if not idxs:
            continue
        pad = [idxs[-1]] * (BATCH - len(idxs))
        noise = []
        for i in idxs + pad:
            torch.manual_seed(i)
            noise.append(torch.randn(1, KLAT, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
        video = pipe.inference(noise=torch.cat(noise), text_prompts=[prompts[i] for i in idxs + pad])
        for bi, i in enumerate(idxs):
            fr = (video[bi].permute(0, 2, 3, 1).float() * 255).byte().cpu().numpy()
            tmp = f"{D}/{TAG}_p{i:03d}.mp4.tmp.mp4"
            imageio.mimsave(tmp, fr, fps=16, quality=8)
            os.rename(tmp, f"{D}/{TAG}_p{i:03d}.mp4")
        print(f"DONE batch {b0 // BATCH + 1}/{(len(pids)+BATCH-1)//BATCH}", flush=True)
    print(f"{TAG} finals complete", flush=True)


if __name__ == "__main__":
    main()
