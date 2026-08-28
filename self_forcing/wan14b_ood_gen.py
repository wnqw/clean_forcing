"""Stage 6a of the 14B scale-up: OOD video generation — finals protocol at 14B, 32-prompt subset.

32 strided prompts from the finals-128 selection (every 4th of finals128/prompts_used.txt =
extended indices 320,340,...,940 — untouched by training, which used pool indices < 320).
50 s un-seeded T2V: 201 latents, block-causal, CFG 6 / shift 8, seed = finals prompt index
(matches wan_ttc_finals.py), batch 1 (14B KV cache). Two arms via env:
  ARM=abase  -> deploy-selected adapted base            (tag abase14)
  ARM=av2    -> adapted base + v2 corrector MERGED at full gain (alpha* flat 0.889-0.991,
                same flat-gate/full-gain rule as 1.3B)  (tag av2_14)
Resumable per video. Output: wan_cache/wan14b_ood/{tag}_p{idx:03d}.mp4 (fps 16, q8).
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import imageio.v2 as imageio
import torch
from wan.modules.lora import apply_lora, lora_parameters, LoRALinear
from wan14b_common import CausalDiffusionInferencePipeline14B, load_cfg, write_timing

DEVICE = "cuda"
KLAT = 201
N = int(os.environ.get("N", 32))
STRIDE = 128 // N
D = "wan_cache/wan14b_ood"
FINALS = "wan_cache/finals128/prompts_used.txt"
ADAPTED = os.environ.get("ADAPTED_BASE")
ARM = os.environ.get("ARM", "abase")
TAG = {"abase": "abase14", "av2": "av2_14"}[ARM]
V2 = os.environ.get("V2", "wan_cache/wan14b_lora_v2.pt")


def merge_lora(model):
    """Static merge at gain 1 (paper deploy: corrector baked into weights, zero overhead)."""
    n = 0
    for name, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                w = child.base.weight.data.float() + child.B.weight.data.float() @ child.A.weight.data.float()
                child.base.weight.data.copy_(w.to(child.base.weight.dtype))
                setattr(module, cname, child.base)
                n += 1
    return n


@torch.no_grad()
def main():
    assert ADAPTED, "set ADAPTED_BASE"
    os.makedirs(D, exist_ok=True)
    cfg = load_cfg()
    torch.set_grad_enabled(False)
    finals = [ln.strip() for ln in open(FINALS) if ln.strip()]
    assert len(finals) == 128
    pids = list(range(0, 128, STRIDE))[:N]

    pipe = CausalDiffusionInferencePipeline14B(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    model = pipe.generator.model
    sd = torch.load(ADAPTED, map_location="cpu")["merged"]
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    if ARM == "av2":
        apply_lora(model, rank=16)
        lw = torch.load(V2, map_location="cpu")["lora"]
        lw = list(lw.values()) if isinstance(lw, dict) else lw
        for p, w in zip(lora_parameters(model), lw):
            p.data.copy_(w.to(p.device, p.dtype))
        nm = merge_lora(model)
        print(f"v2 corrector merged into {nm} linears (full gain)", flush=True)
    print(f"OOD arm {ARM} tag={TAG} | {N} prompts (finals idx {pids[0]}..{pids[-1]} stride {STRIDE})", flush=True)

    t0, ndone = time.time(), 0
    for pi in pids:
        out = f"{D}/{TAG}_p{pi:03d}.mp4"
        if os.path.exists(out):
            continue
        tc = time.time()
        torch.manual_seed(pi)
        noise = torch.randn(1, KLAT, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16)
        video = pipe.inference(noise=noise, text_prompts=[finals[pi]])
        fr = (video[0].permute(0, 2, 3, 1).float() * 255).byte().cpu().numpy()
        tmp = out + ".tmp.mp4"
        imageio.mimsave(tmp, fr, fps=16, quality=8)
        os.rename(tmp, out)
        ndone += 1
        print(f"DONE {TAG}_p{pi:03d} in {time.time() - tc:.0f}s", flush=True)
    write_timing(f"ood_{ARM}", time.time() - t0, max(ndone, 1), {"klat": KLAT, "n": N})
    print(f"{TAG} OOD complete", flush=True)


if __name__ == "__main__":
    main()
