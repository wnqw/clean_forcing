"""CF-row finals rollouts: CF base (tag cfb) and CF base + corrector (tag cfc)
on the locked 128 finals prompts, 50 s / 201 latents, un-seeded T2V, seed = prompt idx.

Protocol footnote: CF base runs under OUR protocol — rolling 21-frame KV window
(local_attn_size=21, sink 0), 20-step UniPC, shift 5 — identical to the feasibility
gate rollouts (not CF's 50-step default).

Env: TAG=cfb|cfc  LORA=<ckpt for cfc>  KLAT=201  N=128 (first-N prompts, deterministic)
     BATCH=1  OUTD=<video dir>
Resumable (skips existing mp4s). Run from Self-Forcing repo.
"""
import os
import time

import imageio.v2 as imageio
import torch

from cf_common import DEVICE, PROMPTS, ROW, load_cf_pipe, load_lora

TAG = os.environ.get("TAG", "cfb")
LORA = os.environ.get("LORA")
KLAT = int(os.environ.get("KLAT", 201))
N = int(os.environ.get("N", 128))
BATCH = int(os.environ.get("BATCH", 1))
OUTD = os.environ.get("OUTD", os.path.join(ROW, "finals"))


@torch.no_grad()
def main():
    os.makedirs(OUTD, exist_ok=True)
    torch.set_grad_enabled(False)
    pipe = load_cf_pipe()
    if LORA:
        load_lora(pipe.generator.model, LORA, scale=1.0)
    prompts = [ln.strip() for ln in open(PROMPTS) if ln.strip()]
    assert len(prompts) == 128
    pids = list(range(N))  # deterministic first-N subset, completable later

    for b0 in range(0, len(pids), BATCH):
        idxs = [i for i in pids[b0:b0 + BATCH] if not os.path.exists(f"{OUTD}/{TAG}_p{i:03d}.mp4")]
        if not idxs:
            continue
        pad = [idxs[-1]] * (BATCH - len(idxs))
        noise = []
        for i in idxs + pad:
            torch.manual_seed(i)
            noise.append(torch.randn(1, KLAT, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
        t0 = time.time()
        video = pipe.inference(noise=torch.cat(noise), text_prompts=[prompts[i] for i in idxs + pad])
        for bi, i in enumerate(idxs):
            fr = (video[bi].permute(0, 2, 3, 1).float() * 255).byte().cpu().numpy()
            tmp = f"{OUTD}/{TAG}_p{i:03d}.mp4.tmp.mp4"
            imageio.mimsave(tmp, fr, fps=16, quality=8)
            os.rename(tmp, f"{OUTD}/{TAG}_p{i:03d}.mp4")
        print(f"DONE {TAG} {idxs} ({fr.shape[0]} frames, {time.time()-t0:.0f}s)", flush=True)
    print(f"{TAG} finals complete ({N} prompts, KLAT={KLAT})", flush=True)


if __name__ == "__main__":
    main()
