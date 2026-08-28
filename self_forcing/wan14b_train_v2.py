"""Stage 4 of the 14B scale-up: corrector v2 — port of wan_train_v2.py to Wan2.1-T2V-14B.

Identical recipe: init from v1 LoRA, DAgger aggregation (round-0 + corrector-active round-1
pools, uniform draw), drift-contraction term (committed corrected x0 spliced into the next
block's history WITH grad) at CW_LOSS=0.5, LR 2e-4, warmup 40, grad-clip 1.0, 600 steps, K=21.
14B deltas: pipeline/config from wan14b_common; ADAPTED_BASE required; step-tagged snapshots +
val-peak snapshot (coordinator rule); KILL BAR printed at end (v2 val dag-R^2 >= v1's) +
wan_cache/wan14b_v2_result.json.
Run: ADAPTED_BASE=... V1_VAL_R2=<v1 final> python -u wan14b_train_v2.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time
import numpy as np
import torch
from wan.modules.lora import apply_lora, set_lora_scale, lora_parameters
from wan14b_common import CausalDiffusionInferencePipeline14B, load_cfg, write_timing

DEVICE = "cuda"
LOSS_MODE = os.environ.get("LOSS_MODE", "both")
K, W = int(os.environ.get("K", 21)), 9
STEPS = int(os.environ.get("STEPS", 600))
LR = 2e-4
WARMUP = 40
CW_LOSS = float(os.environ.get("CW_LOSS", 0.5))
EVAL_EVERY = 100
OUT = "wan_cache"
POOLS = os.environ.get("POOLS", "wan14b_pairs_synth.pt,wan14b_pairs_dagger.pt").split(",")
INIT = os.environ.get("INIT", "wan14b_lora_v1.pt")
CKPT = os.environ.get("CKPT", "wan14b_lora_v2.pt")
V1_VAL_R2 = float(os.environ.get("V1_VAL_R2", "nan"))


def velocity_tf(pipe, cond, history, x0_cur, z_t, t):
    Wh, nf = history.shape[1], z_t.shape[1]
    clean_x = torch.cat([history, x0_cur], 1)
    noisy = torch.cat([history, z_t], 1)
    ts = torch.cat([torch.zeros((1, Wh), device=z_t.device, dtype=torch.float32),
                    torch.full((1, nf), float(t), device=z_t.device, dtype=torch.float32)], 1)
    flow, _ = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts, clean_x=clean_x)
    return flow[:, Wh:]


def main():
    cfg = load_cfg()
    torch.set_grad_enabled(True)
    pipe = CausalDiffusionInferencePipeline14B(cfg, device=torch.device(DEVICE)).to(dtype=torch.bfloat16).cuda()
    pipe.corrector = None
    ab = os.environ.get("ADAPTED_BASE")
    assert ab, "set ADAPTED_BASE"
    sd_ab = torch.load(ab, map_location="cpu")["merged"]
    pipe.generator.model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd_ab.items()}, strict=False)
    print(f"adapted base loaded: {ab}", flush=True)
    model = pipe.generator.model
    apply_lora(model, rank=16)
    sd = torch.load(os.path.join(OUT, INIT), map_location="cpu")["lora"]
    sd = list(sd.values()) if isinstance(sd, dict) else sd
    for i, p in enumerate(lora_parameters(model)):
        p.data.copy_(sd[i].to(p.device, p.dtype))
    model.gradient_checkpointing = True
    print(f"14B v2 LOSS_MODE={LOSS_MODE} CW={CW_LOSS} | init {INIT} | pools {POOLS}", flush=True)
    opt = torch.optim.AdamW(lora_parameters(model), lr=LR)
    START_STEP = 1
    pp = os.path.join(OUT, CKPT + ".partial")
    if os.environ.get("RESUME", "1") == "1" and os.path.exists(pp):
        pd = torch.load(pp, map_location="cpu")
        for p, w in zip(lora_parameters(model), pd["lora"]):
            p.data.copy_(w.to(p.device, p.dtype))
        START_STEP = pd["step"] + 1
        print(f"resumed from partial step {pd['step']}", flush=True)

    pools = [torch.load(os.path.join(OUT, p), map_location="cpu") for p in POOLS]
    caps = pools[0]["captions"]
    N = pools[0]["gt"].shape[0]
    nval = max(4, N // 8)
    train_ids, val_ids = list(range(N - nval)), list(range(N - nval, N))
    nfb = pipe.num_frame_per_block
    ks = list(range(W, K - 2 * nfb + 1))
    sched = pipe._initialize_sample_scheduler(torch.zeros(1, K, 16, 60, 104, device=DEVICE, dtype=torch.bfloat16))
    sigmas, tsteps = sched.sigmas.to(DEVICE).float(), sched.timesteps.to(DEVICE).float()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        cond_cache = {c: pipe.text_encoder(text_prompts=[caps[c]]) for c in range(N)}

    def draw(c):
        pool = pools[int(rng.integers(len(pools)))]
        return (pool["gt"][c:c + 1].to(DEVICE).to(torch.bfloat16),
                pool["gen"][c:c + 1].to(DEVICE).to(torch.bfloat16), int(rng.choice(ks)))

    def losses(c, grad):
        gtc, genc, k = draw(c)
        idx = int(rng.integers(len(tsteps)))
        t, sig = tsteps[idx], sigmas[idx]
        gt_h, gen_h, x0 = gtc[:, k - W:k], genc[:, k - W:k], genc[:, k:k + nfb]
        eps = torch.randn(x0.shape, device=DEVICE, dtype=x0.dtype)
        z_t = (1 - sig) * x0 + sig * eps
        with torch.no_grad():
            set_lora_scale(model, 0.0)
            v_clean = velocity_tf(pipe, cond_cache[c], gt_h, x0, z_t, t)
            v_base = velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
        rt = v_clean - v_base
        set_lora_scale(model, 1.0)
        gctx = torch.enable_grad() if grad else torch.no_grad()
        with gctx:
            v_corr = velocity_tf(pipe, cond_cache[c], gen_h, x0, z_t, t)
            l_dag = ((v_corr - v_clean) ** 2).sum() / ((rt ** 2).sum() + 1e-8)
            l_con = torch.zeros((), device=DEVICE)
            if LOSS_MODE == "both":
                x0_hat = z_t - sig * v_corr
                k2 = k + nfb
                idx2 = int(rng.integers(len(tsteps)))
                t2, sig2 = tsteps[idx2], sigmas[idx2]
                x0_next = genc[:, k2:k2 + nfb]
                eps2 = torch.randn(x0_next.shape, device=DEVICE, dtype=x0_next.dtype)
                z2 = (1 - sig2) * x0_next + sig2 * eps2
                h2_gen = torch.cat([genc[:, k2 - W:k], x0_hat.to(genc.dtype)], 1)
                with torch.no_grad():
                    set_lora_scale(model, 0.0)
                    v_clean2 = velocity_tf(pipe, cond_cache[c], gtc[:, k2 - W:k2], x0_next, z2, t2)
                    v_base2 = velocity_tf(pipe, cond_cache[c], genc[:, k2 - W:k2], x0_next, z2, t2)
                set_lora_scale(model, 1.0)
                v_corr2 = velocity_tf(pipe, cond_cache[c], h2_gen, x0_next, z2, t2)
                l_con = ((v_corr2 - v_clean2) ** 2).sum() / (((v_clean2 - v_base2) ** 2).sum() + 1e-8)
        return l_dag, l_con

    @torch.no_grad()
    def val_loss(n=8):
        s = 0.0
        for _ in range(n):
            l_dag, _ = losses(int(rng.choice(val_ids)), grad=False)
            s += l_dag.item()
        return s / n

    best_val = float("inf")
    t0 = time.time()
    for step in range(START_STEP, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        l_dag, l_con = losses(int(rng.choice(train_ids)), grad=True)
        loss = l_dag + CW_LOSS * l_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        opt.step()
        if step % 200 == 0:
            torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)], "step": step}, pp)
            torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)], "step": step},
                       os.path.join(OUT, CKPT.replace(".pt", f"_step{step}.pt")))   # step-tagged
        if step % EVAL_EVERY == 0 or step == 1:
            vl = val_loss()
            if vl < best_val:
                best_val = vl
                torch.save({"lora": [p.detach().cpu() for p in lora_parameters(model)],
                            "step": step, "val_dag_loss": vl},
                           os.path.join(OUT, CKPT.replace(".pt", "_valpeak.pt")))
            print(f"step {step:4d} | dag {l_dag.item():.4f} | con {float(l_con):.4f} "
                  f"| val dag-loss {vl:.4f} (R^2 {1-vl:+.3f})", flush=True)

    torch.save({"lora": {i: p.detach().cpu() for i, p in enumerate(lora_parameters(model))}},
               os.path.join(OUT, CKPT))
    vl = val_loss(16)
    r2 = 1 - vl
    bar_ok = (not np.isnan(V1_VAL_R2)) and r2 >= V1_VAL_R2
    verdict = "PASS" if bar_ok else ("UNKNOWN(V1_VAL_R2 unset)" if np.isnan(V1_VAL_R2) else "FAIL")
    with open(os.path.join(OUT, "wan14b_v2_result.json"), "w") as f:
        json.dump({"final_val_dag_r2": r2, "best_val_dag_r2": 1 - best_val, "v1_val_r2": V1_VAL_R2,
                   "verdict": verdict, "steps": STEPS, "pools": POOLS, "cw_loss": CW_LOSS,
                   "adapted_base": ab, "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    write_timing("train_v2", time.time() - t0, STEPS - START_STEP + 1)
    print(f"v2 done | final val dag-R^2 {r2:+.3f} (best {1-best_val:+.3f}) vs v1 {V1_VAL_R2:+.3f} "
          f"| KILL BAR: {verdict} | saved {OUT}/{CKPT}", flush=True)


if __name__ == "__main__":
    main()
