"""Stage 6b: OOD scoring for the 14B arms — MUSIQ Delta-drift + post-hoc guards + contact strips.

Per tag (abase14, av2_14), over the 32 OOD mp4s in wan_cache/wan14b_ood/:
  MUSIQ      — per-frame at latent stride (every 4th pixel frame) -> {tag}_pXXX.npy;
               Delta = MUSIQ(first 20%) − MUSIQ(overall), SEM over prompts  (score_finals_row.py)
  post-hoc   — DINO latesim (anchoring), lag2s identity, cuts, pulse@12  (posthoc_metrics.py,
               no SF-reference pairing at 14B)
  saturation — mean-|sat − ref| drift curve summary (in-domain metric on OOD videos)
  contact    — 8-prompt x 10-frame contact sheet per tag -> outputs_wan14b/contact_{tag}.png
Output: wan_cache/wan14b_ood_scores.json. Run from repo root: TAGS=abase14,av2_14.
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import glob
import json
import time
import imageio.v2 as imageio
import imageio.v3 as iio3
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda"
D = "wan_cache/wan14b_ood"
TAGS = os.environ.get("TAGS", "abase14,av2_14").split(",")
NEXPECT = int(os.environ.get("N", 32))
LAG = 12
CONTACT_N = 8
OUT_JSON = "wan_cache/wan14b_ood_scores.json"
STRIPS = "outputs_wan14b"

sem = lambda x: float(np.std(x) / np.sqrt(len(x)))  # noqa: E731


def musiq_row(files, musiq):
    deltas, overalls = [], []
    for f in files:
        npy = f.replace(".mp4", ".npy")
        if os.path.exists(npy):
            pf = np.load(npy)
        else:
            vid = iio3.imread(f)[::4]
            scores = []
            for i in range(0, len(vid), 16):
                batch = torch.from_numpy(vid[i:i + 16]).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
                with torch.no_grad():
                    scores.append(musiq(batch).squeeze(-1).float().cpu().numpy())
            pf = np.concatenate(scores).astype(np.float32)
            np.save(npy, pf)
        n20 = max(1, len(pf) // 5)
        deltas.append(pf[:n20].mean() - pf.mean())
        overalls.append(pf.mean())
    return np.array(deltas), np.array(overalls)


@torch.no_grad()
def posthoc_row(files, dino, mean, std):
    late, lag2, cuts, pulse, satdrift = [], [], [], [], []
    for f in files:
        reader = imageio.get_reader(f)
        lum, diffs, feats, sats, prev = [], [], [], [], None
        for i, fr in enumerate(reader):
            g = fr.astype(np.float32) / 255.0
            lum.append(g.mean())
            cmax, cmin = g.max(axis=2), g.min(axis=2)
            sats.append(((cmax - cmin) / (cmax + 1e-6)).mean())
            if prev is not None:
                diffs.append(np.abs(g - prev).mean())
            prev = g
            if i % 16 == 0:
                x = torch.from_numpy(g).permute(2, 0, 1)[None].cuda()
                x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
                feats.append(F.normalize(dino((x - mean) / std).float(), dim=-1))
        reader.close()
        fe = torch.cat(feats)                                  # (~50,768) at 1 fps
        n5 = max(1, len(fe) // 5)
        late.append(float((fe[-n5:] @ fe[:n5].T).mean()))
        lag = 2                                                # 2 s at 1 fps sampling
        lag2.append(float((fe[:-lag] * fe[lag:]).sum(-1).mean()))
        dif = np.array(diffs)
        thr = max(0.10, dif.mean() + 5 * dif.std())
        cuts.append(int((dif > thr).sum()))
        lu = np.array(lum) - np.mean(lum)
        denom = (lu * lu).sum() + 1e-9
        pulse.append(float((lu[:-LAG] * lu[LAG:]).sum() / denom))
        sats = np.array(sats)
        ref = sats[:8].mean()
        satdrift.append(float(np.abs(sats - ref)[81:].mean()))
    return {"latesim": (float(np.mean(late)), sem(np.array(late))),
            "lag2s": (float(np.mean(lag2)), sem(np.array(lag2))),
            "cuts": (float(np.mean(cuts)), sem(np.array(cuts))),
            "pulse": (float(np.mean(pulse)), sem(np.array(pulse))),
            "sat_drift": (float(np.mean(satdrift)), sem(np.array(satdrift)))}


def contact_sheet(files, tag):
    rows = []
    step = max(1, len(files) // CONTACT_N)
    for f in files[::step][:CONTACT_N]:
        vid = iio3.imread(f)[::4]
        idx = np.linspace(0, len(vid) - 1, 10).astype(int)
        row = np.concatenate([vid[i][::4, ::4] for i in idx], axis=1)   # 4x downscale
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    path = os.path.join(STRIPS, f"contact_{tag}.png")
    imageio.imwrite(path, sheet)
    return path


def main():
    import pyiqa
    import timm
    os.makedirs(STRIPS, exist_ok=True)
    torch.set_grad_enabled(False)
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    dino = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    out = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    t0 = time.time()
    for tag in TAGS:
        files = sorted(glob.glob(f"{D}/{tag}_p*.mp4"))
        assert len(files) == NEXPECT, f"{tag}: expected {NEXPECT} videos, got {len(files)}"
        deltas, overalls = musiq_row(files, musiq)
        ph = posthoc_row(files, dino, mean, std)
        sheet = contact_sheet(files, tag)
        out[tag] = {"n": len(files),
                    "musiq": [float(overalls.mean()), sem(overalls)],
                    "delta_drift": [float(deltas.mean()), sem(deltas)],
                    "delta_per_video": [round(float(x), 3) for x in deltas],
                    **{k: [round(v[0], 4), round(v[1], 4)] for k, v in ph.items()},
                    "contact_sheet": sheet}
        json.dump(out, open(OUT_JSON, "w"), indent=1)
        print(f"RESULT {tag}: MUSIQ {overalls.mean():.1f}±{sem(overalls):.2f} | "
              f"Delta {deltas.mean():+.2f}±{sem(deltas):.2f} | latesim {ph['latesim'][0]:.3f} | "
              f"lag2s {ph['lag2s'][0]:.3f} | cuts {ph['cuts'][0]:.1f} | pulse {ph['pulse'][0]:.3f} | "
              f"sat_drift {ph['sat_drift'][0]:.3f}", flush=True)
    out["_wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()
