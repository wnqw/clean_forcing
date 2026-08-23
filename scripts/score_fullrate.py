"""Full-frame-rate MUSIQ/Delta re-score (stride-1, every frame) for fairness vs the latent-stride
protocol; per-config npz saved so partial progress survives interrupts."""
import os
import glob
import imageio.v3 as iio
import numpy as np
import torch, pyiqa

torch.set_num_threads(8)
D = "./self_forcing/wan_cache/finals128"
TAGS = os.environ.get("TAGS", "av2s").split(",")
musiq = pyiqa.create_metric("musiq", device="cuda")

for tag in TAGS:
    out = f"{D}/{tag}_fullrate.npz"
    done = dict(np.load(out))["means"].tolist() if os.path.exists(out) else []
    files = sorted(glob.glob(f"{D}/{tag}_p*.mp4"))
    means, deltas = list(done), []
    for f in files[len(done):]:
        vid = iio.imread(f)
        xs = torch.from_numpy(vid).permute(0,3,1,2).float().div(255).cuda()
        with torch.no_grad():
            pf = torch.cat([musiq(xs[i:i+16]).flatten() for i in range(0, len(xs), 16)]).cpu().numpy()
        means.append(pf.mean())
        np.savez(out, means=np.array(means))
    for f, m in zip(files, means):
        pass
    # recompute deltas in one pass (cheap re-read of curves not stored; do full properly)
    deltas = []
    for f in files:
        vid = iio.imread(f)
        xs = torch.from_numpy(vid).permute(0,3,1,2).float().div(255).cuda()
        with torch.no_grad():
            pf = torch.cat([musiq(xs[i:i+16]).flatten() for i in range(0, len(xs), 16)]).cpu().numpy()
        deltas.append(pf[:len(pf)//5].mean() - pf.mean())
        np.save(f.replace(".mp4", "_fullrate.npy"), pf)
    m, d = np.array(means[:len(files)]), np.array(deltas)
    sem = lambda x: x.std()/np.sqrt(len(x))
    print(f"RESULT fullrate {tag}: MUSIQ {m.mean():.1f}±{sem(m):.2f} | Delta {d.mean():+.2f}±{sem(d):.2f}", flush=True)
