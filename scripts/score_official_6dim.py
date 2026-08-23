"""Official VBench custom-input 6-dim scoring for a finals config row (calibrated scale).

Must run from a neutral cwd (NOT Self-Forcing/ — its utils/ shadows torch.hub's) in a fresh process.
Usage: python score_official_6dim.py --tag attc
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
SF = "./self_forcing"
VB = "./VBench"
sys.path.insert(0, VB)
sys.path.insert(0, SF)
import vbench_decord_shim  # noqa: E402

vbench_decord_shim.install()
import torch  # noqa: E402
from vbench import VBench  # noqa: E402

OUT = f"{SF}/vbench_finals128"
DIMS = ["subject_consistency", "background_consistency", "aesthetic_quality",
        "imaging_quality", "motion_smoothness", "dynamic_degree"]
CAL = {"subject_consistency": (0.1462, 1.0), "background_consistency": (0.2615, 1.0),
       "motion_smoothness": (0.706, 0.9975), "dynamic_degree": (0.0, 1.0),
       "aesthetic_quality": (0.0, 1.0), "imaging_quality": (0.0, 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tag = args.tag

    vdir = f"{OUT}/videos_{tag}"
    os.makedirs(vdir, exist_ok=True)
    for link in glob.glob(f"{vdir}/*.mp4"):
        if not os.path.exists(os.path.realpath(link)):
            os.remove(link)
    vids = sorted(glob.glob(f"{SF}/wan_cache/finals128/{tag}_p*.mp4"))
    assert len(vids) == 128, f"expected 128 videos for {tag}, got {len(vids)}"
    for v in vids:
        dst = os.path.join(vdir, os.path.basename(v))
        if not os.path.lexists(dst):
            os.symlink(v, dst)

    vb = VBench(torch.device("cuda"), os.path.join(VB, "vbench", "VBench_full_info.json"), f"{OUT}/results")
    tab = json.load(open(f"{OUT}/table.json")) if os.path.exists(f"{OUT}/table.json") else {}
    tab.setdefault(tag, {})
    for dim in DIMS:
        try:
            vb.evaluate(videos_path=vdir, name=f"{tag}_{dim}", dimension_list=[dim], mode="custom_input")
            r = json.load(open(f"{OUT}/results/{tag}_{dim}_eval_results.json"))
            raw = r[dim][0]
            lo, hi = CAL[dim]
            tab[tag][dim] = 100 * (raw - lo) / (hi - lo)
            print(f"RESULT {tag} {dim}: {tab[tag][dim]:.2f}", flush=True)
        except Exception as e:  # noqa: BLE001 — score remaining dims, report the failure
            print(f"FAIL {tag} {dim}: {str(e)[:120]}", flush=True)
        json.dump(tab, open(f"{OUT}/table.json", "w"), indent=1)
    print(f"{tag} official scoring done", flush=True)


if __name__ == "__main__":
    main()
