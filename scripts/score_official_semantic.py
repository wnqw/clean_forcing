"""Official VBench custom-input scoring: temporal_flickering + overall_consistency (semantic axis).

Extends score_official_6dim.py to the two remaining custom-input dims that are meaningful on
free-form prompts. overall_consistency = ViCLIP text-video alignment vs the real generation prompt
(passed via prompt_list; the 6-dim script could skip this because its dims ignore text).

Not scored, by design:
  - object_class/multiple_objects/scene/appearance_style/color/spatial_relationship: rejected by
    VBench.check_dimension_requires_extra_info in custom_input mode (need labeled categories).
  - human_action: parses the Kinetics label from the video FILENAME -> garbage on {tag}_pNNN.mp4.
  - temporal_style: byte-identical computation to overall_consistency on custom prompts
    (same ViCLIP video-text sim vs the full prompt; differs only in intended prompt content).

Must run from a neutral cwd (NOT self_forcing/) in a fresh process; set VBENCH_DIR to your VBench checkout.
Usage: python score_official_semantic.py --tag attc [--dims temporal_flickering,overall_consistency] [--limit N]
"""
import argparse
import glob
import json
import os
import re
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF = os.environ.get("SF_REPO", os.path.join(_REPO, "self_forcing"))
VB = os.environ.get("VBENCH_DIR", os.path.join(_REPO, "VBench"))
sys.path.insert(0, VB)
sys.path.insert(0, SF)
import vbench_decord_shim  # noqa: E402

vbench_decord_shim.install()

# openai-clip does `from pkg_resources import packaging`; newer setuptools dropped it
import packaging  # noqa: E402
import pkg_resources  # noqa: E402

if not hasattr(pkg_resources, "packaging"):
    pkg_resources.packaging = packaging
import torch  # noqa: E402
from vbench import VBench  # noqa: E402

OUT = f"{SF}/vbench_finals128"
DIMS = ["temporal_flickering", "overall_consistency"]
# official leaderboard normalization, VBench/scripts/constant.py
CAL = {"temporal_flickering": (0.6293, 1.0), "overall_consistency": (0.0, 0.364)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dims", default=",".join(DIMS))
    ap.add_argument("--limit", type=int, default=128)
    args = ap.parse_args()
    tag, dims = args.tag, args.dims.split(",")

    vids = sorted(f for f in glob.glob(f"{SF}/wan_cache/finals128/{tag}_p*.mp4")
                  if re.match(rf".*/{tag}_p\d{{3}}\.mp4$", f))[:args.limit]
    assert len(vids) == args.limit, f"expected {args.limit} videos for {tag}, got {len(vids)}"
    prompts = open(f"{SF}/wan_cache/finals128/prompts_used.txt").read().splitlines()
    assert len(prompts) == 128

    vdir = f"{OUT}/videos_{tag}" if args.limit == 128 else f"{OUT}/videos_{tag}_lim{args.limit}"
    os.makedirs(vdir, exist_ok=True)
    for link in glob.glob(f"{vdir}/*.mp4"):
        keep = (os.path.exists(os.path.realpath(link))
                and re.match(rf".*/{tag}_p\d{{3}}\.mp4$", link)
                and os.path.realpath(link) in set(map(os.path.realpath, vids)))
        if not keep:
            os.remove(link)
    prompt_map = {}
    for v in vids:
        base = os.path.basename(v)
        dst = os.path.join(vdir, base)
        if not os.path.lexists(dst):
            os.symlink(v, dst)
        prompt_map[base] = prompts[int(base.split("_p")[-1][:3])]

    vb = VBench(torch.device("cuda"), os.path.join(VB, "vbench", "VBench_full_info.json"), f"{OUT}/results")
    tab_path = f"{OUT}/table_semantic.json"
    tab = json.load(open(tab_path)) if os.path.exists(tab_path) else {}
    tab.setdefault(tag, {})
    for dim in dims:
        try:
            vb.evaluate(videos_path=vdir, name=f"{tag}_{dim}", dimension_list=[dim],
                        prompt_list=prompt_map, mode="custom_input")
            r = json.load(open(f"{OUT}/results/{tag}_{dim}_eval_results.json"))
            raw = r[dim][0]
            lo, hi = CAL[dim]
            tab[tag][dim] = 100 * (raw - lo) / (hi - lo)
            tab[tag][dim + "_raw"] = raw
            print(f"RESULT {tag} {dim}: {tab[tag][dim]:.2f} (raw {raw:.4f})", flush=True)
        except Exception as e:  # noqa: BLE001 — score remaining dims, report the failure
            print(f"FAIL {tag} {dim}: {str(e)[:200]}", flush=True)
        json.dump(tab, open(tab_path, "w"), indent=1)
    print(f"{tag} semantic scoring done", flush=True)


if __name__ == "__main__":
    main()
