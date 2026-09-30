# Clean Forcing: Drift-Resistant Autoregressive Video Diffusion with a Frozen Base

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://clean-forcing.github.io/static/pdfs/clean_forcing.pdf) | [![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=github)](https://clean-forcing.github.io/) | [![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-HuggingFace-yellow)](https://huggingface.co/illustro1/clean-forcing) | [![Results](https://img.shields.io/badge/%F0%9F%A4%97%20Videos%20%26%20Measurements-HuggingFace-yellow)](https://huggingface.co/datasets/illustro1/clean-forcing-results)

<p align="center">
    <br>
    <img src="assets/pipeline.png" width="100%"/>
    <br>
</p>

This repository is for the Clean Forcing method introduced in the following paper:

> **Clean Forcing: Drift-Resistant Autoregressive Video Diffusion with a Frozen Base** \
> [Wenqing Wang](https://wenqing-wang.netlify.app/)<sup>1</sup>, [Joonghyuk Shin](https://joonghyuk.com/)<sup>2</sup>, [Jonathan Tremblay](https://jtremblay.org/)<sup>3\*</sup>, [Chan Hee Song](https://chanh.ee/)<sup>3\*</sup>, and [Yun Fu](https://www1.ece.neu.edu/~yunfu/)<sup>1\*</sup> \
> <sup>1</sup>Northeastern University, <sup>2</sup>Seoul National University, <sup>3</sup>NVIDIA \
> <sup>\*</sup>Co-last authors

## Contents
1. [Abstract](#abstract)
2. [Setup](#setup)
3. [Checkpoints](#checkpoints)
4. [Inference](#inference)
5. [Train](#train)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Repository Layout](#repository-layout)
9. [Citation](#citation)
10. [Acknowledgments](#acknowledgments)

## Abstract
Autoregressive (AR) video diffusion models generate streaming video of arbitrary length by conditioning
each new chunk of frames on those already generated. However, they suffer from exposure bias, where a
model trained on clean context must continue from its own imperfect outputs, and errors compound until
quality collapses within seconds (drift). Existing methods either require cluster-scale base retraining,
apply training-free inference corrections with inconsistent results, or train frozen-base correctors for
sampling and cache errors. We instead find that drift is largely deterministic and therefore learnable,
as holding the noisy state fixed while varying only the history reveals a counterfactual velocity gap
that is \~95% systematic across noise seeds.

Based on this observation, we introduce *Clean Forcing*, which trains a 5.9M-parameter LoRA corrector on
the *frozen* base model to regress this gap, with a closed-loop stage that adds DAgger-style aggregation
and a drift-contraction objective to expose the corrector to its own rollouts. Because the clean
histories can be the base model's own single-shot generations, training needs *no real videos*, and the
merged corrector adds no inference cost. In 50 s text-to-video generation, Clean Forcing reduces Δ-drift
from +11.3 to +1.65 without real video and to **−0.16** with 40 real clips, improving on the best
published result of +3.57 with over **150×** less data. Extensive experiments demonstrate that Clean
Forcing outperforms prior methods in aesthetic quality and data efficiency with reduced drift for
long-horizon video generation.

## Setup
One A100/H100-class GPU (≥ 48 GB), bf16.

- Step 1: Create the `clean_forcing` conda environment, install the dependencies, and download the
  public Wan2.1-T2V-1.3B base into `self_forcing/wan_models/`:
```bash
bash SETUP.sh
```

- Step 2 (optional): For the external-base experiments, download the public
  [Causal-Forcing](https://huggingface.co/zhuhz22/Causal-Forcing) checkpoint (see the commented line in `SETUP.sh`).

## Checkpoints
All checkpoints are on Hugging Face at [`illustro1/clean-forcing`](https://huggingface.co/illustro1/clean-forcing).
From the repository root:
```bash
huggingface-cli download illustro1/clean-forcing --include "weights/*" --local-dir .                     # 14 corrector LoRAs -> weights/
huggingface-cli download illustro1/clean-forcing adapted_base_4000.pt --local-dir self_forcing/wan_cache  # adapted 1.3B base
(cd weights && sha256sum -c SHA256SUMS)
```
The same repository holds the adapted Wan2.1-14B base (`wan14b_adapted_base_4000.pt`). See `RUN.md`
for which base each LoRA applies to.

## Inference
Generate one 50 s video (832×480, 16 fps) with or without the corrector:
```bash
# with Clean Forcing
LORA=weights/lora_r_phi_v2_both_adapt.pt PROMPT="a corgi surfing a wave at sunset" python demo_generate.py
# the same prompt on the uncorrected base (it drifts within 10-20 s)
LORA=none PROMPT="a corgi surfing a wave at sunset" python demo_generate.py
```
Use `lora_r_phi_v2s_adapt.pt` for the zero-real-video corrector, and `KLAT=120` for a \~30 s clip.

## Train
All commands run from `self_forcing/`. The full pipeline takes about 3 GPU-days and uses no real videos.
```bash
# 1. Synthetic reference clips (bidirectional single-shot generations of the base)
python wan_gen_synthetic.py                          # -> wan_cache/synth_clips.pt

# 2. Light causal adaptation of the base (Diffusion-Forcing objective, LoRA r64 merged into the base)
STEPS=6000 python wan_train_adapt.py                 # -> wan_cache/adapted_base_{2000,4000,6000}.pt; deploy 4000

# 3. Drift pairs (clean references + drifted rollouts)
ADAPTED_BASE=wan_cache/adapted_base_4000.pt python wan_build_pairs_synth.py   # -> pairs_synth_adapt.pt

# 4. One-step corrector (counterfactual clean-history loss)
PAIRS=pairs_synth_adapt.pt CKPT=lora_r_phi_synth_adapt.pt \
  ADAPTED_BASE=wan_cache/adapted_base_4000.pt python wan_train_synth.py

# 5. Closed-loop corrector: DAgger pool with the one-step corrector active, then drift contraction
CORRECTOR=wan_cache/lora_r_phi_synth_adapt.pt OUT=pairs_synth_dagger_adapt.pt \
  ADAPTED_BASE=wan_cache/adapted_base_4000.pt python wan_build_pairs_synth.py
K=21 POOLS=pairs_synth_adapt.pt,pairs_synth_dagger_adapt.pt INIT=lora_r_phi_synth_adapt.pt \
  CKPT=lora_r_phi_v2s_adapt.pt LOSS_MODE=both ADAPTED_BASE=wan_cache/adapted_base_4000.pt \
  python wan_train_v2.py                             # -> Clean Forcing (zero real videos)
```
Please refer to `RUN.md` for details, and to `cf_external_base/` for the Causal-Forcing base.

## Evaluation
128 held-out MovieGen prompts (`prompts_finals128.txt`), un-seeded 50 s text-to-video. From `self_forcing/`:
```bash
LORA=wan_cache/lora_r_phi_v2s_adapt.pt TAG=av2s N=128 ADAPTED_BASE=wan_cache/adapted_base_4000.pt \
  python ../scripts/eval_corrector_subset.py         # MUSIQ, Δ-drift and videos
python ../scripts/score_official_6dim.py             # official VBench
TAGS=av2s,abase,sfd,skyr N=128 python ../scripts/posthoc_metrics.py   # drift metrics (appendix)
python ../scripts/make_paper_figures.py              # paper figures
```
`paper_tables.md` lists every number in the paper with its source. `user_study/analysis.py` reproduces
the user-study table. All generated videos and per-video measurements are on Hugging Face at
[`illustro1/clean-forcing-results`](https://huggingface.co/datasets/illustro1/clean-forcing-results).

## Results
The same frozen base before (top) and after (bottom) merging the Clean Forcing corrector:

<p align="center"><img src="assets/teaser.jpg" width="100%"/></p>

Frames over the full 50 s, same prompt and seed (rows top to bottom: adapted base, context noise,
history guidance, Self Forcing, Clean Forcing zero real, Clean Forcing +real):

<p align="center"><img src="assets/comparison_frames_p098.jpg" width="100%"/></p>

<details>
<summary>More comparisons</summary>

<p align="center"><img src="assets/comparison_frames_p101.jpg" width="100%"/></p>
<p align="center"><img src="assets/comparison_frames_p113.jpg" width="100%"/></p>
<p align="center"><img src="assets/comparison_frames_p040.jpg" width="100%"/></p>
<p align="center"><img src="assets/comparison_frames_p008.jpg" width="100%"/></p>
<p align="center"><img src="assets/comparison_frames_p028.jpg" width="100%"/></p>
<p align="center"><img src="assets/comparison_frames_p032.jpg" width="100%"/></p>
<p align="center"><img src="assets/comparison_frames_p030.jpg" width="100%"/></p>

</details>

Videos are on the [project page](https://clean-forcing.github.io/).

## Repository Layout
- `self_forcing/`: modified [Self-Forcing](https://github.com/guandeh17/Self-Forcing) tree with the
  block-causal Wan2.1 pipeline (`pipeline/causal_diffusion_inference.py`, corrector hooks), LoRA
  (`wan/modules/lora.py`), and the training entry points (`wan_*.py`).
- `scripts/`: evaluation, VBench scoring, and figures.
- `cf_external_base/`: training and evaluation on the Causal-Forcing base.
- `user_study/`: anonymized responses and analysis.
- `demo_generate.py`, `SETUP.sh`, `RUN.md`: demo, environment, and runbook.

## Citation
If our work or code helps you, please consider citing our paper. Thank you!
```bibtex
@article{wang2026cleanforcing,
  title={Clean Forcing: Drift-Resistant Autoregressive Video Diffusion with a Frozen Base},
  author={Wang, Wenqing and Shin, Joonghyuk and Tremblay, Jonathan and Song, Chan Hee and Fu, Yun},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgments
In this code we refer to the following codebases: [Self-Forcing](https://github.com/guandeh17/Self-Forcing),
[Wan2.1](https://github.com/Wan-Video/Wan2.1), and [Causal-Forcing](https://huggingface.co/zhuhz22/Causal-Forcing).
We gratefully thank the authors for their wonderful work.

## License
Code is released under the Apache License 2.0 (`LICENSE`). `self_forcing/` is a modified copy of
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) and keeps its Apache-2.0 license; the Wan2.1
base models and the Causal-Forcing checkpoint are distributed by their authors under Apache-2.0.
