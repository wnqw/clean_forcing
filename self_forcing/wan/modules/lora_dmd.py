"""
Dual-adapter LoRA for the Axis-2 DMD baseline (AutoRefiner-style: noise + distribution
matching, NO clean-history teacher). One frozen base carries TWO independent LoRA adapters:
  'gen'    = the refiner G (what we ship / eval)
  'critic' = the fake-score network (trained on G's samples; the DMD critic)
Exactly one adapter is active per forward (or None = frozen base = the DMD 'real' score).
Kept separate from lora.py so the Axis-1 checkpoints/scripts are unaffected.
"""
import torch
import torch.nn as nn


class DualLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        def mk():
            A = nn.Linear(base.in_features, rank, bias=False)
            B = nn.Linear(rank, base.out_features, bias=False)
            nn.init.normal_(A.weight, std=1.0 / rank)
            nn.init.zeros_(B.weight)              # zero-init -> starts as base
            return A, B
        self.Ag, self.Bg = mk()
        self.Ac, self.Bc = mk()
        self.active = None                        # None | 'gen' | 'critic'

    def forward(self, x):
        out = self.base(x)
        if self.active == "gen":
            out = out + self.Bg(self.Ag(x.to(self.Ag.weight.dtype))).to(out.dtype)
        elif self.active == "critic":
            out = out + self.Bc(self.Ac(x.to(self.Ac.weight.dtype))).to(out.dtype)
        return out


def apply_dual_lora(model, rank=16, targets=("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o")):
    wrapped = []
    for mname, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            if isinstance(child, nn.Linear) and any(t in full for t in targets):
                setattr(module, cname, DualLoRALinear(child, rank).to(child.weight.device))
                wrapped.append(full)
    return wrapped


def set_active(model, which):                     # None | 'gen' | 'critic'
    for m in model.modules():
        if isinstance(m, DualLoRALinear):
            m.active = which


def adapter_parameters(model, which):
    ps = []
    for m in model.modules():
        if isinstance(m, DualLoRALinear):
            if which == "gen":
                ps += list(m.Ag.parameters()) + list(m.Bg.parameters())
            elif which == "critic":
                ps += list(m.Ac.parameters()) + list(m.Bc.parameters())
    return ps


def gen_state_dict(model):
    """Save the 'gen' adapter in the same {i: tensor} order as lora.lora_parameters,
    so wan_ablation_eval / drift-curve loaders (which apply single-adapter LoRA) can read it."""
    return {i: p.detach().cpu() for i, p in enumerate(adapter_parameters(model, "gen"))}
