"""
LoRA adapter for the Wan DiT — the idea-#6 corrector r_phi as low-rank deltas on the
frozen base's self-attention projections (inherits the base's history-conditioning).

v_{theta+LoRA}(z_t, h_gen) trained -> v_theta(z_t, h_clean). Base frozen; only A,B trained.
B zero-init => LoRA starts as identity (scale-any == base). `scale` is a runtime gain
(0 for the clean-history teacher pass; alpha(t) for the corrected pass).
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / rank)
        nn.init.zeros_(self.B.weight)          # start as zero delta == base
        self.scale = 1.0                       # runtime gain; 0 == base

    def forward(self, x):
        out = self.base(x)
        if self.scale != 0:
            delta = self.B(self.A(x.to(self.A.weight.dtype)))  # LoRA path in A/B dtype (fp32)
            out = out + self.scale * delta.to(out.dtype)
        return out


def apply_lora(model, rank=16, targets=("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o")):
    wrapped = []
    for mname, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            if isinstance(child, nn.Linear) and any(t in full for t in targets):
                setattr(module, cname, LoRALinear(child, rank).to(child.weight.device))
                wrapped.append(full)
    return wrapped


def set_lora_scale(model, s):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.scale = s


def lora_parameters(model):
    ps = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            ps += list(m.A.parameters()) + list(m.B.parameters())
    return ps


def num_lora_params(model):
    return sum(p.numel() for p in lora_parameters(model))
