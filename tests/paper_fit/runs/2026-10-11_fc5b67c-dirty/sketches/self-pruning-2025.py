"""LEMBAS-GPU self-pruning: signed cyclic net plus random added edges."""

import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)
# Final signed edge list with cycles, plus 3 random unsigned non-edges
# (drawing them is upstream of kpnn2); `added` stays on the caller's table.
net = pd.DataFrame([r.split() for r in (
    "L1 R1 1|L2 R1 1|L2 R2 1|R1 K1 1|R2 K2 1|K1 K3 1|K2 K3 -1|K3 K1 -1|"
    "K3 T1 1|K2 T2 0|T1 K2 1|K3 T3 -1|R1 K2 0|K1 T2 0|T3 K1 0"
).split("|")], columns=["source", "target", "sign"]).astype({"sign": int})
net["added"] = [False] * 12 + [True] * 3
TF = ["T1", "T2", "T3"]
spec = kpnn2.parse_adjacency(net)
n, nnz = spec.state_dim, len(spec.source_index)
slot = [spec.edge_location(s, t)[0] for s, t in zip(net.source, net.target)]
sign = torch.zeros(nnz).index_put_((torch.tensor(slot),),
                                   torch.tensor(net.sign.values).float())


class KeepMask(nn.Module):  # CONTEXT's keep-mask; here for edge ablation
    def __init__(self, nnz):
        super().__init__()
        self.register_buffer("keep", torch.ones(nnz))

    def forward(self, w):
        return w * self.keep


class Lembas(nn.Module):
    def __init__(self, spec, steps=150, tol=1e-5):
        super().__init__()
        self.core = kpnn2.PackedLinear(
            spec.source_index, spec.target_index, n, n,
            identity=spec.fingerprint, constraint=KeepMask(nnz))
        self.register_buffer("inp", torch.as_tensor(spec.input_index))
        self.register_buffer("out", torch.as_tensor(  # GLUE: TFs by name,
            [spec.node_units(t).start for t in TF]))  # GLUE: sinks or not
        self.out_scale = nn.Parameter(torch.full((len(TF),), 1.2))
        self.steps, self.tol = steps, tol

    def act(self, z):  # Michaelis-Menten-like, leak 0.01
        return torch.where(z < 0, 0.01 * z, torch.where(
            z <= 0.5, z, 1 - 0.25 / z.clamp(min=0.5)))

    def forward(self, x):
        u = x.new_zeros(x.shape[0], n).index_copy(-1, self.inp, 3 * x)
        h = torch.zeros_like(u)
        for _ in range(self.steps):
            z = self.core(h) + u
            h, prev = self.act(z), h
            if (h - prev).abs().max() < self.tol:
                break
        return h[:, self.out] * self.out_scale, h, z


def rho(core, v, d=1.0, k=5):  # GLUE: spectral radius, power iteration
    for _ in range(k):  # GLUE
        v = d * (core(v) - core.bias)  # GLUE: (f'(z) .) W v, no dense W
        r, v = v.norm(dim=-1), v / v.norm(dim=-1, keepdim=True)  # GLUE
    return r  # GLUE


model = Lembas(spec)
with torch.no_grad():
    model.core.weight.copy_((0.1 + 0.1 * torch.rand(nnz)) * torch.where(
        sign < 0, -1.0, 1.0))
    model.core.weight.mul_(0.9 / rho(model.core, torch.rand(2, n), k=50  # GLUE
                                     ).max())  # GLUE
X = pd.DataFrame(torch.rand(8, 2).numpy(), columns=["L2", "L1"])
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
pred, h, z = model(x)
w = model.core.effective_weight()
fprime = torch.where(z < 0, 0.01, torch.where(  # GLUE: f'(z) for the
    z <= 0.5, 1.0, 0.25 / z.clamp(min=0.5) ** 2)).detach()  # GLUE: Jacobian
loss = (nn.functional.mse_loss(pred, torch.rand(8, 3))
        + 0.1 * torch.relu(-sign * w).sum() + 1e-6 * w.pow(2).sum()
        + torch.exp(rho(model.core, torch.rand(2, n), fprime[:2]).mean()))
loss.backward()
net["w"] = w.detach()[slot].numpy()  # trained weight of each named row
print(net.groupby("added").w.apply(lambda s: s.abs().mean()).round(4))
with torch.no_grad():  # single-edge ablation of the added edges
    base = model(x)[0]
    for i in net.index[net.added]:
        model.core.constraint.keep[slot[i]] = 0
        net.loc[i, "ko"] = float((model(x)[0] - base).abs().max())
        model.core.constraint.keep[slot[i]] = 1
print(spec.input_nodes, spec.output_nodes, net[net.added].round(4))
