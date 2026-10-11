"""DTox under seeded repeats, shuffled labels, and label-equal inputs."""

import math

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerLRP
from captum.attr._utils.lrp_rules import EpsilonRule
from torch import nn

import kpnn2

torch.manual_seed(42)
# Final DTox hierarchy: targets feed pathways at any depth; one root unit
# reads the top pathways (adding it is upstream). Set sizes from its files.
e = pd.DataFrame([p.split() for p in (
    "t1 P11|t2 P11|t3 P11|t3 P12|t4 P12|t5 P2|t6 P2|t7 P2|t8 P1|t8 P2|"
    "P11 P1|P12 P1|P1 root|P2 root").split("|")], columns=["source", "target"])
size, smax = {"P11": 3, "P12": 6, "P1": 9, "P2": 12}, 12
spec = kpnn2.parse_layered(e, widths={  # GLUE: N104 units from set size
    p: 1 if s < 5 else round(1 + 19 * math.log(s / 5) / math.log(smax / 5))
    for p, s in size.items()})  # GLUE
layer_of = {n: spec.node_units(n)[0] for n in spec.hidden_nodes}


class DTox(nn.Module):
    def __init__(self, spec, gen):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint, generator=gen) for h in spec.hops)
        self.acts = nn.ModuleList(nn.ReLU() for _ in spec.hops[:-1])
        self.aux = nn.ModuleDict({n: nn.Linear(u.stop - u.start, 1)
                                  for n in spec.hidden_nodes
                                  for _, u in [spec.node_units(n)]})

    def forward(self, x, aux=False):
        saved = {0: x}
        for i, hop in enumerate(self.spec.hops):
            z = self.hops[i](kpnn2.gather_hop_inputs(saved, hop))
            saved[hop.target_layer] = self.acts[i](z) if i < len(
                self.acts) else z
        heads = {n: self.aux[n](saved[l][:, u]) for n in self.aux
                 for l, u in [self.spec.node_units(n)]}
        return (z, heads) if aux else z


def lrp(model, i, x):  # Captum has no default rule for PackedLinear and
    for h in model.hops:  # deletes rules after every call
        h.rule = EpsilonRule(0.1)
    return LayerLRP(model, model.hops[i]).attribute(x, target=0).detach()


X = pd.DataFrame(torch.rand(40, 8).numpy(), columns=[f"t{i}" for i in range(
    8, 0, -1)])
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
y = torch.randint(0, 2, (40, 1)).float()
bce = nn.functional.binary_cross_entropy_with_logits
runs = {}
for setup in ("real", "shuffled", "deterministic"):
    for seed in range(3):  # paper: 51 runs per setup
        gen = torch.Generator().manual_seed(seed)
        lab = torch.randint(0, 2, (40, 1), generator=gen).float() if (
            setup == "shuffled") else y
        xin = lab.expand_as(x).clone() if setup == "deterministic" else x  # GLUE
        model = DTox(spec, gen)
        out, heads = model(xin, aux=True)
        loss = bce(out, lab) + 0.5 * sum(bce(hd, lab) / len(
            spec.layer_nodes[layer_of[n]]) for n, hd in heads.items())
        loss.backward()
        pos = lab[:, 0] == 1
        runs[setup, seed] = pd.concat([kpnn2.map_node_attributions(
            lrp(model, i, xin[pos]), spec, hop_output=h).groupby(  # GLUE: N105
            "node").sum().mean("observation").to_series()  # GLUE: unit sum
            for i, h in enumerate(spec.hops[:-1])])
S = pd.DataFrame(runs)  # GLUE: N33 nodes x (setup, seed)
both = S[["real", "shuffled"]]  # N102: quantile-normalize, real - control
qn = both.rank(method="first").astype(int).apply(  # GLUE: N102
    lambda r: pd.Series(np.sort(both.values, 0).mean(1)[r - 1], r.index))  # GLUE
corrected = qn["real"].mean(1) - qn["shuffled"].mean(1)  # GLUE: N102
el = spec.to_edgelist()
deg = el.source.value_counts().add(el.target.value_counts(),  # GLUE: N103
                                   fill_value=0)[S.index]  # GLUE
print(S.round(3), S.corr().round(2).iloc[:3, :3], sep="\n")
print(corrected.round(3).to_dict(), S["real"].mean(1).corr(deg))  # GLUE
