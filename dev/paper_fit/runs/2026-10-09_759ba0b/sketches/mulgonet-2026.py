"""MULGONET: BP and MF branches on one shared input, a dense head."""

import pandas as pd
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from torch import nn

import kpnn2

torch.manual_seed(42)
OMICS = ["meth", "amp", "del", "exp"]
# Final edgelists, 3 levels (paper: 5). How they were derived from GO
# (depth cut, adjacent levels, gene-set size filter, pruning, one input
# per omics) is upstream of kpnn2.
terms = {"bp": "B11 B1|B12 B1|B111 B11|B111 B12|B121 B12",
         "mf": "M11 M1|M12 M1|M111 M11|M121 M12"}
genes = {"bp": {"B111": ["g1", "g2"], "B121": ["g2", "g3"]},
         "mf": {"M111": ["g1", "g4"], "M121": ["g3"]}}
specs = {ns: kpnn2.parse_layered(pd.DataFrame(
    [p.split() for p in terms[ns].split("|")] + [[f"{g}_{o}", t] for t, gs
     in genes[ns].items() for g in gs for o in OMICS],
    columns=["source", "target"])) for ns in terms}
X = pd.DataFrame(torch.rand(10, 20).numpy(), columns=[
    f"{g}_{o}" for o in OMICS for g in ["g1", "g2", "g3", "g4", "g5"]])

class Branch(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint) for h in spec.hops)
        self.acts = nn.ModuleList(nn.Tanh() for _ in spec.hops)
        self.drops = nn.ModuleList(nn.Dropout(0.5 if i == 0 else 0.1)
                                   for i in range(len(spec.hops) - 1))

    def forward(self, x):
        saved = {0: x}
        for i, hop in enumerate(self.spec.hops):
            h = self.acts[i](self.hops[i](kpnn2.gather_hop_inputs(saved, hop)))
            saved[hop.target_layer] = self.drops[i](h) if i < len(
                self.drops) else h
        return h


model = nn.ModuleDict({ns: Branch(s) for ns, s in specs.items()})
head = nn.Sequential(nn.Linear(sum(s.layer_dims[-1] for s in specs.values()),
                               16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid())
net = lambda bp, mf: head(torch.cat([model["bp"](bp), model["mf"](mf)], 1))  # noqa: E731
xs = tuple(torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, s)])
           for s in specs.values())
y = torch.randint(0, 2, (10, 1)).float()
loss = nn.functional.binary_cross_entropy(net(*xs), y) + 1e-4 * sum(
    b.hops[0].effective_weight().pow(2).sum() for b in model.values())
loss.backward()
model.eval()
pos = (y[:, 0] == 1).numpy()  # recurrence-positive samples
attr = IntegratedGradients(net).attribute(xs, n_steps=20)
inp = pd.concat([kpnn2.map_node_attributions(a.detach(), s, axis="inputs")
                 [pos].sum("observation").to_series() for a, s in zip(  # GLUE
                     attr, specs.values())]).groupby(level=0).sum()  # GLUE
hid = {(ns, h.target_layer): kpnn2.map_node_attributions(
    LayerIntegratedGradients(net, model[ns].acts[i]).attribute(
        xs, n_steps=20).detach(), s, hop_output=h)[pos].sum("observation")  # GLUE
    for ns, s in specs.items() for i, h in enumerate(s.hops)}
print({ns: s.layer_nodes[1:] for ns, s in specs.items()})
print(inp.round(4).head(6), hid["bp", 3].to_series().round(4))
