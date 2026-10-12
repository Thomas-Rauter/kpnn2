"""CellTICS: Reactome masked tanh layers; nets over the bottom 1..L layers vote."""

import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)
CLASSES = ["B", "T", "NK"]
# Final edgelist, 3 levels (paper: 5). Root, depth cut, copy chains,
# propagated leaf genes, and marker-gene inputs are upstream of kpnn2.
e = pd.DataFrame([p.split() for p in (
    "ENSG1 R111|ENSG2 R111|ENSG2 R112|ENSG3 R112|ENSG4 R12_copy1|"
    "ENSG5 R12_copy1|ENSG5 R211|ENSG6 R211|R111 R11|R112 R11|"
    "R12_copy1 R12|R211 R21|R11 R1|R12 R1|R21 R2").split("|")],
    columns=["source", "target"])
spec = kpnn2.parse_layered(e)
L = len(spec.hops)


class Net(nn.Module):  # bottom k graph layers, own weights, own output
    def __init__(self, spec, k):
        super().__init__()
        self.hops = nn.ModuleList(kpnn2.MaskedLinear(
            h.to_mask(), identity=spec.fingerprint) for h in spec.hops[:k])
        self.acts = nn.ModuleList(nn.Tanh() for _ in range(k))
        self.out = nn.Linear(spec.layer_dims[k], len(CLASSES))

    def forward(self, x):
        saved = {0: x}
        for i, hop in enumerate(spec.hops[:len(self.hops)]):
            saved[hop.target_layer] = self.acts[i](self.hops[i](
                kpnn2.gather_hop_inputs(saved, hop)))
        return self.out(saved[len(self.hops)]), saved


X = pd.DataFrame(torch.rand(30, 7).numpy(), columns=[
    "ENSG6", "ENSG5", "ENSG9", "ENSG4", "ENSG3", "ENSG2", "ENSG1"])
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
y = torch.randint(0, len(CLASSES), (30,))
nets = [Net(spec, k) for k in range(1, L + 1)]
for net in nets:
    logits, _ = net(x)
    nn.functional.cross_entropy(logits, y).backward()  # + L2 on all weights
    net.eval()
votes = torch.stack([net(x)[0].argmax(1) for net in nets])
pred = votes.mode(0).values  # majority vote over the L networks
labels = [CLASSES[i] for i in y]
gaps = []
for net in nets:
    saved = net(x)[1]
    for h in spec.hops[:len(net.hops)]:
        act = kpnn2.map_node_attributions(saved[h.target_layer].detach(),
                                          spec, hop_output=h)
        m = act.assign_coords(cls=("observation", labels)).groupby(  # GLUE
            "cls").mean().to_pandas()  # GLUE: N18 per-class mean
        top = m.apply(lambda c: c.nlargest(2), axis=0)  # GLUE: N107
        gaps.append(pd.DataFrame({"cls": m.idxmax(), "gap": top.max()  # GLUE
                                  - top.min()}))  # GLUE: top minus second
g = pd.concat(gaps)
g = g[g.gap >= 0.01]  # GLUE: N107 cutoff (paper 0.1; toy weights untrained)
g.index = g.index.str.replace(r"_copy\d+$", "", regex=True)  # GLUE: N108
print(spec.layer_nodes[1:], pred.tolist()[:10])
print(g.sort_values("gap").groupby(level=0).tail(1).round(3))  # GLUE: N108
