"""DeepHisCoM: one bias-free MLP per pathway (own width and depth), beta = score."""

import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)
COV = ["age", "sex"]
members = {"hsa00010": "m1 m2 m3", "hsa00020": "m3 m4", "hsa00230": "m5 m6",
           "hsa00330": "m6 m7 m8"}
layerinfo = {"hsa00010": (3, 1), "hsa00020": (0, 0),  # (width, depth) as
             "hsa00230": (2, 2), "hsa00330": (3, 1)}  # in layerinfo.csv
D = max(d for _, d in layerinfo.values())
rows, widths, ranks = [], {}, {}
for p, (w, d) in layerinfo.items():  # members -> p_h1 -> ... -> p_hd -> p
    chain = [f"{p}_h{j}" for j in range(1, d + 1)] + [p]
    rows += [(m, chain[0]) for m in members[p].split()]
    rows += list(zip(chain, chain[1:]))
    widths |= {c: w for c in chain[:-1]}
    ranks |= {c: j for j, c in enumerate(chain, 1)} | {p: D + 1}
e = pd.DataFrame(rows, columns=["source", "target"])
ranks |= {m: 0 for m in e.source if m.startswith("m")}
spec = kpnn2.parse_layered(e, widths=widths, ranks=ranks)  # pathways on top
P = spec.layer_dims[-1]


class DeepHisCoM(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            bias=False, identity=spec.fingerprint) for h in spec.hops)
        self.act, self.drop = nn.LeakyReLU(0.2), nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(P)
        self.out = nn.Linear(P + len(COV), 1)
        with torch.no_grad():
            for h in self.hops:
                h.weight.fill_(0.01)
            self.out.weight.zero_()
            self.out.bias.fill_(0.001)

    def forward(self, x, cov):
        saved = {0: x}
        for i, hop in enumerate(self.spec.hops):
            saved[hop.target_layer] = self.drop(self.act(self.hops[i](
                kpnn2.gather_hop_inputs(saved, hop))))
        f = self.bn(saved[len(self.spec.hops)])
        f = f / f.norm()  # one norm over the batch and all pathways
        return torch.sigmoid(self.out(self.drop(torch.cat([f, cov], 1))))


X = pd.DataFrame(torch.randn(32, 10).numpy(), columns=[
    "m8", "m7", "m6", "m5", "m4", "m3", "m2", "m1", "age", "sex"])
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
cov = torch.as_tensor(X[COV].to_numpy())
y = torch.randint(0, 2, (32, 1)).float()
betas = {}
for run in range(3):  # run 0: real phenotype; then permuted phenotypes
    torch.manual_seed(42 + run)
    lab = y if run == 0 else y[torch.randperm(32)]
    model = DeepHisCoM(spec)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(5):
        opt.zero_grad()
        nn.functional.binary_cross_entropy(model(x, cov), lab).backward()
        opt.step()
    betas[run] = kpnn2.map_node_attributions(  # beta per named pathway
        model.out.weight[0, :P].detach(), spec, layer=len(spec.hops)
    ).to_series()
print(spec.layer_nodes, [h.source_layers for h in spec.hops])
print(pd.DataFrame(betas).round(5))
