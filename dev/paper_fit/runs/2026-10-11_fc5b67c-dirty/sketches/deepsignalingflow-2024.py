"""DeepSignalingFlow: per-edge scalars on shared channel maps, up and down."""

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import kpnn2

torch.manual_seed(42)
# Final graph: KEGG gene-gene edges plus drug-target edges both ways (the
# signaling filter and dropping genes without omics are upstream).
gg = pd.DataFrame([p.split() for p in (
    "G0 G1|G1 G2|G2 G3|G3 G1|G2 G4|G4 G5|G5 G6|G6 G4|G3 G6|G6 G7"
).split("|")], columns=["source", "target"])
dt = pd.DataFrame([("DA", "G1"), ("DB", "G4"), ("DC", "G5")],
                  columns=["source", "target"])
edges = pd.concat([gg, dt, dt.rename(columns={"source": "target",
                                              "target": "source"})])
spec = kpnn2.parse_adjacency(edges)
n, src, tgt = spec.state_dim, *map(torch.as_tensor, (spec.source_index,
                                                     spec.target_index))
drug_edge = torch.zeros(len(src), dtype=torch.bool)
for s, t in pd.concat([dt, dt[["target", "source"]].set_axis(
        ["source", "target"], axis=1)]).itertuples(index=False):
    drug_edge[list(spec.edge_location(s, t))] = True
deg_in = torch.bincount(tgt, minlength=n).clamp(min=1)  # GLUE: N81 mean over
deg_out = torch.bincount(src, minlength=n).clamp(min=1)  # GLUE: in-edges


class DrugFixed(nn.Module):  # drug-target edges held at 1 (CONTEXT freeze)
    def forward(self, w):
        return torch.where(drug_edge, torch.ones_like(w), w)


class Layer(nn.Module):  # x: (batch, channels, n); channels share each edge
    def __init__(self, cin, cout):
        super().__init__()
        self.up = kpnn2.PackedLinear(src, tgt, n, n, bias=False,
                                     constraint=DrugFixed())
        self.down = kpnn2.PackedLinear(tgt, src, n, n, bias=False,
                                       constraint=DrugFixed())
        self.m = nn.ModuleList(nn.Linear(cin, cout, bias=False)
                               for _ in range(3))  # up, down, self

    def forward(self, x):
        y = [m(x.transpose(1, 2)).transpose(1, 2) for m in self.m]
        h = torch.cat([self.up(y[0]) / deg_in, self.down(y[1])  # GLUE
                       / deg_out, y[2]], 1)  # GLUE: N81 divide by degree
        return F.leaky_relu(F.normalize(h, dim=1), 0.1)


layers = nn.ModuleList([Layer(4, 4), Layer(12, 4), Layer(12, 36)])
P1, P2 = nn.Parameter(torch.randn(108, 16)), nn.Parameter(torch.randn(16, 16))
omics = pd.DataFrame(torch.rand(9, 2).numpy(), columns=["rna", "cnv"], index=[
    "G5", "G1", "G7", "G9", "G3", "G0", "G2", "G6", "G4"])
om = torch.as_tensor(omics.reindex(spec.nodes, fill_value=0).to_numpy().T,
                     dtype=torch.float32)  # GLUE: N45 data onto state nodes
pairs = [("DA", "DB"), ("DB", "DC"), ("DA", "DC")]
tg = {d: set(dt.target[dt.source == d]) for d in dt.source}


def flags(a, b):  # per sample: target of either drug, target of both
    either = [(v in tg[a]) | (v in tg[b]) for v in spec.nodes]  # GLUE
    both = [(v in tg[a]) & (v in tg[b]) for v in spec.nodes]  # GLUE
    return torch.tensor([either, both]).float()


x = torch.stack([torch.cat([flags(a, b), om]) for a, b in pairs])  # (3,4,n)
pos = {d: spec.node_units(d).start for d in tg}  # GLUE: N82 drug nodes
ia, ib = (torch.tensor([pos[p[i]] for p in pairs]) for i in (0, 1))  # GLUE
h = nn.BatchNorm1d(4)(x)
for layer in layers:
    h = layer(h)
r = torch.arange(len(pairs))
ha, hb = h[r, :, ia], h[r, :, ib]  # GLUE: N82 read nodes chosen per sample
score = ((ha @ P1 @ P2 @ P1.T) * hb).sum(1)  # bilinear decoder
F.mse_loss(score, torch.rand(len(pairs))).backward()
flow = torch.stack([torch.stack([l.up.weight, l.down.weight]).abs()
                    for l in layers]).mean((0, 1)).detach()  # slot = edge
el = spec.to_edgelist().assign(score=flow.numpy())[~drug_edge.numpy()]
deg = pd.concat([el.groupby(c).score.sum() for c in  # GLUE: N72 weighted
                 ("source", "target")]).groupby(level=0).sum()  # GLUE: degree
print(spec.input_nodes, spec.output_nodes, h.shape)
print(el.round(3).head(4), deg.round(3).to_dict())
