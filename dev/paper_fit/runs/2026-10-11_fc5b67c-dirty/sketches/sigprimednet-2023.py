"""SigPrimedNet: gene -> circuit layer whose prior only zeroes the initial kernel."""

import pandas as pd
import torch
from captum.attr import LRP
from torch import nn

import kpnn2

torch.manual_seed(42)
TYPES = ["B", "T", "NK", "Mono"]
# Final gene -> circuit membership (HiPathia effector subgraphs upstream).
circ = {"P-hsa04010-28": "egfr kras mapk1", "P-hsa04010-61": "kras mapk1 myc",
        "P-hsa04630-14": "jak1 stat3", "P-hsa04630-55": "jak1 stat1 irf1"}
e = pd.DataFrame([(g, c) for c, gs in circ.items() for g in gs.split()],
                 columns=["source", "target"])
spec = kpnn2.parse_layered(e)
hop = spec.hops[0]
X = pd.DataFrame(torch.rand(40, 10).numpy(), columns=[  # gene filter is off
    "myc", "cd19", "stat1", "egfr", "kras", "cd3e", "jak1", "mapk1", "irf1",
    "stat3"])
col = kpnn2.align_inputs(X.columns, spec)
x = torch.as_tensor(X.to_numpy())
circuits = nn.Linear(X.shape[1], hop.out_features)  # every data gene
with torch.no_grad():  # prior on the initial kernel only, then all train
    m = torch.zeros(hop.out_features, X.shape[1])  # GLUE: N109
    m[:, col] = hop.to_mask()  # GLUE: N109 mask columns by gene name
    circuits.weight.mul_(m)  # GLUE: N109
    circuits.bias.zero_()
model = nn.Sequential(circuits, nn.ReLU(), nn.Linear(hop.out_features,
                                                     len(TYPES)))
y = torch.randint(0, len(TYPES), (40,))
nn.functional.cross_entropy(model(x), y).backward()
act = kpnn2.map_node_attributions(model[:2](x).detach(), spec, layer=1)
pred = [TYPES[i] for i in model(x).argmax(1)]
top = act.to_pandas().apply(  # GLUE: N111 top-15 positive circuits per cell
    lambda r: list(r[r > 0].nlargest(15).index), axis=1)  # GLUE
freq = top.groupby(pred).apply(lambda s: pd.Series(  # GLUE: N111 count per
    [c for cs in s for c in cs]).value_counts().head(10))  # GLUE: type
kernel = pd.DataFrame(circuits.weight.detach(), index=spec.layer_nodes[1],
                      columns=X.columns)  # trained, all entries
lrp = torch.stack([LRP(model[:2]).attribute(x, target=j).detach()
                   for j in range(hop.out_features)])  # one circuit at a time
genes = pd.DataFrame(lrp.mean(1).numpy(), index=spec.layer_nodes[1],
                     columns=X.columns)  # paper: per true cell type
print(act.sizes, (kernel.abs() > 0).sum().sum(), int(hop.to_mask().sum()))
print(freq.round(2).head(6), genes.round(4).iloc[:, :5], sep="\n")
