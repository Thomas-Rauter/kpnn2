"""KPNN: unlayered DAG, one sigmoid unit per node, roots are outputs."""

import pandas as pd
import torch
import xarray as xr
from captum.attr import LayerGradientXActivation
from torch import nn

import kpnn2

torch.manual_seed(42)
# Regulator -> target pairs as in the KPNN edge list; genes carry _gene.
pairs = ("CD4 R1|CD4 R2|CD8 R3|R1 S1|R2 S1|R2 S2|R3 TF3|R1 g8_gene|S1 TF1|"
         "S1 TF2|S2 TF2|S2 TF3|TF1 g1_gene|TF1 g2_gene|TF1 g3_gene|"
         "TF2 g3_gene|TF2 g4_gene|TF2 g5_gene|TF3 g5_gene|TF3 g6_gene|"
         "TF3 g7_gene|TF3 g9_gene|XYZ S2|XYZ S3|S3 TF4|TF4 g6_gene")
e = pd.DataFrame([p.split() for p in pairs.split("|")],
                 columns=["target", "source"])  # data flows child -> parent
data = pd.DataFrame(torch.rand(16, 9).numpy(),
                    columns=[f"g{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 10]])
data.columns = data.columns + "_gene"
labels = ["CD4", "CD8"]
while True:  # GLUE: drop unmatched leaves and unlabeled roots until stable
    keep = ((e.source.isin(data.columns) | e.source.isin(e.target))  # GLUE
            & (e.target.isin(labels) | e.target.isin(e.source)))  # GLUE
    if keep.all():  # GLUE
        break  # GLUE
    e = e[keep]  # GLUE
spec = kpnn2.parse_layered(e)
sib = e.merge(e, on="target").groupby("source_x").source_y.nunique()  # GLUE
kp = sib.map(lambda c: {1: 1.0, 2: 0.9, 3: 0.7}.get(c, 0.5))  # GLUE
keep = [torch.tensor([kp.get(n, 1.0) for n in spec.layer_nodes[  # GLUE
    h.target_layer]], dtype=torch.float32) for h in spec.hops]  # GLUE


class KPNN(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint) for h in spec.hops)
        self.acts = nn.ModuleList(nn.Sigmoid() for _ in spec.hops)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        saved, pre = {0: self.drop(x)}, {}
        for i, hop in enumerate(self.spec.hops):
            z = self.hops[i](kpnn2.gather_hop_inputs(saved, hop))
            h = self.acts[i](z)
            if self.training:  # GLUE: graph-dependent dropout per node
                h = h * torch.bernoulli(keep[i].expand_as(h)) / keep[i]  # GLUE
            saved[hop.target_layer], pre[hop.target_layer] = h, z
        return torch.cat([pre[l][:, u] for l, u in  # GLUE: roots at any depth
                          map(self.spec.node_units, labels)], 1)  # GLUE


model = KPNN(spec)
x = torch.as_tensor(data.to_numpy()[:, kpnn2.align_inputs(data.columns, spec)])
nn.functional.binary_cross_entropy_with_logits(
    model(x), torch.randint(0, 2, (16, 2)).float()).backward()
model.eval()
prob = lambda x: torch.sigmoid(model(x))  # noqa: E731
scores = xr.concat([kpnn2.map_node_attributions(LayerGradientXActivation(
    prob, model.acts[i], multiply_by_inputs=False).attribute(x, target=0
    ).detach(), spec, hop_output=h) for i, h in enumerate(spec.hops)
    if set(spec.layer_nodes[h.target_layer]) & set(spec.hidden_nodes)], "node",
    coords="different", compat="equals")
node_score = scores.sel(node=list(spec.hidden_nodes)).mean(  # GLUE: fold
    "observation")  # GLUE: cells
el = spec.to_edgelist()
el["weight"] = [model.hops[h].effective_weight()[s[0]].item() for h, s in
                (spec.edge_location(a, b) for a, b in zip(el.source, el.target))]
shuffled = kpnn2.align_inputs(data.columns.to_series().sample(frac=1), spec)
print(spec.layer_nodes, spec.output_nodes, len(spec.skips))
print(node_score.to_series().round(4).to_dict(), el.head(3), shuffled)
