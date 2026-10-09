"""PathHDNN: pathway layers as dense masked layers, per-layer SHAP."""

import pandas as pd
import torch
from captum.attr import LayerDeepLiftShap
from torch import nn

import kpnn2

torch.manual_seed(42)
# Final edgelist, 3 levels (paper: 4). How it was derived from Reactome
# (data ancestors, depth cut, copy nodes, propagated memberships) is
# upstream of kpnn2.
e = pd.DataFrame([p.split() for p in (
    "g1_mut R111|g2_mut R111|g2_amp R111|g3_mut R12_copy1|g4_mut R211|"
    "g4_amp R211|R111 R11|R12_copy1 R12|R211 R21|R11 R1|R12 R1|R21 R2"
).split("|")], columns=["source", "target"])
spec = kpnn2.parse_layered(e)
X = pd.DataFrame(torch.randint(0, 2, (12, 7)).float().numpy(), columns=[
    "g1_mut", "g2_mut", "g2_amp", "g3_mut", "g4_mut", "g5_mut", "g6_mut"])
X = X.reindex(columns=X.columns.union(spec.input_nodes), fill_value=0)  # GLUE


class PathHDNN(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.MaskedLinear(
            h.to_mask(), identity=spec.fingerprint) for h in spec.hops)
        self.post = nn.ModuleList(nn.Sequential(
            nn.BatchNorm1d(h.out_features), nn.Dropout(0.5), nn.Tanh())
            for h in spec.hops)
        self.out = nn.Linear(spec.layer_dims[-1], 2)

    def forward(self, x):
        saved = {0: x}
        for i, hop in enumerate(self.spec.hops):
            z = self.hops[i](kpnn2.gather_hop_inputs(saved, hop))
            saved[hop.target_layer] = self.post[i](z)
        return self.out(saved[len(self.spec.hops)])


x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
model = PathHDNN(spec)
nn.functional.cross_entropy(model(x), torch.randint(0, 2, (12,))).backward()
model.eval()
shap = [kpnn2.map_node_attributions(LayerDeepLiftShap(model, layer).attribute(
    x, baselines=x, target=1, attribute_to_layer_input=True).detach(), spec,
    **kw) for layer, kw in [(m, {"hop_input": h}) for m, h in zip(
        model.hops, spec.hops)] + [(model.out, {"layer": len(spec.hops)})]]
score = pd.concat([abs(s).mean("observation").to_series()  # GLUE: fold
                   for s in shap])  # GLUE
table = spec.to_edgelist().merge(score.rename("score"),  # GLUE: one row per
                                 left_on="source", right_index=True)  # GLUE
print(spec.layer_nodes[1:], spec.input_nodes)
print(table)
