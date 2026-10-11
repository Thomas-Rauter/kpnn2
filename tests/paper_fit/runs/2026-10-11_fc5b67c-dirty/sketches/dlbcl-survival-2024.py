"""VNNSurv: Reactome pathways of max(10, in-degree) units, root, Cox head."""

import pandas as pd
import torch
from captum.attr import GradientShap
from torch import nn

import kpnn2

torch.manual_seed(42)
CLIN = ["ipi_high", "age_60", "stage_3"]
# Final edgelist: propagated gene sets, so a gene feeds every kept pathway
# that contains it, at any depth; `root` above the top pathways. Dropping
# small pathways and adding the root is upstream of kpnn2.
sets = {"P111": "g1 g2", "P11": "g1 g2 g3", "P12": "g4 g5",
        "P1": "g1 g2 g3 g4 g5", "P21": "g6 g7", "P2": "g6 g7 g8"}
kids = "P111 P11|P11 P1|P12 P1|P21 P2|P1 root|P2 root"
e = pd.DataFrame([(g, p) for p, gs in sets.items() for g in gs.split()]
                 + [k.split() for k in kids.split("|")],
                 columns=["source", "target"])
deg = e.target.value_counts()  # GLUE: N67 units from member genes + children
widths = {p: max(10, int(deg[p])) for p in sets} | {"root": 16}  # GLUE
spec = kpnn2.parse_layered(e, widths=widths)


class VNNSurv(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint) for h in spec.hops)
        self.post = nn.ModuleList(nn.Sequential(
            nn.Mish(), nn.BatchNorm1d(h.out_features), nn.Dropout(0.1))
            for h in spec.hops)
        self.head = nn.Sequential(nn.Linear(spec.layer_dims[-1] + len(CLIN),
                                            64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, 1))

    def forward(self, x, clin):
        saved = {0: x}
        for i, hop in enumerate(self.spec.hops):
            z = self.hops[i](kpnn2.gather_hop_inputs(saved, hop))
            saved[hop.target_layer] = self.post[i](z)
        return self.head(torch.cat([saved[len(self.spec.hops)], clin], 1))


cols = ["g3", "g1", "g9", "g2", "g8", "g5", "g4", "g7", "g6"] + CLIN
X = pd.DataFrame(torch.randint(0, 2, (16, 12)).float().numpy(), columns=cols)
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
clin = torch.as_tensor(X[CLIN].to_numpy())
model = VNNSurv(spec)
time, event = torch.rand(16), torch.randint(0, 2, (16,)).float()
risk = model(x, clin).squeeze(1)[time.argsort(descending=True)]
loss = -((risk - risk.logcumsumexp(0)) * event[time.argsort(
    descending=True)]).sum() / event.sum()  # Cox partial likelihood
loss.backward()
model.eval()  # paper only: SHAP per input, mean over carriers
sv = GradientShap(model).attribute((x, clin), baselines=(x * 0, clin * 0),
                                   n_samples=8)
imp = kpnn2.map_node_attributions(sv[0].detach(), spec, axis="inputs")
carrier = kpnn2.map_node_attributions(x, spec, axis="inputs") == 1
print(spec.layer_nodes[1:], spec.layer_dims, len(spec.skips))
print(imp.where(carrier).mean("observation").to_series().round(4).to_dict())
