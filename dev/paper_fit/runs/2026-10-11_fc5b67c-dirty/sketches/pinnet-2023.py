"""PINNet: genes -> pathways (scaled by sqrt size) beside a dense branch."""

import pandas as pd
import torch
from captum.attr import DeepLiftShap, LayerDeepLiftShap
from torch import nn

import kpnn2

torch.manual_seed(42)
# Final gene -> pathway membership (MSigDB KEGG; size filter upstream).
sets = {"hsa04110": "CDK4 CCND1 TP53 RB1", "hsa04151": "PIK3CA AKT1 PTEN",
        "hsa05010": "APP PSEN1 MAPT GSK3B AKT1"}
e = pd.DataFrame([(g, p) for p, gs in sets.items() for g in gs.split()],
                 columns=["source", "target"])
spec = kpnn2.parse_layered(e)
hop = spec.hops[0]


def post(o):  # after each branch's linear map
    return [nn.BatchNorm1d(o), nn.LayerNorm(o), nn.Tanh(), nn.Dropout(0.1)]


class PINNet(nn.Module):
    def __init__(self, spec, col, n_genes, n_fc=8):
        super().__init__()
        self.register_buffer("col", torch.as_tensor(col))
        self.pw = kpnn2.PackedLinear(hop.source_index, hop.target_index,
                                     hop.out_features, hop.in_features,
                                     identity=spec.fingerprint)
        self.register_buffer("root_deg", torch.bincount(torch.as_tensor(  # GLUE
            hop.target_index), minlength=hop.out_features).sqrt())  # GLUE: N99
        self.pw_post = nn.Sequential(*post(hop.out_features))
        self.fc1 = nn.Sequential(nn.Linear(n_genes, n_fc),
                                 *post(n_fc))
        self.head = nn.Sequential(nn.Linear(n_fc + hop.out_features, 64),
                                  nn.BatchNorm1d(64), nn.Tanh(),
                                  nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, x):  # x: every gene, in the data's order
        p = self.pw(x[:, self.col]) / self.root_deg  # GLUE: N99
        return self.head(torch.cat([self.fc1(x), self.pw_post(p)], 1))


X = pd.DataFrame(torch.rand(20, 14).numpy(), columns=[
    "MAPT", "TP53", "APOE", "AKT1", "RB1", "CCND1", "PTEN", "BIN1", "APP",
    "PIK3CA", "GSK3B", "CDK4", "PSEN1", "TREM2"])
col = kpnn2.align_inputs(X.columns, spec)  # pathway branch; dense reads all
x = torch.as_tensor(X.to_numpy())
model = PINNet(spec, col, n_genes=X.shape[1])
nn.functional.cross_entropy(model(x), torch.randint(0, 2, (20,))).backward()
model.eval()
bg = x[:6]  # paper: class-balanced training samples
genes = pd.Series(DeepLiftShap(model).attribute(x, baselines=bg, target=1)
                  .detach().abs().mean(0).numpy(), index=X.columns)
pws = kpnn2.map_node_attributions(LayerDeepLiftShap(model, model.pw).attribute(
    x, baselines=bg, target=1).detach(), spec, hop_output=hop)
print(spec.layer_nodes[1], model.root_deg.tolist())
print(genes.round(4).to_dict(), abs(pws).mean("observation").to_series()
      .round(4).to_dict())
