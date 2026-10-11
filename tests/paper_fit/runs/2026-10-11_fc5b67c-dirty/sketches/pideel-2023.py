"""PiDeeL: metabolites -> KEGG pathways as a masked dense layer, dense head."""

import pandas as pd
import torch
from captum.attr import DeepLiftShap, LayerDeepLiftShap
from torch import nn

import kpnn2

torch.manual_seed(42)
# Final metabolite -> pathway membership (built from KEGG upstream).
member = {"Glutamate": "Ala_Asp_Glu D_Gln Arg_Pro", "Glycine": "Gly_Ser_Thr",
          "Serine": "Gly_Ser_Thr Cys_Met", "Aspartate": "Ala_Asp_Glu Arg_Pro",
          "Lactate": "Pyruvate", "Alanine": "Ala_Asp_Glu Pyruvate"}
e = pd.DataFrame([(m, p) for m, ps in member.items() for p in ps.split()],
                 columns=["source", "target"])
spec = kpnn2.parse_layered(e)
hop = spec.hops[0]


class PiDeeL(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fc1 = kpnn2.MaskedLinear(hop.to_mask(), identity=spec.fingerprint)
        self.act1 = nn.ReLU()
        self.head = nn.Sequential(nn.Linear(hop.out_features, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.act1(self.fc1(x)))


def cox(risk, time, event):
    r = risk.squeeze(1)[time.argsort(descending=True)]
    e = event[time.argsort(descending=True)]
    return -((r - r.logcumsumexp(0)) * e).sum() / e.sum()


X = pd.DataFrame(torch.rand(24, 6).numpy(), columns=[
    "Lactate", "Glycine", "Alanine", "Serine", "Glutamate", "Aspartate"])
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
time, event = torch.rand(24), torch.randint(0, 2, (24,)).float()
model = PiDeeL(spec)
cox(model(x), time, event).backward()
# Control: rows of the prior shuffled = metabolites permuted onto the rows.
xs = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(
    X.columns.to_series().sample(frac=1, random_state=0), spec)])
model.eval()
bg = x[:8]
mean = lambda a, s, **kw: kpnn2.map_node_attributions(  # noqa: E731
    a.detach(), s, **kw).mean("observation")
met = mean(DeepLiftShap(model).attribute(x, baselines=bg), spec, axis="inputs")
pw = mean(LayerDeepLiftShap(model, model.act1).attribute(x, baselines=bg),
          spec, hop_output=hop)
flow = spec.to_edgelist().merge(met.to_series().rename("m"),  # GLUE: N97
                                left_on="source", right_index=True)  # GLUE
flow["value"] = flow.m / flow.groupby("target").m.transform(  # GLUE: N97
    "sum") * flow.target.map(pw.to_series())  # GLUE: split pathway score
print(spec.layer_nodes[1], xs.shape)
print(flow.round(4))
