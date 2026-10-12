"""MOViDA: GO terms of 6 units, three gene tables, an enrichment input per term."""

import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)
K = 6
tree = pd.DataFrame([r.split() for r in (
    "root Ta|root Tb|Ta Ta1|Ta Ta2|Ta Tb1|Tb Tb1").split("|")],
    columns=["parent", "child"])
# Own gene list and term-gene table per data type, into the same terms.
ont = {"mut": "Ta1 g1|Ta1 g2|Ta2 g3|Tb1 g4|Ta g5",
       "amp": "Ta1 g2|Tb1 g6|Tb g7", "del": "Ta2 g1|Tb1 g8|root g9"}
rows = [(f"{g}_{d}", t) for d, s in ont.items()
        for t, g in (p.split() for p in s.split("|"))]
# One enrichment score per term enters the term and the term's parent.
terms = sorted(set(tree.parent) | set(tree.child))
es = [(f"{t}_es", t) for t in terms] + [
    (f"{c}_es", p) for p, c in tree.itertuples(index=False)]
e = pd.DataFrame(rows + es + [(c, p) for p, c in tree.itertuples(
    index=False)], columns=["source", "target"])
spec = kpnn2.parse_layered(e, widths={t: K for t in terms})


class MOViDA(nn.Module):
    def __init__(self, spec, n_drug):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint) for h in spec.hops)
        self.norms = nn.ModuleList(nn.BatchNorm1d(h.out_features)
                                   for h in spec.hops)
        dims = [n_drug, 100, 50, K]
        self.drug = nn.Sequential(*[m for a, b in zip(dims, dims[1:]) for m in
                                    (nn.Linear(a, b), nn.Tanh(),
                                     nn.BatchNorm1d(b))])
        self.head = nn.Sequential(nn.Linear(2 * K, K), nn.Tanh(),
                                  nn.BatchNorm1d(K), nn.Linear(K, 1),
                                  nn.LeakyReLU(0.1))

    def forward(self, x, drug):
        saved = {0: x}
        for i, hop in enumerate(self.spec.hops):
            z = self.hops[i](kpnn2.gather_hop_inputs(saved, hop))
            saved[hop.target_layer] = self.norms[i](torch.tanh(z))
        root = saved[len(self.spec.hops)]  # the one root is the top layer
        return self.head(torch.cat([root, self.drug(drug)], 1))


mut = pd.DataFrame(torch.randint(0, 2, (12, 5)).float().numpy(),
                   columns=[f"g{i}_mut" for i in [5, 4, 3, 2, 1]])
cnv = pd.DataFrame(torch.randint(0, 2, (12, 6)).float().numpy(), columns=[
    "g2_amp", "g6_amp", "g7_amp", "g1_del", "g8_del", "g9_del"])
enrich = pd.DataFrame(torch.randn(12, 6).numpy(),
                      columns=[f"{t}_es" for t in terms[::-1]])
X = pd.concat([mut, cnv, enrich], axis=1)
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
model = MOViDA(spec, n_drug=32)
loss = nn.functional.mse_loss(model(x, torch.randn(12, 32)), torch.rand(12, 1))
loss.backward()
print(spec.layer_nodes[1:], spec.output_nodes, len(spec.skips))
print(float(loss.detach()), [h.source_layers for h in spec.hops])
