"""NeST-VNN: gene units over 3 flags, assemblies of k units, aux heads, RLIPP."""

import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from torch import nn

import kpnn2

torch.manual_seed(42)
K, FLAGS = 4, ["mut", "del", "amp"]
# Ontology table as shipped (parent, child, type); the 395 -> 131 filter is
# upstream. Each gene's three flags feed only that gene (one row each).
ont = pd.DataFrame([r.split() for r in (
    "NEST A default|NEST B default|A A1 default|A A2 default|B A2 default|"
    "A1 g1 gene|A1 g2 gene|A2 g3 gene|A2 g4 gene|B g5 gene|A g2 gene|"
    "NEST g6 gene").split("|")], columns=["parent", "child", "type"])
genes = sorted(ont.child[ont.type == "gene"].unique())
e = pd.concat([ont.rename(columns={"child": "source", "parent": "target"}),
               pd.DataFrame([(f"{g}_{f}", g) for g in genes for f in FLAGS],
                            columns=["source", "target"])])
assemblies = set(ont.parent)
spec = kpnn2.parse_layered(e, widths={a: K for a in assemblies})


class NestVNN(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint) for h in spec.hops)
        self.norms = nn.ModuleList(nn.BatchNorm1d(h.out_features)
                                   for h in spec.hops)
        self.drop = nn.Dropout(0.3)  # assembly inputs, paper layer 2 and up
        self.aux = nn.ModuleDict({a: nn.Sequential(
            nn.Linear(K, 1), nn.Tanh(), nn.Linear(1, 1)) for a in assemblies})
        self.final = nn.Sequential(nn.Linear(K, 1), nn.Tanh(), nn.Linear(1, 1))

    def forward(self, x):
        saved = {0: x}
        for i, hop in enumerate(self.spec.hops):
            src = kpnn2.gather_hop_inputs(saved, hop)
            src = self.drop(src) if hop.target_layer >= 4 else src
            saved[hop.target_layer] = self.norms[i](torch.tanh(
                self.hops[i](src)))
        aux = {a: self.aux[a](saved[l][:, u]) for a in self.aux
               for l, u in [self.spec.node_units(a)]}
        return self.final(saved[len(self.spec.hops)]), aux, saved


X = pd.DataFrame(torch.randint(0, 2, (40, 18)).float().numpy(), columns=[
    f"{g}_{f}" for f in FLAGS for g in genes])
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)],
                    ).requires_grad_()
model, y = NestVNN(spec), torch.rand(40, 1)
out, aux, saved = model(x)
loss = nn.functional.mse_loss(out, y) + 0.3 * sum(
    nn.functional.mse_loss(a, y) for a in aux.values())
loss.backward()
model.eval()
out, _, saved = model(x)
layers = sorted(saved)
grads = torch.autograd.grad(out.sum(), [saved[l] for l in layers])
named = [kpnn2.map_node_attributions(g.detach(), spec, layer=l)
         for l, g in zip(layers, grads)]  # plain gradients, inputs included


def probe(z, p):  # GLUE: N76 PCA + ridge fit to the prediction, Spearman
    z = PCA(min(K, *z.shape)).fit_transform(z)  # GLUE
    return spearmanr(RidgeCV().fit(z, p).predict(z), p)[0]  # GLUE


p, el, rlipp = out.detach().numpy().ravel(), spec.to_edgelist(), {}
for a in sorted(assemblies):
    l, u = spec.node_units(a)
    hop = spec.hops[l - 1]
    src = kpnn2.gather_hop_inputs(saved, hop).detach()
    kids = torch.cat([src[:, spec.hop_units(hop, c)]  # each child's units
                      for c in el.source[el.target == a]], 1)
    rlipp[a] = probe(saved[l][:, u].detach(), p) / probe(kids, p)  # GLUE
print(spec.layer_nodes[2:], named[2].sizes)
print({a: round(float(r), 3) for a, r in rlipp.items()})
