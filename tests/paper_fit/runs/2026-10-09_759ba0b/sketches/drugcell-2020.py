"""DrugCell: GO terms of k=6 units, genes at any height, aux heads."""

import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)
K = 6
# Ontology table as shipped: parent, child term or gene, type.
ont = pd.DataFrame(
    [("root", "Ta", "default"), ("root", "Tb", "default"),
     ("Ta", "Ta1", "default"), ("Ta", "Ta2", "default"),
     ("Ta", "Tb1", "default"), ("Tb", "Tb1", "default"),
     ("Ta1", "g01", "gene"), ("Ta1", "g02", "gene"), ("Ta2", "g03", "gene"),
     ("Ta2", "g04", "gene"), ("Tb1", "g05", "gene"), ("Tb1", "g06", "gene"),
     ("Ta1", "g10", "gene"), ("Tb1", "g10", "gene"), ("Ta", "g07", "gene"),
     ("Tb", "g08", "gene"), ("root", "g09", "gene")],
    columns=["parent", "child", "type"],
)
edges = ont.rename(columns={"child": "source", "parent": "target"})
terms = set(edges.target)
spec = kpnn2.parse_layered(edges, widths={t: K for t in terms})
if len(spec.output_nodes) != 1:  # GLUE: DrugCell needs one root
    raise ValueError(spec.output_nodes)  # GLUE


class DrugCell(nn.Module):
    def __init__(self, spec, n_drug):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint) for h in spec.hops)
        self.norms = nn.ModuleList(nn.BatchNorm1d(h.out_features)
                                   for h in spec.hops)
        self.aux = nn.ModuleDict({t: nn.Sequential(
            nn.Linear(K, 1), nn.Tanh(), nn.Linear(1, 1)) for t in terms})
        dims = [n_drug, 100, 50, K]
        self.drug = nn.Sequential(*[m for a, b in zip(dims, dims[1:]) for m in
                                    (nn.Linear(a, b), nn.Tanh(),
                                     nn.BatchNorm1d(b))])
        self.head = nn.Sequential(nn.Linear(2 * K, K), nn.Tanh(),
                                  nn.BatchNorm1d(K), nn.Linear(K, 1),
                                  nn.Tanh(), nn.Linear(1, 1))

    def forward(self, genes, drug):
        saved = {0: genes}
        for i, hop in enumerate(self.spec.hops):
            z = self.hops[i](kpnn2.gather_hop_inputs(saved, hop))
            saved[hop.target_layer] = self.norms[i](torch.tanh(z))
        aux = {t: self.aux[t](saved[l][:, u]) for t in self.aux
               for l, u in [self.spec.node_units(t)]}
        top, units = self.spec.node_units(self.spec.output_nodes[0])
        root = saved[top][:, units]
        return self.head(torch.cat([root, self.drug(drug)], 1)), aux, saved


genes = pd.DataFrame(torch.randint(0, 2, (8, 11)).float().numpy(),
                     columns=[f"g{i:02d}" for i in range(1, 12)])
x = torch.as_tensor(genes.to_numpy()[:, kpnn2.align_inputs(genes.columns,
                                                            spec)])
model = DrugCell(spec, n_drug=64)
y = torch.rand(8, 1)
out, aux, saved = model(x, torch.randint(0, 2, (8, 64)).float())
loss = nn.functional.mse_loss(out, y) + 0.2 * sum(
    nn.functional.mse_loss(a, y) for a in aux.values())
loss.backward()
acts = [kpnn2.map_node_attributions(saved[h.target_layer].detach(), spec,
                                    hop_output=h) for h in spec.hops]
print(spec.layer_nodes, [h.source_layers for h in spec.hops])
print(float(loss.detach()), acts[-1].sizes, list(acts[-1].node.values[:7]))
