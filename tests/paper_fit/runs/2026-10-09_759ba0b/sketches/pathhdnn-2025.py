"""PathHDNN: Reactome by depth, copy chains, dense masked layers, SHAP."""

import networkx as nx
import pandas as pd
import torch
from captum.attr import LayerDeepLiftShap
from torch import nn

import kpnn2

torch.manual_seed(42)
L = 3  # paper: 4
rel = pd.DataFrame([p.split() for p in "R1 R11|R1 R12|R2 R21|R11 R111|"
                    "R111 R1111|R21 R211|R21 R212|R3 R31".split("|")],
                   columns=["parent", "child"])
memb = pd.DataFrame([p.split() for p in "g1_mut R111|g2_mut R1111|"
                     "g2_amp R1111|g3_mut R12|g4_mut R211|g4_amp R211|"
                     "g5_mut R11".split("|")], columns=["feature", "pathway"])
X = pd.DataFrame(torch.randint(0, 2, (12, 7)).float().numpy(), columns=[
    "g1_mut", "g2_mut", "g2_amp", "g3_mut", "g4_mut", "g5_mut", "g6_mut"])
G = nx.DiGraph(rel.to_numpy().tolist())  # GLUE
G = G.subgraph(set(memb.pathway).union(*(  # GLUE: data pathways and their
    nx.ancestors(G, p) for p in memb.pathway))).copy()  # GLUE: ancestors
G.add_edges_from([("root", n) for n, k in G.in_degree() if k == 0])  # GLUE
d = nx.single_source_shortest_path_length(G, "root", cutoff=L)  # GLUE
e = [(c, p) for p, c in G.edges if p != "root" and c in d  # GLUE: adjacent
     and d[c] == d[p] + 1]  # GLUE: layers only
for n in [n for n in d if n != "root" and not any(  # GLUE: bottom nodes
        d.get(c) == d[n] + 1 for c in G[n])]:  # GLUE: and short leaves
    chain = [n] + [f"{n}_copy{i}" for i in range(1, L - d[n] + 1)]  # GLUE
    d.update({c: d[n] + i for i, c in enumerate(chain)})  # GLUE
    e += list(zip(chain[1:], chain[:-1]))  # GLUE: copy feeds original
    below = memb.pathway.isin({n} | nx.descendants(G, n))  # GLUE: propagate
    e += [(f, chain[-1]) for f in memb.feature[below].unique()]
e = pd.DataFrame(e, columns=["source", "target"])
spec = kpnn2.parse_layered(e, ranks={n: L + 1 - d[n] if n in d else 0  # GLUE
                                     for n in {*e.source, *e.target}})  # GLUE
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
