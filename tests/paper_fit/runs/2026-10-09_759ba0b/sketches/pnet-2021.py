"""P-NET: Reactome by depth from a root, copy chains, a head per layer."""

import networkx as nx
import pandas as pd
import torch
from captum.attr import DeepLift, LayerDeepLift
from torch import nn

import kpnn2

torch.manual_seed(42)
L = 3  # pathway levels kept (paper: 5)
rel = pd.DataFrame([("P1", "P11"), ("P1", "P112"), ("P11", "P112"),
                    ("P11", "P111"), ("P2", "P21")], columns=["parent", "child"])
gmt = {"P111": ["g1", "g2"], "P21": ["g3", "g4"], "P112": ["g2", "g4"]}
genes, types = ["g1", "g2", "g3", "g4", "g5"], ["mut", "amp", "del"]
G = nx.DiGraph(rel.to_numpy().tolist())  # GLUE: add a root, depth from it
G.add_edges_from([("root", n) for n, k in G.in_degree() if k == 0])  # GLUE
d = nx.single_source_shortest_path_length(G, "root", cutoff=L)  # GLUE
e = [(c, p) for p, c in G.edges if p != "root" and c in d  # GLUE: adjacent
     and d[c] == d[p] + 1]  # GLUE: layers only
for n in [n for n in d if n != "root" and not any(  # GLUE: leaves above
        d.get(c) == d[n] + 1 for c in G[n])]:  # GLUE: the cut get copies
    chain = [n] + [f"{n}_copy{i}" for i in range(1, L - d[n] + 1)]  # GLUE
    d.update({c: d[n] + i for i, c in enumerate(chain)})  # GLUE
    e += list(zip(chain[1:], chain[:-1]))  # GLUE: copy feeds original
    e += [(g, chain[-1]) for g in gmt[n]]  # gene sets join the deepest
e += [(f"{g}_{t}", g) for g in genes for t in types]  # GLUE: input per type
e = pd.DataFrame(e, columns=["source", "target"])
nodes = set(e.source) | set(e.target)  # GLUE
spec = kpnn2.parse_layered(e, ranks={  # GLUE: rank from depth from root
    n: 0 if n not in d and n not in genes else 1 if n in genes  # GLUE
    else L + 2 - d[n] for n in nodes})  # GLUE


class PNet(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint) for h in spec.hops)
        self.acts = nn.ModuleList(nn.Tanh() for _ in spec.hops)
        self.drops = nn.ModuleList(nn.Dropout(0.5 if i == 0 else 0.1)
                                   for i in range(len(spec.hops)))
        self.heads = nn.ModuleList(nn.Sequential(
            nn.Linear(h.out_features, 1), nn.Sigmoid()) for h in spec.hops)

    def forward(self, x):
        saved, outs = {0: x}, []
        for i, hop in enumerate(self.spec.hops):
            z = self.hops[i](kpnn2.gather_hop_inputs(saved, hop))
            h = self.acts[i](z)
            outs.append(self.heads[i](h))
            saved[hop.target_layer] = self.drops[i](h)
        return torch.cat(outs, 1)


cols = [f"{g}_{t}" for t in types for g in genes]  # data in its own order
X = pd.DataFrame(torch.randint(0, 2, (12, 15)).float().numpy(), columns=cols)
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
model, y = PNet(spec), torch.randint(0, 2, (12, 1)).float()
out = model(x)
loss = sum(w * nn.functional.binary_cross_entropy(out[:, [i]], y)
           for i, w in enumerate([2, 7, 20, 54]))
loss.backward()
model.eval()
top = len(spec.hops) - 1  # explain the head of the top pathway layer
inp = kpnn2.map_node_attributions(DeepLift(model).attribute(
    x, target=top).detach(), spec, axis="inputs").sum("observation")  # GLUE
hid = [kpnn2.map_node_attributions(LayerDeepLift(model, model.acts[i])
       .attribute(x, target=top).detach(), spec, hop_output=h)
       .sum("observation") for i, h in enumerate(spec.hops)]  # GLUE: fold
deg = pd.concat([e.source, e.target]).value_counts()  # GLUE: degree adjust
adj = [abs(s).where(abs(s) <= abs(s).mean() + 5 * abs(s).std(),  # GLUE
       abs(s) / deg[s.node.values].to_numpy()) for s in hid]  # GLUE
sankey = spec.to_edgelist().merge(  # GLUE: scores joined to edges
    adj[1].to_series().rename("score"), left_on="source",  # GLUE
    right_index=True)  # GLUE
print(spec.layer_nodes, out.mean(1)[:3].detach())
print(inp.to_series().round(3).head(), sankey)
