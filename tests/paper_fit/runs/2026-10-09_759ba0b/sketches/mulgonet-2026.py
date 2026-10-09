"""MULGONET: BP and MF branches by depth from a root, one shared input."""

import networkx as nx
import pandas as pd
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from torch import nn

import kpnn2

torch.manual_seed(42)
L, MAX_GENES, OMICS = 3, 3, ["meth", "amp", "del", "exp"]  # paper: 5, 200
go = pd.DataFrame([t.split() for t in (
    "B1 B0 bp|B2 B0 bp|B11 B1 bp|B12 B1 bp|B21 B2 bp|B111 B11 bp|B111 B12 bp|"
    "B112 B11 bp|B121 B12 bp|B211 B21 bp|B1111 B111 bp|M1 M0 mf|M11 M1 mf|"
    "M12 M1 mf|M111 M11 mf|M121 M12 mf").split("|")],
    columns=["term", "parent", "ns"])
ann = pd.DataFrame([t.split() for t in "B111 g1|B111 g2|B121 g2|B121 g3|"
                    "B112 g1|B112 g2|B112 g3|B112 g4|M111 g1|M111 g4|M121 g3"
                    .split("|")], columns=["term", "gene"])
ann = ann[ann.groupby("term").gene.transform("size") <= MAX_GENES]  # GLUE
X = pd.DataFrame(torch.rand(10, 20).numpy(), columns=[
    f"{g}_{o}" for o in OMICS for g in ["g1", "g2", "g3", "g4", "g5"]])


def branch(ns):
    g = nx.DiGraph(go[go.ns == ns][["parent", "term"]].to_numpy().tolist())  # GLUE
    root = [n for n, k in g.in_degree() if k == 0][0]  # GLUE: roots[0]
    d = nx.single_source_shortest_path_length(g, root, cutoff=L)  # GLUE
    e = [(c, p) for p, c in g.edges if p != root and c in d  # GLUE: adjacent
         and d[c] == d[p] + 1]  # GLUE: layers only
    a = ann[ann.term.map(d) == L].merge(pd.Series(OMICS, name="o"),  # GLUE
                                        how="cross")  # GLUE: one per omics
    e = pd.DataFrame(e + list(zip(a.gene + "_" + a.o, a.term)),  # GLUE
                     columns=["source", "target"])
    while not (ok := e.source.isin(X.columns)  # GLUE
               | e.source.isin(e.target)).all():  # GLUE
        e = e[ok]  # GLUE: drop terms with no child below, until stable
    return kpnn2.parse_layered(e, ranks={n: 0 if n in X.columns else  # GLUE
        L + 1 - d[n] for n in {*e.source, *e.target}})  # GLUE


class Branch(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint) for h in spec.hops)
        self.acts = nn.ModuleList(nn.Tanh() for _ in spec.hops)
        self.drops = nn.ModuleList(nn.Dropout(0.5 if i == 0 else 0.1)
                                   for i in range(len(spec.hops) - 1))

    def forward(self, x):
        saved = {0: x}
        for i, hop in enumerate(self.spec.hops):
            h = self.acts[i](self.hops[i](kpnn2.gather_hop_inputs(saved, hop)))
            saved[hop.target_layer] = self.drops[i](h) if i < len(
                self.drops) else h
        return h


specs = {ns: branch(ns) for ns in ("bp", "mf")}
model = nn.ModuleDict({ns: Branch(s) for ns, s in specs.items()})
head = nn.Sequential(nn.Linear(sum(s.layer_dims[-1] for s in specs.values()),
                               16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid())
net = lambda bp, mf: head(torch.cat([model["bp"](bp), model["mf"](mf)], 1))  # noqa: E731
xs = tuple(torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, s)])
           for s in specs.values())
y = torch.randint(0, 2, (10, 1)).float()
loss = nn.functional.binary_cross_entropy(net(*xs), y) + 1e-4 * sum(
    b.hops[0].effective_weight().pow(2).sum() for b in model.values())
loss.backward()
model.eval()
pos = (y[:, 0] == 1).numpy()  # recurrence-positive samples
attr = IntegratedGradients(net).attribute(xs, n_steps=20)
inp = pd.concat([kpnn2.map_node_attributions(a.detach(), s, axis="inputs")
                 [pos].sum("observation").to_series() for a, s in zip(  # GLUE
                     attr, specs.values())]).groupby(level=0).sum()  # GLUE
hid = {(ns, h.target_layer): kpnn2.map_node_attributions(
    LayerIntegratedGradients(net, model[ns].acts[i]).attribute(
        xs, n_steps=20).detach(), s, hop_output=h)[pos].sum("observation")  # GLUE
    for ns, s in specs.items() for i, h in enumerate(s.hops)}
print({ns: s.layer_nodes[1:] for ns, s in specs.items()})
print(inp.round(4).head(6), hid["bp", 3].to_series().round(4))
