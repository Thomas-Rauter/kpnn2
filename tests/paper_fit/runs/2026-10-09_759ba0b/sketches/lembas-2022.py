"""LEMBAS: signed cyclic network, one recurrent unit per node, knockouts."""

import networkx as nx
import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)
net = pd.DataFrame([r.split() for r in (
    "L1 R1 1|L2 R1 1|L2 R2 1|L3 R2 1|R1 K1 1|R2 K2 1|K1 K3 1|K2 K3 -1|"
    "K3 K1 -1|K3 T1 1|K2 T2 0|T1 K2 1|K3 T3 -1|K9 K1 1|K1 K5 1"
).split("|")], columns=["source", "target", "sign"]).astype({"sign": int})
LIG, TF = ["L1", "L2", "L3"], ["T1", "T2", "T3"]
G = nx.DiGraph(net[["source", "target"]].to_numpy().tolist())  # GLUE
ok = set().union(*({n} | nx.descendants(G, n) for n in LIG)) & set(  # GLUE
    ).union(*({n} | nx.ancestors(G, n) for n in TF))  # GLUE: reachability
net = net[net.source.isin(ok) & net.target.isin(ok)]  # GLUE
spec = kpnn2.parse_adjacency(net)
n, nnz = spec.state_dim, len(spec.source_index)
sign = torch.zeros(nnz)
for s, t, sg in net.itertuples(index=False):
    sign[list(spec.edge_location(s, t))] = sg
dt = kpnn2.parse_layered(pd.DataFrame({  # viability variant: drug targets
    "source": ["D1", "D1", "D2"], "target": ["K1", "K3", "K2"]}))


class Lembas(nn.Module):
    def __init__(self, spec, steps=150, tol=1e-6):
        super().__init__()
        self.core = kpnn2.PackedLinear(spec.source_index, spec.target_index,
                                       n, n, identity=spec.fingerprint)
        self.register_buffer("inp", torch.as_tensor(spec.input_index))
        self.register_buffer("out", torch.as_tensor(  # GLUE: TFs by name,
            [spec.node_units(t).start for t in TF]))  # GLUE: sinks or not
        self.out_scale = nn.Parameter(torch.full((len(TF),), 1.2))
        hop = dt.hops[0]
        self.drug = kpnn2.PackedLinear(hop.source_index, hop.target_index,
                                       hop.out_features, hop.in_features,
                                       bias=False, identity=dt.fingerprint)
        self.register_buffer("tgt", torch.as_tensor(  # GLUE: drug targets
            [spec.node_units(t).start for t in dt.layer_nodes[1]]))  # GLUE
        self.steps, self.tol = steps, tol

    def act(self, z):  # Michaelis-Menten-like
        return torch.where(z < 0, 0.01 * z, torch.where(
            z <= 0.5, z, 1 - 0.25 / z.clamp(min=0.5)))

    def forward(self, x, d, offset=0.0):
        u = x.new_zeros(x.shape[0], n).index_copy(-1, self.inp, 3 * x) + offset
        u = u.index_add(-1, self.tgt, self.drug(d))
        h = torch.zeros_like(u)
        for _ in range(self.steps):  # unrolled; LEMBAS uses an adjoint
            h, prev = self.act(self.core(h) + u), h
            if (h - prev).abs().sum() < self.tol:
                break
        return h[:, self.out] * self.out_scale, h


model = Lembas(spec)
with torch.no_grad():
    model.core.weight.copy_((0.1 + 0.1 * torch.rand(nnz)) * torch.where(
        sign < 0, -1.0, 1.0))
    A = torch.zeros(n, n).index_put(  # GLUE: dense matrix for the
        (torch.as_tensor(spec.target_index), torch.as_tensor(  # GLUE:
            spec.source_index)), model.core.weight)  # GLUE: spectral radius
    model.core.weight.mul_(0.8 / torch.linalg.eigvals(A).abs().max())  # GLUE
X = pd.DataFrame(torch.rand(6, 3).numpy(), columns=["L1", "L2", "L9"])
X = X.reindex(columns=X.columns.union(spec.input_nodes), fill_value=0)  # GLUE
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)],
                    dtype=torch.float32)
D = pd.DataFrame(torch.rand(6, 2).numpy(), columns=["D2", "D1"])
d = torch.as_tensor(D.to_numpy()[:, kpnn2.align_inputs(D.columns, dt)])
pred, h = model(x, d)
w = model.core.effective_weight()
loss = nn.functional.mse_loss(pred, torch.rand(6, 3)) + 1e-2 * torch.relu(
    -sign * w).sum()
loss.backward()
with torch.no_grad():
    base, _ = model(x[:1], d[:1])
    ko, _ = model(x[:1].repeat(n, 1), d[:1].repeat(n, 1),
                  offset=-3 * torch.eye(n))
effect = kpnn2.map_node_attributions(ko - base, spec, dims=("node", "tf"),
                                     coords={"tf": TF})
params = spec.to_edgelist().assign(weight=w.detach().numpy())
print(spec.nodes, spec.input_nodes, spec.output_nodes)
print(effect.to_pandas().round(3), params.head(4))
