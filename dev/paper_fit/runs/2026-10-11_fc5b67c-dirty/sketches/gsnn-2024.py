"""GSNN: state on edges, a private network per node, unrolled residual steps."""

import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)
C, STEPS = 4, 3  # hidden units per function node; unrolled steps (paper 10-20)
# Final typed graph (OmniPath restriction, drug and output pruning upstream).
g = pd.DataFrame([p.split() for p in (
    "D1 P1|D2 P2|E1 R1|P1 P2|P2 P3|P3 P1|P3 R1|P2 R2|R1 P3|R1 L1|R2 L2"
).split("|")], columns=["source", "target"])
inp, out = {"D1", "D2", "E1"}, {"L1", "L2"}
fn = sorted((set(g.source) | set(g.target)) - inp - out)
# Edge-state graph, also upstream: each function -> function edge u->v is a
# node of its own; v's hidden block reads it, u's hidden block writes it.
# Input edges collapse into input -> hidden; an output edge is named after
# its output node (each output node has one in-edge).
rows = []
for u, v in g.itertuples(index=False):
    if u in inp:
        rows.append((u, f"{v}_h"))
    elif v in out:
        rows.append((f"{u}_h", v))
    else:
        rows += [(f"{u}_h", f"{u}->{v}"), (f"{u}->{v}", f"{v}_h")]
spec = kpnn2.parse_adjacency(pd.DataFrame(rows, columns=["source", "target"]),
                             widths={f"{f}_h": C for f in fn})
n = spec.state_dim
hid = torch.cat([torch.arange(n)[spec.node_units(f"{f}_h")]  # GLUE: hidden
                 for f in fn])  # GLUE: units, node-major
is_hid = torch.zeros(n, dtype=torch.bool).index_fill(0, hid, True)  # GLUE


class GSNN(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.cores = nn.ModuleList(kpnn2.PackedLinear(
            spec.source_index, spec.target_index, n, n,
            identity=spec.fingerprint) for _ in range(STEPS))
        with torch.no_grad():  # He-normal from each unit's present fan-in
            for c in self.cores:
                c.weight.normal_().mul_(2 ** 0.5 * c.init_bound())
        self.register_buffer("inp", torch.as_tensor(spec.input_index))
        self.norm = nn.LayerNorm(C, elementwise_affine=False)
        self.act, self.drop = nn.ELU(), nn.Dropout(0.1)

    def forward(self, x):
        s = x.new_zeros(x.shape[0], n).index_copy(-1, self.inp, x)
        states = []
        for core in self.cores:  # W1 and W3 slots share one PackedLinear
            z = core(s.masked_fill(is_hid, 0))[:, hid]  # GLUE: edges -> hidden
            z = self.norm(z.view(-1, len(fn), C)).flatten(1)  # GLUE: per node
            hu = s.new_zeros(s.shape).index_copy(-1, hid, self.drop(  # GLUE
                self.act(z)))  # GLUE
            s = s + core(hu).masked_fill(is_hid, 0)  # GLUE: hidden -> edges
            states.append(s)
        return s[:, spec.output_index] / STEPS, states


X = pd.DataFrame(torch.rand(6, 4).numpy(), columns=["E1", "D2", "D1", "M9"])
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
model = GSNN(spec)
pred, states = model(x)
nn.functional.mse_loss(pred, torch.rand(6, 2)).backward()
edge_states = kpnn2.map_node_attributions(  # every step, named by edge
    [s.detach() for s in states], spec).sel(node=~is_hid.numpy())
print(spec.output_nodes, pred.shape, n, len(spec.source_index))
print(edge_states.isel(step=-1, observation=0).to_series().round(3).to_dict())
