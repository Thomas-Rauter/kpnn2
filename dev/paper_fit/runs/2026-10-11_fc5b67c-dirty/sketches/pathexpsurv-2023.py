"""PathExpSurv: prior gene -> pathway edges, then every pair with L1 on the rest."""

import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)
genes = [f"g{i}" for i in range(1, 11)]
prior = pd.DataFrame([(g, p) for p, gs in {
    "P1": "g1 g2 g3", "P2": "g3 g4 g5", "P3": "g6 g7"}.items()
    for g in gs.split()], columns=["source", "target"])
# Every gene x pathway pair is a candidate edge (writing the rows is
# upstream); the prior is a set of slots, found by name.
full = pd.MultiIndex.from_product([genes, ["P1", "P2", "P3"]]).to_frame(
    index=False, name=["source", "target"])
spec = kpnn2.parse_layered(full)
hop = spec.hops[0]
is_prior = torch.zeros(len(hop.source_index), dtype=torch.bool)
for s, t in prior.itertuples(index=False):
    is_prior[list(spec.edge_location(s, t)[1])] = True


class Phase(nn.Module):  # prior slots always on; the others times beta
    def __init__(self):
        super().__init__()
        self.register_buffer("is_prior", is_prior)
        self.register_buffer("beta", torch.tensor(0.0))

    def forward(self, w):
        return w * torch.where(self.is_prior, 1.0, self.beta)


class PathExpSurv(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(hop.in_features)
        self.sc = kpnn2.PackedLinear(
            hop.source_index, hop.target_index, hop.out_features,
            hop.in_features, bias=False, identity=spec.fingerprint,
            constraint=Phase())
        self.bn = nn.BatchNorm1d(hop.out_features)
        self.out = nn.Linear(hop.out_features + 1, 1, bias=False)

    def forward(self, x, event):
        p = torch.tanh(self.bn(self.sc(self.bn_in(x))))
        return torch.tanh(self.out(torch.cat([p, event[:, None]], 1)))


def cox(risk, time, event):
    r = risk.squeeze(1)[time.argsort(descending=True)]
    e = event[time.argsort(descending=True)]
    return -((r - r.logcumsumexp(0)) * e).sum() / e.sum()


X = pd.DataFrame(torch.rand(30, 10).numpy(), columns=genes[::-1])
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
time, event = torch.rand(30), torch.randint(0, 2, (30,)).float()
selected = []
for seed in range(3):  # paper: 100 runs on 90% subsets
    torch.manual_seed(42 + seed)
    model = PathExpSurv(spec)
    with torch.no_grad():
        model.sc.weight.uniform_(0, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for epoch in range(4):
        if epoch == 2:  # phase 2: other pairs on, starting at zero
            with torch.no_grad():
                model.sc.weight[~is_prior] = 0
                model.sc.constraint.beta.fill_(1.0)
        w = model.sc.effective_weight()
        loss = cox(model(x, event), time, event) + (
            0.1 * w[is_prior].std() if epoch < 2 else
            0.001 * w[~is_prior].abs().sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.sc.weight.clamp_(min=0)
    selected.append(model.sc.effective_weight().detach() > 0)
el = spec.to_edgelist().assign(  # GLUE: N94 selection count per named edge
    count=torch.stack(selected).sum(0).numpy())  # GLUE
cut = el["count"].nlargest(int(len(prior) * 1.2)).min()  # GLUE
expanded = el[el["count"] >= cut]  # GLUE: the new prior, as an edgelist
print(spec.layer_nodes[1], int(is_prior.sum()), len(el))
print(expanded.merge(prior, how="left", indicator=True).round(3))
