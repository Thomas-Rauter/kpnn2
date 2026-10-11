"""G2B2P: SNPs -> one private MLP per gene (20 -> 128 -> 1) -> dense head."""

import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)
H = 8  # hidden units of each gene's own network (paper: 128)
# Final SNP -> gene table of one outer split. Selecting the genes (elastic
# net) and their SNPs (eQTL top 20) from the training data is upstream.
eqtl = {"Zm01": ["s1", "s2", "s3"], "Zm02": ["s3", "s4", "s5"],
        "Zm03": ["s6", "s7", "s8"], "Zm04": ["s2", "s8", "s9"]}
rows = [(s, f"{g}_h") for g, ss in eqtl.items() for s in ss] + [
    (f"{g}_h", g) for g in eqtl]  # gene network: own hidden node, then latent
spec = kpnn2.parse_layered(pd.DataFrame(rows, columns=["source", "target"]),
                           widths={f"{g}_h": H for g in eqtl})


class G2B2P(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(kpnn2.PackedLinear(
            h.source_index, h.target_index, h.out_features, h.in_features,
            identity=spec.fingerprint) for h in spec.hops)
        self.act = nn.Sigmoid()
        self.final = nn.Sequential(nn.Linear(spec.layer_dims[2], 16),
                                   nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, latent=None):
        saved = {0: x}
        for i, hop in enumerate(self.spec.hops):
            z = self.hops[i](kpnn2.gather_hop_inputs(saved, hop))
            saved[hop.target_layer] = self.act(z) if i == 0 else z
        lat = saved[2] if latent is None else latent
        return self.final(lat), saved[2]


X = pd.DataFrame(torch.randint(0, 3, (32, 10)).float().numpy(),
                 columns=[f"s{i}" for i in [9, 1, 2, 3, 4, 5, 6, 7, 8, 10]])
expr = pd.DataFrame(torch.randn(32, 5).numpy(),  # measured, own column order
                    columns=["Zm04", "Zm02", "Zm05", "Zm01", "Zm03"])
x = torch.as_tensor(X.to_numpy()[:, kpnn2.align_inputs(X.columns, spec)])
e = torch.as_tensor(expr[list(spec.layer_nodes[2])].to_numpy())  # GLUE: N55
model, y = G2B2P(spec), torch.randn(32, 1)
pred, lat = model(x)
zc = lambda t: (t - t.mean(0)) / t.std(0)  # noqa: E731
loss = nn.functional.mse_loss(pred, y) + 0.1 * (  # soft loss: 1 - Pearson
    1 - (zc(lat) * zc(e)).mean(0)).mean()
loss.backward()
# Inference: spec saved with the weights; inputs aligned by name, not width.
blob = {"spec": spec.to_dict(), "state": model.state_dict()}
spec2 = kpnn2.LayeredSpec.from_dict(blob["spec"])
model2 = G2B2P(spec2)
model2.load_state_dict(blob["state"])
Xt = X.sample(frac=1, axis=1, random_state=0)  # test columns, other order
xt = torch.as_tensor(Xt.to_numpy()[:, kpnn2.align_inputs(Xt.columns, spec2)])
# Paper only: clamp one gene latent to mean +- a*sd, score the mean change.
with torch.no_grad():
    y0, lat = model2(xt)
    dy = []
    for j in range(lat.shape[1]):
        ch = [(model2(xt, lat.index_fill(1, torch.tensor([j]), float(
            lat[:, j].mean() + s * 1.0 * lat[:, j].std())))[0].mean()
            - y0.mean()).abs() for s in (1, -1)]
        dy.append(sum(ch) / 2)
score = kpnn2.map_node_attributions(torch.stack(dy), spec2, layer=2)
print(spec.layer_nodes[1:], spec.layer_dims, float(loss.detach()))
print(score.to_series().round(4).to_dict())
