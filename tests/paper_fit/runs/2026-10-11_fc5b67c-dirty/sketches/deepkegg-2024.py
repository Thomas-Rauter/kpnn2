"""DeepKEGG: one feature -> pathway layer per omics, attention over the batch."""

import pandas as pd
import torch
from captum.attr import InputXGradient
from torch import nn

import kpnn2

torch.manual_seed(42)
# Final membership tables; keeping pathways present in both the gene and
# the miRNA annotation is upstream of kpnn2.
genesets = {"hsa00010": "TP53 EGFR KRAS", "hsa04010": "EGFR KRAS MYC",
            "hsa04110": "TP53 CDK4 MYC", "hsa05200": "TP53 KRAS CDK4 PTEN"}
mirsets = {"hsa00010": "miR-1 miR-2", "hsa04010": "miR-2 miR-3",
           "hsa04110": "miR-4", "hsa05200": "miR-1 miR-4 miR-5"}
table = lambda sets: pd.DataFrame(  # noqa: E731
    [(f, p) for p, fs in sets.items() for f in fs.split()],
    columns=["source", "target"])
specs = {"snv": kpnn2.parse_layered(table(genesets)),
         "mrna": kpnn2.parse_layered(table(genesets)),
         "mirna": kpnn2.parse_layered(table(mirsets))}


class Attn(nn.Module):  # mixes the samples of a batch; no bias
    def __init__(self, d, k=8):
        super().__init__()
        self.w = nn.Parameter(torch.rand(3, d, k))

    def forward(self, x):
        q, k, v = (x @ w for w in self.w)
        return v @ torch.softmax(k.T @ q / k.shape[1] ** 0.5, -1)


class DeepKEGG(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.kegg = nn.ModuleList(kpnn2.PackedLinear(
            s.hops[0].source_index, s.hops[0].target_index,
            s.hops[0].out_features, s.hops[0].in_features,
            identity=s.fingerprint) for s in specs.values())
        self.attn = nn.ModuleList(Attn(s.layer_dims[1]) for s in specs.values())
        self.head = nn.Sequential(nn.Linear(8 * len(specs), 32), nn.Tanh(),
                                  nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, *xs):
        return self.head(torch.cat([a(torch.tanh(k(x))) for k, a, x in zip(
            self.kegg, self.attn, xs)], 1))


data = {"snv": pd.DataFrame(torch.randint(0, 2, (20, 7)).float().numpy(),
                            columns="KRAS TP53 MYC EGFR PTEN CDK4 BRAF".split()),
        "mrna": pd.DataFrame(torch.randn(20, 6).numpy(),
                             columns="PTEN CDK4 MYC KRAS EGFR TP53".split()),
        "mirna": pd.DataFrame(torch.randn(20, 5).numpy(), columns=[
            f"miR-{i}" for i in [5, 4, 3, 2, 1]])}
xs = tuple(torch.as_tensor(d.to_numpy()[:, kpnn2.align_inputs(d.columns, s)])
           for d, s in zip(data.values(), specs.values()))
model, y = DeepKEGG(specs), torch.randint(0, 2, (20, 1)).float()
loss = nn.functional.binary_cross_entropy(model(*xs), y) + 1e-3 * sum(
    k.effective_weight().pow(2).sum() for k in model.kegg)
loss.backward()
pos = y[:, 0] == 1  # recurrence samples, scored as one batch
gxi = InputXGradient(model).attribute(tuple(x[pos] for x in xs))
scores = {o: kpnn2.map_node_attributions(a.detach(), s, axis="inputs").sum(
    "observation") for (o, s), a in zip(specs.items(), gxi)}  # GLUE: N18
print({o: s.layer_nodes[1] for o, s in specs.items()})
print({o: v.to_series().round(4).to_dict() for o, v in scores.items()})
