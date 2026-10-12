"""DeepGAMI: SNP (eQTL) and TF (GRN) masked layers into one gene list."""

import pandas as pd
import torch
from captum.attr import IntegratedGradients, LayerConductance
from captum.attr import NeuronConductance
from torch import nn

import kpnn2

torch.manual_seed(42)
# Final edge tables. Hidden genes = eQTL targets with a GRN edge too, and a
# gene that is also a TF input gets its own hidden name (`_g`); upstream.
eqtl = pd.DataFrame([p.split() for p in (
    "rs1 GRIN2A_g|rs2 GRIN2A_g|rs2 DRD2_g|rs3 DRD2_g|rs4 TCF4_g|rs5 TCF4_g"
).split("|")], columns=["source", "target"])
grn = pd.DataFrame([p.split() for p in (
    "TCF4 GRIN2A_g|EGR1 GRIN2A_g|EGR1 DRD2_g|TCF4 TCF4_g|SOX2 TCF4_g"
).split("|")], columns=["source", "target"])
specs = [kpnn2.parse_layered(eqtl), kpnn2.parse_layered(grn)]
assert specs[0].layer_nodes[1] == specs[1].layer_nodes[1]  # one gene order


class DeepGAMI(nn.Module):
    def __init__(self, masks, n_fc=16, n_cls=2):
        super().__init__()
        self.l1, self.l2 = (kpnn2.MaskedLinear(m) for m in masks)
        g = masks[0].shape[0]
        self.alpha, self.beta = (nn.Parameter(torch.tensor(v))
                                 for v in (0.0, 0.1))
        self.ffn = nn.Sequential(nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(2 * g, n_fc), nn.ReLU(),
                                 nn.Linear(n_fc, n_cls))

    def forward(self, x1, x2, mode="both"):
        h1, h2 = self.l1(x1), self.l2(x2)
        h2 = self.alpha * h1 + self.beta if mode == "cg" else h2
        s = h1 + h2
        return self.ffn(torch.cat([h1 + s, h2 + s], 1)), h1, h2


snp = pd.DataFrame(torch.randint(0, 3, (24, 6)).float().numpy(),
                   columns=["rs5", "rs4", "rs3", "rs2", "rs1", "rs9"])
tf = pd.DataFrame(torch.randn(24, 4).numpy(),
                  columns=["SOX2", "EGR1", "TCF4", "FOXP2"])
xs = [torch.as_tensor(d.to_numpy()[:, kpnn2.align_inputs(d.columns, s)])
      for d, s in zip((snp, tf), specs)]
model = DeepGAMI([s.hops[0].to_mask() for s in specs])
control = DeepGAMI([torch.bernoulli(0.2 + 0.6 * s.hops[0].to_mask())
                    for s in specs])  # noisy-prior ablation
y = torch.randint(0, 2, (24,))
out, h1, h2 = model(*xs)
loss = nn.functional.cross_entropy(out, y) + nn.functional.mse_loss(
    model.alpha * h1 + model.beta, h2)  # auxiliary imputation loss
loss.backward()
model.eval()
ig = IntegratedGradients(lambda a, b: model(a, b)[0]).attribute(
    tuple(xs), target=1)
feat = [abs(kpnn2.map_node_attributions(a.detach(), s, axis="inputs")).mean(
    "observation") for a, s in zip(ig, specs)]  # GLUE: N18 mean |IG|
gene = kpnn2.map_node_attributions(LayerConductance(
    lambda a, b: model(a, b)[0], model.l1).attribute(
    tuple(xs), target=1).detach(), specs[0], hop_output=specs[0].hops[0])
nc = NeuronConductance(lambda a, b: model(a, b)[0], model.ffn[0])
per_gene = [nc.attribute(tuple(xs), neuron_selector=j, target=1,  # gene j's
                         attribute_to_neuron_input=True)  # h1 + (h1 + h2)
            for j in range(specs[0].layer_dims[1])]
links = pd.concat([s.to_edgelist().merge(abs(kpnn2.map_node_attributions(
    torch.stack([a[k].detach() for a in per_gene]), s, axis="inputs",
    dims=("gene", "observation", "node"), coords={"gene": s.layer_nodes[1]})
    .mean("observation")).to_series().rename("score").reset_index(),
    left_on=["source", "target"], right_on=["node", "gene"])[
    ["source", "target", "score"]] for k, s in enumerate(specs)])
print(specs[0].layer_nodes[1], float(loss.detach()), control.l2.mask.sum())
print([f.to_series().round(4).to_dict() for f in feat], links.round(4))
