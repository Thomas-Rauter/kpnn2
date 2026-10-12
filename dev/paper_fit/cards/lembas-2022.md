# lembas-2022: LEMBAS (knowledge-embedded recurrent signaling network)

| Field | Value |
|-------|-------|
| Paper | Nilsson A, Peters JM, Meimetis N, Bryson B, Lauffenburger DA, 2022. Artificial neural networks enable genome-scale simulations of intracellular signaling. Nature Communications 13:3069. 10.1038/s41467-022-30684-y |
| Code | `https://github.com/Lauffenburger-Lab/LEMBAS` at `2034125daa67aab2e89440d9c0a5c2edc9c06e04` |
| Other code | Zenodo snapshot 10.5281/zenodo.6532706 (cited in the paper, not fetched). `Demo/src/bionetwork.py` copies `Model/bionetwork.py` with the activations inlined. No other LEMBAS repository found by a GitHub name search. |
| Framework | Python 3.7.10, PyTorch 1.6.0, NumPy 1.20.2, SciPy 1.6.2 (pinned in `readme.md`). The recurrence runs in NumPy/SciPy inside a hand-written `torch.autograd.Function`. |
| License | MIT (`LICENSE`) |
| Full text | Europe PMC full-text XML of PMC9163072 (abstract, Results model sections, Methods, captions, code availability). Equations 1-3 are images and were not read. |
| Extracted | 2026-10-09, Claude Code, claude-opus-5-5 |

## Summary

The model predicts transcription factor (TF) activities from ligand
concentrations. A curated, signed, directed protein signaling network from
OmniPath (KEGG, InnateDB, or SIGNOR as approved sources) gives the
connectivity. Each node is one recurrent unit and each edge one weight in a
sparse square matrix. The state is iterated from zero to a steady state with
the ligand input and a bias added at every step; the TF nodes' steady-state
values are the outputs. Training penalizes weights whose sign contradicts the
prior and the matrix's spectral radius. Trained networks are probed with
simulated knockouts and a sensitivity analysis. A variant adds drug,
expression, and mutation inputs and a viability readout.

## Prior-knowledge graph

- Source: OmniPath, archive of 2021-06-21, human core set. Edges are kept if
  their `sources` field names KEGG (`Network Construction/trimKeggModel.py:82-92`),
  KEGG or InnateDB (`trimMacrophageModel.py:82-88`), or KEGG, SIGNOR, or
  manual curation (`trimLigandModel.py:75-96`). Ligand-receptor edges come
  from a separate table (`trimLigandModel.py:99-105`).
- Cleaning (`Network Construction/extractPKN.py`): self loops removed
  (110-112); reversible interactions become two opposite directed edges
  (115-119); duplicate source-target pairs merged by OR-ing signs (9-19,
  121-132); edges both activating and inhibiting lose their sign (170-171);
  only reviewed UniProt proteins kept (174-177).
- Size: macrophage experiment model has 1262 nodes and 6594 edges (paper,
  Fig. 6c). Node types: ligand, receptor, signaling protein, TF. Directed,
  signed with three states (activating, inhibiting, unknown;
  `Model/bionetwork.py:762-767`), cycles kept (feedback loops motivate the
  recurrent design), no self loops.
- Pruning: nodes with no directed path from any ligand or none to any TF are
  removed (`trimLigandModel.py:5-45`, applied at 107 and 136); edges into
  ligands are removed (110-113). Pass-through nodes A→X→A are removed
  repeatedly (`trimKeggModel.py:47-72`, applied at 108), but not for the
  experimental network (`trimLigandModel.py:108` is commented out).
- Data to nodes: ligand and TF columns match node names (UniProt IDs) by
  identity; `numpy.intersect1d` keeps only columns that are nodes and drops
  the rest silently (`Model/macrophageNet.py:41-49`). Nodes with no input
  column get zero input (`Model/bionetwork.py:616-618`).
- Built at: `Model/bionetwork.py:753-789` (edge table → index arrays and
  sign masks).

## Architecture

- Layers: none. One square recurrent matrix over all nodes, rows = targets,
  columns = sources (`Model/bionetwork.py:471-472`). Update
  `h ← f(A·h + x + b)` with constant projected input `x` and per-node bias
  `b` (135, 147-149), from `h = 0` (137), for up to `iterations` steps (100
  or 150; `Model/macrophageNet.py:27`, `Model/ligandScreenKO.py:78`),
  stopping early once the summed change is below 1e-6 (141-145). Only the
  final state is used.
- Units per node: one, own bias, shared activation (474-475). Default is the
  "Michaelis-Menten-like" function: slope 0.01 below 0, identity on
  [0, 0.5], `1 - 0.25/x` above (`Model/activationFunctions.py:11-16`).
  Leaky ReLU and sigmoid are options (`Model/bionetwork.py:477-488`).
- Input: each ligand column times a per-input factor (3, frozen by
  `Model/macrophageNet.py:53`) is written at its node's index
  (`Model/bionetwork.py:604-618`). Output: TF node states gathered by index
  times a trainable per-output scale, initialized 1.2 (621-638).
- Connections that skip layers: not applicable.
- Losses on inner nodes: a loss on all node states pushes each node's mean,
  variance, min, and max across samples toward those of a uniform
  distribution (`Model/bionetwork.py:237-262`; `Model/macrophageNet.py:96`,
  103). No auxiliary heads.
- Parts the graph does not constrain: the input and output scales; in the
  viability variant, a per-node expression layer, a per-mutation weight
  projected onto the mutated gene's node, and one linear viability unit over
  named TF states (`Model/viabilityNet.py:47-131`).
- Other structure: viability variant only. Drugs pass through a drug→target
  matrix limited to known pairs; drug, expression, and mutation terms are
  summed into one node input (`Model/viabilityNet.py:10-38`, 145-150).
- Defined at: `Model/bionetwork.py:125-209`, 457-491, 604-638.

## Connectivity mechanism

The network is a `scipy.sparse.csr_matrix` built once from the edge index
arrays (`Model/bionetwork.py:471-472`); the weights are a 1-D
`nn.Parameter` with one entry per edge (474). Absent edges have no storage
and no parameter. Each forward call copies the parameter vector into the
matrix's `.data` in place (130), which assumes the edge order equals SciPy's
CSR order (comment at 471, sort at 781-787).

Forward and backward are NumPy code in a custom autograd function
(125-194). The backward pass iterates the adjoint
`g ← f'(z) ⊙ (Aᵀ·g + ∂L/∂h)` to its own fixed point with a tanh-saturated
clip each step (16-21, 162-183). The weight gradient exists only for present
edges, `Σ_samples h[source]·g[target]` (191); the bias gradient is `Σ g`
(192). Gradients are taken at the steady state, not through a stored
trajectory.

Initialization: `|w| ~ U(0.1, 0.2)`, negated for inhibitory edges; bias
1e-3, or 1 for nodes whose in-edges are all inhibitory (552-561). Weights
are then scaled to spectral radius 0.8 (573-576; 0.7 in
`Model/ligandScreenCrossValidation.py:64`).

Signs are soft: the loss adds `MoAFactor · Σ|w|` over weights whose sign
contradicts the prior; unsigned edges are free (`Model/bionetwork.py:500-505`,
`Model/macrophageNet.py:100`). An exponential penalty acts on the spectral
radius of `A` with each edge scaled by `f'` at its target for one random
sample (`Model/bionetwork.py:211-234`). The radius and its gradient
`Re(w[target]·v[source] / wᵀv / phase)` per edge come from a sparse
eigensolver plus a shifted solve on `Aᵀ` for the left eigenvector (28-120).

Viability variant: the drug→target matrix is dense with a 0/1 mask,
multiplied in place at the start of every batch
(`Model/viabilityNet.py:296`, 469). The forward pass uses the stored weight,
so absent entries get gradients and updates that the next batch erases.

## Interpretation

- Knockout and knock-in: for one condition, the projected input is repeated
  once per node and a constant is added to that node's pre-activation (-3 or
  +3, `Model/ligandScreenKO.py:8-24`, 66-67; -5 on internal nodes,
  `Model/synthNetKO.py:88`, 103-113). Score: change of each TF output versus
  the unperturbed run, per node. Combined across cross-validation fold models
  by the median and ranked by absolute effect on one TF
  (`Model/ligandScreenKO.py:97-135`); tabled with gene names (136).
- Sensitivity (elasticity): each node's raw steady-state pre-activation
  `A·h + b + x` is raised by 0.001 of itself; score is `(ΔTF/TF) / 0.001`,
  zeroed where the reference TF is below 0.01
  (`Model/ligandScreenSensitivity.py:8-30`, 72, 104-113). Median across fold
  models (115), then max and min across conditions per internal node
  (126-128). Both analyses use the full node state that the model returns
  (`Model/bionetwork.py:204-209`).
- Edge weights: `saveParam` writes each weight with source and target name,
  each bias, and the projection scales to a TSV (808-846). `loadParam` reads
  that format back by name (848-877) to build the hand-parameterized
  reference model for synthetic data (`Demo/main.py:29-30`).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `Model/viabilityNet.py:10-38` | Viability variant: each drug feeds its known protein targets through a drug-target table matched by name. |
| N09 Dense head after the graph layers | `Model/viabilityNet.py:99-131` | Viability variant: one linear unit with bias over the named TF states predicts viability. |
| N10 One unit per node | `Model/bionetwork.py:474-475` | Each protein is one state with its own bias and the shared activation. |
| N12 Parameters only for present edges | `Model/bionetwork.py:471-474`, 191 | One weight per edge in a sparse matrix; gradients are computed only for edges. |
| N17 Map scores to node names | `Model/ligandScreenKO.py:136` | Knockout effects are tabled by gene name of the perturbed node. |
| N28 Prune nodes that cannot reach an output | `Network Construction/trimLigandModel.py:5-18`, 33-45 | Nodes with no directed path to any TF are removed. |
| N31 Read out node activations | `Model/bionetwork.py:204-209` | All node states are returned for losses and perturbation analyses. |
| N32 Edge weights labeled by edge | `Model/bionetwork.py:808-846` | Weights, biases, and scales are saved with source and target names. |
| N33 Combine scores across training repeats | `Model/ligandScreenKO.py:97-135`, `Model/ligandScreenSensitivity.py:115` | Knockout and sensitivity scores are combined across cross-validation fold models by the median. |
| N42 Read a signed directed graph with cycles | `Model/bionetwork.py:753-789` | The edge table has source, target, and activating/inhibiting flags; cycles are kept and unsigned edges allowed. |
| N43 Iterate a cyclic graph to steady state | `Model/bionetwork.py:135-157`, 162-194 | All node states update together from zero with constant input and bias until they stop changing; loss and gradients use the steady state. |
| N44 Edge signs as a soft constraint | `Model/bionetwork.py:500-505`, 552-553 | Weights start with the prior sign and an L1 penalty acts on weights of the wrong sign. |
| N45 Feed inputs into named graph nodes | `Model/bionetwork.py:604-618` | Each ligand column is scaled and added to the pre-activation of the node with its name at every step. |
| N46 Read outputs from named graph nodes | `Model/bionetwork.py:621-638` | TF node states, selected by name, are the outputs, each with a trainable scale. |
| N47 Keep a recurrent edge matrix contractive | `Model/bionetwork.py:211-234`, 573-576 | Weights are scaled to spectral radius 0.8 at start and an exponential spectral-radius penalty is added in training. |
| N48 Penalize node-state distributions | `Model/bionetwork.py:237-262` | Each node's mean, variance, min, and max across samples are pushed toward a uniform distribution's. |
| N49 Simulated node knockout | `Model/ligandScreenKO.py:8-24` | A large negative (or positive) offset is added to one node's pre-activation, for every node in a batch. |
| N50 Relative output sensitivity to each node | `Model/ligandScreenSensitivity.py:8-30`, 104-113 | Each node's pre-activation is raised by a small fraction of itself and the relative output change is divided by it. |
| N51 Set parameters from a named edge table | `Model/bionetwork.py:848-877`, `Demo/main.py:29-30` | Weights and biases are loaded by source and target name to build a reference model. |
| N52 Prune nodes no input can reach | `Network Construction/trimLigandModel.py:20-45` | Nodes with no directed path from any ligand are removed. |
| N53 Re-zero masked weights before each step | `Model/viabilityNet.py:296`, 469 | Viability variant: the dense drug-target matrix is multiplied by its 0/1 mask in place at each batch start. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Clean OmniPath edges (self loops, reversible, duplicates, sign conflicts) | `Network Construction/extractPKN.py:108-179` | ~70 |
| Prune by reachability; remove A→X→A pass-through nodes | `Network Construction/trimLigandModel.py:5-45`, `trimKeggModel.py:47-72` | ~65 |
| Edge table → index arrays, sign masks, node order | `Model/bionetwork.py:753-789` | ~35 |
| Sparse recurrent forward and adjoint backward | `Model/bionetwork.py:16-21`, 125-194 | ~75 |
| Spectral radius and its gradient on the edge list | `Model/bionetwork.py:28-120`, 211-234, 573-576 | ~110 |
| Sign-aware initialization and sign violations | `Model/bionetwork.py:500-511`, 547-571 | ~30 |
| Align data columns to nodes; input scatter and output gather by name | `Model/macrophageNet.py:41-49`, `Model/bionetwork.py:604-638` | ~45 |
| Save and load parameters by edge name | `Model/bionetwork.py:808-877` | ~70 |
| Knockout and sensitivity batches | `Model/ligandScreenKO.py:8-24`, `Model/ligandScreenSensitivity.py:8-30` | ~40 |
| Drug-target mask and per-batch re-masking | `Model/viabilityNet.py:10-38`, 296 | ~30 |

## Fragile spots

- Weight order: weights are written into the CSR `.data`
  (`Model/bionetwork.py:38`, 130). This is right only if the edge order
  equals SciPy's storage order (comment at 471; workaround for a SciPy
  ordering change at 780-783). A mismatch moves weights to wrong edges
  silently.
- Shared matrix: forward, backward, and spectral-radius code overwrite the
  same CSR `.data` (38, 130, 166); `Model/ligandScreenSensitivity.py:17`
  resets it by hand.
- No convergence check on exit: if the iteration has not settled by
  `iterations`, the last state is used (`Model/bionetwork.py:141-149`); the
  backward pass assumes a contractive matrix and relies on clipping (183).
- Duplicate edges: `makeNetworkList` sums duplicate source-target rows
  (777), so a summed sign code such as 1 + 0.1 matches neither sign test
  (767). `trimLigandModel.py:106` drops only exact duplicate rows.
- Drug mask: re-masking runs at the start of a batch
  (`Model/viabilityNet.py:296`), so after the last optimizer step (323)
  absent drug-target entries keep one update.

## Out of scope

- Loss terms other than the above: MSE on TFs, L2 on weights and biases,
  ligand-bias and projection-scale penalties (`Model/macrophageNet.py:98-110`).
- Training loop: Adam, one-cycle learning rate, optimizer reset every 200
  epochs, input noise scaled by learning rate (`Model/macrophageNet.py:64-125`).
- Data: TF activities from DoRothEA/viper (`TF activities*/`), ligand design
  matrices; synthetic data and ODE comparisons (`Model/synthNet*.py`,
  `Model/ODEsimulation.py`); MATLAB timing tests; plotting.

## Paper vs code

- Sensitivity step: paper 0.01; code 0.001 (`Model/ligandScreenSensitivity.py:72`).
- Knockout offset: paper -5; macrophage -3 (`Model/ligandScreenKO.py:66`),
  synthetic -5 (`Model/synthNetKO.py:88`).
- Iterations: paper fixed 150; code stops early at a 1e-6 change
  (`Model/bionetwork.py:142-145`), 100 in `Model/macrophageNet.py:27`.
- L2: paper 1e-8; code 1e-6 (`Model/ligandScreenCrossValidation.py:17`,
  `Model/macrophageNet.py:19`). The anti-zero term `Σ 1/(w²+0.5)` is
  described for synthetic data only but also appears in
  `Model/macrophageNet.py:105`.
- Spectral prescale: paper 0.8; experimental CV script 0.7
  (`Model/ligandScreenCrossValidation.py:64`).
- Pass-through node removal is described for all networks but is commented
  out for the experimental one (`Network Construction/trimLigandModel.py:108`).

## Key excerpts

`Model/bionetwork.py:141-149`

```python
        for i in range(parameters['iterations']):
            if i>40: #normally takes around 40 iterations to reach steady state
                if i>41:
                    if numpy.sum(numpy.abs(xhat-xhatBefore))<1e-6:
                        break            
                xhatBefore = xhat.copy()            
            xhat = A.dot(xhat)
            xhat += bIn
            xhat = activation(xhat, parameters['leak'])
```

`Model/bionetwork.py:471-475`

```python
        #for this to work as intended network list must be sorted on index 0
        self.A = scipy.sparse.csr_matrix((weights.detach().numpy(), networkList), shape=(size, size), dtype='float64')

        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
```

`Model/ligandScreenKO.py:8-12`

```python
def generateKO(Yin, knockOutLevel):
    nrKO = Yin.shape[1]
    inputKO = Yin.repeat(nrKO, 1)
    inputKO = inputKO + knockOutLevel * torch.eye(nrKO)
    return inputKO
```

## Open questions

- Size of the synthetic (KEGG-only) network: not in the text read; data
  files not opened.
- Whether TF nodes have outgoing edges (outputs not always sinks): needs the
  model TSVs, not read.
- Equations 1-3 are images in the XML; the code forms are cited instead.
- Whether the Zenodo snapshot differs from HEAD (last commit 2025-01-24, "fix
  bug causing issues with autograd test"): not checked.
