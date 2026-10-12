# self-pruning-2025: LEMBAS on GPU, self-pruning of added edges

| Field | Value |
|-------|-------|
| Paper | Nordenstorm O, Baghdassarian H, Lauffenburger DA, Nilsson A, 2025. Biologically informed neural network models are robust to spurious interactions via self-pruning. bioRxiv. 10.1101/2025.10.24.684155 |
| Code | `https://github.com/AvlantNilssonLab/LEMBAS_GPU` at `35ff041d0765e9032c2672fa7fc4612df1002684` |
| Other code | Original CPU LEMBAS (`https://github.com/Lauffenburger-Lab/LEMBAS`), copied under `original_code/`; PyPI `LEMBAS-re` built from `LEMBAS/`; trained models on Zenodo record 17425598 (not downloaded) |
| Framework | Python, PyTorch `>=2.1.0` (`setup.py:37`), NumPy, SciPy, pandas |
| License | MIT per `setup.py:23` and the paper; no LICENSE file in the repository |
| Full text | bioRxiv v1 full text (HTML) |
| Extracted | 2026-10-10, Claude Code, claude-opus-5-5 |

## Summary

The paper reimplements LEMBAS, a recurrent network with one state per
signaling protein and one recurrent weight per prior-knowledge
interaction, for the GPU with dense matrix products and autograd. It
then asks whether training removes false interactions: for each of 50
models per setting it adds 100 random edges to the prior, trains on
ligand inputs to predict transcription-factor (TF) activities (two
macrophage datasets, one synthetic KEGG-based dataset) at three L2
strengths, and compares trained weight magnitudes of added and real
edges. It also zeroes edges in trained models and measures the output
change. The repository also holds an experiment, not in the paper,
that adds random edges to a layered Reactome network.

## Prior-knowledge graph

- Source: signed signaling networks from the original LEMBAS work
  (`macrophage-Model.tsv`, `ligandScreen-Model.tsv`,
  `KEGGnet-Model.tsv`), read at `benchmark/self_pruning_figure_macrophage_and_syn_setup.py:82,134,150`. The files
  are not in the repository.
- Size: not in the repository. The paper says 100 added edges are
  12 %, 1.7 %, and 1.5 % of the edges of the low-coverage, synthetic,
  and high-coverage networks (about 830, 5,900, and 6,700 edges).
  Directed, signed (activating, inhibiting, unknown), cycles kept.
- Edge table: `source`, `target`, `stimulation`, `inhibition`; the
  flags become `mode_of_action` 1, −1, or 0.1 for unknown, so unknown
  edges survive the sparse constructor
  (`LEMBAS/model/model_utilities.py:51-59`). Duplicate rows are summed
  (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:800-801`).
- Nodes: every source or target name, sorted and indexed in that order
  (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:789-790`); an optional ban list drops
  named nodes (`:785-786`), unused here.
- Data to nodes: ligand and TF columns are kept only if their name is a
  node (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:728-729`); others are dropped
  silently. Nodes without data get zero external input.
- Added edges: 100 pairs drawn uniformly over all nodes, rejecting
  existing edges but not self-loops, appended with unknown sign
  (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:28-37,165-180`). A degree-weighted variant
  (no self-loops) and an all-pairs variant are switched off by a
  hard-coded string (`:157,182-224`). Added pairs are kept as
  `random_edges` (`:226`) and saved with each checkpoint
  (`LEMBAS/benchmarking_version/self_prune_train_figure.py:256`); the prior's edges as `edge_real`
  (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:160`).
- Built at: `LEMBAS/benchmarking_version/self_prune_bionetwork.py:748-805`, `:157-232`.

## Architecture

- Layers: none. One step `x ← f(W x + b + u)` updates all nodes from
  `x = 0`, up to 150 times, stopping early when the largest change is
  below `tolerance` (checked every 10 steps after step 20)
  (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:344-373`; `benchmark/self_pruning_figure_macrophage_and_syn_setup.py:175`). Only the
  final state is used.
- Units per node: one state, one bias, a shared Michaelis–Menten-like
  activation with leak 0.01 (`LEMBAS/model/activation_functions.py:11-32`).
- Connections that skip layers: not applicable; any node may feed any
  node, itself included.
- Outputs or losses at inner layers: none.
- Inputs: each ligand column times a scale (3, frozen) is written into
  the node of that name (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:68-85`;
  `benchmark/self_pruning_figure_macrophage_and_syn_setup.py:204`).
- Outputs: TF node states, each times a trainable scale initialized to
  1.2 (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:580-631`). The package version adds a
  trainable bias per output (`LEMBAS/model/bionetwork.py:571-572,596`);
  the self-pruning version has none.
- Parts the graph does not constrain: only the input and output scales.
- Other structure: the Reactome experiment stacks one `nn.Linear` per
  connectivity matrix of an external `binn` package, with ReLU and a
  dense output layer (`self_prune_BINN/binn_moreedges.py:23-39,148`).
- Defined at: `LEMBAS/benchmarking_version/self_prune_bionetwork.py:39-245,580-840`.

## Connectivity mechanism

The recurrent weight is a dense `n × n` `nn.Parameter` (targets ×
sources), filled at edge positions and zero elsewhere
(`LEMBAS/benchmarking_version/self_prune_bionetwork.py:288-304`); a boolean mask marks absent
entries (`:274-286`). The forward pass uses the stored weight in
`torch.mm` without the mask (`:364`). Absent entries therefore get
gradients from the fit and spectral losses and Gaussian gradient noise
(`:971-985`), Adam updates them, and `force_sparcity` sets them back
to zero after every optimizer step (`:375-376`, called at
`LEMBAS/benchmarking_version/self_prune_train_figure.py:298`). Added edges are ordinary entries of
the same matrix.

Initialization: `0.1 + 0.1·U(0,1)` per edge, negated for inhibiting
edges; unknown and added edges stay positive
(`LEMBAS/benchmarking_version/self_prune_bionetwork.py:263-264`). Bias 1e-3, or 1 for a node whose
inputs are all negative (`:266-270`). The matrix is rescaled to
spectral radius 0.9 (`:327-342`; `benchmark/self_pruning_figure_macrophage_and_syn_setup.py:205`).

Edge terms in the loss (`LEMBAS/benchmarking_version/self_prune_train_figure.py:276-288`): L1 on
weights whose sign contradicts a known sign, weight 0.1, unknown and
added edges exempt (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:447-494`); L2 on all
recurrent weights and biases, `1e-6 ×` a command-line factor, the knob
the paper varies (`:409-426`; `benchmark/self_pruning_figure_macrophage_and_syn_setup.py:180,219-237`); a penalty on
the spectral radius of `f'(x)·W` at the current states, by 5
power-iteration steps on 2 samples and 2 probes (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:518-577`;
`benchmark/self_pruning_figure_macrophage_and_syn_setup.py:182`). The node-state uniformity penalty is multiplied by
0 (`LEMBAS/benchmarking_version/self_prune_train_figure.py:288`); the package loop keeps it
(`LEMBAS/model/train.py:223,228`).

In the Reactome experiment `torch.nn.utils.prune.custom_from_mask`
multiplies each stored weight by its 0/1 mask in every forward pass
(`self_prune_BINN/binn_moreedges.py:29-31`).

## Interpretation

- During training, every 5 epochs: mean `|w|` of real and of added
  edges, and each set's share below the 5th percentile of pooled `|w|`
  (`LEMBAS/benchmarking_version/self_prune_train_figure.py:326-345`). Weights are read at
  (target, source) index pairs, not by name. Full state dicts are
  saved at epoch 0, every `max_iter/20` epochs, and the last epoch
  (`LEMBAS/benchmarking_version/self_prune_train_figure.py:252-257`).
- Below-median share (Fig. 2e): per model, share of added and of real
  edges below the median nonzero weight
  (`data_to_report/fig_2_3/sup_fig_X_Y.py:49-64`). Real edges come from
  `ground_truth_{type}.txt`, not in the repository (`:30-31`).
- Weight distributions (Fig. 3): at the middle checkpoint, CDFs of `|w|`
  for added and real edges, an area summary per model, and a one-sided
  Wilcoxon test across models
  (`data_to_report/fig_2_3/fig_3_S_9_S_10_S_11.py:17-24,76-98,219`).
  Edges can be binned by mean endpoint degree; one bin is in use
  (`:34-41,49,100-111`).
- Cumulative edge ablation: zero nonzero weights in order of `|w|`, in
  chunks of 300 (30 for synthetic), rerun, report mean absolute output
  deviation; curves averaged over 50 models
  (`data_to_report/fig_2_3/fig_2_g.py:121-143`).
- Single-edge ablation: zero one edge at a time (added edges plus a
  sample of real ones), record the largest output change against the
  intact model (`data_to_report/fig_2_3/fig_2d_zero_out_scatter.py:167-201`).
- Repeats: 50 models per dataset and L2 level, seeds 0–49, each with its
  own added edges (`benchmark/self_pruning_figure_macrophage_and_syn_setup.py:228-246`).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N42 Read a signed directed graph with cycles | `LEMBAS/model/model_utilities.py:51-59`; `LEMBAS/benchmarking_version/self_prune_bionetwork.py:748-805` | Source, target, and sign flags become a cyclic signed edge list. |
| N43 Iterate a cyclic graph to steady state | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:358-373` | All states update together from zero until the change is below a tolerance. |
| N10 One unit per node | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:264-270,366` | Each protein is one state with its own bias and the shared activation. |
| N45 Feed inputs into named graph nodes | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:68-85,728` | Ligand columns enter the nodes of the same name, times a frozen scale. |
| N46 Read outputs from named graph nodes | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:614-631,729` | TF node states, each with a trainable scale, are the outputs. |
| N53 Re-zero masked weights before each step | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:375-376`; `LEMBAS/benchmarking_version/self_prune_train_figure.py:294-298` | The dense recurrent weight is re-masked in place after each optimizer step. |
| N44 Edge signs as a soft constraint | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:263-264,447-494` | Signed init and an L1 penalty on sign violations; unknown and added edges are free. |
| N47 Keep a recurrent edge matrix contractive | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:327-342,518-577` | Spectral radius set to 0.9 at start; the Jacobian's spectral radius is penalized. |
| N13 Regularize edge weights per layer | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:409-426`; `benchmark/self_pruning_figure_macrophage_and_syn_setup.py:180` | L2 strength on the one recurrent edge matrix is the variable that drives self-pruning. |
| N48 Penalize node-state distributions | `LEMBAS/model/train.py:223,228`; `LEMBAS/benchmarking_version/self_prune_train_figure.py:288` | Kept in the package training loop; multiplied by 0 in the self-pruning runs. |
| N22 Mask a dense weight | `self_prune_BINN/binn_moreedges.py:29-31` | The Reactome experiment masks each `nn.Linear` with a pruning mask. |
| N59 Add random edges to the prior | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:28-37,157-226`; `self_prune_BINN/binn_moreedges.py:76-104` | 100 unsigned non-edges per model, with their indices kept for comparison. |
| N32 Edge weights labeled by edge | `LEMBAS/benchmarking_version/self_prune_train_figure.py:326-345`; `data_to_report/fig_2_3/fig_3_S_9_S_10_S_11.py:97-98` | Trained weights are read per edge at index pairs and split into real and added sets. |
| N60 Edge ablation | `data_to_report/fig_2_3/fig_2_g.py:121-137`; `data_to_report/fig_2_3/fig_2d_zero_out_scatter.py:178-192` | Edges are zeroed singly or cumulatively by size and the output change is measured. |
| N33 Combine scores across training repeats | `benchmark/self_pruning_figure_macrophage_and_syn_setup.py:228-246`; `data_to_report/fig_2_3/fig_2_g.py:143` | Edge-weight statistics and ablation curves are pooled or averaged over 50 seeds. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Sign flags to one value, unknown as 0.1 | `LEMBAS/model/model_utilities.py:51-59` | ~10 |
| Edge table to sorted node index and edge list | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:748-805` | ~30 |
| Sample and append random edges, keep indices | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:28-37,157-232` | ~50 |
| Dense weight, absence mask, sign mask | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:274-325` | ~25 |
| Re-zero absent entries after each step | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:375-376` | 2 |
| Sign-violation penalty | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:447-494` | ~15 |
| Inputs and outputs matched to nodes by name | `LEMBAS/benchmarking_version/self_prune_bionetwork.py:68-85,614-631,728-729` | ~15 |
| Real and added edge weights during training | `LEMBAS/benchmarking_version/self_prune_train_figure.py:326-345` | ~20 |
| Random edges per layer, by row and column degree | `self_prune_BINN/binn_moreedges.py:76-122` | ~45 |

## Fragile spots

- Absent entries are nonzero between `optimizer.step()` and
  `force_sparcity()`, and the forward pass has no mask; a loop that
  skips the call lets absent edges carry signal
  (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:364,375-376`).
- Uniform sampling accepts self-loops and edges into input nodes or out
  of output nodes; whether an added edge can reach an output is not
  checked (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:33`).
- Single-edge ablation samples `randperm(edge_list[1][0])` real edges,
  the source index of the first edge, not the edge count
  (`data_to_report/fig_2_3/fig_2d_zero_out_scatter.py:173`).
- Cumulative ablation measures deviation from the first ablated run,
  not the intact model (`data_to_report/fig_2_3/fig_2_g.py:129-136`).

## Out of scope

- Loss: MSE on TF activities (`LEMBAS/benchmarking_version/self_prune_train_figure.py:276`).
- Training loop: Adam, warm-up and decay capped at 1e-3, optimizer
  state reset every 200 epochs, batch size 1000, input noise off
  (`LEMBAS/benchmarking_version/self_prune_train_figure.py:156,193,221,235,364-365`).
- Gradient noise `g + γ·N(0, I)`, γ = 1e-9 (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:971-985`).
- Data loading and split (`LEMBAS/benchmarking_version/self_prune_train_figure.py:24-65`).
- CPU versus GPU timing and gradient checks (`benchmark/`,
  `LEMBAS/benchmarking_version/benchmark_*.py`); plotting (`data_to_report/`).

## Paper vs code

- Paper: Fig. 3 distributions differ by a Kolmogorov–Smirnov test. Code:
  one-sided Wilcoxon on per-model CDF areas
  (`data_to_report/fig_2_3/fig_3_S_9_S_10_S_11.py:219,424`); KS appears
  only in `benchmark/benchmarking_bias.ipynb` and `ligand_data/test.py`.
- Paper: magnitudes compared with the median. Code: signed weights
  (`data_to_report/fig_2_3/sup_fig_X_Y.py:49-64`).
- Paper: optional output bias in the GPU version. The self-pruning model
  has none (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:614-631`).
- Paper: state regularization removed. Code: computed and multiplied by
  0 (`LEMBAS/benchmarking_version/self_prune_train_figure.py:288`).
- The Reactome experiment in `self_prune_BINN/` is not in the paper.

## Key excerpts

`LEMBAS/benchmarking_version/self_prune_bionetwork.py:358-376`

```python
        X_bias = X_full.T + self.bias # this is the bias with the projection_amplitude included
        X_new = torch.zeros_like(X_bias) #initialize all values at 0
        

        for t in range(self.training_params['max_steps']): # like an RNN, updating from previous time step
            X_old = X_new
            X_new = torch.mm(self.weights, X_new) # scale matrix by edge weights
            X_new = X_new + X_bias  # add original values and bias       
            X_new = self.activation(X_new, self.training_params['leak'])
            
            if (t % 10 == 0) and (t > 20):
                diff = torch.max(torch.abs(X_new - X_old))    
                if diff.lt(self.training_params['tolerance']):
                    break

        Y_full = X_new.T
        return Y_full
    def force_sparcity(self):
        self.weights.data.masked_fill_(mask = self.mask, value = 0.0) # fill non-interacting edges with 0
```

`LEMBAS/benchmarking_version/self_prune_train_figure.py:288-298`

```python
            total_loss = fit_loss + sign_reg + ligand_reg + param_reg + stability_loss +  0 * uniform_reg

            total_loss.backward()

            mod.add_gradient_noise(noise_level = hyper_params['gradient_noise_level'])
                
            optimizer.step()
            # store
            cur_eig.append(spectral_radius)
            cur_loss.append(fit_loss.item())
            mod.signaling_network.force_sparcity()
```

## Open questions

- Node and edge counts: network files are not in the repository; only
  the paper's percentages give edge counts.
- Which sampling mode produced the published models: the mode is a
  hard-coded string set to uniform
  (`LEMBAS/benchmarking_version/self_prune_bionetwork.py:157`), but the
  figure scripts read `parameter_bayesian_moa_study_*` folders
  (`data_to_report/fig_2_3/fig_2_g.py:112`) while training writes
  `parameter_study_*`
  (`benchmark/self_pruning_figure_macrophage_and_syn_setup.py:238`).
  The paper does not say. Zenodo models not downloaded.
- `ground_truth_*.txt` (real-edge indices for the figure scripts): not
  in the repository; assumed to be the prior's edges, not checked.
- The full set of L2 factors: command-line arguments; the Fig. 2d driver
  uses 1 for high coverage and 10 for the others
  (`data_to_report/fig_2_3/fig_2d_zero_out_scatter.py:208-212`).
- How the external `binn` package builds the layered Reactome matrices:
  not in this repository.
