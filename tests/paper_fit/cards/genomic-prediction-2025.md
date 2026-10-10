# genomic-prediction-2025: BINN for genomic prediction (G2B2P)

| Field | Value |
|-------|-------|
| Paper | Kontolati K, Gladstone RJ, Davis IW, Pickering E, 2025. Biology-informed neural networks learn nonlinear representations from omics data to improve genomic prediction and interpretability. arXiv. 10.48550/arXiv.2510.14970 |
| Code | Zenodo record 10.5281/zenodo.17477083, archive `binns_code.zip` (3.3 GB, md5 `1ef084f3cf1293c5d4cfd1198356bf95` as published by Zenodo). Only source files, one `metadata.json`, and one ONNX model were read via HTTP range requests; the archive was not downloaded whole, so no sha256 was computed. |
| Other code | None public. The model class, losses, and training loop are imported from a package `deepconcept` (`deepconcept.model.biognn`, `deepconcept.model.loss`) that is not in the archive, not on PyPI, and not found by GitHub repository or code search. |
| Framework | Python, PyTorch (ONNX exports report producer `pytorch` 2.6.0); bytecode is CPython 3.12. No pinned requirements file. |
| License | Zenodo record: CC-BY-4.0. Paper: CC BY-NC-ND 4.0. |
| Full text | arXiv HTML v1 (Methods 4.1-4.3, Results 2.1, Fig. 1-2 captions, Code Availability). Formulas read from the HTML alt text. |
| Extracted | 2026-10-10, Claude Code, claude-opus-5-5 |

Citation conventions in this card:

- `data_prep.py`, `utils.py` mean `src_binn/data_prep.py`, `src_binn/utils.py`.
- `model.py:N` means line N of `src_binn/model.py`. That file is not in the archive. Only `src_binn/__pycache__/model.cpython-312.pyc` is, and the line numbers come from its bytecode line table (read with `marshal` and `dis`, not executed). Its header records a 54,653-byte source.
- `ONNX` means `binn_l1_ratio_0.1_paper/pretrained_models/variables_20percent/split_1/models/Anthesis.sp.MI/BINN_MSE/fold_0/best_model.onnx`. It is cited by ONNX node or initializer name, read with a minimal protobuf reader (no onnx install, no inference run).
- `notebook` means `inference-demo.ipynb`, cited by code-cell index.

## Summary

The model predicts maize flowering-time traits (days to anthesis and silking, two locations) from SNP genotypes. Gene expression is used only to build the network and, optionally, to supervise hidden values during training. It is not needed at inference. Per trait and per outer data split, an elastic net on expression selects about 1,000 genes. An eQTL scan then links each gene to SNPs, and the first 20 SNPs per gene become that gene's inputs. Each gene is a small private MLP (20 SNPs → 128 sigmoid units → 1 latent). The 1,041 gene latents are concatenated and passed to a dense integrator (128 ReLU → 1). The paper interprets the model by clamping each gene latent and measuring the change in prediction, but that code is not public.

## Prior-knowledge graph

- Source: derived from the training data, not from a curated database. Genes are kept when `ElasticNetCV` (`l1_ratio` 0.1 in the shipped models) gives them a nonzero coefficient for the trait (`utils.py:99-124`; `l1_ratio = 0.1` in notebook cell 6). Per gene, an eQTL scan (rMVP, run by an R script) nominates SNPs (`utils.py:183-217`). The R script `src/run_eQTL_with_rMVP.R` (`utils.py:214`) is not in the archive. The first `num_snps` rows of each gene's marker table are used (`data_prep.py:1028`; `num_snps = 20`, notebook cell 6). Selection runs separately for each of 5 outer splits (`data_prep.py:1420-1436`). Genotypes are the WiDiv panel, MAF ≥ 0.05, LD-pruned at 0.5 (file names in the archive listing). Gene IDs are maize `Zm00001eb…`.
- Size: one bipartite layer, SNP → gene. In the inspected model there are 1,041 genes, each with 20 SNP inputs, giving an input width of 20,820 (`ONNX`: 1,041 `core.pathway_nets.*` blocks; `metadata.json`: `"input_dim": 20820`). Directed, unsigned, acyclic. One SNP may feed several genes; the code counts the reuse (`data_prep.py:1032`).
- Data to nodes: allele suffixes are stripped from SNP column names (`data_prep.py:922`). For each gene that has a marker table, its SNP columns are copied out under the names `"{gene}_{snp}"`, and the blocks are concatenated in gene-list order (`data_prep.py:1022-1038`). A shared SNP therefore appears once per gene. Genes in the list without a marker table are dropped with a warning (`data_prep.py:1008-1012`). SNPs that no gene selected are not in the input at all. Measured expression is collected per gene in the same order (`data_prep.py:1043-1055`); genes without expression only print a warning (`data_prep.py:1050-1051`).
- Built at: `data_prep.py:1006-1057` (input blocks), `data_prep.py:1263-1271` (SNP → gene mask), `data_prep.py:1281-1346` (optional gene → module mask).

## Architecture

- Layers: SNP inputs → one gene layer → dense integrator → one output (`ONNX` node sequence; `model.py:278-286`). The model takes a list of masks (`model.py:259`) and passes all of them when there is more than one (`model.py:280`). A second mask, gene → co-expression module, is built when a module file is given (`data_prep.py:1281-1346`). How the private model stacks several masks is unclear.
- Units per node: each gene is its own MLP. In the inspected model it is Linear(20→128), Sigmoid, then Linear(128→1) (`ONNX` initializers `core.pathway_nets.j.network.0.weight` (128, 20) and `network.3.weight` (1, 128)). The arguments are `activation='sigmoid'`, `out_activation=False`, `output_size=1` (`model.py:268-270`), and `hidden_layers` comes from the config (`metadata.json`: `[128]`). Module index 2 of each subnetwork does not appear in the export; its role is unclear (possibly dropout).
- Connections that skip layers: none.
- Outputs or losses at inner layers: an optional soft loss compares gene latents with measured expression (`model.py:148-150`: `partial(BinnSoftLoss, lam=config['binn_lambda'])`; also `BinnHardLoss`). Each batch carries the measured expression per gene as `"intermediates"` (`data_prep.py:1240-1247`). The shipped models sit under `BINN_MSE` (`metadata.json`: `"loss": "BINN_MSE"`).
- Parts the graph does not constrain: the integrator `final_net`, Linear(1041→128), ReLU, Linear(128→1) (`ONNX` nodes `/core/final_net/*`; `final_hidden_layers` `[128]` in `metadata.json`, passed at `model.py:283`). There is also a residual FCN with `leakyrelu` over SNPs that map to no gene (`model.py:272-274`, `284-286`). It is enabled only if some input row of the first mask is all zero (`model.py:264-265`), which the SNP → gene mask never has (`data_prep.py:1269-1270`).
- Other structure: none.
- Defined at: private `deepconcept.model.biognn.make_binn_model` (imported at `model.py:36`) and `deepconcept.model.model.FCN` (`model.py:35`). The source was not available.

## Connectivity mechanism

The training code passes a 0/1 mask of shape [inputs, genes] (`data_prep.py:1270`) to `make_binn_model(input_dim=…, sparsity_mask=…, pathway_model_class=FCN, …)` (`model.py:278-286`). The code that applies it is private. The ONNX export shows how it is applied. For each gene, one mask column is stored as a constant of shape (20820,) (1,041 initializers of that shape, the inputs of the `/core/NonZero*` nodes). The graph runs `NonZero` → `Transpose` → `Squeeze` to get the gene's input indices, `Gather`s those columns of `x`, and feeds them to that gene's own `Gemm` (`/core/Gather` → `/core/pathway_nets.0/network/network.0/Gemm`, and so on for each gene). So this is gather by index with one small dense network per node. No dense [inputs, genes] weight exists. A SNP that is absent for a gene has no weight in that gene's network and gets no gradient through it. A present SNP → gene edge carries 128 weights, one per hidden unit of the gene's network. Initialization is unclear (private code).

## Interpretation

Paper only; no code in the archive. A search of the source for `sensitiv|clamp|perturb|shap|attribut` found nothing. Methods 4.3 and Algorithm 1 define the method: for each trained model, compute each gene latent's mean μ_j and standard deviation σ_j over test samples. Clamp the latent to μ_j ± a·σ_j for all samples. Score Δy_j = ½(|ŷ⁺ − ŷ₀| + |ŷ⁻ − ŷ₀|), where each ŷ is the mean prediction (eq. 9-10). Average Δy_j over the models that contain gene j (5 outer splits × 5 inner folds, eq. 11) and rank genes. The paper also correlates gene latents with measured expression (Fig. 2e). Granularity: hidden node (gene). Names come from the gene-list order (`gene_name_list_*.txt`, `data_prep.py:1007`).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `data_prep.py:1022-1038` | Each gene takes the SNPs named in its eQTL table; a SNP shared by genes feeds each of them; unselected SNPs are dropped. |
| N09 Dense head after the graph layers | `model.py:283`; `ONNX` `/core/final_net/*` | Gene latents go through Linear(1041→128), ReLU, Linear(128→1). |
| N54 Private subnetwork per node | `model.py:268-270`; `ONNX` `core.pathway_nets.j.network.{0,3}` | Each gene is its own 20→128→1 sigmoid MLP that emits one latent. |
| N55 Supervise node states with measured values | `model.py:148-150`; `data_prep.py:1043-1055`, `1240-1247` | Optional loss ties gene latents to measured expression, matched by gene name, weighted by `lam`. |
| N56 Derive the graph from training data | `utils.py:99-124`, `183-217`; `data_prep.py:1028`, `1420-1436` | Genes come from an elastic net and SNP links from an eQTL scan, run per outer split. |
| N57 Dense branch for features outside the graph | `model.py:264-265`, `272-274`, `284-286` | A residual FCN over SNPs that map to no gene; built only when such SNPs exist. |
| N58 Gather each node's inputs by index | `ONNX` `/core/NonZero*`, `/core/Gather*`; `data_prep.py:1270` | Each gene gathers its SNP columns from the nonzero entries of its mask column. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Select genes per split with an elastic net | `utils.py:99-124`; wrappers at `data_prep.py:36`, `212`, `506` (not read in full) | ~26 + wrappers |
| Run the eQTL scan through an external R script | `utils.py:183-217`; callers `data_prep.py:654-762` | ~35 + caller |
| Read per-gene marker tables keyed by gene | `data_prep.py:850-872` | ~23 |
| Build the input matrix as one column block per gene | `data_prep.py:1006-1057` | ~50 |
| Build the SNP → gene mask by position | `data_prep.py:1263-1271` | ~9 |
| Build an optional gene → module mask | `data_prep.py:1281-1346` | ~65 |
| Carry measured expression per gene into each batch | `data_prep.py:1043-1055`, `1233-1247` | ~25 |
| Pass masks to the model; switch the residual branch | `model.py:259-286` | ~28 |
| Rebuild inputs per split and check width at inference | notebook cell 10 | ~30 of ~110 |

## Fragile spots

- The SNP → gene mask comes from column position alone: `group_idx = (np.arange(D) * K) // D` (`data_prep.py:1269`). It assumes every gene contributes exactly `D / K` columns, in the same order as the measured-expression dict. Column blocks are built from genes that have a marker table (`data_prep.py:1008`), each block keeps up to `num_snps` rows (`data_prep.py:1028`), and `K` counts genes with expression (`data_prep.py:1047-1051`, `1268`). A gene with fewer than 20 markers, with no marker table, or with no expression shifts every later block against the mask, and no name check catches it. The assertion at `data_prep.py:1055` compares the expression dict with the gene list, not with the input columns.
- In the gene → module mask, genes missing from the co-expression file are assigned to a random module (`data_prep.py:1319-1327`); this is only printed.
- The residual branch from the paper is switched on by an all-zero mask row (`model.py:264-265`), and the position-based mask never has one. In this setting, the residual is always absent.
- At inference the notebook rebuilds the input from the data and checks only the width against `metadata.json` (notebook cell 10, `meta_dim != X_test.shape[1]`), not the column names or gene order.

## Out of scope

- Loss: MSE for the shipped models; `BinnSoftLoss` / `BinnHardLoss` are private (`model.py:37`).
- Training loop: private `train_model` with Adam, `lr` 3e-3, `l1_reg` 1e-5, `l2_reg` 1e-5, patience 20 (`model.py:309-323`).
- Data splitting: per-population train/val/test splits and line lists (`data_prep.py:1128-1231`).
- Baselines: ridge regression on genotype (G2P) and on expression (E2P) (`train_G2P`, `train_E2P`, `train_G2P_and_E2P_models` at `model.py:733`, `925`, `1119`).
- Hyperparameter search: `train_binn_tuning` (`model.py:43-179`).
- Ensemble prediction: mean and standard deviation over split × fold ONNX models (notebook cell 10).
- Plotting and t-tests: `utils.py:338-1254`.
- Co-expression modules via WGCNA: `utils.py:219`, `data_prep.py:764`.

## Paper vs code

- Paper eq. 3 adds a residual network on SNPs outside every mask. In code it exists only when some SNP maps to no gene (`model.py:264-265`), which never happens for the eQTL mask, so the shipped models have no residual (`ONNX` has only `pathway_nets` and `final_net`).
- The paper describes masks as feature-to-entity maps (eq. 1). The code derives the input mask from column position, not from names (`data_prep.py:1269`).
- The paper's soft-constrained loss (eq. 7-8) is optional. The shipped maize models are labeled `BINN_MSE`.
- The paper says subnetworks "may have multiple hidden layers". The shipped models have one hidden layer of 128 units (`metadata.json`, `ONNX`).
- The paper's metabolomics/ODE case (Results 2.2) and its masks have no code in the archive.
- The paper's Code Availability says code is "available upon request" for non-commercial use. The later Zenodo record (2026-04-01, CC-BY-4.0) ships inference and data-preparation code but not the model, loss, training, or sensitivity code.

## Key excerpts

`src_binn/data_prep.py:1022-1038` (Zenodo 17477083, CC-BY-4.0)

```python
        for key in keys_to_use:
            sel_markers_df = dfs_sel_markers[key]

            if num_snps is None:
                snp_list = sel_markers_df['SNP'].tolist()
            else:
                snp_list = sel_markers_df['SNP'].iloc[:num_snps].tolist()

            sum_markers += len(snp_list)
            #print(len(snp_list), 'markers selected for', key)
            counter.update(snp_list)
            sub = df_snp_idx[snp_list].copy()
            sub.columns = [f"{key}_{snp}" for snp in snp_list]
            blocks.append(sub)

        # 3) Concatenate side-by-side
        df_G = pd.concat(blocks, axis=1)
```

`src_binn/data_prep.py:1263-1271`

```python
    # Build sparsity masks 
    sparsity_masks: list = []

    # 1) inputs → genes (unchanged logic)
    D = X.shape[1]
    K = len(intermediates)  # number of genes (keep existing dict order)
    group_idx = (np.arange(D) * K) // D  # ranges 0..K-1
    mask_inputs_to_genes = np.eye(K, dtype=int)[group_idx]  # shape [D, K]
    sparsity_masks.append(mask_inputs_to_genes)
```

`src_binn/data_prep.py:1319-1327`

```python
        for i, g in enumerate(gene_names):
            m = gene_to_module.get(g)
            if m is None:
                missing_genes.append(g)
                # Assign randomly to a module
                j = rng.randint(0, M)
            else:
                j = module_to_col[m]
            mask_genes_to_modules[i, j] = 1
```

## Open questions

- How `make_binn_model` builds subnetworks, initializes weights, and stacks a second mask. It lives in private `deepconcept.model.biognn`. Looked in: archive listing, PyPI, GitHub repository and code search.
- The exact form of `BinnSoftLoss`, for example one correlation per gene averaged or one pooled per layer. Eq. 7 in the paper does not settle it. It is private (`deepconcept.model.loss`).
- How the eQTL tables rank SNPs, which decides the "first 20". The R script is not shipped. Reading one marker table failed (Zenodo HTTP 504).
- Whether co-expression (stacked) masks were used for any reported result. The code path exists (`data_prep.py:1281-1346`); the paper points to its Supplementary Material, which was not read.
- What module index 2 inside each gene subnetwork is (absent from the ONNX export).
- Sensitivity analysis code: not in the archive.
