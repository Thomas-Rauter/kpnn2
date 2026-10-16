# drug-toxicity-2022: DTox (deep learning for toxicology)

| Field | Value |
|-------|-------|
| Paper | Hao Y, Romano JD, Moore JH, 2022. Knowledge-guided deep learning models of drug toxicity improve interpretation. Patterns 3(9):100565. 10.1016/j.patter.2022.100565 |
| Code | `https://github.com/yhao-compbio/DTox` at `5adf0cc1fc7cd1abaa8cacae82fd2c133e2ee984`. The graph files DTox reads are built by the authors' `https://github.com/yhao-compbio/ontology` at `7c464d21a93ee950190ede5205ada0db4f3bb530`; citations into that repository start with `ontology:` |
| Other code | `EpistasisLab/DTox` at `10c909b` (later tutorial; its `code/dtox_nn.py`, `code/dtox_lrp.py`, `code/dtox_hierarchy.py` are byte-identical to the official ones); Zenodo 10.5281/zenodo.6808324 (deposit, not fetched); `yhao-compbio/tox_data` (processed data, not fetched); `yhao-compbio/TTox` (makes the input features and the shuffled data sets, not fetched) |
| Framework | Python, PyTorch (not pinned in the official repository; the tutorial's `environment.yml` pins python 3.7.3, torch 1.10.1, pandas 1.3.3, statsmodels 0.9.0); graph construction in R |
| License | GPL-3.0 (`LICENSE.md`; same in `ontology`) |
| Full text | Europe PMC full-text XML of PMC9481960 (methods, Figure 1 caption, code availability) |
| Extracted | 2026-10-16, Claude Code, claude-opus-5-5 |

## Summary

DTox predicts whether a compound is active in one Tox21 toxicity assay. There are 15 assays, one binary model each. The input is 361 features: the probabilities that the compound binds each of 361 target proteins, which the authors' earlier TargetTox random forests infer from its MACCS fingerprint. Each input protein joins the lowest-level Reactome pathways it is annotated to. Pathways join their parent pathways up to a chosen set of top-level Reactome processes (any combination of gene expression, immune system, metabolism, and signal transduction), and those feed one sigmoid output unit. Each pathway is a block of 1–20 ReLU units, sized by the log of its protein count, and each block has its own sigmoid auxiliary head trained on the same label. For interpretation, LRP runs from the output to every unit, unit scores sum into pathway scores, and every root-to-protein path is scored. A path counts as significant for a compound when it beats models retrained on shuffled labels.

## Prior-knowledge graph

- Source: Reactome, downloaded August 2019 (paper). The parent–child pairs come from `ReactomePathwaysRelation.txt`, the protein-to-pathway annotations at every level from `UniProt2Reactome_All_Levels.txt`, and there is a list of roots (`ontology:src/run/run_get_knowledge_hierarchy_reactome.R:13-16`, `ontology:src/generate_root_file.R:11-26`). Annotations are kept for Homo sapiens only, with isoform suffixes cut from the UniProt accessions and duplicates removed (`ontology:src/get_knowledge_hierarchy.R:28-33`). Filtering runs in three steps:
  1. Drop pathways with fewer than `min_size` annotated proteins, counting every Reactome protein, not only the inputs. A relation row survives only when its parent and child both pass. Then keep only descendants of the chosen roots (`ontology:src/functions.R:53-77`).
  2. Call "lowest level" the pathways that, after step 1, appear as children and never as parents (`ontology:src/get_knowledge_hierarchy.R:57-61`).
  3. Keep the lowest-level pathways that have at least one input protein, plus all their ancestors (`ontology:src/functions.R:180-204`).

  `min_size` is tuned over 5 and 20; 10 is built but excluded (`ontology:src/run/run_get_knowledge_hierarchy_reactome.R:39`, `src/run/run_dtox_implementation.R:47-49`). The roots come in all 15 non-empty combinations of the four processes, each with its own hierarchy files (`ontology:src/generate_root_file.R:19-26`). Two optional variants go unused: Reactome reactions below the lowest pathways (`re_1`) are excluded from the runs (`src/run/run_dtox_implementation.R:50-52`), and the MACCS-bit layer below the proteins (`st_1`) is not in the paper runs, which read `_st_0_` files (`src/run/dtox_compound_target_probability_tox21_implementation1.sh:1`).
- Size: 361 input proteins. An optimal model has on average 412 pathway nodes and 45,623 parameters (paper, Results). Edge counts not found. The graph is directed, unsigned, and acyclic. A pathway can have several parents, because Reactome is a DAG.
- Data to nodes: inputs are matched to pathways by UniProt accession (`ontology:src/functions.R:182-185, 220-223`). Inputs join only lowest-level pathways, and inner pathways read only child pathways (`ontology:src/functions.R:218-237`). A protein in no kept lowest-level pathway stays in the input layer with no outgoing edge: every feature is layer 0 (`ontology:src/get_knowledge_hierarchy.R:68-71`), and LRP gives it relevance 0 (`src/dtox_lrp.py:233-235`). Step 3 drops pathways with no input below them. In the network, data column `i` feeds input node `i` by position, with no name check (`src/dtox_nn.py:89-91`, `src/dtox_data.py:51`).
- Built at: `ontology:src/get_knowledge_hierarchy.R:22-123`, with helpers at `ontology:src/functions.R:46-351`. It writes the `_layer`, `_node`, `_knowledge_by_name`, `_knowledge_by_node`, `_node_size`, and `_root` TSV files (`ontology:src/get_knowledge_hierarchy.R:111-123`), which `src/dtox_hierarchy.py:103-134` reads.

## Architecture

- Layers: layer 0 holds every input feature and layer 1 the lowest-level pathways. Each later layer holds the parents whose children all sit in earlier layers, which makes a node's layer its height above the data (`ontology:src/functions.R:250-274`). Nodes are numbered in layer order (`ontology:src/functions.R:276-287`). The forward pass ignores layers: it computes nodes one by one in number order, each from the concatenated outputs of its children (`src/dtox_nn.py:93-102`). The layer index only weights each node's auxiliary loss (`src/dtox_hierarchy.py:82-87`, `src/dtox_loss.py:29-31`).
- Units per node: each input is 1 column. Pathway p gets round(1 + (N_max − 1)·log(S_p/S_min)/log(S_max/S_min)) ReLU units, where:
  - N_max = 20;
  - S_p is the number of Reactome proteins annotated to p;
  - S_min is the `min_size` threshold;
  - S_max is the largest size in the hierarchy.

  A node with S_p < S_min gets 1 unit; that covers the inputs, whose size defaults to 1 (`ontology:src/functions.R:338-340`, `src/dtox_hierarchy.py:28-40`). A node's input width is the sum of its children's unit counts (`src/dtox_hierarchy.py:50-61`). An edge joins every unit of the child block to every unit of the parent block.
- Connections that skip layers: yes. A parent reads children from any lower layer. Layers come from height, so a child can sit several layers down; the child lists alone build this (`src/dtox_nn.py:95-97`). Root blocks sit in different layers and are concatenated into the output unit (`src/dtox_nn.py:104-107`).
- Outputs or losses at inner layers: every pathway block, roots included, has a sigmoid head Linear(N_p, 1) (`src/dtox_nn.py:33-49, 73-74, 102`). The loss is BCE(output) + α·Σ_p BCE(head_p)/n_layer(p), where n_layer(p) is the number of nodes in p's layer (`src/dtox_loss.py:25-34`), and α ∈ {0.1, 0.5, 1} (paper). The heads do not feed the prediction.
- Parts the graph does not constrain: the output unit, Linear(sum of root units, 1) plus sigmoid over the concatenated roots (`src/dtox_nn.py:76, 104-107`). The input features come from a separate model (TargetTox) that is not trained with DTox.
- Other structure: none (no dropout, normalization, recurrence, or attention). The baseline is a fully connected MLP with one dense layer per DTox hidden layer, each as wide as that layer's total units (`src/mlp/mlp.py:34-45`, `src/mlp/mlp_learning.py:52-72`, `src/dtox_hierarchy.py:88-90`).
- Defined at: `src/dtox_nn.py:12-108`; sizes at `src/dtox_hierarchy.py:12-66`.

## Connectivity mechanism

Each pathway node is one `nn.Linear(M_in, M_out)` followed by ReLU, where M_in is the summed width of that node's children (`src/dtox_nn.py:12-29, 68-71`). The forward pass looks up the children's output tensors by the stored child-index list and concatenates them (`src/dtox_nn.py:95-100`). No full weight matrix exists, so absent edges have no parameter and no gradient. Child lists are stored sorted by node number (`ontology:src/functions.R:312-317`), and LRP slices relevance back to the children in the same order (`src/dtox_lrp.py:249-256`). Initialization is PyTorch's default for each `nn.Linear`, so the scale comes from that node's own present fan-in (`src/dtox_nn.py:20`); no custom init found. Training uses Adam (lr 0.001, batch 32) with `weight_decay` = λ = 1e-4 on every parameter, biases, heads, and output included (`src/dtox_learning.py:22, 50`, `src/dtox_data.py:39`). It stops early on the test total loss (patience 20, at most 200 epochs; `src/dtox_learning.py:63-114`). The seed is fixed with `torch.manual_seed(0)` (`src/dtox.py:42`).

## Interpretation

- Method: LRP on the trained model, for active compounds (label 1) only (`src/interpret_dtox.py:48-57`). Relevance starts as the sigmoid output value (`src/dtox_lrp.py:216, 226`), and nodes are visited in reverse number order (`src/dtox_lrp.py:229`). The rules:
  - Generic rule, for the output unit and for a node whose first child is a pathway: z = a·(w + γ·w⁺)ᵀ, and relevance is divided by z + ε·std(z). The std runs over all samples and units of that node, and there is no bias term (`src/dtox_lrp.py:70-100`). γ and ε are each tuned over {0.001, 0.01, 0.1}; results are reported at γ = 0.001, ε = 0.1 (paper).
  - Bounded-input rule, for a node whose first child is an input, with l = 0 and h = 1 (`src/dtox_lrp.py:139-171`, `src/interpret_dtox.py:57`).
  - An αβ rule exists as an option (`src/dtox_lrp.py:104-135`).

  A zero denominator is smoothed by adding 0.001 × the smallest nonzero absolute value to whole rows of activations and whole columns of weights (`src/dtox_lrp.py:16-66`). A child's relevance is the sum over its parents (`src/dtox_lrp.py:231-232`).
- Granularity: per sample, per unit (named `<node>_<unit>`) and per node (the sum over its units), inputs included (`src/dtox_lrp.py:237, 260-268`). Every path from each root down to an input is enumerated (`src/dtox_lrp.py:281-305`). A path scores Σ log10(max(R_node, 0)) over its nodes, input and root included, so one nonpositive node makes the score −inf (`src/dtox_lrp.py:309-321`).
- Names: path ids are node numbers joined as `a_b_c`, and they are renamed with the node names from the `_node.tsv` file (`src/interpret_dtox.py:68-80, 101-105`). Node-level relevance columns carry node numbers only (`src/dtox_lrp.py:268`).
- Significance: the same LRP and path scores run on every label-shuffled null model (200 in the paper), with the same structure and query compounds (`src/interpret_dtox.py:84-98`). For each compound–path pair, p = #(null ≥ observed)/#null. BH-FDR runs over all pairs pooled, and pairs with FDR < 0.05 are kept (`src/dtox_lrp.py:327-361`).
- Repeats: one trained model per assay and setting (seed 0); no averaging across seeds.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `ontology:src/get_knowledge_hierarchy.R:25`, `ontology:src/functions.R:46-80` | Reactome parent–child pathway pairs form the hierarchy. |
| N05 Filter nodes by gene-set size | `ontology:src/functions.R:53-61` | Pathways with fewer than 5 or 20 annotated proteins (tuned) are dropped with their edges. |
| N06 Connect features to nodes by membership | `ontology:src/functions.R:218-226` | Input proteins join pathways through UniProt-to-Reactome annotations matched by accession. |
| N20 Keep only nodes above the data | `ontology:src/functions.R:180-204` | Only lowest-level pathways with an input protein, and their ancestors, are kept. |
| N131 Keep the subgraph below chosen roots | `ontology:src/functions.R:63-77`, `ontology:src/generate_root_file.R:19-26` | Each of 15 root combinations gives its own hierarchy, and the combination is a tuned hyperparameter. |
| N132 Features join only childless nodes | `ontology:src/get_knowledge_hierarchy.R:57-61`, `ontology:src/functions.R:218-237` | Proteins connect only to pathways with no child pathway; inner pathways read child pathways only. |
| N74 Layer nodes by height above the data | `ontology:src/functions.R:250-274`, `src/dtox_loss.py:29-31` | A pathway sits one layer above its highest child, and its layer's node count sets its auxiliary-loss weight. |
| N26 Unlayered DAG in topological order | `src/dtox_nn.py:93-102` | Nodes are computed one at a time in layer-sorted order, each reading children from any lower layer. |
| N104 Units per node set by gene-set size | `src/dtox_hierarchy.py:28-40` | Each pathway gets 1–20 units from the log of its Reactome protein count. |
| N12 Parameters only for present edges | `src/dtox_nn.py:68-71` | Each pathway's weight matrix spans only its children's units. |
| N58 Gather each node's inputs by index | `src/dtox_nn.py:95-100` | Children's outputs are picked by a stored index list and concatenated before the node's own Linear. |
| N70 Initialize from each unit's present fan-in | `src/dtox_nn.py:20` | Implicit: PyTorch's default init of each per-node Linear scales by that node's present fan-in. |
| N40 Auxiliary output head on every node | `src/dtox_nn.py:73-74, 102`, `src/dtox_loss.py:25-34` | Every pathway block has a sigmoid head whose BCE term is weighted α/(nodes in its layer). |
| N133 One output over root nodes of any depth | `src/dtox_nn.py:76, 104-107` | The chosen top-level processes sit in different layers and are concatenated into one sigmoid output. |
| N115 Dense twin of the graph layer | `src/mlp/mlp.py:34-45`, `src/mlp/mlp_learning.py:52-72` | The baseline is a dense MLP whose layers are as wide as the matched DTox layers' total units. |
| N35 Randomized prior as a control | `ontology:src/shuffle_hierarchy.R:28-46`, `src/run/run_dtox_shuffle.R:39-45` | Each parent's children are redrawn from the same layers, keeping the per-layer counts. |
| N105 Layer-wise relevance propagation | `src/dtox_lrp.py:70-100, 139-171, 229-256` | A γ/ε rule runs between blocks and a bounded-input rule into the proteins; a node's score is the sum over its units. |
| N17 Map scores to node names | `src/interpret_dtox.py:68-80` | Path ids made of node numbers are renamed with node names. |
| N134 Score every root-to-input path | `src/dtox_lrp.py:281-321` | Every root-to-protein path gets Σ log10 of its nodes' clipped relevance, per compound. |
| N101 Null scores from shuffled-label runs | `src/interpret_dtox.py:84-98` | Path scores from models retrained on shuffled labels form the null. |
| N135 Empirical p-values from null runs | `src/dtox_lrp.py:336-359` | p is the share of null scores ≥ the observed one; BH-FDR runs over all pairs, and FDR < 0.05 is kept. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Filter pathways by size and by descent from the chosen roots | `ontology:src/functions.R:46-80` | ~35 |
| Keep pathways above the input proteins | `ontology:src/functions.R:173-207` | ~35 |
| Build input→lowest-pathway and child→parent lists | `ontology:src/functions.R:211-240` | ~30 |
| Layer nodes by height and number them | `ontology:src/functions.R:244-290` | ~45 |
| Convert names to node numbers | `ontology:src/functions.R:294-323` | ~30 |
| Node sizes and root ids | `ontology:src/functions.R:327-351` | ~25 |
| Hierarchy driver and file output | `ontology:src/get_knowledge_hierarchy.R:22-123` | ~100 |
| Root combinations | `ontology:src/generate_root_file.R:15-39` | ~25 |
| Read hierarchy files; units and input widths per node; layer sizes | `src/dtox_hierarchy.py:12-134` | ~125 |
| Node-by-node forward pass over child lists | `src/dtox_nn.py:84-108` | ~25 |
| Layer-wise auxiliary loss weights | `src/dtox_loss.py:25-34` | ~10 |
| LRP across node blocks, with child slicing | `src/dtox_lrp.py:174-270` | ~95 |
| Path enumeration and path scores | `src/dtox_lrp.py:273-323` | ~50 |
| Empirical p-values and FDR | `src/dtox_lrp.py:326-361` | ~35 |
| Map path ids to node names | `src/interpret_dtox.py:65-80, 101-105` | ~20 |
| Shuffled hierarchy control | `ontology:src/shuffle_hierarchy.R:16-50` | ~35 |

## Fragile spots

- Data columns feed input nodes by position (`src/dtox_nn.py:89-91`, `src/dtox_data.py:51`). The column order must equal the input order in `_node.tsv`, and nothing checks it.
- `src/dtox_hierarchy.py:50-61` uses row `i` of the merged size and layer table as node number `i` for `node_children[i]` and `module_size[i]`. That holds only while the `_node_size` and `_layer` files list every node in node order, as `ontology:src/functions.R:282, 342` writes them.
- At `ontology:src/functions.R:314`, a child name missing from the node map turns into NA, and `sort()` drops it, so the edge vanishes without a message.
- `src/dtox_lrp.py:241` picks the input-layer rule from a node's first (lowest-numbered) child alone. It assumes a node's children are all inputs or all pathways, which the construction at `ontology:src/functions.R:218-237` guarantees.
- `src/dtox_loss.py:30-31` pairs head `k` with the `k`-th entry of the layer-size array from `src/dtox_hierarchy.py:81-87`. That pairing relies on the pandas merge keeping node order.
- The layering loop at `ontology:src/functions.R:259-274` has no exit when a pathway can never be placed, for example under a cycle in the relation table.
- Zero-denominator smoothing at `src/dtox_lrp.py:52-64` shifts whole rows of activations and columns of weights, so relevance is not exactly conserved.
- The shuffled hierarchy redraws children from all nodes of a layer. Layer 0 includes inputs that have no edge. Node sizes, layers, and roots stay from the original files (`ontology:src/shuffle_hierarchy.R:30-46`, `src/run/run_dtox_shuffle.R:39-45`), so a node can end up with no parent while its auxiliary head still trains.

## Out of scope

- Input features: TargetTox random forests on MACCS fingerprints (separate repository).
- Training loop, early stopping (`src/early_stop.py`), and the grid search over `min_size` × roots × α (90 settings).
- Splits: 4:1 learning/validation, made outside this repository; 7:1 train/test, stratified, seed 0 (`src/dtox_data.py:54`).
- Baselines: random forest and gradient boosting (`src/simple/`), LIME, and read-across.
- Validation analyses in R (LINCS expression, standard pathway–receptor patterns, viability, DILI): `src/analysis_*`.
- Virtual screening of about 700,000 compounds: `src/predict_dtox.py`.
- Plotting.

## Paper vs code

- Unit-count formula: the paper defines S_min as the smallest pathway size in the network. The code uses the `min_size` threshold (`src/dtox_hierarchy.py:36`).
- L2: the paper puts λ‖W‖² in the loss. The code uses Adam `weight_decay` on every parameter, biases and heads included (`src/dtox_learning.py:50`).
- Generic LRP rule: the paper writes ε·SD[...]. The code uses ε times the std of the node's z over batch and units, and drops the bias (`src/dtox_lrp.py:91-92`). The paper says relevance is redistributed "in equal amounts"; the code redistributes it in proportion to each rule's contributions.
- FDR: the paper reports significant paths "for each active compound". The code corrects across all compound–path pairs at once (`src/dtox_lrp.py:353-356`).

## Key excerpts

`src/dtox_nn.py:93-107`

```python
		for scs in range(self.input_size, self.combine_size):
			# obtain input values for the current node  
			scs_children = self.node_children[scs]
			scs_children_output_list = [layer_result[sc] for sc in scs_children]
			scs_children_output = torch.cat(scs_children_output_list, 1)
			# feed input values into root loss-oriented structure to compute node output 
			scs_net_id = scs - self.input_size
			layer_result[scs] = self.net[scs_net_id](scs_children_output)
			# feed input values into auxiliary loss-oriented structure to compute auxiliary node output
			auxiliary_result[scs_net_id] = self.auxiliary[scs_net_id](layer_result[scs])
		# obtain values for root node  
		root_output_list = [layer_result[srn] for srn in self.root_node]
		root_output = torch.cat(root_output_list, 1)
		# feed values into root structure to compute final output   
		y_pred = self.output_layer(root_output)
```

`src/dtox_hierarchy.py:28-40`

```python
	for nsv in node_size_vec:
		# for gene modules, each module contains 1 neuron  
		if nsv < min_path:
			module_size.append(1)
		# for pathway modules, the number of neurons varies between 1 and 'max_module', depends on the pathway size 
		else:
			# adjust pathway size by min/max in log scale
			if log_scale:
				nsv_module = 1 + (max_module - 1) * np.log(nsv/min_path)/np.log(max_node_size/min_path)
			# adjust pathway size by min/max
			else:
				nsv_module = 1 + (max_module - 1) * (nsv-min_path)/(max_node_size-min_path)	 
			module_size.append(int(round(nsv_module)))
```

`ontology:src/functions.R:259-273`

```r
	while(length(sorted_paths) != length(all_paths)){
		# identify all parents of current layer pathways
		c_id <- which(relation_mat[,2] %in% path_layer[[i]])
		parent_paths <- unique(relation_mat[c_id, 1])
		# check if all children of identified parent pathways have been sorted in the previous layers
		parent_path_id <- sapply(path_children[parent_paths], function(pcpp){
			pcpp_diff <- setdiff(pcpp, sorted_paths)
			if(length(pcpp_diff) == 0)	return(1)
			else	return(0)
		})
		# only assign the parent pathways whose children have been all sorted to the next available layer  
		i <- i + 1
		path_layer[[i]] <- parent_paths[parent_path_id == 1]
		# update pathways that have been sorted 
		sorted_paths <- union(sorted_paths, path_layer[[i]])
```

`src/dtox_loss.py:29-33`

```python
		auxi_loss = 0
		for lapy in range(0, len(auxiliary_pred_y)):
			auxi_loss += 1/float(self.layer_size[lapy]) * self.BCELoss(auxiliary_pred_y[lapy], y) 
		# multiply auxiliary loss by coefficient, add two parts together 
		total_loss = root_loss + self.alpha * auxi_loss
```

`src/dtox_lrp.py:309-317`

```python
	module_relevance_df1 = module_relevance_df.copy()
	module_relevance_df1[module_relevance_df1 < 0] = 0
	# perform log10 transformation  
	log_r_df = np.log10(module_relevance_df1) 
	# iterate by path, compute observed relevance score of each path  
	path_score_list = []
	for pa in path:
		# sum the log10-transformed relevance score of all nodes in the path
		pa_score = log_r_df.iloc[:, pa].sum(axis = 1)
```

## Open questions

- Layer and edge counts per optimal model: these sit in data files under `data/` and the `ontology` hierarchy outputs, which were not read.
- How the labels (null models) and the input values (feature-shuffle control) were shuffled: the data comes from TTox folders (`src/run/run_dtox_null.R:11`, `src/run/run_dtox_feature_shuffle.R:11`), whose code was not fetched.
- Whether the data columns follow the input order of `_node.tsv`: the data is built in TTox and was not checked.
- The number of null models used: the code reads it from a list file (`src/interpret_dtox.py:84`) that was not read. The paper says 200.
- The root list `downloads/reactome/root_pathways.txt` in `ontology` is a data file and was not read. The four process names come from the paper.
