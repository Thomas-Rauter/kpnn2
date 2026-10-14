# graphgonet-2022: GraphGONet

| Field | Value |
|-------|-------|
| Paper | Bourgeais V, Zehraoui F, Hanczar B, 2022. GraphGONet: a self-explaining neural network encapsulating the Gene Ontology graph for phenotype prediction on gene expression. Bioinformatics 38(9):2504-2511. 10.1093/bioinformatics/btac147 |
| Code | `https://forge.ibisc.univ-evry.fr/vbourgeais/GraphGONet.git` at `8df7b1488e567f4aa6a10e52e05ab93a65d86d3f` (commit date 2024-08-20, after publication) |
| Other code | none found (a web search for a GitHub mirror found only this forge repository) |
| Framework | Python 3.6.7, PyTorch 1.7.1, PyTorch Geometric 1.6.3, torch-scatter 2.0.5, networkx 2.5, obonet 0.2.6, captum 0.3.1 (`requirements.txt:1-22`); R/Bioconductor through rpy2 for annotations |
| License | Apache 2.0 (`LICENSE:1-2`); the README badge says MIT (`README.md:5`) |
| Full text | Bioinformatics landing page (open access): abstract, Methods, Fig. 1 caption, code availability. Supplementary Information (graph preprocessing, Table S3) not read |
| Extracted | 2026-10-14, Claude Code, claude-opus-5-5 |

## Summary

GraphGONet classifies gene expression samples (TCGA, 12 tumor types;
microarray, cancer vs non-cancer). The prior is the GO Biological Process
DAG with gene annotations, one unit per term. A masked gene-to-term layer
gives each annotated term its own weighted gene sum. Terms are computed
from the most specific upward as tanh(w_G·gene sum + w_N·mean of children),
with w_G and w_N shared by all terms. A per-sample top-k by absolute value
keeps a few terms for one linear output layer; a term's score is its
activation times its output weight.

## Prior-knowledge graph

- Source: `go-basic.obo`, downloaded at run time, read with obonet as a
  child→parent multigraph (`notebooks/1-Build_GONet_graph_part2.ipynb:180`).
  MF and CC are deleted below their roots (`part2.ipynb:1544-1827`). No
  relation-type filter found; is_a, part_of, regulates stay, unsigned
  (`scripts/utils.py:110-111`). Annotations: Bioconductor
  `AnnotationDbi.select` on `org.Hs.eg.db` (Ensembl, TCGA) or
  `hgu133plus2.db` (probes), NA rows dropped, BP only
  (`notebooks/1-Build_GONet_graph_part1.ipynb:421-422,1499,1669,1685`);
  evidence codes ignored (`part2.ipynb` cell 30).
- Pruning: childless BP terms without genes go, then their ancestors with
  no gene and no annotated descendant (`part2.ipynb:1681,1966-1977,2078`).
  This is the "entire" graph (`:2209`).
- Most specific annotations: a gene's annotation to a term is dropped when
  the gene is also annotated to a descendant (`part2.ipynb:6429-6441`),
  after duplicates and terms outside the graph go (`:6309,6349`).
- "Truncated" graph, the default (`scripts/GraphGONet.py:474-479`): every
  childless term is removed once (`part3.ipynb:410,1764`); a term that
  becomes childless gets its descendants' genes plus its own
  (`:1897-1931`); other terms keep only their own (`:1869`).
- Size (README flags): TCGA 10,636 terms, 8,288 with genes, 18,427 genes
  (`README.md:51`); microarray 10,663, 8,249, 36,834 probes (`:46`,
  commented out). Edge count not determined. Acyclic (checked,
  `part2.ipynb` cells 23, 62, 77), unsigned, several parents per term.
- Data to nodes: a 0/1 gene × term table, rows in dataset order
  (`part2.ipynb:6884-6938`). Genes with no kept annotation are dropped by
  a saved column mask (`:6562-6583`) applied to every split
  (`scripts/utils.py:57-58`, `scripts/GraphGONet.py:32-38`). Terms without
  genes get a zero gene input (`scripts/base_model.py:438-442`). Terms are
  relabeled to integers, annotated first, in table-column order
  (`part2.ipynb:7981-7994,9052`; `part3.ipynb:3001-3016,4080`); the map is
  saved as JSON (`part3.ipynb` cell 53).
- Built at: `part1.ipynb`, `part2.ipynb`, `part3.ipynb`; edge index in
  `scripts/utils.py:99-119`.

## Architecture

- Layers: none stored. Each forward pass starts at the childless terms
  (`scripts/base_model.py:89-100`). It then computes, wave by wave, every
  term whose children are all done (`scripts/base_model.py:120-173`). A
  term's wave is its longest path down to a childless term. A
  level-from-root table is written (`part2.ipynb:5476,5704`) but the model
  never reads it.
- Units per node: one. The gene layer gives each annotated term one value
  h_G(v) with its own bias (`scripts/base_model.py:404`). Propagation has
  one channel (`scripts/GraphGONet.py:493-496`; children flattened to
  scalars at `scripts/base_model.py:165`).
- Connections that skip layers: implicit. A term averages all its
  children, whatever their wave, and genes enter annotated terms at any
  depth (`scripts/base_model.py:105-110,161-171`).
- Outputs or losses at inner layers: none; after selection the output
  reads terms of every depth (`scripts/base_model.py:429,453-465`).
- Parts the graph does not constrain: the shared scalars `lin_r`, `lin_l`
  (`scripts/base_model.py:50-52`) and the linear output `fc2` (`:429`).
- Other structure: per-sample top-k of term values by absolute value,
  ratio r, PyG `topk` (`scripts/base_model.py:215`); kept values go back
  to their positions in a zero vector (`:292-312`); random-k is the
  control (`:261-271`). README runs use r = 0.001 (`README.md:51`), i.e.
  11 of 10,636 terms per sample.
- Message-passing part (named, not covered further): term-to-term edges
  have no own weight. All terms share w_N and a bias on the children's
  mean, and w_G on their gene sum; childless terms get tanh(h_G(v))
  (`scripts/base_model.py:99,110,166-171`). The class is "inspired from
  the modules Graph Convolutional Layers from pytorch-geometric" (`:12`).
- Defined at: `scripts/base_model.py:13-175` (propagation), `:347-471`.

## Connectivity mechanism

Gene → term: `fc1` is a dense `Linear(n_genes, n_annot)`
(`scripts/base_model.py:404`). The transposed 0/1 table, a non-trainable
`Parameter` (`:400-401`), is multiplied into the weight once at
construction (`:407-408`) and into its gradient after every backward pass,
before `optimizer.step()` (`scripts/GraphGONet.py:211-213`). The forward
pass never uses it. Absent entries start at 0, get zero gradient, and stay
0 under Adam, RMSprop, or SGD momentum without weight decay (`:154-159`).
Init is PyTorch's default over the full gene count; the bias is not masked.

Term → term: each wave gathers the children of the terms it computes by
the edge index (`scripts/base_model.py:149-153`) and averages them with
`scatter(..., reduce="mean")` (`:162-169`); absent edges are never
gathered, no edge has a parameter. Every sample's graph is a clone
(`scripts/GraphGONet.py:106-109`); the traversal uses the first sample's
edges, offset per sample (`scripts/base_model.py:76-78,118-119,150-159`).

## Interpretation

- Method: Captum `LayerGradientXActivation` on the selection module's
  output (`notebooks/Interpretation.ipynb:4968`), target the predicted
  class for TCGA (`:5032-5034`). With a linear `fc2` this is the paper's
  R_jc = h_select,j · w_jc. Per term and sample; only selected terms are
  nonzero (`:8014`).
- Names: index → GO ID through the saved JSON (`:973,8051`); names from
  goatools `GODag` (`:4205,8587`). A per-sample table lists relevance,
  activation, and output weight (`:8182`).
- Across samples: a forward hook reads the selected activations (`:4997`);
  each kept term's sign is one-hot coded "+"/"−" per sample
  (`:6701-6705`), summed per predicted class, and the 10 most frequent
  terms are plotted (`:6972,7516`, cells 94-101).
- Repeats: one model, one batch with the whole test set (cells 17, 71).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `part2.ipynb:180,1544-1827` | GO is read from OBO as a child→parent DAG and cut to BP by deleting the MF and CC subtrees. |
| N06 Connect features to nodes by membership | `part2.ipynb:6562-6583,6884-6938`; `scripts/utils.py:57-58` | Genes join terms through an annotation table by Ensembl or probe ID. Genes with no kept annotation are dropped. Genes join terms at any depth. |
| N10 One unit per node | `scripts/base_model.py:404`; `scripts/GraphGONet.py:493-496` | Each term is one tanh unit. Its own bias comes from the gene layer (annotated terms only), plus one bias all terms share. |
| N15 Attribution to hidden nodes | `Interpretation.ipynb:4968,5032` | Each selected term gets a relevance score per sample. |
| N17 Map scores to node names | `Interpretation.ipynb:973,8051,8587` | Integer term indices are mapped to GO IDs and names. |
| N20 Keep only nodes above the data | `part2.ipynb:1681,1966-1977,2078` | Childless terms without genes, and ancestors with no annotated descendant, are removed. |
| N21 Propagate memberships up the hierarchy | `part3.ipynb:1918-1931` | After the cut, each term that becomes childless takes the genes of its removed descendants. |
| N26 Unlayered DAG in topological order | `scripts/base_model.py:120-173` | A term is computed once all its children are done, with no stored layers. It reads its genes and children of any depth. |
| N31 Read out node activations | `Interpretation.ipynb:4997,6701` | A forward hook reads the selected term activations, labeled by GO ID. |
| N39 Mask weights at start and gradients each step | `scripts/base_model.py:407-408`; `scripts/GraphGONet.py:211-213` | The gene-to-term weight is masked once at construction and its gradient after every backward pass. |
| N74 Layer nodes by height above the data | `scripts/base_model.py:89-100,120-173` | Waves start at the childless terms. A term's wave is its longest path down to one. |
| N81 Average over incoming edges | `scripts/base_model.py:166-168` | A term's children are averaged: their sum divided by its number of children in the prior. |
| N111 Per-class frequency of top-k active nodes | `Interpretation.ipynb:6701-6705,6972,7516` | Kept terms (top-k by absolute value) are counted per predicted class, split by sign. The 10 most frequent are reported. |
| N121 Keep each feature's most specific memberships | `part2.ipynb:6429-6441` | A gene's annotation to a term is dropped when the gene is also annotated to a descendant. |
| N122 Cut the childless nodes of a DAG | `part3.ipynb:410,1764,1869` | The default network drops every childless GO term once. |
| N123 Shared weights on node-to-node edges | `scripts/base_model.py:50-52,110,166-171` | All term-to-term edges share one weight on the children's mean. Gene-to-term edges have weights of their own. |
| N124 Output layer reads every node | `scripts/base_model.py:292-312,429` | Terms of every depth feed one linear output layer. |
| N125 Per-sample top-k node selection | `scripts/base_model.py:215,292-312` | Per sample, only the ⌈r·n⌉ terms with the largest absolute value reach the output. The rest are zero. |
| N126 Gradient times hidden activation | `Interpretation.ipynb:4968,5032` | A term's relevance is its activation times the gradient of the predicted class's logit. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Query gene-to-GO annotations through rpy2/Bioconductor | `part1.ipynb`, cells 4-35 | ~50 |
| Read GO, keep BP, prune unannotated branches | `part2.ipynb`, cells 15-82 | ~60 |
| Most specific annotations, gene mask, 0/1 table | `part2.ipynb`, cells 113-146 | ~45 |
| Relabel terms to integers, annotated first; save map | `part2.ipynb`, cells 149-156; `part3.ipynb`, cells 49-55 | ~25 |
| Cut childless terms, move genes to new leaves | `part3.ipynb`, cells 25-47 | ~30 |
| networkx → edge index, cloned per sample; gene sums into node slots | `scripts/utils.py:99-119`; `scripts/GraphGONet.py:96-120`; `scripts/base_model.py:436-442` | ~40 |
| Mask gene-to-term weight at init and gradients | `scripts/base_model.py:399-408`; `scripts/GraphGONet.py:211-213` | ~10 |
| Wave-by-wave DAG traversal per forward pass | `scripts/base_model.py:72-175` | ~100 |
| Top-k selection, scattered back to term positions | `scripts/base_model.py:188-312` | ~100 |
| Map integer indices to GO IDs and names | `Interpretation.ipynb`, cells 40-47, 112, 123 | ~15 |

## Fragile spots

- Node alignment is by position: gene-layer outputs fill node slots
  0..n_annot−1 (`scripts/base_model.py:439-442`), right only because the
  notebooks number annotated terms first (`part3.ipynb:3001-3016`). The
  counts `--n_nodes`, `--n_nodes_annotated` are passed by hand and default
  to the microarray values (`scripts/GraphGONet.py:483-492`).
- Truncated graph: only new leaves receive genes (`part3.ipynb:1918-1931`),
  so a gene annotated only to cut terms whose parents keep other children
  gets an all-zero row, yet stays an input, because the gene mask comes
  from the entire graph (`part2.ipynb:6562-6583`).
- Edges of the obonet multigraph are listed without keys
  (`scripts/utils.py:110-111`); a pair joined by two relation types would
  count the child twice in the mean. Not checked for `go-basic.obo`.
- The mask holds only through the training loop's gradient step
  (`scripts/GraphGONet.py:211-213`); weight decay or another loop would
  make absent entries nonzero with no error.
- The interpretation table's weight column is `fc2.weight[0, ...]` (class
  0) while relevance targets the predicted class
  (`Interpretation.ipynb:5032,8182`).
- `part2.ipynb` cell 149 reads `matrix_connection_original`, never defined
  there (`:6962`); cell 141 calls `.remove` on a numpy array
  (`:6552,6640`). The notebook relies on state from elsewhere.

## Out of scope

- Loss: cross-entropy or BCE with logits, optional class weights
  (`scripts/GraphGONet.py:161-174`).
- Training loop, early stopping, checkpoints (`scripts/GraphGONet.py:185-430`,
  `scripts/utils.py:123-208`).
- Loading preprocessed `.npz` splits (`scripts/utils.py:21-88`) and
  sub-sampling training samples with a fixed split seed (`:59-67`).
- Plotting (`Interpretation.ipynb` cells 101, 125; `part2.ipynb` cell 108).

## Paper vs code

- Eq. 1 has no bias in the term-to-term step. The code's `lin_l` adds a
  bias that all terms share (`scripts/base_model.py:34,50`).
- The paper calls each hidden layer a GO level, most specific first. The
  code assigns no levels and computes waves by height above the childless
  terms (`scripts/base_model.py:120-173`).
- The paper does not say how gene-to-term connectivity is enforced.
- The paper sums selection counts over patients and models (e.g. 100
  models, r = 0.01) and clusters them; the notebook uses one model, and
  no clustering code was found.
- Graph preprocessing is in the unread Supplementary Information; the
  childless-term cut and most-specific filter come from the code only.

## Key excerpts

`scripts/base_model.py:399-408`

```python
        # Convert adjacency matrix to a non-trainable torch tensor
        adj_mat_fc1 = torch.tensor(adj_mat_fc1, dtype=torch.float).t()
        self.adj_mat_fc1 = Parameter(adj_mat_fc1, requires_grad=False)

        # Define the first fully connected layer (gene to GO annotation mapping).
        self.fc1 = Linear(in_features=n_genes, out_features=n_nodes_annot)

        # Apply the mask to the weights of the first layer, ensuring only connections specified in the adjacency matrix are active.
        with torch.no_grad():
            self.fc1.weight.mul_(self.adj_mat_fc1)
```

`scripts/GraphGONet.py:211-213`

```python
            with torch.no_grad():
                model.fc1.weight.grad.mul_(model.adj_mat_fc1)
            optimizer.step()
```

`scripts/base_model.py:161-171`

```python
            # Prepare the inputs for aggregation: extract the neighbors' embedding of fathers
            mask = adj_mat[0, :]

            # Take the most updated embedding of the neighboring
            children = out[mask].view(-1)
            out[next_visits] += self.lin_l(
                scatter(src=children, index=adj_mat[1, :], reduce=self.aggr)[
                    next_visits
                ][:, None]
            )
            out[next_visits] = self.nonlinearity(out[next_visits])
```

`notebooks/1-Build_GONet_graph_part2.ipynb:6429-6441` (cell source)

```python
%%time
gp=file_go.groupby(keyRef)
for ensembl in list(gp.groups.keys()):
    go_associated=gp.get_group(ensembl)["GO"].tolist()
    indexes = list(gp.groups[ensembl]) #rows id
    current_idx=0
    while current_idx<len(go_associated):
        go_visited = go_associated[current_idx]
        idx = indexes.pop(0)
        successors = set(nx.ancestors(graph, go_visited))
        if len(successors.intersection(set(go_associated)))!=0:
            file_go=file_go.drop(idx) #remove 'go_visited' as at least one child is connected to the current gene.
        current_idx+=1
```

## Open questions

- Edge and wave counts: only in notebook outputs and Supplementary Table
  S3, neither read.
- The README "no selection" run (`README.md:68-72`) may fail at this
  commit. `self.selection` is set only when a selection is given
  (`scripts/base_model.py:416-423`), but `forward` reads it every time
  (`:453`). Not run.
- Which commit produced the published results: the pinned commit postdates
  publication, and the docstrings were added later (`README.md:33`).
- Relation types: the code filters none. Whether the Supplementary
  Information restricts them is unknown.
