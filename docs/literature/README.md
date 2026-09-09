# Literature notebooks

Frozen reproductions of one published result per paper.
Docs CI renders the stored outputs. It does not re-execute
these notebooks.

## Re-run

1. Download the paper files into
   `.literature-data/<paper>/` (gitignored).
2. From a clone with the `docs` extra:

   ```bash
   python scripts/docs_notebooks.py --literature
   ```

Set `KPNN2_LITERATURE_DATA` if the data live outside the
clone. Default is `<repo>/.literature-data`.

## Fortelny and Bock 2020

SIM1 demo from
<https://medical-epigenomics.org/papers/fortelny2019/>
(same files as `Download_Data/Download_SIM1.sh` in
[epigen/KPNN](https://github.com/epigen/KPNN)).

```bash
DEST=.literature-data/fortelny-bock-2020
mkdir -p "$DEST"
BASE=https://medical-epigenomics.org/papers/fortelny2019
for f in SIM1_ClassLabels.csv SIM1_Data.csv SIM1_Edgelist.csv
do
  curl -fL -o "$DEST/$f" "$BASE/$f"
done
(cd "$DEST" && sha256sum -c ../../docs/literature/fortelny-bock-2020.sha256)
```

`SIM1_Data.csv` is about 100 MB. Do not add it to git.
