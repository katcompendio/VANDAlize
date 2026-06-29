# Data

## Dataset CSVs (in git)

The VANDA dataset. Each per-condition file has columns `statements` (text) and
`label`, balanced 85 harmful / 85 benign = 170 rows.

**Label convention: `1` = harmful, `0` = benign.**

| File | Condition | n | Source |
|---|---|---|---|
| `english_1.csv`  | English  | 170 | Aya Red-teaming (Cohere Labs) |
| `filipino_1.csv` | Filipino | 170 | Aya Red-teaming, translated |
| `taglish_1.csv`  | Taglish  | 170 | **Native-speaker translation** |
| `*_2.csv` | (second run / expanded) | varies | as above |

Parallel views (same statements aligned across languages, for inspection):
- `parallel_harmful_statements.csv` — `id, english, filipino, taglish, harm category`
- `parallel_benign_statements.csv`  — `id, english, filipino, taglish`

**Provenance.** English and Filipino harmful/benign prompts come from the Aya
Red-teaming dataset (Cohere Labs) on HuggingFace. The **Taglish translations were
produced by native Filipino speakers** (the student authors and their families),
not machine-translated or synthetically code-switched — this is the dataset's
core methodological contribution. Harmful statements carry a `harm category`
label (e.g. Discrimination & Injustice; Violence, Threats & Incitement).

> Note: some `statements` contain embedded newlines inside quoted fields. Read
> with a proper CSV parser (`pandas.read_csv`), not line counting.

## Label files (in git)

`*_labels.pt` — label vectors per condition, dict `{"<condition> run <k>": tensor[n]}`.
Byte-identical to the `label` column of the matching CSV (verified).

## Activation files (NOT in git)

Per-layer last-token residual-stream activations, dict
`{"<condition> run <k>": {layer_idx: tensor[n, hidden]}}`. Large (~130 MB each
for a 13B model), excluded via `.gitignore`. Obtain by either:

1. **Regenerate** — `notebooks/01_extract_activations.ipynb` (GPU + gated weights),
   ~15–20 min per model on a free Colab T4.
2. **Download** — HuggingFace mirror (link to add on release).

Expected filenames the analysis notebook looks for (LLaMA-2-13B):

```
English_1_activations_all_layers.pt   English_1_nat_labels.pt    key "english run 1"
filipino_1_activations_all_layers.pt  filipino_1_nat_labels.pt   key "filipino run 1"
taglish_1_activations_all_layers.pt   taglish_1_nat_labels.pt    key "taglish run 1"
```

SEA-LION files use `SEA`/`SEAv3` prefixes
(e.g. `SEAv3English_1_activations_all_layers.pt`).
