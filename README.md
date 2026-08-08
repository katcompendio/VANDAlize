# VANDAlize

**Safety Degradation of Large Language Models Through Taglish-Based Inputs**

A mechanistic interpretability study of how an LLM's internal "this request is
harmful" representation behaves under Tagalog–English code-switching (Taglish),
the register tens of millions of Filipinos speak natively.

We probe last-token residual-stream activations with a difference-of-means probe
(Marks & Tegmark, *Geometry of Truth*) across English, Filipino, and Taglish, in
three models: LLaMA-2-13B, SEA-LION v1, and SEA-LION v3.

## Key findings

- A linear harm direction exists in English, Filipino, and Taglish (in models
  that develop one at all), but its **magnitude is ~1.5× weaker** in the two
  Filipino-family conditions than in English.
- In the cross-condition transfer matrix, **AUROC stays high (0.93–0.99) while
  accuracy drops** in cross-condition cells: the ranking of harmful-vs-benign
  transfers across languages, but the decision threshold does not. The failure
  is threshold *miscalibration* driven by magnitude attenuation, not a wrong or
  missing harm direction.
- The English-vs-Tagalog magnitude asymmetry **persists in SEA-LION v3**, a model
  explicitly trained on Tagalog — targeted training data does not close the gap.
- Whether a linear harm direction is present at all is **base-architecture
  sensitive**: absent in the Llama-2-based SEA-LION v1, present in the
  Llama-3-based SEA-LION v3.

Because English-calibrated safety systems set their threshold where the English
signal is strong, the weaker Taglish signal lands below that threshold — an
accidental, population-level differential-protection failure for Filipino
code-switching speakers, not an adversarial jailbreak.

## Repository layout

```
vandalize/
├── notebooks/
│   ├── 01_extract_activations.ipynb   # GPU; extract activations from model weights
│   └── 02_analysis.ipynb              # CPU; reproduces every figure & table
├── src/
│   └── vandalize.py                   # probe + analysis functions (imported by 02)
├── data/
│   ├── *_labels.pt                    # label vectors (0=harmful, 1=benign)
│   └── README.md                      # where the prompts and activations live
├── figures/                           # written by 02_analysis.ipynb
├── requirements.txt
└── README.md
```

## Reproducing the results

**Fast path (no GPU, ~2 min)** — reproduce all figures from pre-extracted
activations:

1. Obtain the activation files (see `data/README.md`) and place them in `data/`.
2. Open `notebooks/02_analysis.ipynb`, run top to bottom.
3. Figures are written to `figures/`.

To analyze a different model, edit the `CONDITIONS` dict in Section 2 of
`02_analysis.ipynb` to point at that model's activation files. All analysis code
is model-agnostic.

**Full path (GPU + gated weights)** — regenerate activations from scratch:

1. Open `notebooks/01_extract_activations.ipynb` on a GPU runtime.
2. Provide a HuggingFace token (via the `HF_TOKEN` environment variable or the
   prompt) with access to the gated model.
3. Set `MODEL_NAME` and `CONDITION`, point `CSV_PATH` at the prompt CSV, run.
4. Repeat per (model, condition) pair, then run `02_analysis.ipynb`.

## Data convention

- Prompt CSVs have columns `statements` (text) and `label` (`1` = harmful,
  `0` = benign), balanced 85/85.
- Harm direction = mean(harmful) − mean(benign); projections are higher for
  harmful prompts. The **norm** of this raw difference vector is the
  "signal strength" / magnitude reported throughout. (The probe is sign-symmetric
  in the label convention; metrics are identical either way.)
- Probe layer is 14 (chosen a priori from prior moral-valence work).

## Models

| Model | HuggingFace ID | Base | Layers | Hidden |
|---|---|---|---|---|
| LLaMA-2-13B | `meta-llama/Llama-2-13b-hf` | Llama-2 | 40 | 5120 |
| SEA-LION v1 | `aisingapore/Llama-SEA-LION-v2-...` | Llama-2 | — | — |
| SEA-LION v3 | `aisingapore/Llama-SEA-LION-v3-8B` | Llama-3 | 32 | 4096 |

## Citation

If you use this code or data, please cite this repository. A paper is in preparation.

## Acknowledgments

Built on the ARENA linear-probes module and the difference-of-means probe from
Marks & Tegmark (*The Geometry of Truth*). Taglish translations were produced by
native speakers.
