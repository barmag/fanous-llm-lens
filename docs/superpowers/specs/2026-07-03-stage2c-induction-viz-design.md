# Stage 2c induction visualization rebuild — design

**Date:** 2026-07-03
**Notebook:** `notebooks/education/stage2c_induction_tinystories.ipynb`
**Status:** approved approach (A), pending spec review

## Problem

The notebook trains a 2-layer transformer on TinyStories and scores the induction
circuit correctly, but the visualization (cell-10) is broken in five ways:

1. **Stimulus/form mismatch.** Eval sequences are uniform-random token IDs (correct
   for measuring the circuit), but the node-link diagram decodes them into token
   boxes — producing gibberish that can never read semantically.
2. **False legend.** Arrows are colored `blue if k < q else orange` with the label
   "orange=forward/induction". Attention is causal (`k ≤ q` always), so orange never
   appears — and induction *is* backward attention. The legend teaches a wrong
   mental model.
3. **Arrows point at nothing.** Token boxes exist only at y=1; arrows terminate at
   y=0 where nothing is drawn, so the attended-to token is invisible.
4. **Out-of-bounds rendering.** The subplot figures contain only annotations and
   zero traces; Plotly cannot autorange annotations, so content spills outside the
   plot area. Sequences up to ~44 tokens in ~500 px panels overlap regardless.
5. **Overload.** Twenty figures × two panels dumped in a loop — no focal point.

## Audience and constraints

- **Personal learning notebook** for the project owner (AuDHD). Not (yet) part of
  the bilingual curriculum ladder — no RTL scaffolding required.
- Concept must be built **from zero**: the mechanism, the evidence-reading, and the
  in-context-learning connection all need constructing, not just illustrating.
- **Static, stepwise, predictable** spine: one idea per panel, the *same* running
  example carried through, nothing that requires re-mapping between figures.
  One optional interactive **sandbox** cell at the end.
- Every new panel's markdown cell **maps the code to the source papers**:
  - Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits*
    (prev-token heads, QK/OV circuits, K-composition, two-layer term expansion)
  - Olsson et al. 2022, *In-context Learning and Induction Heads*
    (induction head definition = prefix matching + copying, repeated-random-token
    evaluation, ICL score)
- Runs end-to-end against the existing checkpoint on the iGPU in <10 min.

## What stays, what goes

| Cells | Fate |
|---|---|
| 1–6 (setup, data, tokenizer, training, load) | untouched |
| 7–9 (paper patterns, `induction_from_patterns`, score table) | untouched — the scoring is correct |
| 10 (node-link loop + dead `node_link_diagram`) | **deleted** |
| 11 (heatmap pair) | absorbed into Panels 2–3 |
| 12 (fresh-pattern verification) | untouched — quantitative backstop for Panel 5 |

## New structure

### Intro markdown (rewritten)

States the 5-panel route upfront (predictable structure), acknowledges model
limits (500-vocab BPE, 2L/2H, TinyStories only), and adds a **paper map table**:
notebook section → paper concept → paper section, so the reader always knows
which part of which paper the code in front of them implements.

### Helper cell: one shared visual language

```
attn_heatmap(pattern, tokens, highlight=None, title=...) -> go.Figure
```

- `go.Heatmap` trace (real trace → autorange works; bounds bug structurally gone)
- Blues colorscale, y-axis reversed, "rows = query (destination), cols = key
  (source)" subtitle — same convention as the sibling reference notebooks
- **Decoded tokens as axis tick labels** (monospace)
- Hovertemplate: `query 'Tom' → key 'Lily': 0.83`
- `highlight`: list of (q, k) cells outlined with shapes, optional caption arrow

Panels 2, 3, 5 and the sandbox all render through this one helper.

### Example-picker cell

Deterministically selects the running example: iterate a shortlist of handcrafted
TinyStories-style prompts (e.g. `"Tom and Lily went to the park. Tom and"` →
target `Lily`); accept the first whose repeated bigram survives the 500-vocab BPE
as clean tokens **and** whose target lands in the model's top-3 at the second
occurrence. Print the chosen prompt, its tokenization, and the top-3. Raise a
clear error naming all rejected candidates if none pass.

### Panel 1 — the behavior (why care)

Token strip (`go.Scatter` markers+text; first `A B` highlighted, second `A`
highlighted, `?` box) + top-5 next-token probability bars at the second `A`.
**Paper hook:** Olsson et al.'s behavioral definition of in-context learning —
the model gets *better* at predicting later in the context because it can reuse
what already appeared.

### Panel 2 — L0 prev-token head

`attn_heatmap` of the L0 head selected by diagonal score (same `diag_score`
device as the reference notebook), diagonal cells outlined.
Message: "every token learns who is directly behind it."
**Paper hook:** Elhage et al. — previous-token heads as the key ingredient the
first layer contributes; QK circuit says *where* to look, OV says *what* to move.

### Panel 3 — L1 induction head

`attn_heatmap` of the best L1 head, the single cell (query = second `A`,
key = first `B`) circled and captioned: "look at what *followed* the first
occurrence." The full stripe is visible; the circled cell anchors it.
**Paper hook:** Olsson et al.'s induction head definition — **prefix matching**
(attend to the token after a previous occurrence of the current token) — and the
note that our `induction_from_patterns` score is exactly a prefix-matching score
computed on known source positions.

### Panel 4 — composition schematic (no model data)

Static trace-based diagram (same Scatter+annotation device as the project's
prediction tree): L0 stamps "`A` is behind me" into `B₁`'s residual position →
`A₂`'s query matches that stamp → attention lands on `B₁` → OV copies `B` into
the prediction.
**Paper hook:** Elhage et al.'s **K-composition** — the L1 head's key is computed
from residual content *written by the L0 head*, which is why the circuit needs
two layers and cannot exist in a one-layer model (their term-expansion argument).

### Panel 5 — the punchline: it works on gibberish

One random-token pattern's L1 heatmap through the same helper (gibberish tick
labels shown deliberately), plus the existing score summary. Random tokens stop
being a rendering bug and become the point: the circuit is an **algorithm over
positions and repetitions, not memorized bigrams**.
**Paper hook:** Olsson et al. evaluate induction heads on *repeated random
sequences* for precisely this reason, and argue induction heads are the main
driver of in-context learning in small models; cell-12's fresh-seed patterns are
our version of their held-out evaluation.

### Sandbox cell (last)

`explore(text: str)` → tokenizes, runs the model, renders side-by-side L0/L1
heatmaps via `attn_heatmap`, prints the induction score for the best L1 head.
Pre-filled with one example call. Markdown cell before it: what to try (repeat a
name, break the repetition, vary the gap) and what to expect.

### Closing markdown

Recap of the five steps + pointer back to the two papers, and a note on how this
connects to `stage2_dash2_composition_induction_reference.ipynb` (K-composition
measured there via weight products, observed here via attention patterns).

## Error handling

- Example picker fails loudly with the list of rejected prompts (never silently
  falls back to a random-token sequence for panels 1–4).
- Helper asserts `len(tokens) == pattern.shape[-1]`.
- All figures produced by helpers with real traces; no annotation-only figures.

## Verification

- Run the notebook end-to-end against the existing checkpoint (no retrain).
- Eyeball every figure: labels inside bounds, stripe visible, circled cell
  correct.
- Confirm the chosen example's L1 stripe matches the score table's best head.
- Clear outputs before commit (project convention).
