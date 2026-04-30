# Glossary

A working list of terms used in this repo. Define once here; link from code and
notebooks rather than re-explaining in every file.

## Mechanistic interpretability terms

**Circuit** — a subgraph of a neural network's computation that implements an
identifiable algorithm (e.g., the IOI / "indirect object identification" circuit
in GPT-2 small). A circuit is typed: it has inputs, outputs, and intermediate
nodes (heads, MLP neurons, residual stream positions).

**Feature** — a direction (or set of directions) in a representation space that
corresponds to a human-interpretable concept. Sparse autoencoder (SAE) features
are the most common modern instance.

**Residual stream** — the running sum of inputs and component outputs that flows
through a transformer layer-by-layer. Most modern interp framing treats the
residual stream as the central "communication channel" between components.

**Hook / HookedTransformer** — TransformerLens's mechanism for inserting
read/write callbacks at any internal point in the model's forward pass. The
foundation of nearly all interp tooling here.

**Probing** — training a small classifier on a model's intermediate activations
to test what information is linearly decodable at that layer. Cheap, blunt,
informative.

**Activation patching** — replacing a model's internal activations at a specific
location with activations from a different forward pass, to isolate which
components are causally responsible for a behaviour.

## Arabic terms

**MSA — Modern Standard Arabic** — the formal written Arabic used across the
Arabic-speaking world (news, books, formal speech). Native to no one; learned
in school. *Fusha* (الفصحى) is the colloquial name in Arabic.

**Masri / Egyptian Arabic** (مصري) — the colloquial Arabic spoken in Egypt;
the most widely understood Arabic dialect across the region due to Egyptian
media reach. Distinct from MSA in vocabulary, syntax, and phonology.

**Dialect (lahja, لهجة)** — the regional/colloquial variants of Arabic. Masri,
Levantine, Gulf, Maghrebi, etc. Models trained primarily on MSA-heavy corpora
typically degrade on dialectal inputs in ways worth investigating.

**Buckwalter / ISO-233** — Latin-script transliteration schemes for Arabic.
Useful when storing prompts so non-Arabic-readers can still skim diffs.

## Substrate terms

**Strix Halo** — AMD's high-end APU platform combining Zen 5 CPU cores with a
Radeon 8060S iGPU and ~24-32GB of unified memory. Runs ROCm; appears to torch as
a CUDA device via the ROCm/HIP compatibility layer.

**ROCm** — AMD's open-source GPU compute stack, the rough equivalent of CUDA.
Most CUDA Python code Just Works™ via `torch.cuda.is_available()` returning
True; CUDA-only C++ libraries (xformers, certain flash-attention variants) do
not.

---

*Add terms as they enter the repo. Brevity over completeness — link to canonical
sources for deep treatments rather than reproducing them here.*
