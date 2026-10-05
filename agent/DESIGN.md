# Design Intuition
Why the research should work, in the terms used to convince a skeptical colleague at a whiteboard. USAGE.md says how to run the code; this says what the code is trying to be true about. Intuition first — equations and hyperparameters belong in `configs/experiment/` and the paper draft.

Below the line:
- **Three-sentence pitch** — problem, mechanism, consequence, no hedging. If it takes more, the design is not designed yet; write the three sentences first and let the rest elaborate them.
- **The problem** — setting, observed failure, evidence it is not a tuning artifact, and why prior work does not already fix it. A problem no existing method fails at is not a problem statement.
- **Core intuition** — the idea as one image or analogy; the causal story for why it helps; and what it predicts that the baseline does not. That last item is what separates "it worked" from "it worked for the reason claimed".
- **Why this might be wrong** — per claim, the cheapest falsifying observation and whether it has been run. Plus confounds that could produce the same gain (parameter count, effective batch size, data seen, eval protocol) and the most boring explanation for a win, with the ablation that rules it out.
- **Load-bearing decisions** — only choices where a different answer would move a number: alternatives rejected, why, and when to revisit. Everything else is a config value. This is what stops a future agent re-litigating a settled question.
- **Evaluation logic** — why the metrics can detect the mechanism, what they are blind to, which diagnostics test it directly, and which protocol details break comparability if they drift. The numbers themselves go in SOTA.md.
- **Idea → code** — where each component lives. A wrong pointer costs more than a missing one.
- **Open questions** — ranked by what a wrong guess costs. Promote one to a direction in ACTIVE.md rather than answering it here.

Maintenance:
- Rewrite in place, never append; superseded versions go to `agent_legacy/design/YYYY-MM-DD_<slug>.md`. This file states the current belief, not its history.
- A mechanism added without its failure modes is an unfinished edit.
- Directions graduate here from ACTIVE.md once they survive their success criterion.

##### MODIFY UNDER THIS LINE #####

## Evaluation logic
- **The bank monitor is ambiguous (spec line 181; Amendment 1, S24).** "Drop any terms where one of the nodes has a single child" can mean either node (the literal reading) or only the parent (the step is exactly zero only when the parent has a single child). "Averaged over each rank" can mean over species or over unique parent-child edges. From family to genus the two drop rules keep about 96% or 54% of the edges (68,574 genera before S9; 67,859 after, where the parent rule keeps 95.8%), so the choice moves the numbers a lot.
  - Pilot 1 logs all four variants. The headline is the parent rule averaged over unique edges, chosen by the spec's owner.
  - These numbers deserve close scrutiny from the owner before anyone reads meaning into them.
  - No decision rule uses this monitor: not the LR sweep, not checkpoint selection, not the verdict.
- **Encode noise sets the resolution of integer counts (eval gate, 2026-10-04).** cuDNN's choice of convolution algorithm for the patch embedding changes fp16 embeddings by about 1e-6 in cosine. That moves iNat21 correct counts by up to about 17 images per rank between two encodes of the same weights. Under benchmark mode off, BioCLIP 1 itself re-encodes to 70,193 species-correct, against the cached 70,186.
  - The ±1-point bar (1,000 images, S31) is far above this noise.
  - S32's floor has no margin, so a model within about 20 images of 70,186 can land on either side depending on the encode setting. Decided (Amendment 2 A2.4, 2026-10-05): the M3 eval records its setting, and all six baselines are re-evaluated under that same setting, so the floor and the comparison cells are same-setting integer counts.

## Idea → code (pilot 1, 2026-10-04)
- Image store and decode chain: `scripts/store/build_store.py`, reader `src/data/tol_store.py` (uint8 crops; `to_model_input` = ToTensor + Normalize).
- Catalog, lineage drop, S9 list: `src/data/tol_catalog.py`. Train tree (species keys, ancestors, n_a, P_s, g_s, held-out 5%) and ToL-val candidates: `src/data/taxonomy.py`.
- Species groups and batches of B images, the last one of an epoch short (one group per species per batch): `src/data/tol_sampler.py`; loaders: `src/data/tol_datamodule.py`.
- LoRA on q,v rows of `in_proj_weight` through `parametrize`, logit_scale clamp: `src/models/bioclip_lora.py`.
- fp64 bank (EMA, ancestors, root, refresh, zero-sum check, bank monitor, L*, drift): `src/models/bank.py`.
- BFL and flat contrastive losses in distributed form, group means, penalty with 1/g_s: `src/models/losses.py`.
- Training step (manual optimization, hand-averaged gradients, S16 monitor, collapse monitor), refresh, ToL-val, checkpoints: `src/models/p1_module.py`; ToL-val protocol: `src/eval/tolval.py` on top of `src/eval/zeroshot.py` (the iNat21 harness and its exact-count gate).
- Smoke tests and gates: `scripts/smoke/p1_smoke.py`, `scripts/smoke/s13_compare.py`, `scripts/smoke/eval_gate.py`.
