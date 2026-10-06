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

## Why this might be wrong (first evidence, 2026-10-05)
- **The level-restricted objective alone loses species accuracy.** Arm (c), BioCLIP 1 with LoRA trained on the seven-level BFL loss and no penalty, gains the coarse ranks within one epoch on deduplicated ToL-val (kingdom 46 → 98%, class 6 → 93%) and then loses species accuracy every epoch: 40.23% untouched, 41.59% after the first epoch, 36.20%, 31.86% at lr 1e-4; worse and unstable at 3e-4 (`audit/2026-10-05_M2_sweep_results.md`). The loss's own species and genus terms rise while its coarse terms fall, and the image embeddings contract into a narrower cone (mean pairwise cosine 0.14 → 0.71, participation ratio 137 → 75, mostly in the first 250 steps). This is the failure mode of spec line 174, before any penalty is applied.
  - Two candidate causes, each with its test. (1) Batch composition: the grouped plan packs about 890 species with about 9 images each into a batch of 8,192, so the coarse terms see large same-taxon clusters; the lr 1e-4 sampler control (ask A14, `data.train.sampler=random`, about 7,750 species per batch) tests this, running since 2026-10-06. (2) The objective itself: seven equal-weight levels where the coarse terms start large; if the control declines the same way, this is the cause, whatever the batches. A lower learning rate (3e-5) is a third test, not run; the sweep's better rate is its lower edge (spec line 161 lists only 1e-4 and 3e-4; the owner decides).
  - What it does not yet show: whether the ToL-val decline transfers to iNat21 (10,000 species, the verdict's metric); that is seen once per arm at M3, through the S27 checkpoint. And whether the penalty (arm d) changes it: the orthogonality constraint acts on the fine-rank steps, so it could either protect species directions or not; the (d) vs (c) comparison is the pilot's question.
- **Confound to keep in view.** ToL-val species is a 131,143-way task with 268,609 images, mostly of species with few training images; iNat21 species is 10,000-way. A model can lose ToL-val species while holding iNat21 species, or the reverse. Only M3's one iNat21 eval per arm settles the verdict.

## Evaluation logic
- **The bank monitor is ambiguous (spec line 181; Amendment 1, S24).** "Drop any terms where one of the nodes has a single child" can mean either node (the literal reading) or only the parent (the step is exactly zero only when the parent has a single child). "Averaged over each rank" can mean over species or over unique parent-child edges. From family to genus the two drop rules keep about 96% or 54% of the edges (68,574 genera before S9; 67,859 after, where the parent rule keeps 95.8%), so the choice moves the numbers a lot.
  - Pilot 1 logs all four variants. The headline is the parent rule averaged over unique edges, chosen by the spec's owner.
  - These numbers deserve close scrutiny from the owner before anyone reads meaning into them.
  - No decision rule uses this monitor: not the LR sweep, not checkpoint selection, not the verdict.
- **Checkpoint selection follows ToL-val species (S27).** The iNat21 checkpoint of each arm is the epoch with the most species-correct on deduplicated ToL-val, so when accuracy peaks early the selected model is an early one: for the lr 1e-4 sweep run it is currently `epoch_00.ckpt` (111,727), with every later epoch lower. The verdict then compares a short-trained model with the baselines; this is what the rule asks for, and it should be stated with the result.
- **Encode noise sets the resolution of integer counts (eval gate, 2026-10-04).** cuDNN's choice of convolution algorithm for the patch embedding changes fp16 embeddings by about 1e-6 in cosine. That moves iNat21 correct counts by up to about 17 images per rank between two encodes of the same weights. Under benchmark mode off, BioCLIP 1 itself re-encodes to 70,193 species-correct, against the cached 70,186.
  - The ±1-point bar (1,000 images, S31) is far above this noise.
  - S32's floor has no margin, so a model within about 20 images of 70,186 can land on either side depending on the encode setting. Decided (Amendment 2 A2.4, 2026-10-05): the M3 eval records its setting, and all six baselines are re-evaluated under that same setting, so the floor and the comparison cells are same-setting integer counts.

## Idea → code (pilot 1, 2026-10-04)
- Image store and decode chain: `scripts/store/build_store.py`, reader `src/data/tol_store.py` (uint8 crops read with `os.pread`; `to_model_input` = ToTensor + Normalize).
- Catalog, lineage drop, S9 list: `src/data/tol_catalog.py`. Train tree (species keys, ancestors, n_a, P_s, g_s, held-out 5%) and ToL-val candidates: `src/data/taxonomy.py`.
- Species groups and batches of B images, the last one of an epoch short (one group per species per batch): `src/data/tol_sampler.py` (`make_random_epoch_plan` is a diagnostic mode, ask A14, not the spec's batches); loaders: `src/data/tol_datamodule.py`.
- LoRA on q,v rows of `in_proj_weight` through `parametrize`, logit_scale clamp: `src/models/bioclip_lora.py`.
- fp64 bank (EMA, ancestors, root, refresh, zero-sum check, bank monitor, L*, drift): `src/models/bank.py`.
- BFL and flat contrastive losses in distributed form, group means, penalty with 1/g_s: `src/models/losses.py`.
- Training step (manual optimization, hand-averaged gradients, S16 monitor, collapse monitor), refresh, ToL-val, checkpoints: `src/models/p1_module.py`; ToL-val protocol: `src/eval/tolval.py` on top of `src/eval/zeroshot.py` (the iNat21 harness and its exact-count gate).
- Smoke tests and gates: `scripts/smoke/p1_smoke.py`, `scripts/smoke/s13_compare.py`, `scripts/smoke/eval_gate.py`.
