Written 2026-10-02.
(sha256 8f2ee5ea1769ed2effa0ec55a8e7f844b9d69c5541695a46a9bf9e174a1bfd0f).
Do not edit. Changes go in an Amendments section at the bottom, each dated, with the reason
and whether any results had been seen.

Numbers and paths I give you are from memory. Verify each one against disk before relying on it, and stop and tell me if one does not match.

Copy ~/bdbk/repos/tol_embed/src/open_clip before starting. Do not install via standard pip.

Starter files to reference **read-only** for implementation at ~/bdbk/repos/tol_embed/scripts/:
- encode_cache.py: refactor, it is too long. If it helps, split logic for encoding and caching inat21 and tol10m, and make room for refactoring for tol200m in the future.
- data_paths.py: prioritize this repo's configured paths over what's defined here.

In tol_embed, read only the files the spec names. Everything else there (the older PILOT_PROMPT drafts, HANDOFF files, audits/) is history and may contradict the spec. If anything there disagrees with specs/pilot1.md, the spec wins; ask me if the difference matters.

### Project idea
BioCLIP 1 and 2 (https://arxiv.org/abs/2311.18803; https://arxiv.org/pdf/2505.23883) train a CLIP backbone contrastively on a taxonomical dataset (TreeOfLife-10M or -200M) to classify the taxon of an image at a given rank (e.g., species level, genus level, ...). "Beyond Flat Labels" (BFL, https://arxiv.org/pdf/2606.21838) improves the objective by enforcing level-restricted contrastive learning, supervised by groups of positives rather than the naive InfoNCE single-positive setup (e.g., at the family level, all images sharing a certain family count as positives for the truncated taxonomic label; and the taxonomic label (truncated up until family) is a unique positive for all those images, in a deduplicated taxonomic label set). We argue that the BioCLIP and BFL objectives are geometrically unconstrained (BFL less so, but there is potential headroom nonetheless), while RCME (applies entailment learning: https://arxiv.org/pdf/2506.21476) is an example of too strong an inductive bias for regularizing the BioCLIP 1/2 embedding space. We propose cross-level regularization based on the intuition that "new" information gained while moving down one lineage of the Tree of Life is orthogonal to the accumulated information, and we argue this for the image side rather than the text taxonomic label side. That is, define a species prototype as the mean of the L2-normalized image embeddings of that species in a dataset, then define every internal node of the Tree of Life (leaf nodes are species, root is the representation of Eukarya, internal nodes are prototypes at the genus level, family level, etc.) as a species-uniform mean of species prototypes belonging to the internal node's subtree. Call these internal nodes ancestor prototypes (or specifically, genus prototype, family prototype, etc.). Center all prototypes around the Eukarya prototype (the mean of all species prototypes) such that species prototypes are zero-mean. Then whiten the embedding space (https://arxiv.org/pdf/2406.01506) [ask me more about this]. Now, treat new information from an ancestor at one level to a descendant in a lower level as the difference between the ancestor and descendant prototypes, essentially a displacement vector. Treat accumulated information as the ancestor prototype, essentially a position vector. Across all images in the dataset that each (displacement, position) pair for descendant=species s, ancestor=kingdom/phylum/class/order/family/genus taxon: enforce cos^2(displacement, position) towards 0.

### 1. Claim and decision
- What the pilot decides: Whether our hierarchical orthogonality regularizer improves BioCLIP 1's accuracy competitively on zero-shot hierarchical classification (across the seven ranks from kingdom to species). The baselines to compete against are: BioCLIP 1, BioCLIP 1 + RCME, BioCLIP 1 + BFL. Their rows are shown below, measured in our harness, not from any paper.

- Reproduced baseline evaluation: Each model under its best form, except BFL which is fixed at `photo`.
  - photo: "A photo of [truncation at the current level in the tree being evaluated]."
  - lineage: "[truncation up to current level]"

| rank | clip-l14 `photo` | bioclip2 `photo` | rcme `lineage` | bioclip1 `photo` | bfl-euc `photo` | bfl-hyp `photo` |
|---|---|---|---|---|---|---|
| kingdom | 66.26 | 77.79 | 86.70 | 74.66 | **99.01** | 98.95 |
| phylum | 36.50 | 70.32 | 86.35 | 43.88 | **98.58** | 98.48 |
| class | 28.49 | 37.71 | 54.77 | 18.17 | **91.85** | 91.78 |
| order | 10.63 | 51.48 | 46.21 | 13.41 | **89.92** | 79.62 |
| family | 8.73 | 54.42 | 43.64 | 24.37 | **85.33** | 46.55 |
| genus | 9.77 | **78.11** | 66.67 | 46.12 | 75.84 | 23.90 |
| species | 3.27 | **89.12** | 71.66 | 70.19 | 68.78 | 31.96 |
| **average** | 23.38 | 65.56 | 65.14 | 41.54 | **87.04** | 67.32 |

- The bar for arm (d): In our own reproduction on iNaturalist2021-val, BioCLIP 1 + BFL has ~90% zero-shot classification accuracy from kingdom down to order (see table), while performing ~3 points worse than BioCLIP 1 + RCME at the species level. Therefore, the target for this pilot is: competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons. If confused, don't make a verdict and just report everything faithfully for human review.


### 2. Data
- Training data: TreeOfLife-10M `train` split (5,908,775 samples). Path: /u/liv/bdbk/data/tol10m. Drop any rows with an incomplete taxonomic lineage. Build the bank of entries from the train split only. Keep `split == "train"`, no `train_small`.
- Validation data: TreeOfLife-10M `val` split (310,899 samples). Same path. Deduplicate it (find any val images with >=0.98 cosine similarity with same-species train-split images). Drop rows with incomplete lineage.
- Shard composition at /u/liv/bdbk/data/tol10m. Training data loader must skip `image_set_61`-`63` and extract the train images from `image_set_60`. The bank seeding and bank refresh must also filter to train ids.
    | shards | contents |
    |---|---|
    | image_set_01–59 | all train |
    | image_set_60 | mixed: 8,775 train / 91,222 val |
    | image_set_61–63 | all val |
- The shards are 32 GB gzip streams that read at 23-27 img/s each and can't be accessed at random. Extract train and val with the eval transform (224 px) into a store readable by uuid in any order, keeping the transformed pixels lossless (re-saved JPEGs would fail the cache smoke test below). Store in ~/bdbk/data. The sampler and the refresh read from that store. Streaming shards can't give "all images of a small species in one group."
- How each image gets its text: ToL shards are `image_set_01..63.tar.gz` with bare `<uuid>.jpg` images and no paired text. Taxonomy is joined from `catalog.csv` via `treeoflife_id == uuid`. This needs a smoke test that reproduces either the BioCLIP 1 encodings of the dataset or some reasonable classification metrics based on the embeddings. `/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1/emb.f16.npy` holds BioCLIP 1's embeddings of 6,219,459 ToL-EOL images keyed by uuid in ids.txt. Smoke test: encode 1,000 images in fp16 autocast through the new data pipeline (same dtype as the cache), look up the same uuids in the cache, and require cosine >= 0.9999 for every row.
- Class string template: (a) As per BioCLIP 1's training convention, train every arm with the text input `"a photo of [truncated label up to that level]."`, (b) alternative for reporting during evaluation only, `"[truncated label]"`. E.g. For the fox species: kingdom -> `"a photo of Animalia."`, order -> `"a photo of Animalia Chordata Mammalia Carnivora."`
- Taxons (at any level above species) with a single child: its step is exactly zero, so cosine is undefined, and it should be skipped by the penalty, but kept for the contrastive objective. Log count at every rank.
- Samples/Rows with missing ranks: Drop entirely, and log count.
- Evaluation data: iNaturalist 2021 (iNat21) val split, which is outside ToL-10M. Path: /u/liv/bdbk/data/inat21.
- Same image transform for batch and bank. State and choose the eval transform for both training and eval.


### 3. Model
- Checkpoint: BioCLIP 1, b/16 backbone. Path: /u/liv/bdbk/ckpt/bioclip1. Load with `local-dir:` schema.
- Assertions for ckpt loading: expect `visual.output_dim == 512`, parameter count ~149.6M
- Fine-tuning on ToL-10M: Fine-tune via LoRA on the attention q,v projection matrices through all layers in both towers with `alpha = 2r`. Because BioCLIP 1 is built on OpenCLIP, its `Attention` class packs q,k,v projections into `in_proj_weight: (3*dim, dim)`. Implement by hand-rolling adapter pairs for attention blocks and adding to the q,v blocks of `in_proj_weight`. Other learnable parameters: `logit_scale`
  - Assertion: Trainable params should be ~1% or less of the total params. 
  - Smoke test: adapters at init should produce bit-identical embeddings to the frozen BioCLIP 1 model (both in the same precision mode); set one adapter to a nonzero value and the embedding changes.
  - Capacity control: Full fine-tune
  - Separate arm: In addition to LoRA on both towers and the `logit_scale`, also unfreeze `visual.proj`, `text_projection`. Save for later.


### 4. Objective
- TreeOfLife Bank:
  - Bank entries must be at least fp32: bf16/fp16 round the EMA increments to zero. fp64 makes the zero-sum assertion below exact.
  - Batch estimate of species entry `muhat_s`:
    - Role: Differentiable, the child side of the penalty
  - Species entry `m_s`:
    - Initialized as average over the same species of L2-normalized BioCLIP 1 embeddings of ToL-10M train-split images with a complete lineage. Subtract the bank root (species-uniform mean) from every position before the penalty and bank monitor.
    - Computed as: EMA over past batches
    - Role: Detached, derives every ancestor in the bank
    - Update rule:
    ```
    delta_s = β_s * (muhat_s.detach() - m_s)          # the bank's own increment
    m_s    <- m_s + delta_s
    for each ancestor a on s's chain:
    m_a <- m_a + delta_s / n_a       # n_a = species beneath a (in the taxonomy, not in the batch)
    ```
    - `β_s = K_s/N_s` so that bank update rate is directly proportional to the proportion of images represented in the batch for species `s`, and each species gets roughly one full turnover per epoch summed over its image splits.
    ```
    g_s = ceil(N_s / K)  # number of balanced groups for species s
    K_s = floor(N_s / g_s) or ceil(N_s / g_s)
    
    ```
    - Effectively, the bank's species entry is nudged (at every training step it appears in) by the pre-parameter update-derived in-batch species prototype, off by one step deliberately.
  - Within epochs: EMA. After each epoch: exact recomputation of the full bank (<2 GPU-hours), then diff to obtain the EMA proxy vs. fresh drift. This proxy-vs.-fresh drift can be used to tune `β_s`.
  - Assertion by definition: for every parent, species-count-weighted child deviations (steps) sum to zero throughout training. `m_a == (1/n_a) * sum_{s under a} m_s` (up to rounding if the bank is fp32)

- Total loss `L = L_con + lambda * L_B`. After LR sweep is complete, set and fix `lambda` so that the gradient norm-ratio monitor gives 0.2 (we choose 0.2 for now), averaged over steps 50-250 at the chosen LR (see Monitors); queue `lambda` sweep based on ToL-10M val accuracy in the future.
- Contrastive objective:
  - BFL level-restricted contrastive objective
  - Flat: species-level only, but also with group-positives. In a batch, images sharing the same species are a group of positives for the label, and the deduplicated label is the unique positive for all those images. Same positive/negative setup as BFL.
- Canonical penalty:
  ```
  L*  =  (1/|S|) Σ_s  ℓ*_s
  ℓ*_s = Σ_{p ∈ P_s} cos²( mu_s − m_{a_p(s)} ,  m_{a_p(s)} )
  P_s = {p : n_{a_p(s)} ≥ 2}
  ```
  `a_p(s)` is the rank-`p` ancestor of species `s`. `P_s` is the set of valid ranks `p` kept for species `s` with at least two species beneath the ancestor at that rank. That is, for every species `s`, skip penalty terms with an ancestor that has only a single species beneath it. This produces a zero denominator in the penalty term.
- Per-batch penalty:
  ```
  L_B = (1/B) * Σ_{s ∈ B} ℓ_s(muhat_s)
  ℓ_s(muhat_s) = Σ_{p ∈ P_s} <muhat_s - m_{a_p(s)}, m_{a_p(s)} - r>² / (den_{s,p} ‖m_{a_p(s)} - r‖²)

  den_{s,p} = ‖muhat_s − m_a‖².detach()      if K_s = N_s or K >= N_s
        = ‖m_s − m_a‖²                   otherwise
  den_{s,p} = max(den_{s,p}, δ=1e-5)
  ```
  `K_s = N_s` meaning the in-batch group of images holds all of `s`'s images: exact, fresh from current parameters. `otherwise` `muhat_s` is replaced by the bank entry `m_s` so there is no numerator-denominator coupling of the sampling noise, but is costed by some staleness from the bank entry. Flooring `den_{s,p}` by `δ` prevents near-zero denominators from breaking things.
  - Control: No penalty; (level-restricted / flat) contrastive only.
  - It may happen that at different levels in the tree, the room for enforcing orthogonality between step and parent position is different. For now we enforce the penalty with uniform weighting across the six level-pairs (kingdom ~ genus paired with species).
  - n_batch = number of images in the batch of the species of image i, so that every species that appears in the batch contributes uniformly to the batch-level penalty, even if that species appears multiple times in the batch
  - quantities in the penalty that carry gradient: only `muhat_s`. however, because `m_a` is an EMA of `m_s` over all species `s` under `a`, and cos^2 is optimized over the batch and thus over the entire dataset per epoch, it suffices that only the positions of the leaf nodes in the tree are moving, because their movements propagate upward through the tree via EMA and move their ancestor positions.
  - every parent position is computed as a species-uniform mean (i.e., given all species positions, you can either compute each rank one at a time *weighted by how many species are under the child* (family from genera (but each genus is weighted by how many species it has), order from classes), or directly from species with each species leaf uniformly-weighted (family from species, order from species).), while each species position is a mean of the embeddings of images showing that species. parent positions are held in a bank as EMAs. Initialization: initialized as species-uniform means of L2-normalized BioCLIP 1's image embeddings. Update rule: when a species prototype receives an update `delta`, ancestor receives `delta * (1 / its species count)`.
  - at penalty time, for each term `(s, a)`: the parent `a` is centered by the bank root `r`, `m_a - r`. The step `muhat_s - m_a` cancels out the centering. Nodes stay uncentered in the bank and for the contrastive objective, only centered when read for the penalty. L2-normalize every image embedding before any species mean: in bank seeding (the cache is unnormalized, with row norms 5.95-17.91), in the refresh, and for muhat_s. Subtract the bank root (the species-uniform mean) from every position before the penalty and before the bank monitor.
  - every bank entry is keyed by its full lineage prefix (`|`-joined path as in the cache's vocab.json), not by name.
  - set up the bank by doing a full-pass of the dataset (for now: TreeOfLife-10M; and not the full dataset, just the subset at /u/liv/bdbk/data/tol10m) through BioCLIP 1.
  - Degenerate solutions: Shrinking the parent position is prevented because `m_a` is detached and read from an EMA. Sibling (same immediate parent) orthogonality is impossible because sibling steps sum to zero by construction. Representation collapse can be monitored by logging participation ratio of the batch embedding covariance `(sum of eigenvectors)^2 / sum of squared eigenvectors`.
- Whitening: To account for significant levels of anisotropy in any CLIP-style embedding space (i.e. work shows it is a narrow, high-dim cone), we apply the inductive bias that vector representations of species/ancestor positions and ancestor-species steps which are computed from BioCLIP 1's image embeddings should be orthogonalized in a whitened space where embedding coordinates are uncorrelated and thus represent decomposable species-level semantics. Applying the penalty on whitened embeddings is invariant to any invertible linear map of the embedding space. Recompute the whitener every epoch from the bank of embeddings and tree node positions. Let `x_i` be the L2-normalized image embedding of image `i`, `s(i)` its species, and `m_s = mean_{i in s} x_i` the species prototype. Three scatter matrices are available:
    ```
    Σ_W = (1/N) * sum_i (x_i - m_s(i)) (x_i - m_s(i))^T   within-species
    Σ_B = Cov over species of m_s   (species-uniform)   between-species
    Σ_T = Cov over images of x_i  =  Σ_W + Σ_B(image-weighted)  total
    ```
    
    - They are the two halves of the ANOVA decomposition and their sum. On bioclip1 the split is 48.6% within / 51.4% between; on bfl-euclidean 33.0% / 67.0% -- level-restricted training pulled images toward their species prototypes.

    - For this first stage of the pilot, `W=I`.
    
    - A whitener is `W = Σ^{-1/2}` (precision: float64), applied to every prototype and position before the cosine. Whitening by a covariance means measuring angles in the units of that covariance:

    - within-species -- angles in units of the model's own within-species noise (the Fisher / LDA metric). Orthogonal means: the noise projected onto the step direction is uncorrelated with the noise projected onto the position directio.
    - between-species -- angles in units of how the taxon population spreads. In these coordinates `Cov_species(z) = I`, so two directions are orthogonal iff the species-population coordinates along them are uncorrelated. Step-perpendicular-to-position then says, literally: across the population of taxa, how far a species sits along its new direction is uncorrelated with how far its parent sits along the parent's direction. This is the cleanest match to the sentence "new information is uncorrelated with accumulated information", and it is Park's causal-inner-product reading.
    - total -- the standard PCA-whitening compromise; no story to defend.

- Batch mining, constant across all arms:
  - `K = {8, 16}` is the maximum number of images representing a species in a batch.
  - Every species is sampled and counted towards the penalty except for a random 5% held-out set (they stay in the bank and the contrastive objective, but not the penalty). For every species `s`, group all its `N_s` images and then divided into `ceil(N_s/K)` balanced splits. At `K=8`, a species with `N_s=9` gets split into two groups of 4 and 5 images. Re-partition each species' images randomly every epoch. Penalty terms with an ancestor with exactly one species beneath it is skipped, as defined by the set of ranks `P_s`.
  - With 2~4 A100s and a global batch size `B=8192`, we construct a batch by sampling the balanced splits of species-specific images.
  - Justification: Since we cannot recompute all image embeddings and prototypes at each training step, we want each species in the batch to be well-represented by its total population of images in the dataset, so that the penalty terms push embeddings toward a direction that accurately reflects where the prototypes actually are. For tail species with few images in the dataset, we include all the images to eliminate sampling noise and limit noise to something data-shaped. For head species with numerous images, we sample subsets of roughly equal size about `N_s / ceil(N_s/K)` and further reduce sampling noise by replacing a denominator in the penalty terms (see Penalty). At the same time, we want each batch to have sufficiently many negatives (i.e. different species) for contrastive signal.
  - DDP concerns: keep each species group on the same DDP rank and in one micro-batch. Keep a full bank copy on each GPU. When updating bank, all-gather sparse `(s, delta_s)` pairs (only species that appeared in batch) across GPU ranks so that every per-species bank increment is made on every GPU's bank copy. Bank refresh is done on every GPU copy. Each GPU normalizes `L_B` by local batch size, DDP averages gradients. All-gather features and label IDs across GPU for contrastive negatives.


### 5. Run matrix
- Arms:
  - Loss: (a) flat contrastive only, (b) flat contrastive + penalty, (c) level-restricted contrastive only, (d) level-restricted contrastive + penalty 
    - Compare: (d) vs (c), (b) vs (a)
    - Order of experiments: (c), (d), then (a), (b)
  - Fine-tuning: LoRA fine-tuning (attention q,v) with `r = 16, alpha = 32` and `logit_scale` learnable; queue (i) full fine-tuning and (ii) LoRA fine-tuning that additionally learns `visual.proj`, `text_projection`
- Max 10 epochs.
- Seed: 42

### 6. Optimization
- Time budget for pilot: 24 hours on 4 A100s.
- Batch size: globally 8192 (more if possible; want maximum GPU memory utilization during training) on 2~4 A100s. `gpu_a100` nodes are 2 A100s each; 4 gpus means 2 nodes
- Learning rate: sweep 1e-4, 3e-4 on (c) arm. Decide based on ToL-10M val accuracy (species-level, but log all ranks and the 7-rank mean) after 3 epochs. Use the full training's scheduler (total steps is counted over 10 epochs). Then run the chosen LR run to completion, and run remaining arms on the chosen LR.
- Warmup: 1~5% of total steps
- Weight decay: 0.0
- Gradient clipping: 1.0
- bf16 mixed precision (Lightning `precision="bf16-mixed"`, never `"bf16-true"`, which rounds away small LoRA and bank updates). Compute penalty in fp32, casting before any mean or subtraction, for precision working with near-identical vectors. Set `torch.set_float32_matmul_precision("highest")`: src/train.py sets `"medium"`, which runs fp32 matmuls (penalty, eval) at bf16/TF32 precision.


### 7. Evaluation
- Benchmark: iNat21 val split.
- Precision: fp16 autocast for images, fp32 for text and scoring. Matches old eval harness in repos/tol_embed.
- Per-level classification: image as input; truncated taxonomic label (see the two class string templates in section 2; "a photo of [truncation]." is the convention) as valid class strings; L2-normalize both sides' embeddings and take the argmax cosine against the image embedding. Zero-shot top-1 accuracy. Report performance on both templates (with/out "a photo of"), but the decisive one is the "a photo of [label]." used in our training.
- Evaluate zero-shot per-level classification (top-1 accuracy) on iNat21 val split at the end of training, using the checkpoint chosen based on ToL10M val.
- Eval harness: reference ~/bdbk/repos/tol_embed/scripts/zeroshot_ranks.py. Gate implementation by asserting that it reproduces exact correct counts against the JSONs, for all six model columns. Require identical n_classes, n_eval, and correct counts. JSONs: /projects/bdbk/liv/repos/tol_embed/results/ZEROSHOT_RANKS_inat21-val.json, …__forms.json, …__bfl.json.
- Pilot failure mode: Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification. This signals losing fine-grained semantics. Refer to the bar in section 1.


### 8. Monitors
- Every loss term: where applicable, the penalty and contrastive objective within every level. Penalty has six terms, level-restricted contrastive has seven terms.
- Gradient: cosine between the penalty gradient and the contrastive gradient, and their norm ratio. Shows if the penalty is doing anything.
- Embedding collapse: log the participation ratio of the batch embedding covariance `(sum of eigenvalues)^2 / sum of squared eigenvalues` (effective dimension) plus mean pairwise cosine between batch embeddings and the mean embedding norm. Collapse shows up as effective dimension falling toward 1 and mean pairwise cosine rising toward 1.
- From the bank (rather than the batch), compute and log `cos^2(m_{a_{p+1}(s)} - m_{a_p(s)}, m_{a_p(s)})` for `p=1..5`, averaged over each rank. Drop any terms where one of the nodes has a single child (not just a single species but a single child taxon). This measures representation orthogonality in consecutive ranks kingdom->phylum, phylum->class, etc. Where applicable, use whitened embeddings as for the penalty. Note that this is a mean over ranks and is expected to be noisy over coarse ranks with fewer nodes (ToL-EOL train: 12 kingdoms, 79 phyla, 283 classes, etc.).


### 9. Infrastructure
- Smoke runs:
  - One training step on 1000 images: loss is finite and both terms non-zero
  - The bank's zero-sum assertion passes on an arbitrary small subtree
  - Penalty computed twice on the same batch gives the same number
  - 100 steps with penalty weight at 0 reproduces the control arm's loss curve
  - Before training: Eval harness runs end-to-end on iNat21-val and reproduces all rows on the cached embeddings (/projects/bdbk/liv/repos/tol_embed/results/probe_cache/inat21-val/bioclip2/, and probe_cache_b16/inat21-val/<ckpt>/ for the other five), and for re-encoded images, every row at cosine >= 0.9999 against the cache.
- The bank and optimizer state need to be checkpointed alongside the model at every ckpt, because it updates with training. Reasonable checkpointing cadence works.
- Environemnt: bioclip conda env; prefer A100s; allocation from `configs/paths/default.yaml`.

### Amendments

#### Amendment 1 (2026-10-04): decisions from the spec preflight

- **Sources.**
  - Decided by the spec's owner on 2026-10-02, in section 3 of `audit/2026-10-02_spec_preflight.md`.
  - Decided on 2026-10-04, in section 6 of `audit/2026-10-04_B2_B3_check.md`, together with the readings in section 5 of that report.
  - The S9 rule was confirmed by the owner on 2026-10-04, in these words: "confirmed: the new rule for S9 "does this species field contain a real epithet?" is more exhaustive while correctly keeping species fields that contain a real species epitheti plus a placeholder/tag/author-name."
- **No results seen.** No training run or evaluation of pilot 1 had started when any item below was decided.
- **Precedence.** Where this block conflicts with the text above it, this block governs. Checkers compare the implementation against this block first.
- **References.** Line numbers refer to the text above, which is the 90a7d38 version. Item IDs (S1, S2, …) are those of the preflight report.
- **Later changes.** A later change to anything here gets its own dated amendment below this one. That includes moving a directory back to `/projects/bdbk`.

##### Paths

- **P1. Repo location.**
  - The repo moved to `/u/liv/repos/taxa_maze` on 2026-10-04, off the over-quota `/projects/bdbk` allocation.
  - The old path `~/bdbk/repos/taxa_maze` (= `/projects/bdbk/liv/repos/taxa_maze`) is now a symlink to it, and stays valid.
  - Nothing else moves: the data, the BioCLIP 1 checkpoint and tol_embed stay at the paths above, and pilot 1 only reads them. The owner decided on 2026-10-04 not to move tol_embed.
  - The store and the run outputs are the one other path change (S1).
  - Reason: VAST blocks writes to an allocation once its soft-quota grace period ends. For `/projects/bdbk` that is about 2026-10-07 14:35 CDT.

##### Data

- **S1 (lines 49, 191). The store and the run outputs.**
  - Store the extracted images as a few large uint8 memmap files (N × 224 × 224 × 3) plus a uuid-to-row index. Each crop is stored after Resize, CenterCrop and RGB conversion, and before ToTensor and Normalize.
  - The store and every run output live on the `/u` filesystem under `/u/liv`, not under `~/bdbk/data`. Run outputs are Hydra run directories, logs, checkpoints, wandb files, Slurm logs and smoke-test outputs. The same goes for `HF_HOME` and `WANDB_DIR` in the job scripts.
  - Test: a path's resolved form (`readlink -f`) must not start with `/projects/bdbk`. So nothing goes under `/u/liv/bdbk`, which is a symlink into `/projects/bdbk`.
  - The store holds every complete-lineage EOL train and val image, including the rows S9 drops, so the S9 flag can be turned off without a rebuild.
  - Reason: `/projects/bdbk` is over its soft file quota (P1). uint8 crops are exactly what the eval transform hands to ToTensor, so they are lossless (line 49).
- **S52 (lines 41-42, 47, 49, 74). Oversize images.**
  - Keep the 215 shard images that PIL refuses by default: raise `PIL.Image.MAX_IMAGE_PIXELS` when building the store.
  - Encode the 206 of them with a complete lineage fresh (fp32 weights, fp16 autocast, as the cache was made), so the bank seed (line 74) and the ToL-val dedup (S26) cover them.
  - The line-50 smoke test samples only uuids that have a cache row.
  - Line 47's val count for `image_set_60` becomes 91,225, the tar's count, because the 3 oversize val images are kept.
  - Asserted counts:
    - 5,908,775 train and 310,899 val EOL rows;
    - 5,372,586 and 282,542 after the lineage drop;
    - 5,298,907 and 278,613 after the S9 drop with the flag on.
  - Reason: lines 41-42 count these images, and the spec's only drop rule is an incomplete lineage.
- **S9 (lines 41, 42, 53). Species fields with no real species epithet.**
  - **Rule.** A row counts as missing its species rank (line 53) when its species field contains no real species epithet. Such a row is dropped from training (bank, sampler and labels) and from ToL-val before deduplication.
  - **The frozen list.** The dropped rows are exactly those whose full seven-rank key is listed in `specs/pilot1_s9_missing_species_keys.txt`.
    - The file holds 8,084 keys, one per line, sorted, UTF-8, ending with a newline. Its sha256 is `3ac09b97abc94748a3368c4d96992aaad2eb590ef63823e98213232ca31fbf64`.
    - It is built by `scripts/preflight/s9_final_keys.py` from the census in `audit/2026-10-02_S9_epithet_census.md`.
    - The list governs, including where the census missed a case.
  - **Dropped kinds** (keys, train images):
    - single-word author names (7,307, 71,302), for example `latreille`;
    - single-word placeholders (681, 1,566), for example `sp.`, `xx`, `species`, `nov.`, `?`, and cultivar names in quotes;
    - author-only strings of several words (46, 477), for example `tourn. ex`;
    - subgenus, section and series labels (21, 161), for example `subg. urostigma`;
    - `sp.` plus a tag or locality (10, 107), for example `sp. a`, `sp. 2`, `sp. silk`;
    - morph labels, cultivar markers and cultivar names (16, 46), for example `morph white`;
    - machine tags with nothing else (3, 20).
  - **Kept, because the field contains a real epithet:**
    - `cf.`, `aff.` or `nr.` plus an epithet;
    - an epithet plus a doubt mark (`gracilis?`);
    - an epithet with an author, tag or placeholder appended (`cabritii fischer`, `sagittifolia spp.`);
    - an epithet with a stray period (`deliciosa.`);
    - subspecies, hybrids, and `sp.` plus an epithet (`sp. attenuata`).
  - **Flag.** A config flag controls the S9 drop, and every run logs its value. With the flag off, no S9 row is dropped; the lineage drop of lines 41, 42 and 53 still applies. With the flag on, 73,679 train and 3,929 val images are dropped. Log the dropped keys and image counts.
  - **Recounts.** Every count derived from the image or species lists is computed after the drop:
    - steps per epoch: 647 at B = 8192, which is 646 full batches plus one short batch, as in preflight reading S7;
    - warmup steps: 65;
    - group counts, and the held-out 5% (line 143).
  - Line 94 counts the λ window in steps, so it stays at steps 50-250.
  - Reason: such a field names no species. As a species label, it puts a fake class beside the real species of its genus in the contrastive loss, and a non-species leaf in the penalty. The largest one, `Bombus|latreille` (10,152 images), was a genus label.
- **S26 (line 42). ToL-val dedup.**
  - Use the frozen BioCLIP 1 cache, L2-normalized in fp32 or higher, with the full seven-rank key, against complete-lineage train images.
  - Compute it once, after the S9 drop, save it, and log the count. The 206 oversize images get fresh fp16 encodes for this step (S52).
  - Reason: only the BioCLIP 1 cache exists before training, and a fixed set keeps the LR and checkpoint choices comparable.

##### Model and loss

- **S20 (line 61). logit_scale.** It is learnable, and clamped to [0, ln 100] after every optimizer step. It starts at ln 100, so it can only fall. Reason: this follows the training convention of BioCLIP 1's code.
- **S12 (lines 70, 92, 187). Zero-sum tolerance.**
  - The bank is fp64. The zero-sum check uses an absolute tolerance of 1e-10. This replaces line 70's "exact" and line 92's "==": rounding is allowed in fp64 too.
  - On a small real subtree (for example, a family with at least 3 genera and uneven species counts), after at least 100 real updates, assert for every node that max |m_a − (1/n_a) Σ m_s| ≤ 1e-10 and |Σ_c n_c (m_c − m_a)| ≤ 1e-10 × n_a.
  - Apply the same check during training, and log the largest value seen.
  - Reason: incremental and fresh means add in different orders, so they never match bit for bit. Gaps reach about 3e-15 in fp64, while a real bug moves m_a by about 1e-6 or more.
- **S21 (lines 52, 102, 181). Penalty skip rule.** The penalty uses P_s exactly as line 102 defines it, including ancestors that have one child taxon. Line 52's single-child rule applies only to the bank monitor (line 181). Log line 52's single-child counts per rank. Reason: line 143 names P_s as the definition.
- **S22 (lines 100, 107, 117). Penalty weighting.**
  - Each in-batch species term is weighted by 1/g_s, where g_s is the species' number of groups per epoch: L_B = (1/B) Σ_{s∈B} ℓ_s(muhat_s) / g_s. This matches L* (line 100) up to a constant.
  - Line 117's uniform weight within a batch is replaced by 1/g_s, and n_batch is not applied as an extra factor.
  - Line 107's unweighted form stays available behind a config flag, which is recorded in each run's config. A test checks that both forms agree when every g_s = 1.
  - Pilot 1 runs only the 1/g_s form, and λ is calibrated under it. A run with line 107's form needs its own λ calibration.
  - Note: the penalty is species-uniform, so it may barely move individual images of head species.
  - Reason: under line 107, a species weighs as much as its number of groups (up to 299 at K = 16), unlike the canonical L*.
- **S15 (lines 115, 178). Controls keep the bank.**
  - Arms (a) and (c) keep the full bank: EMA updates and the per-epoch refresh.
  - They log every monitor: the penalty and its six terms, the gradient monitor (S16) and the bank monitor. The penalty stays out of their loss.
  - Reason: λ (line 94) can only come from the sweep this way, and all four arms then have the same monitors.
- **S25 (lines 91, 122, 146, 165). Bank refresh.**
  - The per-epoch refresh is split across GPUs, under fp16 autocast with fp32 weights, instead of line 165's bf16. Values are cast to fp32 or fp64 before normalizing and averaging. fp64 per-species sums and counts are all-reduced, so every GPU holds the same bank.
  - Seed the bank from the cache, plus fresh encodes of any train image the cache lacks (S52). The BioCLIP 1 cache counts as line 122's full pass.
  - Apply EMA deltas in a fixed order. Log the step-0 gap between the bf16 and fp16 species means as the drift floor.
  - Reason: this is the most exact encode that fits line 91's 2 GPU-hour bound, and it matches the seed cache and the eval precision.

##### Batching

- **S2 (lines 141-142).** K = 16 in every arm. There is no K sweep.
- **S8 (lines 144, 159-160).** B = 8192 in every run, on 4 A100s: 2 nodes × 2 GPUs, 2,048 per GPU. B is not raised. Report the spare memory, which goes to checkpointing fewer blocks.
- **S3 (lines 84, 142-143).** Under S2 and S8 no extra rule is needed. The sampler puts each species' groups in distinct batches, so a batch holds at most one group per species, and an assertion checks this every epoch.
- Reason for all three: K = 16 at B = 8192 satisfies lines 84, 142 and 143 for every species. After S9 the largest species, `Pandion|haliaetus` (4,775 images), needs 299 groups, against 647 steps per epoch.

##### Optimization

- **S19 (lines 158-165).** AdamW: betas (0.9, 0.98), eps 1e-6, weight decay 0, one parameter group (LoRA and logit_scale share the LR). Reason: this is the open_clip default for ViTs that BioCLIP 1 used.
- **S5 (lines 161-162).**
  - Linear warmup over 1% of total steps (65 steps after S9), then cosine decay to 0 at the end of epoch 10, updated every optimizer step.
  - The same schedule applies to both sweep runs and every arm, and is fixed before the sweep.
  - Reason: this is BioCLIP 1's schedule. The shortest allowed warmup keeps nearly all of line 94's window at the chosen LR.
- **S6 (line 94). λ calibration.**
  - Each penalized arm gets its own λ = 0.2 / mean_t r_t, over steps 50-250. Here r_t = ‖∇L_B‖ / ‖∇L_con‖, without λ, as measured by the S16 monitor.
  - λ for (d) comes from the chosen (c) sweep run, and λ for (b) from arm (a). Log the realized ratio λ·r_t in (b) and (d).
  - Reason: line 94 fixes the ratio at 0.2, and the flat and level-restricted losses have different gradient sizes.
- **S29 (lines 155, 159, 161). Time budget.**
  - The 24 hours are an estimate, not a cap. Measure the step and refresh times in the smoke runs and report the projected total before the sweep.
  - Do not stall a run waiting for approval of the hours.
  - Do not cut epochs to fit the budget: every arm trains all 10 epochs. The losing sweep run still stops after epoch 3 (line 161).
  - Reason: line 161 runs the chosen schedule to completion over 10 epochs, and the spec has no rule for cutting a run short.

##### Evaluation

- **S4 (lines 161, 165, 172). ToL-val accuracy.**
  - At each rank, the candidates are the taxa present in the deduplicated, complete-lineage val set (after S9), and all its images are scored.
  - Template "a photo of [label].". Precision as line 170: fp16 autocast for images, fp32 for text and scoring. Lightning's bf16 autocast (line 165) is kept out of validation.
  - Also log the lineage template and seen-species-only accuracy, as extras.
  - Reason: this applies the iNat21 rule to ToL-val. The candidates are the benchmark's own taxa, and the template and precision are the decisive ones of lines 170-171.
- **S27 (line 172). Checkpoint selection.**
  - The iNat21 checkpoint is the epoch with the best species top-1 on deduplicated ToL-val with "a photo of [label].", over the ends of epochs 1-10. A tie goes to the earlier epoch.
  - Also log the seven-rank mean, and epoch 0 (untouched BioCLIP 1) as a reference only.
  - Keep every epoch's checkpoint (model, LoRA, bank and optimizer).
  - Reason: line 161 uses species-level ToL-val accuracy for the only other ToL-val decision, and line 171 makes "a photo of" the decisive template.
- **S31 (line 37). The ±1-point bar.**
  - Apply it per rank, with three labels: better (ahead by more than 1 point), matched (within 1 point), worse (behind by more than 1 point).
  - "Beating" means better. "Competitive" and "matched or better" mean matched or better.
  - The baselines are the table cells: RCME lineage, BFL-Euc photo and BioCLIP 1 photo. "BFL" in the genus and species clause means BFL-Euc.
  - Compare integer counts of correct images.
  - Reason: a ±1 margin sorts results into better, tie and worse. "Levels" in line 37 reads per rank, and "(see table)" points to the cells. Integer counts avoid rounding traps.
- **S32 (lines 37, 174). Floor versus margin.**
  - The two rules stay separate. The floor is BioCLIP 1's count, 70,186 correct, with no margin.
  - Flag the failure mode if species correct < 70,186 while the coarse ranks improve over BioCLIP 1.
  - In the band of 69,186-70,185 correct, report "matched under the bar, below the floor" and make no call.
  - Reason: both rules stay as written. Line 37 says "If confused, don't make a verdict and just report everything faithfully", and line 174 names BioCLIP 1's own value.

##### Monitors

- **S16 (line 179). Gradient monitor.**
  - Calibrate on all trainable parameters: LoRA in both towers, plus logit_scale. Also log the ratio and cosine for the image-tower LoRA alone.
  - Take each loss's gradient from its own backward pass on the same forward. Average it across GPUs by hand, as DDP would, and read it before clipping.
  - Keep the number of GPUs fixed. Use the contrastive all-gather that passes gradients back, so the summed gradient equals the one-GPU gradient (preflight reading S17).
  - Log at every step from 50 to 250, then every 50 steps.
  - Reason: the gradient that the optimizer and clipping see covers every trained parameter, and line 61 trains logit_scale. The image-only log shows whether the text tower or logit_scale drives the ratio.
- **S24 (line 181). Bank monitor.**
  - Log four numbers per rank: two drop rules times two averages.
    - The literal rule drops a term when either node has a single child. The parent rule drops a term when the parent has a single child, whatever the child has.
    - The averages are over unique parent-child edges, and over species.
  - The headline is the parent rule, averaged over unique edges.
  - Compute it on the refreshed bank after each epoch, and on the EMA bank at log steps.
  - This monitor's definition is ambiguous, and its numbers deserve closer scrutiny from the spec's owner. No decision rule uses it.
  - Reason: the owner's choice. The step is exactly zero only when the parent has a single child.

##### Smoke tests

- **S13 (line 189). λ = 0 reproduces the control.**
  - Run in deterministic mode: `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Log which operations warn.
  - Run the control twice. If the two runs are bit-identical, the λ = 0 run must be bit-identical too. If not, its per-step loss gap (total and each level) must stay within the largest gap between the two control runs.
  - Test (d) at λ = 0 against (c), and (b) at λ = 0 against (a).
  - Use seed 42, the same data order, B = 8192 on 4 GPUs and LR 1e-4, with the full penalty code running: bank EMA, all-gather and monitors.
  - Reason: "reproduces" most naturally means exact equality. This demands it whenever the hardware can deliver it, and otherwise uses the measured run-to-run noise as the bar. Running the full penalty code shows that λ = 0 leaks nothing into training.

#### Amendment 2 (2026-10-05, M1): decisions after the M1 checker

- **Sources.** Decided by the spec's owner on 2026-10-05, after reading the M1 checker report (`audit/2026-10-05_M1_checker.md`) and the response to it (`audit/2026-10-05_M1_checker_response.md`). The owner's words are recorded in `agent/ACTIVE.md`, asks A11 and A12. P and O numbers below are the checker report's item IDs; S numbers are Amendment 1's.
- **Results seen.** No training run of any arm had started. Seen: the M1 smoke-test and timing numbers (the six S13 runs, the single-GPU tests, the 2-node timing run and the drift-floor passes), and the eval gate's re-encode of the six baseline checkpoints under the harness's default setting (job 257672), including BioCLIP 1's species count of 70,193 correct against the cached 70,186.
- **Precedence.** Where this block conflicts with the text above it, including Amendment 1, this block governs.

##### Run geometry

- **A2.1 (S8; lines 144, 159-160; checker P3).** B = 8192 on 2 A100s on one node, that is 1 node × 2 GPUs with 4,096 images per GPU, for every run from M2 on. All 24 transformer blocks of each tower stay activation-checkpointed; the spare memory goes to the larger per-GPU batch, not to checkpointing fewer blocks. The M1 smoke runs (S13, grad_1v4) were made on 2 nodes × 2 GPUs and stay the M1 evidence; S16's fixed GPU count holds within every run, and the grad_1v4 test shows that the gradient does not depend on the number of GPUs. No run measures the new geometry before the sweep. Reason: one-node jobs schedule more easily on `gpu_a100` and need no inter-node communication (the 2-node timing run had 18 unexplained stalls of 32-591 s), and peak memory at 2,048 images per GPU was 24.9 GiB of 80.

##### Time budget

- **A2.2 (S29; lines 155, 159; checker O1, O3).** No dedicated measurement of step and refresh times before the sweep, and no projected total before it. The sweep runs report their own step, refresh and evaluation times as they go. The end-to-end smoke run (60 steps, then refresh, ToL-val and a checkpoint) runs alongside the sweep to exercise the epoch-end path, not to time it, and the sweep does not wait for it. The rest of S29 stands: every arm trains all 10 epochs, and no run stalls for approval. Reason: the owner's choice; the measurement feeds no decision.

##### Epoch numbering

- **A2.3 (S27; lines 161, 172, 191; checker O7).** Checkpoint files and `decisions.json` use Lightning's 0-based epoch index: `epoch_00.ckpt` to `epoch_09.ckpt` are the ends of S27's training epochs 1-10, and `epoch_02.ckpt` is the end of the sweep's third epoch (line 161). The untouched model's reference evaluation is keyed `init` (metrics `val_init/...`). The S27 pick is over `epoch_00` to `epoch_09`, ties to the earlier file. The per-step log's `epoch` field uses the same index. Reason: one convention everywhere, matching Lightning's.

##### Evaluation

- **A2.4 (S31, S32; lines 26-37, 173-174; checker O2).** The M3 iNat21 evaluation records its encode setting: the cuDNN benchmark and deterministic flags, the batch size, the autocast dtype and the library versions. All six baseline columns of lines 26-35 are re-evaluated by the same harness under the same setting during M3, and S31's comparison cells and S32's floor are taken from these re-evaluated integer counts of correct images, not from the cached table. The setting is the harness default: fp32 weights, fp16 autocast for images, fp32 text and scoring, batch 512, `cudnn.benchmark` off, `cudnn.deterministic` on. The cached table and its exact reproduction (line 173, gate Part A) stay as the harness gate, and S32's "70,186" now reads "BioCLIP 1's re-evaluated species count". Reason: two encodes of the same weights differ by up to about 17 correct images per rank from cuDNN's algorithm choice alone, and the S32 floor has no margin.

##### Decisions recorded without a rule change

- **P2 (S16).** The logged gradient cosine is computed with autocast off, in fp64. The norms and the ratio r_t were already fp32.
- **O6 (line 165; code-review note N-9).** The model is put in train mode at the start of training; the refresh and the evaluations restore the mode they find. No numeric effect: dropout is 0, there is no batch norm, and the attention fast path is off under autocast.
- **P4 (S9).** The flag-off S9 path (prep files and loaders) stays unimplemented; the owner holds this decision.
- **O4.** The code behind M1 is committed and pushed before the first M2 job starts; `.gitignore` no longer ignores `scripts/`.
- **O9.** The documentation inaccuracies the checker found are fixed; small clock-stamp discrepancies are left as they are.
