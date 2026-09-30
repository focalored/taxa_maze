Written 2026-09-30.
(sha256 8f2ee5ea1769ed2effa0ec55a8e7f844b9d69c5541695a46a9bf9e174a1bfd0f).
Do not edit. Changes go in an Amendments section at the bottom, each dated, with the reason
and whether any results had been seen.

Numbers and paths I give you are from memory. Verify each one against disk before relying on it, and stop and tell me if one does not match.

Copy ~/bdbk/repos/tol_embed/src/open_clip before starting. Do not install via standard pip.

Starter files to reference **read-only** for implementation at ~/bdbk/repos/tol_embed/scripts/:
- encode_cache.py: refactor, it is too long. If it helps, split logic for encoding and caching inat21 and tol10m, and make room for refactoring for tol200m in the future.
- data_paths.py: prioritize this repo's configured paths over what's defined here.

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