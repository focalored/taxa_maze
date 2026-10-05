# Documentation For Usage & Operations
This document should map out current state of the codebase, guidelines for usage, and pointers on how to reproduce core functionalities. A human or agent should be able to read this document and cleanly be able to set up everything including data preprocessing, training, evaluation, ablation studies, and visualization. This document should be regularly pruned to ensure it gives recommendations that describe the code as it exists now rather than as it once did.

Below the line:
- **Harness gaps**, first and brief: what is referenced but not yet implemented, so a fresh clone knows what will fail. Delete rows as they are filled, then the section.
- **Environment**: conda env, every env var the configs read and what breaks without it, marker files the entrypoint requires, where datasets live on this cluster.
- **Quickstart** in increasing cost: a CPU smoke test that exercises the full path, single GPU, multi GPU. Copy-pasteable as written.
- **Config composition**: the entrypoint config, the data contract it enforces, package syntax for wiring datamodules into train/val/test, group vs leaf overrides, and any ordering that leaks into behaviour.
- **Data**: one row per datamodule — expected input layout, what it is for, which is the default. Then the preprocessing to reach that layout, and per-module behaviour that silently changes what the model sees (filters, dedup, resampling, how a streaming epoch is defined).
- **SLURM**: partitions and which to prefer, the allocation to charge, a working sbatch script, and the launcher/config invariants that must agree.
- **Outputs**: run directory layout, what lands where, how loggers are selected.
- **Evaluation**: how to reproduce a reported number, and how the reported protocol differs from any cheaper in-training proxy. Conflating the two makes SOTA.md meaningless.
- **Sweeps and visualization**: multirun usage, one experiment config per claim, and how tables and figures are produced (provtab for LaTeX).
- **Gotchas**: the sharp edges that cost an hour — defaults that are not what you want, values that must be set per experiment, prompts that hang batch jobs, anything pinned at import time.
- **Where things go**: the directory map, so new code and scratch work land correctly without asking.

Maintenance:
- Update in the same turn as the change that invalidated it; a wrong instruction here gets followed.
- Document only what has actually been run. Mark untested commands as untested, or leave them out.
- Cite `path:line` for claims about behaviour, and re-verify pointers when the target file changes.
- Write down every path a result depends on — `agent/`, `logs/`, `audit/`, `*.json`, `*.parquet`, `*.ckpt` are gitignored.
- Delete stale instructions rather than annotating them. Exception: a retired procedure that produced an artifact still in use goes to `agent_legacy/usage/` with a note on what replaced it.
- Keep it readable start to finish. A section outgrowing that should push detail into a docstring or a runnable `scripts/` example, leaving a pointer here.

##### MODIFY UNDER THIS LINE #####

## Harness gaps
- The first official runs are the M2 LR sweep: arm (c), 3 epochs each, at 1 node × 2 A100s. lr 1e-4 is job 264726 (`p1c_lvl_lr1e-4_s42_v1`, started 2026-10-05 14:08 from commit `7c1832f`, old `np.memmap` reader). lr 3e-4 is job 265500 (`p1c_lvl_lr3e-4_s42_v2`, started 17:12 from `eba8444`, `os.pread` reader); it replaces 264727, cancelled after a loader stall, and uses `version=2` so its wandb id is fresh. Before them, the training step, bank, monitors and schedule had run for 100 steps in the six S13 runs (`logs/smoke/lam0/`, 2 × 2 A100s). The per-epoch refresh, ToL-val evaluation and checkpointing completed once on the pre-commit code at 2 × 2 (job 263082, 2026-10-05 02:11: 60 steps, refresh 2,467 s, ToL-val 83 s, `epoch_01.ckpt` under the old 1-based naming). The end-to-end smoke job 264728 (`configs/experiment/p1/smoke_timing.yaml`) timed out at 17:16, stuck in its refresh on the old reader.
- The λ (`model.lam`) of arms (b) and (d) is `???` until the matching control's `decisions.json` supplies `lambda_calibration.lam_0.2` (Amendment 1 S6).
- `src/data/{coco_local,image_text,parquet,webdataset}_module.py` and `src/data/processing.py` come from an unrelated earlier codebase. Nothing in pilot 1 uses them.

## Environment
- The spec is `specs/pilot1.md`. Its amendments at the bottom govern wherever they conflict with the text above them, the later one first: Amendment 1 (2026-10-04, the preflight decisions) and Amendment 2 (2026-10-05, after the M1 checker: 1 node × 2 GPUs, 0-based epoch numbering, baselines re-evaluated under the M3 encode setting, no S29 measurement). S9's frozen list of dropped species keys is `specs/pilot1_s9_missing_species_keys.txt` (8,084 keys; the sha256 is in Amendment 1).
- Repo location (since 2026-10-04): `/u/liv/repos/taxa_maze`, in the home quota. The old path `~/bdbk/repos/taxa_maze` (= `/projects/bdbk/liv/repos/taxa_maze`) is a symlink to it and still works, so older docs keep it. It moved because `/projects/bdbk` is over its soft file quota, and VAST blocks writes there once the grace period ends (B3, resolved 2026-10-04; block in `agent_legacy/active/2026-10-04_asks-A4-A6_blocks-B2-B3.md`). Home's ACL admits only `liv`, so collaborators use GitHub (`focalored/taxa_maze`).
- Conda env `bioclip` at `/u/liv/envs/bioclip`: python 3.10, torch 2.4.0+cu121, numpy 1.26.4, pandas 2.2.2.
- Added on 2026-10-02 with every existing package pinned (`pip install -c <pip freeze>`), so nothing was upgraded: lightning 2.6.6 (with pytorch-lightning 2.6.6, torchmetrics 1.9.0, lightning-utilities 0.15.3), hydra-core 1.3.7, omegaconf 2.3.1, hydra-colorlog 1.2.0 (with colorlog 6.12.0), rootutils 1.0.7, rich 15.0.0, plus their dependencies (aiohttp and friends, antlr4-python3-runtime 4.9.3, python-dotenv 1.2.4, Pygments 2.21.0). `rich` was not on the original list. The template's `RichModelSummary` callback and `src/utils/rich_utils.py` import it. Also added, the same way, for the wandb logger: wandb 0.30.0 with pydantic 2.11.10 and the opentelemetry packages (15 packages in all).
- Marker file: `.project-root` at the repo root. `src/train.py` calls `rootutils.setup_root(..., indicator=".project-root")`. That sets `PROJECT_ROOT` to the repo root inside the process, overriding the `PROJECT_ROOT=/u/liv/bdbk` that `~/.hpc_env.sh` exports. `configs/paths/default.yaml` builds `root_dir`, `data_dir` and `log_dir` from it, so run outputs land in `<repo>/logs/<task_name>/runs/<stamp>_j<jobid>/` by default; every pilot-1 experiment sets its own `hydra.run.dir` (see "Run naming" and the Quickstart). `paths.data_dir` resolves to `<repo>/data/`, not `~/bdbk/data`, so dataset paths must be configured explicitly.
- Default logger: `csv` (`configs/train.yaml`). Official runs use wandb (see "Outputs" below).
- Slurm jobs source `scripts/env.sh` first. It runs `source "$HOME/.hpc_env.sh"` and `conda activate bioclip`, sets `HF_HOME=/u/liv/.cache/huggingface`, `WANDB_DIR=<repo>/logs/wandb` and `PYTHONPATH=<repo>`, then changes into the repo. Account `bdbk-tgirails` (`configs/paths/default.yaml: allocation`).
- Pilot-1 paths live in `configs/paths/default.yaml`: `tol10m_dir`, `inat21_dir`, `bioclip1_ckpt` and `tol_embed_dir` are read only, and `p1_data_dir` (`/u/liv/data/taxa_maze`) holds what pilot 1 writes. Scripts outside Hydra read them through `src/utils/paths.py:load_paths`. `assert_outside_projects` refuses any write that resolves under `/projects/bdbk` (Amendment 1 S1).

## Quickstart (pilot 1)
- Smoke tests (1 GPU, A100): `sbatch --account=bdbk-tgirails --partition=gpu_a100 --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=00:45:00 --wrap "bash scripts/smoke/p1_smoke.sh <test>"` with `<test>` one of `adapter`, `cache`, `one_step`, `penalty_twice`, `penalty_forms`, `zero_sum`, `grad_1v4` (on `gpu_a100`, `grad_1v4 --world-run 4` needs `--nodes=2 --ntasks-per-node=2 --gres=gpu:2` and `srun`, as job 257774 did). Each writes `logs/smoke/<test>/<stamp>_j<jobid>/{result.json,log.txt}` and exits nonzero on failure; for `grad_1v4`, `result.json` comes from whichever of the 1-GPU and 4-GPU halves runs second.
- S13 λ = 0 runs (100 deterministic steps; the M1 runs were made on 2 × 2 A100s before Amendment 2, so a repeat of that layout needs `trainer.num_nodes=2` and `sbatch --nodes=2`): `sbatch --time=01:00:00 scripts/p1/train.sh experiment=p1/smoke_lam0 model.arm=<a|b|c|d> tag=<name>` with `tag` one of `c_run1`, `c_run2` (arm c), `d_lam0` (arm d), `a_run1`, `a_run2` (arm a), `b_lam0` (arm b); then `python scripts/smoke/s13_compare.py` reads those six runs' `steps.jsonl` (100 steps each) and writes `logs/smoke/lam0/s13_result.json`.
- End-to-end smoke (Amendment 2 A2.2; runs alongside M2, not a timing measurement): `sbatch --time=03:00:00 scripts/p1/train.sh experiment=p1/smoke_timing` (60 steps of arm (d) at λ = 1, then refresh, ToL-val and a checkpoint).
- LR sweep (M2, line 161): `sbatch --job-name=p1c_lr1e-4 scripts/p1/train.sh experiment=p1/c_lvl model.lr=1e-4 trainer.max_epochs=3`, and the same with `3e-4`. The schedule stays the 10-epoch one (`model.epochs: 10`: 6,470 steps, warmup 65, cosine decay to about 6e-8 × base at the last step, 6,469); `trainer.max_epochs` only stops the run, after `epoch_02.ckpt`. The winner continues with `ckpt_path=logs/p1/<run_name>/checkpoints/last.ckpt trainer.max_epochs=10` (same `run_name`, same wandb id).
- Official run: `sbatch scripts/p1/train.sh experiment=p1/c_lvl model.lr=1e-4` (see "Run naming"). Penalized arms need `model.lam=<value>`.

## Config composition
- `configs/train.yaml` composes `model: p1` and `data/p1_tol@data.train`; pilot 1 has no `data.val.*` datamodule because the ToL-val evaluation runs inside `P1Module.on_train_epoch_end`, so `src/train.py` calls `trainer.fit(model, datamodule=train_dm)` when `cfg.data.val` is empty.
- `configs/experiment/p1/{a_flat,b_flat_pen,c_lvl,d_lvl_pen}.yaml` set the arm, `trainer: p1_ddp` (1 node × 2 GPUs with 4,096 images per GPU since Amendment 2 A2.1, bf16-mixed, `ManualSyncDDPStrategy`: a DDP process group without the gradient-averaging wrapper, because the module averages gradients by hand), `callbacks: p1` (every epoch kept) and `logger: wandb`. `smoke_lam0.yaml` and `smoke_timing.yaml` are the smoke variants (csv logger, refresh and evaluation off in `smoke_lam0`).
- `run_name` is composed in the experiment file from `model.lr` through the `sci` resolver registered in `src/train.py` (`1e-4`, `3e-4`, `2.5e-4`). `src/train.py` refuses a fresh start when `logs/<task_name>/<run_name>/checkpoints/` already holds files; resume with `ckpt_path=...` or bump `version`.
- Every run writes `run_info.json` (arm, λ, LR, S9 flag, K, B, world, steps per epoch, warmup, parameter counts), `data_stats.json`, `steps.jsonl` (every logged scalar per step as an exact float64 hex; its `epoch` field is Lightning's 0-based epoch index, the same index that names the checkpoints `epoch_00.ckpt` to `epoch_09.ckpt` and keys `decisions.json`, where `init` is the untouched model's reference evaluation; Amendment 2 A2.3) and, whenever ToL-val runs (so not in `smoke_lam0`), `decisions.json` (per-epoch ToL-val counts for both templates, the r_t series and the λ calibration) into the Hydra run directory.

## Data
### Pilot 1 inputs (`/u/liv/data/taxa_maze/p1/`, built 2026-10-04 and 2026-10-05 by `scripts/p1/prepare.sh`)
- `heldout_species_s9.txt`: the 5% held-out species (18,322 of 366,439), drawn once with seed 42; shared by all arms.
- `oversize_emb.f16.npy` and `oversize_uuids.txt`: fresh BioCLIP 1 encodes (fp32 weights, fp16 autocast) of the 206 store rows the cache lacks (S52).
- `bank_seed_s9.npy` (366,439 × 512 fp64) with `bank_seed_s9.json`: per-species means of L2-normalized cache rows (5,298,714 from the cache, 193 fresh); the json records the species-list sha256 the training code checks.
- `tolval_dedup_drop.txt` and `tolval_dedup.json`: S26, 10,004 of the 278,613 post-S9 val images have cosine ≥ 0.98 to a same-species train image and are dropped; 268,609 remain.
- `drift_floor.json`: S25, the step-0 gap between bf16 and fp16 species means (parts 1 and 3 from job 257594, parts 0 and 2 from the rerun 262306, merge 262307; written 2026-10-05 00:30; see SLURM, "Floor rerun").
- `data_stats_train.json`: the train tree after S9: 366,439 species, 5,298,907 images, 11 / 77 / 281 / 1,267 / 7,100 / 67,859 nodes per rank, 2,166,307 valid penalty terms, 576,825 groups per epoch at K = 16.
- The S9 flag is `data.train.s9_drop` (default true). A flag-off run is not possible yet: the datamodule stops with `NotImplementedError`, and `prepare.py` hard-codes the drop and writes its `_s9` files into `p1/`. Pilot 1 runs with the flag on; the flag-off path stays unimplemented while the user holds the decision (Amendment 2, P4).

### Image store (`/u/liv/data/taxa_maze/store/`, built 2026-10-04)
- Contents: every complete-lineage EOL train and val image, 5,372,586 + 282,542 = 5,655,128 rows (851 GB), including the rows S9 drops (Amendment 1 S1, S52).
- Layout:
  - `shards/image_set_XX.u8`: raw uint8, shape (n, 224, 224, 3), rows in tar order.
  - `shards/image_set_XX.uuids.txt`: one uuid per row.
  - `shards/image_set_XX.done.json`: per-shard counts, timings and the oversize list.
  - `index.parquet`: uuid, split, shard, row.
  - `catalog.parquet`: the 6,219,674 EOL train/val catalog rows with the seven ranks and `complete`.
  - `store_summary.json`: the merge checks.
  - `not_in_cache.csv`: the 206 rows the BioCLIP 1 cache lacks (193 train, 13 val).
- Each crop is taken after Resize(224, bicubic), CenterCrop(224) and RGB conversion, and before ToTensor. The decode is the cache's: `Image.open`, `load()`, `convert("RGB")`, with PIL's pixel limit lifted and truncated JPEGs rejected (`scripts/store/build_store.py`). `src/data/tol_store.py:to_model_input` finishes ToTensor and Normalize. The M1 code review decoded 300 rows each of shards 01, 60 and 63 from the finished store: 900 of 900 bit-identical to `preprocess_val`, max difference 0.0 (`audit/2026-10-04_M1_code_review.md`).
- Reading: `ImageStore` (`src/data/tol_store.py`) reads each row with `os.preadv` from one descriptor per shard and process, with `POSIX_FADV_RANDOM`, rows in file order, and `POSIX_FADV_DONTNEED` after each row so reads leave no page cache (since 2026-10-05, commit `eba8444`). It returns bytes identical to the earlier `np.memmap` reader on the production loaders. Do not read the store through `np.memmap` in DataLoader workers: with every worker mapping the whole ~850 GB store over NFS, workers stalled in the kernel for up to 3 h (`audit/2026-10-05_lr3e-4_stall.md`).
- Rebuild: `bash scripts/store/build_store.sh`. It submits the catalog step, a 63-task array on `cpu` (16 CPUs, 48 GB each, about 4-10 minutes per shard) and the merge. Logs go to `logs/store/`. The 2026-10-04 build was jobs 257377 (array) and 257378 (merge), about 20 minutes wall time. Merge checks: every complete uuid landed exactly once; 6,219,674 tar members, of which 564,546 incomplete were skipped; 0 decode errors; the 206 images PIL refuses by default equal the 206 rows missing from the cache.

## Evaluation
### iNat21-val harness and gate (spec lines 170-173, 190)
- `src/eval/zeroshot.py` is the shared harness and copies tol_embed's `scripts/zeroshot_ranks.py` operation by operation:
  - class strings come from `class_text(parts, form)`; "photo" is `"a photo of <lineage>."`, "lineage" has no template and no period;
  - text is fp32 with no autocast, in batches of 256;
  - scoring is the fp32 cosine argmax in chunks of 8,192;
  - matmul precision "highest" is asserted.
- `src/eval/checkpoints.py` registers the six table checkpoints. bioclip2 loads with `local-dir:` from the read-only HF snapshot under `/u/liv/bdbk/cache/hf/hub/`, so nothing is downloaded. `src/data/inat21.py` holds the val images in the caches' row order.
- Gate: `mkdir -p logs/smoke/eval_gate && sbatch scripts/smoke/eval_gate.sh`. It writes `result.json` and `gate.log` to `logs/smoke/eval_gate/<stamp>_j<jobid>/`. Report: `audit/2026-10-04_eval_gate.md`.
  - Part A replays the cached embeddings: 84 of 84 cells (6 models × 2 templates × 7 ranks) match the tol_embed JSONs exactly (job 257672).
  - `logs/smoke/eval_gate/baseline_counts.json` holds the six table columns as integer correct counts reproduced from the caches (gate Part A). Since Amendment 2 A2.4, the S31 comparison cells and the S32 floor come from a re-evaluation of all six baselines under the M3 encode setting (fp16 autocast for images, batch 512, cuDNN benchmark off, deterministic on), recorded with the M3 eval; the cached counts (floor 70,186) stay the harness gate.
  - Part B re-encodes 600,000 images (all six models) and counts as passed on job 257703, by the user's decision of 2026-10-04. That run used the caches' own settings (`--batch 256 --cudnn-benchmark`), and every row was bit-identical to the caches.
  - With the default batch 512 and benchmark mode off (job 257672), 6 rows fall below 0.9999, the lowest at 0.99978. That comes from fp16 rounding in a different convolution algorithm.
- cuDNN rule of thumb: with `torch.backends.cudnn.benchmark = False` (PyTorch's and Lightning's default, used in training) fp16 encodes are reproducible from run to run but not bit-identical to the caches. Two encodes of the same model can differ by up to about 17 correct images per rank from the algorithm choice alone.

### ToL-val (in training; Amendment 1 S4, S27)
- `src/eval/tolval.py` runs before training (the untouched model, logged as `val_init/...` and keyed `init` in `decisions.json`) and after each epoch's refresh (`val/...`, keys `0` to `9`). It scores the 268,609 deduplicated, post-S9 val images against every taxon present in that set. Images are encoded in fp16 autocast with fp32 weights; text and scoring are fp32. Both templates are scored, and the seen-species subset is a separate extra.
- The decisive number is `val/tol/photo/species`. `decisions.json` holds the integer counts per epoch, form and rank, plus each epoch's checkpoint file. The S27 pick (best species count over `epoch_00` to `epoch_09`, ties to the earlier file; Amendment 2 A2.3) reads that file.
- Untouched-model reference (`init` since A2.3; timing job 257654 ran the earlier code, which keyed it `"0"` and logged `val_epoch0/`): BioCLIP 1 photo species 108,062 of 268,609 (40.23%). That is far below its iNat21 figure because every val species is a candidate, not 10,000.

## SLURM
- Partitions: `cpu` for the store, prep and comparisons; `gpu_a100` (2 A100s per node) for everything with a GPU. The H100 `gpu` partition needs the user's permission for every job (ACTIVE.md A10).
- Job scripts: `scripts/store/build_store.sh`, `scripts/p1/prepare.sh`, `scripts/p1/train.sh` (1 node × 2 GPUs, `srun python src/train.py "$@"`; the old 2 × 2 layout needs `--nodes=2` and `trainer.num_nodes=2`), `scripts/smoke/p1_smoke.sh`. All source `scripts/env.sh` first. `scripts/preflight/` keeps only `s9_final_keys.py` (named by Amendment 1 S9) with its input generator `s9_epithet_census.py`, and `eval_anova_tol.{py,sh}` (the Σ_W / Σ_B split for the whitening question); the other preflight helpers were retired on 2026-10-05 to `agent_legacy/scripts/` (README there).
- Lightning reads the geometry from Slurm (`SLURM_PROCID`, node list), so `trainer.devices × trainer.num_nodes` must equal `--nodes × --ntasks-per-node`; `src/train.py` aborts otherwise.
- Deterministic smoke runs (`trainer.deterministic: warn`) make Lightning set `CUBLAS_WORKSPACE_CONFIG=:4096:8` itself; two backward kernels still warn that they are non-deterministic, flash attention and memory-efficient attention (recorded per run in `s13_result.json`).
- Floor rerun. If a `p1prep-floor` array task times out, its dependents show `DependencyNeverSatisfied`: `scancel` them, then reuse the `sbatch` lines of `scripts/p1/prepare.sh` with three changes: distinct nodes for the parts (two parts on one node did not finish in 2 h, against 25-28 min for one part alone, 2026-10-04): one job with `--nodes=2 --ntasks-per-node=1 --gres=gpu:1` running `srun bash -c '... floor --part $((SLURM_PROCID * 2)) --parts 4'` guarantees that (job 262306); `--time=03:00:00`; and the merge's dependency set to `afterok:<that job>`. Then `sbatch --dependency=afterok:<merge> --time=03:00:00 --job-name=p1-e2e-fixed scripts/p1/train.sh experiment=p1/smoke_timing`.

## Outputs

### Loggers
- Smoke tests use `logger=csv`, the default in `configs/train.yaml`. Lightning metrics go to `<run dir>/csv/`. Script-style smoke tests (no Lightning trainer) write `log.txt` and `result.json` to `logs/smoke/<test>/<stamp>_j<jobid>/` (`scripts/smoke/p1_smoke.py`).
- Official runs use `logger=wandb`. The experiment config selects it, never the command line. Project `taxa_maze`, entity `multimodal_lab` (`configs/logger/wandb.yaml`). The logger config was tested on 2026-10-02 in offline mode. Credentials come from `~/.netrc`, written by `wandb login` on 2026-10-02. Home is shared storage, so batch jobs on compute nodes see them too.
- Decision numbers (ToL-val correct counts per epoch and template, r_t, λ) are written to `decisions.json` in the run directory, so audits do not depend on wandb access. The iNat21 eval of the chosen checkpoint is a separate script (M3, not written yet); it records its encode setting and re-evaluates the six baselines under the same setting (Amendment 2 A2.4).

### Run naming (official runs)
Pattern: `p<pilot><arm>_<loss>[_<deviation>...]_lr<lr>_s<seed>_v<n>`. Fields are separated by `_`, and no field value contains `_`.
1. `p1c`: the pilot number, then the spec's arm letter (spec section 5).
2. `<loss>`: the arm's objective in words. `flat` is arm (a), `flat-pen` is (b), `lvl` is (c), and `lvl-pen` is (d). This repeats the letter on purpose, so a name reads without the spec open.
3. `<deviation>`: zero or more tokens, present only when a setting departs from the pilot's frozen defaults. The pilot-1 defaults are: LoRA r=16 on q,v plus `logit_scale`; K and B as the spec fixes them; and lambda calibrated per spec line 94. Tokens appear in this order:
   1. fine-tuning scope: `full`, `lora16-proj`
   2. K: `k16`
   3. global batch: `b16k`
   4. lambda: `lam0.05`
   5. whitener: `whiten-between`, `whiten-within`, `whiten-total`

   Add a token to this list before its first use.
4. `lr<lr>`: the peak LR in its shortest scientific form: `1e-4`, `3e-4`, `2.5e-4`.
5. `s<seed>`: the seed.
6. `v<n>`: the iteration, starting at 1. Bump it for a fresh restart of the same settings, for example after a bug fix. A resume from checkpoint keeps the same name. wandb ids are permanent, so a deleted run's name also needs a bump.

Pilot 1's official runs:
- The LR sweep is `p1c_lvl_lr1e-4_s42_v1` and `p1c_lvl_lr3e-4_s42_v1`. The winner continues to 10 epochs under the same name.
- Then, at the winning LR: `p1d_lvl-pen_lr<win>_s42_v1`, `p1a_flat_lr<win>_s42_v1`, and `p1b_flat-pen_lr<win>_s42_v1`.
- A later full fine-tune of arm (d) would be `p1d_lvl-pen_full_lr3e-4_s42_v1`.

Where the name is used:
- wandb: `name` and `id` are both the run name, with `resume="allow"`, so resuming continues the same wandb run. `group` is `task_name` (`p1`). The tags are the pilot, the arm letter, and the loss.
- Disk: each launch writes `logs/p1/<run_name>/<stamp>_j<jobid>/` (Hydra logs). Checkpoints live once per run, in `logs/p1/<run_name>/checkpoints/` (`configs/callbacks/p1.yaml`), shared across launches, so a resume and the best-epoch pick see a single directory.
- Guard: a fresh start refuses to run if `logs/p1/<run_name>/checkpoints/` already holds files (`src/train.py`). Either resume with `ckpt_path=...` or bump `version`. This stops two different runs from sharing one wandb id.
- The name is composed from config values, so an LR override renames the run: overriding the LR to 1e-4 turns `p1c_lvl_lr3e-4_s42_v1` into `p1c_lvl_lr1e-4_s42_v1`. The `sci` resolver lives in `src/train.py`. To parse a name, use `name.split("_")`.

## Gotchas
- Store reads: never through `np.memmap` in loader workers (see "Image store", Reading). A run whose GPUs idle while its loader workers spin in the kernel with no NFS reads has this stall; `audit/2026-10-05_stall_diag/probe.py` shows it from inside the job.
- `~/.hpc_env.sh` points `HF_HOME` (`~/bdbk/cache/hf`) and `WANDB_DIR` (`~/bdbk/wandb`) at `/projects/bdbk`. `scripts/env.sh` overrides both to `/u/liv` paths, so source it in every job. Do not edit `~/.hpc_env.sh` itself, because other projects use it.
- File quota on `/projects/bdbk`. `quota -s` mislabels its columns when the allocation is over its soft file quota: the grace clock appears under "File Used", the used count under "Soft", and the soft quota under "Hard", while the real hard limit is dropped. Read the true numbers with `/sw/user/scripts/vastfs quota -qhg tgirails_bdbk /projects/bdbk`, which prints block used/soft/hard, the grace left, then files used/soft/hard (checked 2026-10-02). The overage as of 2026-10-04 is recorded in `agent_legacy/active/2026-10-04_asks-A4-A6_blocks-B2-B3.md`; run the vastfs command for current numbers.
