# Active Research State
Working memory: every user ask, every open direction, and the honest state of each. Read first when picking up a turn, update before ending one.

Below the line:
- **Asks ledger** — every ask verbatim with an ID and date, recorded before acting on it; paraphrase loses intent. Keep any interpretation separate from the quote. Status: OPEN / ACTIVE / BLOCKED / DONE / DROPPED, the last always with a reason.
- **One block per active direction** — asks served, falsifiable hypothesis, success and kill criteria (both set before the first run), what is true now rather than planned, the single next action, artifacts owned (experiment config, run dir, wandb id, `audit/` reports), dated log newest first.
- **Blocked** — on what, since when, what would unblock it.
- **Parking lot** — one line per idea not yet worth a direction.
- **Prune log** — one row per retirement: verdict and destination path. This is what stops a future agent re-running a dead direction.

Maintenance:
- Status changes in the turn the state changes. A stale ACTIVE is worse than no entry.
- Prune on any turn that changed a status, or once the file passes ~200 lines. An ask is prunable when DONE or DROPPED with no live direction referencing it. A direction is prunable when it hits its success criterion (graduate the design to DESIGN.md, the number to SOTA.md), hits its kill criterion, or goes 14 days without a log entry — record that as STALLED, which is a finding about attention, not a failure.
- To prune: write `agent_legacy/active/YYYY-MM-DD_<slug>.md` holding the block verbatim, a verdict (what was tried, the numbers, why it stopped, what to do differently), and every path the result depends on. Then delete the block and add a prune-log row.
- Never delete an ask. Never renumber IDs — they are referenced from directions, `audit/`, and `agent_legacy/`.

##### MODIFY UNDER THIS LINE #####

## Asks ledger

### A1 (2026-10-02) Pilot 1 per specs/pilot1.md. Status: ACTIVE
The spec file is `specs/pilot1.md`. At intake, HEAD was `90a7d388` and the file's sha256 was `544b1accaabc72cbcd1b1867a6c584065c2e304e5334e4e04c6984433f0c5f09`. Never edit it. Amendments go at its bottom.

Verbatim ask (2026-10-02):
> The spec is specs/pilot1.md (committed, frozen). Record it as ask A1 in agent/ACTIVE.md. Never edit it; amendments go at its bottom with date and reason. If you find an error or gap in the spec, stop and tell me; I decide whether it becomes an amendment (dated, with reason, appended at its bottom). Bugs in your own implementation are yours to fix.
>
> Known gaps in the template:
> * src/train.py line 1 is a syntax error.
> * The .project-root marker is missing.
> * The bioclip env lacks lightning, hydra-core, hydra-colorlog, rootutils and omegaconf. Installing them adds packages and upgrades nothing.
> * The default logger is null, but lr_monitor needs one. Set to csv for now.
> * The wandb project and entity are the template author's (moe_db / multimodal_lab). Set project to the existing `taxa_maze`, entity `multimodal_lab`.
> * The checkpoint callback keeps only the newest epoch, but section 7 picks the best ToL-val epoch.
>
> The goal in three milestones:
> 1. The section 9 smoke tests pass and are verified
> 2. The LR sweep has a decision
> 3. All training arms finish, no verdict.
> After each milestone, run the checker and wait for my go.
>
> Checker: A read-only Opus 5.5 subagent, outputs to audit/ and replies with a pointer. It checks:
> * the implementation matches the spec line by line: where each item lives, or which amendment changed it;
> * every claimed smoke-test pass has logs and numbers behind it;
> * the decision rules are untouched: the ±1-point bar, the 70.19 floor, selection on deduplicated ToL-val, and one iNat21 eval per arm;
> * git diff 091b208 -- specs/pilot1.md is empty, or shows only appended amendments.
> It raises problems and never fixes them.
>
> Verdicts: report numbers and interpretations, but don't decide. The final verdict is mine.

Verbatim addition, sent mid-turn (2026-10-02):
> imports of nonexistent files such as src.models.retrieval_module are legacy files from a previous unrelated codebase. you may ignore these file names and structures

Verbatim revision (2026-10-02):
> revision to A1: the checker should check pilot1.md against `90a7d38` as the frozen baseline.

So the checker's fourth check is now: `git diff 90a7d38 -- specs/pilot1.md` is empty, or shows only appended amendments. This replaces the 091b208 baseline in the quote above.

Checker rules the user added on 2026-10-04 (A7):
- S37: "only appended amendments" is tested as a byte prefix. The 90a7d38 file must be a byte prefix of the current spec (`cmp -n <old size> old new`).
- Precedence: where the dated amendment block conflicts with the original spec text, checkers follow the amendment block. User's words: "have any checker prioritize that block over any conflicting statement in the original spec".

My interpretation (not the user's words):
- Milestone 1 means the five "Smoke runs" bullets in spec section 9 pass, each with a log and numbers. Other smoke tests named elsewhere in the spec are prerequisites, and I will log them too: the ToL cache test (section 2), the adapter bit-identity test (section 3), and the eval-harness gate (section 7).
- Milestone 3 ("all training arms") means the four loss arms (a)-(d) with LoRA r=16. The capacity arms (full fine-tuning; LoRA plus `visual.proj`/`text_projection`) are marked "queue" and "save for later" in sections 3 and 5, so I leave them out unless told otherwise.
- I stop at every spec error or gap and ask. I do not resolve spec gaps by choosing a default silently.

### A8 (2026-10-04) Preferred approach to the three milestones. Status: ACTIVE
Verbatim ask (2026-10-04):
> "From now on: quick scripted checks inside the turn. Workflows only for the milestone checker you asked for, or before something irreversible, and never a re-check loop. I saved this as a standing preference in memory and did not log it to agent/ACTIVE.md"
>
> "Fastest path to training:
> Today, after your approval: append the amendment, start the image-store build on CPU nodes (about 1–2 hours), and write the training code while it runs. Testing is limited to what the spec names.
> Tomorrow: the spec's smoke tests, the checker, then your go. The LR sweep can then start: two 3-epoch runs, roughly 1.5 hours each, if A100s are free."
>
> Summarize and write to ACTIVE.md as a preferred approach to reaching the three milestones.

Verbatim addition (2026-10-04):
> add: you may suggest checks or (smoke) tests you think are deserved in addition to what the spec names, in case a result/interpretation may be distorted without them

Done: the summary is under D1, "Preferred approach to the three milestones".

### A10 (2026-10-04) Permission before H100 jobs. Status: ACTIVE (standing rule)
Verbatim ask (2026-10-04):
> pause for user permission before running any jobs, especially full training jobs, on H100s.

My interpretation (not the user's words): ask before every submission to the H100 `gpu` partition, naming the job, GPU count and expected minutes. `gpu_a100` and `cpu` jobs need no ask. The rule is passed to every subagent that may submit jobs. Recorded in memory as well.

### A16 (2026-10-10) Rare Species dataset: download, port, evaluate. Status: DONE (2026-10-10; results in `audit/2026-10-10_rare_species.md`; the reporting decision is in D1's M3 preparation)
Verbatim ask (2026-10-10):
> the following ask is a separate one: download and port this dataset: https://huggingface.co/datasets/imageomics/rare-species
>
> "This dataset was generated concurrently with TreeOfLife-10M, so the process is as described there, with the exception that these images were entirely sourced from EOL, and the species represented were excluded from the TreeOfLife-10M dataset." given this description on HuggingFace, this should be OOD for both BFL and our trained models. i want to evaluate bioclip1, BFL, and some of the existing runs for per-rank classification accuracy on this dataset. this should be able to better answer the gap between bfl and 281184 on inat21 on our eval harness. data path: ~/data/taxa_maze to avoid the soft quota on bdbk.

My interpretation (not the user's words): a diagnostic benchmark outside the spec (the verdict stays iNat21 under S27 and line 172; no amendment needed unless it enters the verdict). Steps: download under `/u/liv/data/taxa_maze/rare_species/`; port to the harness's format (an image list, `codes.i32.npy` (N × 7) and `vocab.json` per rank, as the iNat21 cache has them) so `evaluate_ranks` runs unchanged; check that its species are absent from our ToL-EOL training tree as the card claims; evaluate untouched BioCLIP 1, BFL-Euc (the released checkpoint), and the second-phase runs' epoch checkpoints under A2.4's setting, both text forms; report to `audit/`. Status 2026-10-10 16:50: downloaded and ported; baselines, the untouched cross-check and the sweep and control runs evaluated (job 307305, done); the eight second-phase runs in job 307306 (done 17:14). Table and reading: `audit/2026-10-10_rare_species.md`. Headline: on species none of the models has seen, BFL-Euc is 11.1 points above BioCLIP 1 and our (c)-family runs 7-8 points above it after one epoch; the gap to BFL-Euc at species is about 4 points here against 18 on iNat21, so the iNat21 gap is mostly data; the decline over epochs is 6 points here against 21 on iNat21.


### A17 (2026-10-11) M3 redefined: (d) against (c) on several recipes, ten-epoch curves, λ sweep, ToL-10M in parallel. Status: ACTIVE
Verbatim ask (2026-10-11):
> Answers to pre-M3 questions:
> * agreed that ten-epoch curves should be compared as opposed to S27's best-ckpt comparison
> * lambda needs a sweep rather than loss gradient calibration. and idk what "candidates 60 and 120" are.
> * penalty form: 1/g_s vs. uniform first, queue whitener, refresh cadence, K=8, text-side penalty with frozen image tower for later.
> * arm (b): defer for now.
> * which (c) the penalty sits on: instead of choosing the best (c), we train and evaluate (d) vs. (c) on multiple recipes (each with only one knob turned on) tested in M2: batch composition (random vs. sampled), rank-balanced contrastive objective (uniform, gradient-norm, loss-ratio; skip LogSumExp), two-stage schedule (seven levels for the first 1-2 epochs, then species-only, or decaying weights for coarse ranks), and `{lineage}` training text template. all but the two-stage schedule and training text template have been tested on arm (c), so there are not as many runs to submit. this meaningfully changes "what M3 is" as recorded in the spec and ACTIVE:D1.
> * data confound: while everything above runs, we prepare a complete tol-10m dataset in parallel, with the same data and store paths.
> * eval on inat21 val and rare species together for M3.
>
> instructions:
> * write a living document to audit/ with a table recording live and planned runs for M3
> * draft amendment for pilot1.md as needed (gate)
> * update ACTIVE and agent/ docs and submit jobs

Verbatim approval (2026-10-11):
> approve amendment 4 draft.

Status 2026-10-11 12:52: Amendment 4 appended to `specs/pilot1.md` as drafted, with A4.9 recording the approval and reading the bracketed items as proposed (`audit/2026-10-11_amendment4_draft.md` is the record).

My interpretation (not the user's words): M3 is a matrix, (d) against (c) on six recipes with one knob each from the base recipe, every run to ten epochs with every epoch evaluated on ToL-val, iNat21-val and Rare Species; the five-epoch M2 runs continue to ten epochs from their last checkpoint (the design's own path, untested until now), the random-batch control continues from its third epoch, and the two-stage and lineage cells are fresh runs. "First 1-2 epochs" is read as one coarse epoch (the coarse ranks rise within epoch 0 and hold; species falls from epoch 1); the decaying form is implemented and queued. λ sweep values 10 and 100 around 30; the uniform penalty form at λ 30. "Candidates 60 and 120" were my proposals in D1, not spec values: 30 × 0.2 / 0.112 ≈ 54 would restore S6's ratio, 120 was the next doubling. Arm (a)'s continuation is my addition. "(gate)": the amendment is the gate of M3; I draft it (`audit/2026-10-11_amendment4_draft.md`) and the owner decides whether it is appended (A1); the jobs were submitted on the owner's instruction before approval, as the first second-phase runs were under Amendment 3. The living table is `audit/2026-10-11_M3_runs.md`. ToL-10M: a read-only plan first (inventory, sizes, quota, steps), then the port under `/u/liv/data/taxa_maze/` in the store's format; nothing downloaded until the plan is checked. The runs were submitted at 12:23-12:24 (ten fresh and smoke jobs); the continuations and the two-stage and lineage runs follow their gates.

## Directions

### D1: Pilot 1, hierarchical orthogonality penalty on BioCLIP 1
- Asks served: A1; A17 (M3, 2026-10-11); A16 (A15 done 2026-10-10, pruned).
- Hypothesis (spec section 1): adding the cross-level orthogonality penalty to BioCLIP 1 fine-tuning makes zero-shot classification on iNat21-val competitive across all seven ranks, against BioCLIP 1, BioCLIP 1 + RCME, and BioCLIP 1 + BFL.
- Success bar (quoted from spec section 1): "competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons."
- Failure mode (quoted from spec section 7): "Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification."
- Who decides: the user gives the verdict. I report numbers and interpretations only.
- Spec: `specs/pilot1.md` with Amendments 1-4 at the bottom (60,770 bytes, sha256 `9bc422b8…`; Amendment 4 appended 2026-10-11 12:52 on the owner's approval), the later one first on any conflict; S9 list `specs/pilot1_s9_missing_species_keys.txt`; the preflight report is not edited. Paths: repo `/u/liv/repos/taxa_maze`, data `/u/liv/data/taxa_maze/{store,p1}`, every write under `/u/liv`. Code map: DESIGN.md "Idea → code"; how to run: USAGE.md. Git: `origin/main` at `f362c86` (2026-10-11 12:59: Amendment 4, the M3 code and docs); the running jobs started from the same tree (the fresh (d) runs before the monitor fix, which touches logging only).
- Runs and results: M3 live table `audit/2026-10-11_M3_runs.md` (every M3 run, its state, job and reference, and the agent's decisions of the day); M2: `audit/2026-10-10_M2_runs_table.md` (every M2 run, its reference, both benchmark curves, timing facts). Readings: `audit/2026-10-05_M2_sweep_results.md`, `2026-10-06_A14_sampler_control.md`, `2026-10-08_M2_phase2_round1.md`, `2026-10-08_text_geometry/text_geometry.md`, `2026-10-10_M2_phase2_round2.md`, `2026-10-10_bfl_gap_confounders.md`. M1: `agent_legacy/active/2026-10-05_D1_M1-smoke-evidence.md`, checker `audit/2026-10-05_M1_checker.md`.
- How to work (A8, verbatim in the ledger; A10): quick scripted checks in the turn; workflows only for the milestone checker or before something irreversible; no re-check loops; extra checks are suggested with their cost and the owner decides; A100 jobs need no ask, H100 does. Milestones: M1 done 2026-10-05; M2 was the learning-rate decision plus the second phase under Amendment 3, closed by the owner on 2026-10-10; M3 (redefined by A17 on 2026-10-11; Amendment 4) is arm (d) against arm (c) on several recipes to ten epochs, every epoch evaluated on ToL-val, iNat21-val and Rare Species, the checker, the owner's verdict.
- Closed (M2), 2026-10-10. The owner's statements, with what the runs added:
  - Learning rate: 1e-4 under line 161; 3e-4 is worse and unstable.
  - The shape. Under the level-restricted objective, species accuracy on ToL-val and on iNat21 peaks after the first epoch and falls every epoch after, while kingdom to family rise within the first epoch and hold. The species-only objective (arm a) holds species (+5.3 iNat21 points over BioCLIP 1 after one epoch, +3.4 after five) and loses phylum and kingdom. Owner: the level-restricted objective is the culprit; coarse-rank supervision competes with species supervision.
  - What did not change the shape: batch composition (random batches lift the level, same fall); three rank balancings (gradient-norm gives species the largest weight and the best first epoch, then the same fall; loss-ratio the slowest fall; LogSumExp, family-heavy, worse than uniform); freezing the text tower (lower ceiling, slower fall); the penalty at λ 30 (equal first epoch, slower fall). Only removing the coarse terms removes the decline. Owner: the contrastive side is now mostly tested; alpha or further learning rates would stretch the curve, not change its sign, and are not worth runs.
  - The text anchor: BioCLIP 1's truncated labels are not the centroids of their species' labels at kingdom to family (cos 0.3-0.4 against a 0.15-0.25 baseline; genus 0.74), so image-side LoRA was a recipe test, and as a recipe it is worse under both objectives.
  - Implementation: the loss equals an independent implementation of BFL's Eqs. (1)-(2) to 1e-9; the reader, the batch plan and the eval harness are verified (M1 gate; 9-of-9 reader identity; BioCLIP 1's counts reproduced exactly). Unverifiable without BFL's code: their batch construction, template and temperature handling, and whether their species accuracy fell during training.
  - The gap to the released BFL model (68.78 against our 50.89 at species after five epochs) is confounded by data: BFL trained on ToL-10M including iNat21 train (2.7M images of the val species); our ToL-EOL set holds 8,624 of the 10,000 val species (median 81 images), 13.8% absent. Owner: suspects implementation, LoRA or data; will ask for BFL's code.
  - S27's reading: every (c) and (d) run peaks at epoch 0 on ToL-val, and at that epoch gradient-norm, loss-ratio, uniform and d beat BioCLIP 1 at every rank on iNat21 (species by 1,895-3,990 images, average 79.5-80.6 against 41.5). Amendment 3 is in force (diagnostic iNat21 evals, the second phase's rules, the recipe record, the qualified expectation).
- M3 (A17, 2026-10-11; Amendment 4 in force since 2026-10-11, approved as drafted, record `audit/2026-10-11_amendment4_draft.md`; live table `audit/2026-10-11_M3_runs.md`). The owner's answers to the questions of M2's close, one line each: ten-epoch curves are compared, not S27's checkpoint; λ by sweep (10, 30, 100), not calibration; penalty form 1/g_s against uniform first, whitener, refresh cadence, K = 8 and the text-side penalty later; arm (b) deferred; no single (c) adopted: (d) against (c) on the base recipe, random batches, gradient-norm, loss-ratio (LogSumExp dropped), the two-stage schedule and the bare-lineage training text, one knob per run; iNat21-val and Rare Species together, every epoch; a complete ToL-10M prepared in parallel. The bracketed items of the draft stand as proposed under A4.9 (the bar and the floor read per rank at any compared epoch; A3.4 at every rank on arm (c)'s curve; arm (a) continues, arm (b) deferred; "a photo of" stays the decisive form; Rare Species a diagnostic; P1, P5(1), N3 closed). Still the owner's: BFL's code (the owner's own action); the verdict after M3's checker. The superseded Open/Questions/Preparation blocks: `agent_legacy/active/2026-10-11_D1_M2-close-open-questions-preparation.md`.
- For later (owner, 2026-10-11, verbatim): "\"Only 84,447 of BIOSCAN's 1.1M rows have a full seven-rank lineage\" motivates a different way of dealing with incomplete lineages. one way is to replace missing/placeholder/author-name-only ranks with an `[unknown]` token, we can decide later. adopting it would certainly require re-training existing arms/recipes that get selected." Context: spec line 53 drops rows with missing ranks and S9 drops placeholder species keys; TaxaWalk trains with an `[unknown]` token for missing ranks and leaves it out of its loss (`agent/litreview/Sastry2027-TaxaWalk.md` §4.1); the ToL-10M plan (`audit/2026-10-11_tol10m_plan.md`, question 1) stores only the complete BIOSCAN rows for now.
- Queued, not submitted (owner's order): the decaying two-stage form; the whitener (lines 124-139); the refresh cadence; K = 8; the text-side penalty with a frozen image tower; the set-aside knobs (alpha = r, the loss without dedup and group positives, batch 16,384); one image-side (d); full fine-tuning (line 154); the lower-LR gap (the owner does not want the run).
- True now (2026-10-11 ~13:15): the live table has every job. Running: the six fresh (d) runs (324017, 324019, 324021, 324023, 324025, 324027), the two-stage runs (324042, 324044) and the (c) lineage run (324046). Queued: the (d) lineage run (324048) and the six continuations to ten epochs (324077 c base, 324079 d base, 324081 c gradient-norm, 324083 c loss-ratio, 324085 a, 324087 c random batches), each with an `eval_m3.sh` job behind it. Baselines on iNat21-val under A2.4 done (324029; BioCLIP 1 reproduces 70,193). The resume path was audited (read-only Opus, refuted, toy-reproduced): one real gap, the first continued epoch would have reused epoch 0's batch plan; fixed in `tol_datamodule.py` with a guard in `training_step` before the continuations were submitted (uncommitted). ToL-10M: plan and open questions in `audit/2026-10-11_tol10m_plan.md`; the two missing archives (iNat21 train 240 GB, BIOSCAN-1M cropped_256 7.2 GB) are downloading in CPU job 324090 into `/u/liv/data/taxa_maze/raw/` (home quota 969 of 4,657 GB soft; the plan needs about 673 GB).
- Next action: finish the submissions as their gates clear (continuations after the resume audit; two-stage and lineage runs after the smoke checks), write the ToL-10M plan to `audit/` and start its first step if the quota allows, keep the live table current; run the M3 checker when every run and evaluation is done, then the owner's verdict. Commit when asked.
- Artifacts: runs under `logs/p1/<run_name>/` (Hydra run dir with `steps.jsonl`, `decisions.json`, `run_info.json`; `checkpoints/epoch_NN.ckpt`, `last.ckpt`; `inat21_diag/`), wandb project `multimodal_lab/taxa_maze`; audits under `audit/`; M1-era artifacts in `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md`.
- Log (newest first; older entries in `agent_legacy/active/`):
  - 2026-10-11 13:12: resume audit returned (gap G1, fixed and toy-tested); continuations 324077-324087 submitted; ToL-10M plan received, download job 324089 started; baselines on iNat21-val done (324029). Checker items raised by the audit and left for the owner: none blocking; optional guards (compare the resumed settings with the checkpoint's) not added.
  - 2026-10-11 12:52: Amendment 4 approved by the owner and appended (8,969 bytes; byte-prefix checks against the previous file and 90a7d38 pass); docs updated from draft to in force.
  - 2026-10-11 12:43: smoke runs passed; two-stage and lineage runs submitted (324042-324048, even ids, with eval jobs behind). A monitor gap found by the smoke runs was fixed (`_full_log` keeps the coarse-level monitors after the switch).
  - 2026-10-11 ~12:40: docs updated for M3 (USAGE, DESIGN, this file); the superseded D1 blocks pruned to `agent_legacy/`; Amendment 4 draft and the live table written under `audit/`.
  - 2026-10-11 12:23-12:24: submitted the four smoke runs (324013-324016), the six fresh (d) runs (324017, 324019, 324021, 324023, 324025, 324027) with `eval_m3.sh` behind each, and the six-baseline iNat21 job (324029). Three training runs and the smoke runs started at once.
  - 2026-10-11 12:17: A17 received. Code added: `level_schedule` (switch, decay) with `coarse_epochs`, `train_text_form`, `scripts/p1/eval_m3.sh`; CPU checks of the helpers, the validation and the Hydra composition passed. A read-only Opus workflow launched for the ToL-10M plan and the resume-path audit (N9), each with a refuter.
  - 2026-10-10 ~17:28: M2 checker done (0 blocking, 5 non-blocking, 11 notes). Fixed what is mine (timing ranges, prose numbers, citations line 178 → 96, stale lines, run-naming tokens, library versions in the eval record, the loss check saved as `scripts/p1/check_bfl_loss_reference.py`); the owner's calls listed in the response and under M3 preparation.
  - 2026-10-10 ~17:17: Rare Species evaluations done (307305 14 min, 307306 53 min; 6 baselines, 49 run checkpoints); audit written; A16 DONE. M2 checker launched on the owner's word (Opus, read-only).
  - 2026-10-10 ~16:50: docs prepared for the pause: M2 closed by the owner (checker pending the owner's call), an M3-preparation block under D1, Rare Species status under A16. `agent/1.md` and `agent/2.md` are the owner's copies of ACTIVE.md from 2026-10-08 (ignored by git, untouched).
  - 2026-10-10 ~16:31: A15 marked DONE and pruned to `agent_legacy/active/2026-10-10_ask-A15_c-recipe-second-phase.md` (verdict and paths there); older log entries of today moved to `agent_legacy/active/2026-10-10_D1_log-synthesis-day.md`.
  - 2026-10-10 ~16:21: Rare Species evaluation split on the owner's ask: 307071 (everything in one job) cancelled after 16 min, having written the six baseline files; 307305 evaluates the six baselines, untouched BioCLIP 1 and the sweep and control runs, 307306 the eight second-phase runs (281183, 281184 and their derived runs); results to `logs/p1/rare_species_diag_baselines/` and `<run>/rare_species_diag/`.
  - 2026-10-10 ~16:04: A16 (Rare Species) recorded and started: dataset downloaded (4.1 GB, 11,983 images, 400 species, none in our training tree or in iNat21) and ported to the harness's format under `/u/liv/data/taxa_maze/rare_species/`. The set-aside A15 knobs added to Open (M2) on the owner's ask.
  - Entries from 2026-10-10 14:50 to 16:21 are in `agent_legacy/active/2026-10-10_D1_log-synthesis-day.md`.
  - Entries from 2026-10-08 16:31 to 2026-10-10 15:04 are in `agent_legacy/active/2026-10-10_D1_log-phase2-round2.md`.
  - Entries from 2026-10-06 (the control) to 2026-10-07 (round 1 submitted and validated) are in `agent_legacy/active/2026-10-08_D1_log-phase2-preparation.md`.
  - Entries from 2026-10-05 ~17:20 (the reader switch) to ~23:09 (the sweep's first-round results) are in `agent_legacy/active/2026-10-05_D1_log-sweep-first-round.md`.
  - Entries from 2026-10-05 ~14:15 (M2 launch) to ~17:00 (the reader switch) are in `agent_legacy/active/2026-10-05_D1_log-M2-launch-to-reader-switch.md`.
  - Earlier entries, 2026-10-02 intake through 2026-10-05 ~00:55 (the M1 checker), are in `agent_legacy/active/2026-10-05_D1_log-through-M1.md`.

## Blocked
None (B2 and B3 resolved 2026-10-04; see the prune log).

## Parking lot
- (user, 2026-10-06) Text-side penalty, no bank: "ditch the image bank and the prototype setup, keep image tower frozen, and apply the penalty to the text side- taxonomic label embeddings truncated at each corresponding rank." Each node's position is then the text embedding of its truncated label, computed fresh for the in-batch nodes, so the EMA bank and the species prototypes are not needed. Related to the A5 line on constraining the text side; this is the text-only form with the image tower frozen.
- (A5) Flat decoding. Pilot 1 regularizes the geometry, but it still classifies by a flat nearest-neighbor lookup over all labels at a rank. TaxaWalk (sections 1-2) argues for moving from rank to rank, using the possibly well-structured taxonomic text space of BioCLIP 1/2. Limitation of pilot 1, and a direction after it.
- (A5) Image-side constraints only. Pilot 1 imposes the geometric constraints on image prototypes only. The text side could be constrained too, either separately or jointly with the image side.
- (A5) "TaxaWalk++". A future framework could fold parts of pilot 1's design into TaxaWalk's predictive representation-learning framework. Cheapest concrete form (TaxaWalk review): cross frozen BioCLIP 1, arm (c) and arm (d) with a flat, a top-down and a TaxaWalk decoder (`agent/litreview/taxawalk-and-pilot1.md`).
- (A9) Top-down evaluation. Classify from kingdom down to species, with the candidates at rank t restricted to taxa that share the ranks already predicted. The user expects this to match BFL's "top-down constrained inference"; verify against the BFL paper before relying on that. It would be reported next to the per-rank protocol of spec section 7, not instead of it. Verified 2026-10-05 (TaxaWalk review, against BFL's v1/v2 TeX): it is BFL's rule. The review suggests report-only M3 columns with no new encode, because `src/eval/zeroshot.py` already computes each rank's argmax: top-down, species read-off, oracle-parent accuracy, full-path accuracy and nLCA. The user decides.
- (user, 2026-10-11) Read-off scoring, the protocol TaxaWalk uses for its baselines (§4.1 of `agent/litreview/Sastry2027-TaxaWalk.md`, local only). Verbatim: "For the baselines, we find that scoring each rank separately is unfavourable, since they are trained on full lineages. We instead classify the species against full lineage strings and read the ancestors off the predicted lineage, which keeps their species accuracy and makes their ranks consistent. When ranks above leave a single candidate, such as monotypic genus, we score the rank on the decoded ancestors." TaxaWalk itself scores the seven ranks separately, each among the candidates consistent with the ranks above it (its rollout, Eq. 2), so the paper mixes two protocols, neither of which is ours: spec section 7 takes an independent argmax over truncated labels at every rank. Under read-off, coarse-rank accuracy is bounded by lineage consistency (a wrong species with the right family still counts at family, a wrong kingdom cannot be rescued), and species accuracy is unchanged. Report-only M3 column with no new encodes, like the top-down line above: the species rank already is an argmax over full lineage strings and `top1_correct` returns every image's predicted class, so reading its ancestors off the vocab path and comparing them with the gold truncations is a lookup. The monotypic clause needs a reading before use (mine: when the ranks already read off leave one candidate at a rank, that rank is scored on the read-off ancestor rather than counted as trivially right). The user decides whether it joins the M3 report.
- (A9) Rank-balanced penalty. Replace the plain sum over (species, ancestor) terms in each species' penalty with LogSumExp (Li et al., 2020), so each term's gradient is weighted by a softmax over the term values. TaxaWalk uses this for its next-step MSE loss, a different objective. Question: does it balance the six ranks better than the uniform weighting of spec line 116, and how does it change λ calibration? TaxaWalk review (2026-10-05): its LogSumExp runs over seven batch-level per-rank errors, not over each species' own terms; its τ = 0.1 does not carry over to cos² terms; the gain is small (Figure 4c, species +2.35, measured); and it would change λ by up to about 6×. Position: not for pilot 1; the batch-level form stays open after it.

## Prune log
| date | block | verdict | destination |
|---|---|---|---|
| 2026-10-11 | D1 "Open at M2's close", "Questions for M3", "M3 preparation" | answered by the owner (A17); replaced by the M3 block and the live table `audit/2026-10-11_M3_runs.md` | `agent_legacy/active/2026-10-11_D1_M2-close-open-questions-preparation.md` |
| 2026-10-07 | prune-log rows dated 2026-10-02 to 2026-10-04 (6 rows: M1-era asks, blocked items and directions) | moved out to keep this file near 200 lines | `agent_legacy/active/2026-10-07_prune-log_through-2026-10-04.md` |
| 2026-10-10 | D1 long form (run table, True now, Next action, hypothesis block) and the log entries 2026-10-08 16:31 to 2026-10-10 15:04 | replaced by the Closed/Open/M3 synthesis; numbers moved to `audit/2026-10-10_M2_runs_table.md` | `agent_legacy/active/2026-10-10_D1_before-synthesis.md`, `agent_legacy/active/2026-10-10_D1_log-phase2-round2.md` |
| 2026-10-10 | A15 (diagnose and fix the (c) recipe before M3) | DONE: diagnosed (the shape is the objective's; no fix inside the contrastive objective); unfinished items carried by D1's Open (M2) and Questions for M3 | `agent_legacy/active/2026-10-10_ask-A15_c-recipe-second-phase.md` |
| 2026-10-05 | A11 (first P3/P4 answers) | DONE: decided by A12 (P2, P3 approved; P4 held); findings fed Amendment 2 | `agent_legacy/active/2026-10-05_ask-A11_P3-P4-responses.md` |
| 2026-10-05 | D1 M1 evidence (smoke tests, eval gate, drift floor, timing run) | DONE: M1 closed by the checker and A12 | `agent_legacy/active/2026-10-05_D1_M1-smoke-evidence.md` |
| 2026-10-05 | D1 artifacts through M1 | DONE: listed and verified by the M1 checker; M2 writes elsewhere | `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md` |
| 2026-10-05 | D1 log, intake through the M1 checker | DONE: outcomes summarized in D1 "True now" | `agent_legacy/active/2026-10-05_D1_log-through-M1.md` |
| 2026-10-05 | `scripts/preflight/` one-off helpers (52 files) | DONE: preflight closed by Amendment 1; superseded by `src/` and `scripts/smoke/` | `agent_legacy/scripts/preflight/` (README: `agent_legacy/scripts/README.md`) |
| 2026-10-05 | A3 (literature reviews, four papers) | DONE: all four reviewed, validated and indexed; outcomes in the Parking lot | `agent_legacy/active/2026-10-05_ask-A3_literature-reviews.md` |
| 2026-10-05 | Parking-lot lines "(A3, paper 1-3)" | DONE: their content is in the three topic files (checked by script) | `agent/litreview/whitener-for-hierarchical-orthogonality.md`, `agent/litreview/frozen-clip-hierarchical-geometry-novelty.md`, `agent/litreview/taxawalk-and-pilot1.md` |
| 2026-10-05 | A12 (M1 decisions, Amendment 2, M2 launch) | DONE: all items carried out; P4 hold and A2.4 live in the spec | `agent_legacy/active/2026-10-05_ask-A12_M1-decisions-amendment-2.md` |
| 2026-10-05 | A13 (who writes where; evidence scripts) | FOLDED: the standing rules now live in `agent/litreview/README.md`, below the MODIFY line | `agent_legacy/active/2026-10-05_ask-A13_write-scope.md` |
| 2026-10-05 | D1 log, M2 launch to the reader switch | DONE: outcomes in D1 "True now" and Next action, and in `audit/2026-10-05_lr3e-4_stall.md` | `agent_legacy/active/2026-10-05_D1_log-M2-launch-to-reader-switch.md` |
| 2026-10-06 | A14 (sampler flag and the lr 1e-4 random-batch control) | DONE: flag added (`41380d3`); control ran (266465); random batches lift ToL-val species by 13,700-21,800 images, the decline persists | `agent_legacy/active/2026-10-06_ask-A14_sampler-control.md`, `audit/2026-10-06_A14_sampler_control.md` |
| 2026-10-06 | D1 log, the reader switch to the sweep's first-round results | DONE: outcomes in D1 "True now" and `audit/2026-10-05_M2_sweep_results.md` | `agent_legacy/active/2026-10-05_D1_log-sweep-first-round.md` |
