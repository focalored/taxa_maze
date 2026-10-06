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

### A14 (2026-10-05) Random-batch probe of the species decline. Status: ACTIVE
Verbatim ask (2026-10-05, after the M2 sweep results):
> does the batch sampler have a hydra config flag that can turn it off so that the batch can be sampled completely randomly (many more species than the at most 512 species in a batch)? if so submit a job right now with lr1e-4, max_epoch=3, 10-epoch schedule, and sampler turned off. this would isolate batch composition as the reason for "collapse" in classification over the first three epochs

My interpretation (not the user's words): no such flag existed, so one is added as an opt-in diagnostic, `data.train.sampler=random` (uniformly random batches of B images; images of one species that land in the same batch form its one group there, so S3 still holds). The default `grouped` is the spec's batch plan (S3, S7) and is untouched. The probe is arm (c) at lr 1e-4, `trainer.max_epochs=3` on the 10-epoch schedule, `run_name=p1c_lvl_lr1e-4_s42_rand_v1`; it informs the reading of the sweep and decides nothing under the spec.

## Directions

### D1: Pilot 1, hierarchical orthogonality penalty on BioCLIP 1
- Asks served: A1.
- Hypothesis (spec section 1): adding the cross-level orthogonality penalty to BioCLIP 1 fine-tuning makes zero-shot classification on iNat21-val competitive across all seven ranks, against BioCLIP 1, BioCLIP 1 + RCME, and BioCLIP 1 + BFL.
- Success bar (quoted from spec section 1): "competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons."
- Failure mode (quoted from spec section 7): "Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification."
- Who decides: the user gives the verdict. I report numbers and interpretations only.
- True now (2026-10-06 ~00:22): M1 is DONE (checker `audit/2026-10-05_M1_checker.md`, response `audit/2026-10-05_M1_checker_response.md`, decisions A12 in `agent_legacy/active/2026-10-05_ask-A12_M1-decisions-amendment-2.md`; Amendment 2 appended). M2, first round: both sweep runs finished three epochs at 1 node × 2 A100s (A2.1): 264726 (`p1c_lvl_lr1e-4_s42_v1`, 2026-10-05 14:08-21:29, old reader, 7 h 21 min) and 265500 (`p1c_lvl_lr3e-4_s42_v2`, 17:12-22:17, pread reader, 5 h 05 min). Under line 161 the sweep picks 1e-4: 85,586 species-correct on deduplicated ToL-val after 3 epochs against 42,516 (31.86% vs 15.83%; mean7 68.51 vs 63.98), ahead at every rank. Numbers, curves and reading: `audit/2026-10-05_M2_sweep_results.md`. Both runs deteriorate on ToL-val against untouched BioCLIP 1 (species 108,062, 40.23%): at 1e-4 species peaks after epoch 0 (111,727, 41.59%) and falls 4-5 points per epoch while kingdom to phylum stay at 93-98%; at 3e-4 the training loss itself rose in epoch 1 and the kingdom score collapsed to 5.29% under the photo form at epoch 1 (97.24% at epoch 2). The training loss's species and genus terms rise while its coarse terms fall, and the image embeddings contract (mean pairwise cosine 0.14 → 0.71, participation ratio 137 → 75, mostly in the first 250 steps): line 174's failure mode, in the unpenalized arm (c). No sign of a measurement bug: same `init` counts, bit-identical step 0, training text form = the photo eval form, byte-identical readers. No iNat21 eval of any sweep checkpoint. The lr 1e-4 sampler control (ask A14) is running: job 266465 (`p1c_lvl_lr1e-4_s42_rand_v1`, rails11, since 00:05, `data.train.sampler=random`, commit `c6945b2` plus the uncommitted sampler change); at step 64 its steps take 8.45 s (median) at 60.1 GiB peak, against 6.0 s and 47.4 GiB grouped. Awaiting the user: whether line 161 has a gap (no learning rate in the sweep kept species accuracy; a lower one was not tried). The M2 checker is held until that call. ToL-val has 131,143 candidate species, so its numbers are not comparable with iNat21's 70.19% (10,000 species).
  - Spec: `specs/pilot1.md` with Amendments 1 and 2 at the bottom (47,559 bytes, sha256 `5315446f…`), which govern on any conflict, the later one first. The S9 list is `specs/pilot1_s9_missing_species_keys.txt`. Do not edit the preflight report.
  - Paths: the repo is `/u/liv/repos/taxa_maze` (`~/bdbk/repos/taxa_maze` is a symlink to it). Data, the BioCLIP 1 checkpoint and tol_embed stay on `/projects`, read only. Every write goes to `/u/liv`, never under `/u/liv/bdbk` (S1). The store is `/u/liv/data/taxa_maze/store/` and the prep outputs are in `/u/liv/data/taxa_maze/p1/` (USAGE "Data").
  - Code map: DESIGN.md "Idea → code". How to run each test: USAGE "Quickstart (pilot 1)".
  - M1 evidence (archived 2026-10-05): every smoke test, the eval gate, the drift floor and the timing run, with numbers and job ids, is in `agent_legacy/active/2026-10-05_D1_M1-smoke-evidence.md`; the checker's verification is `audit/2026-10-05_M1_checker.md`.
  - Git (`origin/main`, all pushed): `7c1832f` M1 code and docs (2026-10-05 14:03), `7ab4cbd` O9 doc fixes, `cbd95ac` scripts retired, `9ed4afa` L* floor fix and litreview index rows, `eba8444` the pread reader (17:11). `.gitignore` no longer ignores `scripts/`. `agent/` (except the six docs tracked since the template), `audit/`, `logs/` and `data/` stay untracked, so every path a result depends on is written down here.
- Preferred approach to the three milestones (user, 2026-10-04; ask A8):
  - **How to work.**
    - Check with quick scripts inside the turn: recompute numbers, byte checks, small tests.
    - Run multi-agent workflows only for the milestone checker the user asked for, or before an irreversible step.
    - Never run a re-check loop. If a residual risk remains, state it and fix it later in a dated change rather than running another round.
    - Test only what the spec names: the section 9 smoke runs and the gates of lines 50 and 173 and of section 3.
    - Beyond that, suggest an extra check or smoke test when a result or its interpretation could be distorted without it. Say what it guards against and what it costs, and let the user decide. Do not add it unasked.
    - Ask before submitting any job to the H100 `gpu` partition (A10). A100 and CPU jobs need no ask.
  - **M1 (smoke tests pass and are verified).**
    - Done 2026-10-04: append Amendment 1.
    - Start the image-store build on CPU nodes, which takes about 1-2 hours, and write the training code while it runs.
    - Then run the spec's smoke tests and gates, then the checker, then wait for the user's go.
    - Target: store build and code on 2026-10-04; smoke tests, checker and go on 2026-10-05.
  - **M2 (the LR sweep has a decision).**
    - Run the two 3-epoch sweep runs on arm (c), at 1e-4 and 3e-4, roughly 1.5 hours each if A100s are free. (At 1 × 2 A100s, Amendment 2, the two ran in parallel on two nodes. Measured: 5 h 05 min per run with the pread reader, 7 h 21 min with the old one; `audit/2026-10-05_M2_sweep_results.md` §4.)
    - Decide on species top-1 on deduplicated ToL-val (line 161, S4).
    - Then the checker, then the user's go.
  - **M3 (all arms finish, no verdict).**
    - The winning (c) run continues to 10 epochs, then (d), (a) and (b) in the spec's order (line 153), each with its own λ (S6).
    - No projected hours (S29 as amended by A2.2): the runs report their own times. Do not stall for approval.
    - Then the checker. The user gives the verdict.
- Next action (M2, the LR sweep):
  1. Done: both sweep runs completed three epochs; under line 161 the sweep picks 1e-4 (85,586 vs 42,516 species-correct after epoch 2; `audit/2026-10-05_M2_sweep_results.md` §1).
  2. Awaiting the user (A1: a possible gap in the spec, the user decides): line 161 has no provision for both learning rates degrading species accuracy below the frozen model. Options named in the audit §7: proceed as written, or amend line 161 to add a lower learning rate (one 3-epoch run at 1 × 2 A100s is about 5 h plus queue).
  2b. The lr 1e-4 sampler control (ask A14, user, 2026-10-05): job 266465 (`p1c_lvl_lr1e-4_s42_rand_v1`, rails11, started 2026-10-06 00:05) trains arm (c) at lr 1e-4 for 3 epochs on the 10-epoch schedule with `data.train.sampler=random`: uniformly random batches, about 7,750 distinct species per batch against about 890 in the spec's grouped plan (groups average 9.2 images, not 16). Reading rule, set before its first epoch ends: if its ToL-val species count holds up or rises where the grouped run's fell (111,727 → 97,244 → 85,586), batch composition drives the decline; if it falls the same way, the equal-weight seven-level objective does, whatever the batches. Its step-0 species term is higher (0.72 against 0.23) because each batch holds 9× more species, so compare ToL-val counts, not loss values. Expected epoch ends, from 8.5 s steps plus a 30 min refresh: about 02:30, 04:35 and 06:40. Diagnostic only; it decides nothing under the spec. Watchdog on it until about 06:10 (`watchdog.log`).
  3. After that call: run the M2 checker once (Opus, read-only, to `audit/`; A1's four checks, S37, precedence, the decision's numbers against `decisions.json`), then wait for the user's go.
  4. After the go, if the spec stands: the 1e-4 run resumes with `ckpt_path=logs/p1/p1c_lvl_lr1e-4_s42_v1/checkpoints/last.ckpt trainer.max_epochs=10` (same `run_name`, same wandb id; about 1 h 35 min per epoch with the pread reader); λ for (d) = 0.2 / mean r_t over steps 50-250 of that run = 27.787 (S6; 3e-4 would give 15.222). S27 currently selects `epoch_00.ckpt` (111,727) for arm (c).
  5. The end-to-end smoke run is not resubmitted after 264728's timeout (user, 2026-10-05 ~17:30); the sweep runs' epoch ends exercised the same path three times each.
  - Timing facts kept for M3 planning (no projection is owed, A2.2). At 2 × 2: 3.1-3.5 s per ordinary step, 6.1 s per monitor step, ToL-val 82 s, refresh 41 min with the old reader (job 263082, at night). At 1 × 2: 5.8-6.0 s per ordinary step (median, the same with either reader), 10.5 s per monitor step, 47.4 GiB peak; refresh 25.0-33.5 min with the pread reader (265500: 2,007.6 / 1,708.5 / 1,497.8 s) against 59.1-73.8 min with the old one (264726: 4,425.6 / 3,737.1 / 3,546.2 s) or stuck (264728, unfinished after 2 h 49 min); ToL-val 144-149 s for both forms. An epoch after the first takes 1 h 32-35 min at 1 × 2 with the pread reader (2 h 12 min with the old one). Reader benchmarks on CPU nodes (16 workers, 65,536 fresh rows per run): pread 1,850-3,592 img/s against memmap 1,626-2,187, with less kernel CPU; DONTNEED leaves 0.6 GiB of page cache per 9.2 GiB read against 9.3 GiB without it, at no throughput cost (`audit/2026-10-05_stall_diag/bench_*.txt`).
  - M3 (decided, A2.4): the iNat21 eval records its encode setting and re-evaluates all six baselines under it; the floor and the S31 cells are same-setting counts.
  - M3 reading caveat (TaxaWalk review, `agent/litreview/taxawalk-and-pilot1.md`; the preprint's §4.1 says per-rank scoring is "unfavourable" to models trained on full lineages): at coarse ranks the per-rank protocol may favour the prefix-trained arms over BioCLIP 1 and RCME for reasons unrelated to geometry. The controlled comparisons, (d) vs (c) and (b) vs (a), are unaffected.
- Artifacts: M2 runs write under `logs/p1/<run_name>/` (Hydra run dir with `steps.jsonl`, `decisions.json`, `run_info.json`; `checkpoints/epoch_NN.ckpt` and `last.ckpt`) and to wandb project `multimodal_lab/taxa_maze`. Everything through M1 (audits, preflight and S9 reports, smoke outputs, store and prep paths) is listed in `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md`.
- Log:
  - 2026-10-06 ~00:22: the sampler control 266465 started at 00:05 on rails11 (steps from 00:08). First 65 steps: 8.45 s median (12.6 max), 60.1 GiB peak, `loss/con` 12.14 → 5.81, species image-to-text term 0.72 → 0.46; `run_info.json` records `sampler: random`. Docs updated for the first-round sweep results, the sampler flag and the control (ACTIVE.md, DESIGN.md, USAGE.md).
  - 2026-10-05 ~23:48: A14 recorded and acted on. Added the opt-in `data.train.sampler=random` mode (`src/data/tol_sampler.py:make_random_epoch_plan`, plumbed through `tol_datamodule.py`, `configs/data/p1_tol.yaml`, `run_info.json`); the default `grouped` path is unchanged (hash-identical plan at steps 0 and 646 through `PlanDataset`). Random plan on the real tree: `check_plan` passes, 647 steps, 7,746 species per batch on average (min 6,544), at most 17 images in a group, rank micro-batches 4,096 each. Hydra composition checked (`--cfg job`). Submitted 266465 at 23:47. Code uncommitted; the job reads the working tree when it starts, so `src/` and `configs/` stay untouched until then.
  - 2026-10-05 ~23:09: M2 sweep complete. 264726 (1e-4) ended 21:29 and 265500 (3e-4) ended 22:17, three epochs each, no stall (watchdog probes clean from 17:22 to its last probe at 20:27, when the session that ran it ended; both jobs then ran to completion). Line 161 picks 1e-4 (85,586 vs 42,516 species-correct). Both runs fall below untouched BioCLIP 1 at species (108,062) by epoch 1; 1e-4 peaks at epoch 0 (111,727) and declines; 3e-4 is unstable (loss up in epoch 1, kingdom 5.29% under the photo form at epoch 1). Full tables and reading in `audit/2026-10-05_M2_sweep_results.md`. Raised to the user as a possible gap in line 161; checker held.
  - 2026-10-05 ~19:12: 265500 (3e-4, pread) finished epoch 0 at 19:09: refresh 2,007.6 s (33.5 min; the old reader took 4,425.6 s at 1 × 2 and 41 min at 2 × 2), ToL-val 144 s. ToL-val photo species 40.23% → 31.33% (84,164 of 268,609; −8.90 points); 264726 (1e-4) had 41.59% after epoch 0. Interpretation, not a decision: the training loss at step 646 was lower at 3e-4 (loss/con 4.54 against 4.60, 25-step means), but its species-level image-to-text term rose through the epoch (0.18 at step 100 → 0.26 at step 646; genus 0.60 → 0.65) while kingdom to family fell. At 1e-4 the species term also rises, more slowly (0.14 → 0.18 → 0.22 at step 1293). Checks: the same `init` count in both runs (108,062) and bit-identical step-0 losses (37 of 37). Watchdog: no stall from 17:12 to 19:08; re-armed until 23:12. The decision stays at epoch 2 (line 161, S4).
  - 2026-10-05 ~17:26: 264726 completed epoch 0 at 17:04 (old-reader refresh about 73 min, then ToL-val and `epoch_00.ckpt`): photo species 40.23% → 41.59% (+3,665 images), mean7 22.09 → 70.09. The user asked whether 41.59 is worse than frozen BioCLIP 1; it is +1.36 points above it on the same benchmark; the 70.19% is iNat21 (10,000 species, against ToL-val's 131,143). Ready for compact and a paused session.
  - 2026-10-05 ~17:20: stall handled. `ImageStore` reads with `os.preadv` plus `POSIX_FADV_DONTNEED` (commit `eba8444`); production batches are byte-identical to the old reader (9 of 9: steps 0 and 646 of both ranks, three refresh chunks, two ToL-val chunks; `audit/2026-10-05_stall_diag/equiv_check_1713.txt`). 264727 cancelled at step 306; lr3e-4 restarted as `v2`, job 265500 (17:12, rails14). 264728 timed out in its refresh at 17:16. Watchdog on 264726 and 265500. User decisions: keep 1 × 2 and accept long refreshes; park a two-epoch refresh cadence (Parking lot).
  - Entries from 2026-10-05 ~14:15 (M2 launch) to ~17:00 (the reader switch) are in `agent_legacy/active/2026-10-05_D1_log-M2-launch-to-reader-switch.md`.
  - Earlier entries, 2026-10-02 intake through 2026-10-05 ~00:55 (the M1 checker), are in `agent_legacy/active/2026-10-05_D1_log-through-M1.md`.

## Blocked
None (B2 and B3 resolved 2026-10-04; see the prune log).

## Parking lot
- (user, 2026-10-05) Step-shaped bank-monitor curves. In wandb, the `bank_monitor/parent_species` curves change in steps about every 50 training steps rather than smoothly. Why? Not investigated. One fact to start from: the monitor is only computed at monitor steps (every step from 50 to 250, then every 50th), so the values exist only there; check first whether the step shape is how wandb draws those sparse points.
- (user, 2026-10-05) Bank refresh every two epochs, a fallback if the 1 × 2 refresh stays long. Justify it first with a probe of drift with and without a refresh: for example, after epoch 1, measure the EMA bank's drift against a fresh bank, and against the epoch-0 fresh bank carried one more epoch. Spec line 91 asks for a recomputation after each epoch, so this would need an amendment.
- (A5) Flat decoding. Pilot 1 regularizes the geometry, but it still classifies by a flat nearest-neighbor lookup over all labels at a rank. TaxaWalk (sections 1-2) argues for moving from rank to rank, using the possibly well-structured taxonomic text space of BioCLIP 1/2. Limitation of pilot 1, and a direction after it.
- (A5) Image-side constraints only. Pilot 1 imposes the geometric constraints on image prototypes only. The text side could be constrained too, either separately or jointly with the image side.
- (A5) "TaxaWalk++". A future framework could fold parts of pilot 1's design into TaxaWalk's predictive representation-learning framework. Cheapest concrete form (TaxaWalk review): cross frozen BioCLIP 1, arm (c) and arm (d) with a flat, a top-down and a TaxaWalk decoder (`agent/litreview/taxawalk-and-pilot1.md`).
- (A9) Top-down evaluation. Classify from kingdom down to species, with the candidates at rank t restricted to taxa that share the ranks already predicted. The user expects this to match BFL's "top-down constrained inference"; verify against the BFL paper before relying on that. It would be reported next to the per-rank protocol of spec section 7, not instead of it. Verified 2026-10-05 (TaxaWalk review, against BFL's v1/v2 TeX): it is BFL's rule. The review suggests report-only M3 columns with no new encode, because `src/eval/zeroshot.py` already computes each rank's argmax: top-down, species read-off, oracle-parent accuracy, full-path accuracy and nLCA. The user decides.
- (A9) Rank-balanced penalty. Replace the plain sum over (species, ancestor) terms in each species' penalty with LogSumExp (Li et al., 2020), so each term's gradient is weighted by a softmax over the term values. TaxaWalk uses this for its next-step MSE loss, a different objective. Question: does it balance the six ranks better than the uniform weighting of spec line 116, and how does it change λ calibration? TaxaWalk review (2026-10-05): its LogSumExp runs over seven batch-level per-rank errors, not over each species' own terms; its τ = 0.1 does not carry over to cos² terms; the gain is small (Figure 4c, species +2.35, measured); and it would change λ by up to about 6×. Position: not for pilot 1; the batch-level form stays open after it.

## Prune log
| date | block | verdict | destination |
|---|---|---|---|
| 2026-10-04 | A4, A5, A6 (asks) | DONE; no live direction references them | `agent_legacy/active/2026-10-04_asks-A4-A6_blocks-B2-B3.md` |
| 2026-10-04 | B2 (spec decisions) | RESOLVED: Amendment 1 appended | same file |
| 2026-10-04 | A7 (spec decisions, repo move) | DONE: Amendment 1 appended; repo on /u/liv | `agent_legacy/active/2026-10-04_ask-A7_spec-decisions-and-repo-move.md` |
| 2026-10-04 | B3 (file quota) | RESOLVED for pilot 1: outputs and repo on /u/liv; tol_embed stays (read-only) | `agent_legacy/active/2026-10-04_asks-A4-A6_blocks-B2-B3.md` |
| 2026-10-04 | A9 (parking-lot ideas) | DONE: both ideas are Parking-lot lines | `agent_legacy/active/2026-10-04_ask-A9_parking-lot-ideas.md` |
| 2026-10-04 | A2 (loggers, run naming) | DONE: wandb for official runs, csv for smoke runs, naming in USAGE "Run naming"; the first wandb run comes with M2 | `agent_legacy/active/2026-10-04_ask-A2_loggers-and-run-naming.md` |
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
