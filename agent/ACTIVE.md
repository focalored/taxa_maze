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


## Directions

### D1: Pilot 1, hierarchical orthogonality penalty on BioCLIP 1
- Asks served: A1; A16 (A15 done 2026-10-10, pruned).
- Hypothesis (spec section 1): adding the cross-level orthogonality penalty to BioCLIP 1 fine-tuning makes zero-shot classification on iNat21-val competitive across all seven ranks, against BioCLIP 1, BioCLIP 1 + RCME, and BioCLIP 1 + BFL.
- Success bar (quoted from spec section 1): "competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons."
- Failure mode (quoted from spec section 7): "Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification."
- Who decides: the user gives the verdict. I report numbers and interpretations only.
- Spec: `specs/pilot1.md` with Amendments 1-3 at the bottom (51,801 bytes, sha256 `eaec4f73…`), the later one first on any conflict; S9 list `specs/pilot1_s9_missing_species_keys.txt`; the preflight report is not edited. Paths: repo `/u/liv/repos/taxa_maze`, data `/u/liv/data/taxa_maze/{store,p1}`, every write under `/u/liv`. Code map: DESIGN.md "Idea → code"; how to run: USAGE.md. Git: `origin/main` at `e649803` (2026-10-08) plus today's uncommitted docs.
- Runs and results: `audit/2026-10-10_M2_runs_table.md` (every M2 run, its reference, both benchmark curves, timing facts). Readings: `audit/2026-10-05_M2_sweep_results.md`, `2026-10-06_A14_sampler_control.md`, `2026-10-08_M2_phase2_round1.md`, `2026-10-08_text_geometry/text_geometry.md`, `2026-10-10_M2_phase2_round2.md`, `2026-10-10_bfl_gap_confounders.md`. M1: `agent_legacy/active/2026-10-05_D1_M1-smoke-evidence.md`, checker `audit/2026-10-05_M1_checker.md`.
- How to work (A8, verbatim in the ledger; A10): quick scripted checks in the turn; workflows only for the milestone checker or before something irreversible; no re-check loops; extra checks are suggested with their cost and the owner decides; A100 jobs need no ask, H100 does. Milestones: M1 done 2026-10-05; M2 was the learning-rate decision plus the second phase under Amendment 3, closed by the owner on 2026-10-10; M3 is every arm to ten epochs, one iNat21 eval per arm on its S27 checkpoint, the checker, the owner's verdict.
- Closed (M2), 2026-10-10. The owner's statements, with what the runs added:
  - Learning rate: 1e-4 under line 161; 3e-4 is worse and unstable.
  - The shape. Under the level-restricted objective, species accuracy on ToL-val and on iNat21 peaks after the first epoch and falls every epoch after, while kingdom to family rise within the first epoch and hold. The species-only objective (arm a) holds species (+5.3 iNat21 points over BioCLIP 1 after one epoch, +3.4 after five) and loses phylum and kingdom. Owner: the level-restricted objective is the culprit; coarse-rank supervision competes with species supervision.
  - What did not change the shape: batch composition (random batches lift the level, same fall); three rank balancings (gradient-norm gives species the largest weight and the best first epoch, then the same fall; loss-ratio the slowest fall; LogSumExp, family-heavy, worse than uniform); freezing the text tower (lower ceiling, slower fall); the penalty at λ 30 (equal first epoch, slower fall). Only removing the coarse terms removes the decline. Owner: the contrastive side is now mostly tested; alpha or further learning rates would stretch the curve, not change its sign, and are not worth runs.
  - The text anchor: BioCLIP 1's truncated labels are not the centroids of their species' labels at kingdom to family (cos 0.3-0.4 against a 0.15-0.25 baseline; genus 0.74), so image-side LoRA was a recipe test, and as a recipe it is worse under both objectives.
  - Implementation: the loss equals an independent implementation of BFL's Eqs. (1)-(2) to 1e-9; the reader, the batch plan and the eval harness are verified (M1 gate; 9-of-9 reader identity; BioCLIP 1's counts reproduced exactly). Unverifiable without BFL's code: their batch construction, template and temperature handling, and whether their species accuracy fell during training.
  - The gap to the released BFL model (68.78 against our 50.89 at species after five epochs) is confounded by data: BFL trained on ToL-10M including iNat21 train (2.7M images of the val species); our ToL-EOL set holds 8,624 of the 10,000 val species (median 81 images), 13.8% absent. Owner: suspects implementation, LoRA or data; will ask for BFL's code.
  - S27's reading: every (c) and (d) run peaks at epoch 0 on ToL-val, and at that epoch gradient-norm, loss-ratio, uniform and d beat BioCLIP 1 at every rank on iNat21 (species by 2,000-4,000 images, average 79.5-80.6 against 41.5). Amendment 3 is in force (diagnostic iNat21 evals, the second phase's rules, the recipe record, the qualified expectation).
- Open at M2's close (carried into M3 preparation):
  - Which (c) form goes forward: uniform (the spec's), loss-ratio (slowest fall), or gradient-norm (best first epoch). The owner decides; A3.3 records it.
  - The two-stage schedule (seven levels for the first epoch, then species-only, or a fade of the coarse weights): the one untested knob that could change the shape, since the coarse ranks saturate in the first epoch and arm (a) holds species; about 20 lines on the level weights and one 8 h run; an amendment if adopted. Owner: arguably an M3 item, tried alongside the penalty.
  - Full fine-tuning instead of LoRA (spec line 154 queues it): capacity could change the shape; one 8 h run at a lower rate. Added by the agent; the owner's call.
  - A3.4's rank and checkpoint (S27-selected or final); whether the whole-number λ carries into M3 (A3.3). The lower-LR gap in line 161 stays open on paper; the owner does not want the run.
  - BFL's code: the owner intends to ask; their batch construction, templates, temperature and any per-epoch curves would settle the unverifiable items above.
  - Set aside, not dropped (owner, 2026-10-10): alpha = r (lines 61, 154; config only), the level loss without dedup and group positives (lines 96-97; flag in `losses.py`), batch size 16,384 (S8; 2 × 2 A100s), `{lineage}` training text (line 51(a); flag at `p1_module.py:156`). Each is one 8 h run on the grouped sampler; the owner expects them to stretch the curve rather than change its sign, and would run them after the items above.
- Questions for M3 (remaining and added; the penalty side is largely untested, owner 2026-10-10):
  - The pilot's question, (d) against (c), with the decline in view: the penalty's effect appears over epochs (d ends 6.5 iNat21 points and 11,577 ToL-val images above c-uniform at epoch 4, coarse ranks unchanged; one seed, one λ), so the ten-epoch runs' per-epoch curves carry the comparison; the S27-checkpoint comparison alone is a wash (72.09 against 72.24).
  - λ: one value so far; the realized λ·r_t was 0.112 against S6's 0.200 because the penalty's gradient shrinks once optimized; line 94 queues a λ sweep on ToL-val (candidates 60 and 120). Owner: with the penalty in view there is more to sweep.
  - The penalty's form: 1/g_s against line 107's uniform (S22; own λ); the whitener (lines 124-139; Σ_B per the lit review); the refresh cadence and K = 8 (parked); the text-side penalty with a frozen image tower (parked).
  - Arm (b), flat plus penalty (λ from arm (a)'s 9.26, so 9 or 10): the second controlled pair, untested. Which (c) the penalty sits on, and whether one image-side (d) is worth a run (owner: one of the two, by which weighting wins).
  - The verdict under the data confound: our arms' iNat21 numbers are transfer numbers; the baselines trained with iNat21 train. A2.4's re-evaluation puts every number under one encode setting but cannot remove this; it is stated with the verdict. Decided: A2.4's setting; the TaxaWalk reading caveat (the per-rank protocol may favour prefix-trained arms at coarse ranks; the (d) vs (c) and (b) vs (a) pairs are unaffected).
  - Rare Species (A16) as a reported diagnostic beside iNat21: out-of-distribution species for every model; BFL-Euc 43.74 at species against BioCLIP 1's 32.66 and our best (c) first epoch 41.10 (gradient-norm); the decline over epochs is mild there (`audit/2026-10-10_rare_species.md`). Whether it enters the M3 report is the owner's call; the verdict stays iNat21 unless amended.
- M3 preparation (owner closed M2 on 2026-10-10). M2's checker has not run: A1 asks for it after each milestone before the go, so it is the owner's call to run it now (an Opus read-only subagent, about 40 min, to `audit/`) or to waive it with a dated note.
  - What M3 is (spec lines 153, 161, 172; S27; A2.4; Amendment 3): every arm to ten epochs at lr 1e-4 on the adopted (c) recipe, in the order (c), (d), (a), (b), each with its own λ (S6); the chosen (c) run resumes from `last.ckpt` or restarts under the revised recipe; one iNat21 evaluation per arm on its S27 checkpoint under A2.4's setting, with the six baselines re-evaluated under the same setting; then the checker; then the owner's verdict against S31's bar and S32's floor, stated with the data confound.
  - Decisions before M3 starts (a dated amendment wherever the spec changes): the (c) form (uniform, loss-ratio or gradient-norm; A3.3); A3.4's rank and checkpoint; λ as calibrated or as a whole number (A3.3); whether the two-stage schedule or full fine-tuning is tried first (above); whether Rare Species is reported beside iNat21 in the M3 report (a diagnostic; not in the verdict unless amended).
  - Cost at 1 × 2 A100s with the pread reader: about 16 h per ten-epoch arm, four arms about 64 h plus queue; evaluations minutes (`audit/2026-10-10_M2_runs_table.md`, timing facts).
- Next action: the owner's decisions above; the M2 checker is running (2026-10-10 17:0x, read-only, to `audit/2026-10-10_M2_checker.md`); then M3's first arm. Nothing else is running or queued.
- Artifacts: runs under `logs/p1/<run_name>/` (Hydra run dir with `steps.jsonl`, `decisions.json`, `run_info.json`; `checkpoints/epoch_NN.ckpt`, `last.ckpt`; `inat21_diag/`), wandb project `multimodal_lab/taxa_maze`; audits under `audit/`; M1-era artifacts in `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md`.
- Log (newest first; older entries in `agent_legacy/active/`):
  - 2026-10-10 ~17:17: Rare Species evaluations done (307305 14 min, 307306 53 min; 6 baselines, 49 run checkpoints); audit written; A16 DONE. M2 checker launched on the owner's word (Opus, read-only).
  - 2026-10-10 ~16:50: docs prepared for the pause: M2 closed by the owner (checker pending the owner's call), an M3-preparation block under D1, Rare Species status under A16. `agent/1.md` and `agent/2.md` are the owner's copies of ACTIVE.md from 2026-10-08 (ignored by git, untouched).
  - 2026-10-10 ~16:31: A15 marked DONE and pruned to `agent_legacy/active/2026-10-10_ask-A15_c-recipe-second-phase.md` (verdict and paths there); older log entries of today moved to `agent_legacy/active/2026-10-10_D1_log-synthesis-day.md`.
  - 2026-10-10 ~16:21: Rare Species evaluation split on the owner's ask: 307071 (everything in one job) cancelled while starting; 307305 evaluates the six baselines, untouched BioCLIP 1 and the sweep and control runs, 307306 the eight second-phase runs (281183, 281184 and their derived runs); results to `logs/p1/rare_species_diag_baselines/` and `<run>/rare_species_diag/`.
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
- (A9) Rank-balanced penalty. Replace the plain sum over (species, ancestor) terms in each species' penalty with LogSumExp (Li et al., 2020), so each term's gradient is weighted by a softmax over the term values. TaxaWalk uses this for its next-step MSE loss, a different objective. Question: does it balance the six ranks better than the uniform weighting of spec line 116, and how does it change λ calibration? TaxaWalk review (2026-10-05): its LogSumExp runs over seven batch-level per-rank errors, not over each species' own terms; its τ = 0.1 does not carry over to cos² terms; the gain is small (Figure 4c, species +2.35, measured); and it would change λ by up to about 6×. Position: not for pilot 1; the batch-level form stays open after it.

## Prune log
| date | block | verdict | destination |
|---|---|---|---|
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
