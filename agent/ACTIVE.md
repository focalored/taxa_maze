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

### A3 (2026-10-02) Literature reviews of four papers. Status: ACTIVE
Verbatim ask (2026-10-02). It answered my question on whether to review now or later. I had proposed parking reviews until the first long training run, except papers that bear on a pending gap decision.
> 1. arxiv/2406.01506. The theoretical framework describing orthogonality in hierarchical semantic concepts in LLMs.
> 2. arxiv/2602.11448. The most directly related work on vision-language models (frozen CLIP) which ports the hierarchical orthogonality idea from Paper 1. Different in that they introduce heirarchical sparse coding with SAE.
> 3. agent/litreview/Sastry2027-TaxaWalk. Currently a preprint from my lab.
> 4. arxiv/2506.21476. Code: https://vishu26.github.io/RCME/index.html. Earlier paper by Sastry from my lab. Fine-tuned BioCLIP 1 with a local and a global entailment loss. Pull out their training recipe.

Verbatim follow-up, sent mid-turn (2026-10-02):
> Prioritize the fourth paper on the list. Save the rest in the Parking lot

My interpretation (not the user's words):
- Paper 4 (RCME) is DONE (2026-10-02): `agent/litreview/bioclip1-finetune-recipes.md`, verified against the paper and code (`audit/2026-10-02_rcme_recipe_verification.md`; 19 rows confirmed, 4 contradicted and fixed in place).
- Papers 1-3 sit in the Parking lot below. Each line names the question its review should answer. I launch them when the first long training job starts, unless told otherwise.
- The TaxaWalk preprint is the lab's unpublished work, at `agent/litreview/Sastry2027-TaxaWalk.pdf`. Reviewers read the local PDF only, and never send its contents to an external service.

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

### A12 (2026-10-05) Decisions after the M1 checker; Amendment 2; start M2. Status: ACTIVE
Verbatim ask (2026-10-05, about 13:50):
> write a dated amendment marked with (M1) to pilot1.md that reflects the following decisions before M2 starts. also update all relevant documents in agent/ to reflect the most up-to-date research state.
>
> P2: approve autocast off computation
> P3: approve the 1*2 A100 amendment and the config and script edits, but skip the end-to-end run for measuring memory and time
> P4: i hold my decision
> O1: run 262320 (to cover refresh, ToL-val metrics, ckpt saving, etc.) alongside M2's LR sweeps. no postponing M2.
> O2: record encode setting of M3 iNat21 eval, and evaluate bioclip 1 using the same setting, rather than using the old baseline. approve an amendment so that all baselines are evaluated under the same convolution algorithm setting during M3.
> O3: just skip the measurement. it is not important.
> O4: after all edits are in place and the LR sweep of M2 is ready to fire, commit and push before launching M2
> O7: change checkpoint names and decisions.json to use 0-based epoch, to match Lightning's convention.
> O9: find and fix documentation inaccuracies except for small timestamp discrepancies.
> O6: make minimal edits to fix this.

Verbatim addition, sent mid-turn (2026-10-05, about 14:05):
> you set max_epochs=3 for the LR sweep, but make sure the linear schedule and cosine decay are configured for 10 epochs, same as for a 10-epoch run.

My interpretation (not the user's words):
- I read this as the go on M1: every checker item is decided and M2 is to start right after the commit. P and O numbers are the checker report's (`audit/2026-10-05_M1_checker.md`); A11 held the user's first P3/P4 answers (now in `agent_legacy/active/`).
- Amendment 2 (2026-10-05, M1) holds the spec-level items: A2.1 geometry (S8), A2.2 no S29 measurement, A2.3 0-based epochs with `init` for the untouched model, A2.4 baselines re-evaluated under the M3 encode setting; plus P2, O6, P4, O4 and O9 as decisions without a rule change.
- "262320" is the end-to-end smoke run, which had been cancelled and resubmitted twice (263082 at 2 × 2, cancelled when the geometry changed); it runs at 1 × 2 alongside the sweep (job in the log). Its memory and time numbers are a by-product that nothing waits for.
- The schedule: `_lr(t)` uses `model.epochs × 647 = 6,470` total steps and warmup 65, and `on_fit_start` refuses `model.epochs != 10`; `trainer.max_epochs=3` only stops the sweep run after `epoch_02.ckpt`. At step 1,941 the LR is 0.80 × base; it reaches 0 at step 6,469. Checked numerically on 2026-10-05.
- O9 is done by a read-only Opus review of `agent/` after this turn's edits, then my fixes, in a second commit.

## Directions

### D1: Pilot 1, hierarchical orthogonality penalty on BioCLIP 1
- Asks served: A1.
- Hypothesis (spec section 1): adding the cross-level orthogonality penalty to BioCLIP 1 fine-tuning makes zero-shot classification on iNat21-val competitive across all seven ranks, against BioCLIP 1, BioCLIP 1 + RCME, and BioCLIP 1 + BFL.
- Success bar (quoted from spec section 1): "competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons."
- Failure mode (quoted from spec section 7): "Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification."
- Who decides: the user gives the verdict. I report numbers and interpretations only.
- True now (2026-10-05 ~14:15): M1 is DONE. The checker (`audit/2026-10-05_M1_checker.md`: 0 blockers, 1 should-fix, 3 notes; all 9 smoke-test claims verified; 0 spec deviations; freeze PREFIX_OK), my response (`audit/2026-10-05_M1_checker_response.md`) and the user's decisions (A12) close it. Amendment 2 (M1) is appended to the spec. Code since the checker: fp64 gradient cosine with autocast off (P2), `model.train()` at fit start (O6), 0-based checkpoint names and `decisions.json` keys with `init` for the untouched model (O7), 1 node × 2 GPUs at 4,096 images per GPU (A2.1). M2 starts now: the two sweep runs and the end-to-end smoke run are submitted after the commit (job ids in the log). No training result exists yet.
  - Spec: `specs/pilot1.md` with Amendments 1 and 2 at the bottom (47,559 bytes, sha256 `5315446f…`), which govern on any conflict, the later one first. The S9 list is `specs/pilot1_s9_missing_species_keys.txt`. Do not edit the preflight report.
  - Paths: the repo is `/u/liv/repos/taxa_maze` (`~/bdbk/repos/taxa_maze` is a symlink to it). Data, the BioCLIP 1 checkpoint and tol_embed stay on `/projects`, read only. Every write goes to `/u/liv`, never under `/u/liv/bdbk` (S1). The store is `/u/liv/data/taxa_maze/store/` and the prep outputs are in `/u/liv/data/taxa_maze/p1/` (USAGE "Data").
  - Code map: DESIGN.md "Idea → code". How to run each test: USAGE "Quickstart (pilot 1)".
  - Passed on A100 (each has `result.json` under `logs/smoke/<test>/<stamp>_j<job>/`; the single-GPU tests are job 257773):
    - adapter: bit-identical at init in fp32, fp16 and bf16; only the q and v rows move; 0.653% trainable.
    - cache (line 50): 1,000 rows, min cosine 0.9999943.
    - one_step (S35): arm (d) L_con 10.456, L_B 0.00795; arm (b) L_con 4.134; both gradients nonzero.
    - penalty_twice (S14): bit-identical.
    - penalty_forms (S22): the two forms are bit-identical on the 49 g_s = 1 groups and differ on the mixed batch.
    - zero_sum (S12): 3.0e-15 after 120 updates.
    - grad_1v4 (S17): 1 vs 4 A100s (jobs 257773 and 257774): cosine 0.999988 (contrastive) and 0.999987 (penalty), norms within 0.012%. The earlier H100 pair (257613, 257650) gave 0.999997.
    - S13 (line 189; `logs/smoke/lam0/s13_result.json`, jobs 257775, 257664-257668): PASS. Both control pairs (c_run1/c_run2, a_run1/a_run2) are bit-identical over 100 steps, and the λ = 0 runs (d_lam0, b_lam0) are bit-identical to their controls on the total, the contrastive term and every level's i2t and t2i. Deterministic mode warned about two backward kernels, flash attention and memory-efficient attention.
  - Eval gate (lines 173, 190; `audit/2026-10-04_eval_gate.md`): Part A has 84 of 84 cells exact (job 257672; `logs/smoke/eval_gate/baseline_counts.json` has `all_reproduced: true`). Job 257672 itself exited 1, because its Part B at the brief's settings (batch 512, cuDNN benchmark off) had 6 of 600,000 rows below cosine 0.9999. Part B passes on the rerun at the caches' settings (job 257703, every row bit-identical), which is the user's decision.
  - Drift floor (S25): done 2026-10-05 00:30, `/u/liv/data/taxa_maze/p1/drift_floor.json`. Parts 1 and 3 took 28 min each alone on a node (job 257594); parts 0 and 2 first timed out at 2 h sharing rails12 (GPU idle, loader workers CPU-bound), then ran in 25 min each as one 2-node job with one part per node (262306, submitted 00:00, nodes rails[09,15]); merge 262307. bf16-vs-fp16 species means: mean distance 0.0100, mean 1 − cos 9.6e-5; the fp16 pass vs the cache seed: 7.5e-4 and 5.7e-7. The end-to-end smoke run (`experiment=p1/smoke_timing`, 3 h, excluding rails11-12) runs at 1 × 2 alongside M2 (A2.2; job in the log); its earlier submissions 262320 and 263082 were cancelled before starting when the geometry changed.
  - Timing run 257654 (pre-fix code; `logs/smoke/timing/2026-10-04_17-19-00_j257654/`): timed out at 18:48 after 44 of 60 steps, before refresh and eval. Median step 10.0 s with 18 stalls of 32-591 s (`time/step_s` is measured inside `training_step` on rank 0, so a stall on another rank shows as collective wait; cause not identified; it ran on rails[11,15]). The six S13 runs on rails[13-14] show no stalls: median 6.1-6.2 s per step, max 11-29 s, 100 steps in 8-9 min. The pre-training ToL-val (S27's "epoch 0", untouched BioCLIP 1, logged as `val_epoch0/`) took 349 s (photo species 40.23%, mean7 22.09%). Peak GPU memory in the S13 runs: 24.9 GiB of 80 (S8).
  - Git: the M1 code and docs are committed and pushed to `origin/main` on 2026-10-05 (hash in the log); `.gitignore` no longer ignores `scripts/`. `agent/` (except the six docs tracked since the template), `audit/`, `logs/` and `data/` stay untracked, so every path a result depends on is written down here.
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
    - Run the two 3-epoch sweep runs on arm (c), at 1e-4 and 3e-4, roughly 1.5 hours each if A100s are free. (At 1 × 2 A100s, Amendment 2, about 11 h each; the two run in parallel on two nodes.)
    - Decide on species top-1 on deduplicated ToL-val (line 161, S4).
    - Then the checker, then the user's go.
  - **M3 (all arms finish, no verdict).**
    - The winning (c) run continues to 10 epochs, then (d), (a) and (b) in the spec's order (line 153), each with its own λ (S6).
    - No projected hours (S29 as amended by A2.2): the runs report their own times. Do not stall for approval.
    - Then the checker. The user gives the verdict.
- Next action (M2, the LR sweep):
  1. Sweep runs of arm (c) at 1e-4 and 3e-4, `trainer.max_epochs=3`, 1 node × 2 A100s each (jobs in the log); about 11 h each. Watch the first epoch end, about 3.5 h in: refresh, ToL-val, `epoch_00.ckpt`, `decisions.json`.
  2. The end-to-end smoke run alongside (O1, A2.2): 60 steps of arm (d), then refresh, ToL-val and a checkpoint, about 1.7 h. If it fails at the epoch end, cancel the sweep runs, fix, resubmit.
  3. Decision (line 161, S4): after the third epoch compare `decisions.json["epochs"]["2"]["photo"]["species"]["correct"]` of the two runs; the higher count wins; report every rank and `mean7` too. Record it in D1, then run the M2 checker (A1's four checks, S37, precedence, plus the decision's numbers), then wait for the user's go.
  4. After the go: the winner resumes with `ckpt_path=logs/p1/<run_name>/checkpoints/last.ckpt trainer.max_epochs=10` (same `run_name`, same wandb id); λ for (d) is 0.2 / mean r_t over steps 50-250 from the winner's `decisions.json["lambda_calibration"]` (S6).
  5. O9 follow-up: the doc review's fixes go into a second commit.
  - Timing facts kept for M3 planning (no projection is owed, A2.2): 6.1 s median per step at 2 × 2 on healthy nodes; 349 s per ToL-val pass on 4 GPUs; one fp16 pass over a quarter of the train images took 845 s on one GPU alone on a node and over 2 h with two parts sharing a node, so the loader, not the GPU, bounds the refresh. At 1 × 2 expect about twice the per-run times.
  - M3 (decided, A2.4): the iNat21 eval records its encode setting and re-evaluates all six baselines under it; the floor and the S31 cells are same-setting counts.
- Artifacts:
  - M1: `audit/2026-10-04_M1_code_review.md`, `audit/2026-10-04_M1_code_review_response.md`, `audit/2026-10-04_eval_gate.md`; smoke outputs under `logs/smoke/`; the store and prep outputs under `/u/liv/data/taxa_maze/`.
  - `audit/2026-10-02_preflight_{data,eval,model}.md`, `audit/2026-10-02_reference_behaviour.md`, and `agent/litreview/bfl-level-restricted-loss.md`.
  - S9: `audit/2026-10-02_S9_epithets.md` (the answer), `audit/2026-10-02_S9_epithet_census.md`, `audit/2026-10-02_S9_bioclip1_provenance.md`, `audit/preflight/s9_epithet_census.json` (with the candidate frozen key list), `audit/preflight/S9_bioclip1/`, and `audit/preflight/s9_gbif_cache.json` (14,571 GBIF lookups).
  - `audit/preflight/`: helper outputs, Slurm logs, the gap lists and verdicts.
  - `archive/2026-10-02_preflight-workflow-prompts/`: the workflow scripts, which hold the exact prompts behind those reports.
- Log:
  - 2026-10-05 ~14:15: A12 recorded: the M1 go and the decisions on P2, P3, P4, O1-O4, O6, O7 and O9. Amendment 2 (M1) appended to the spec (47,559 bytes, sha256 `5315446f…`; byte-prefix test PREFIX_OK against 90a7d38 and against the pre-append file). Code: fp64 gradient cosine with autocast off; `model.train()` at fit start; checkpoints `epoch_{epoch:02d}` and `decisions.json` keys 0-based with `init`; `configs/trainer/p1_ddp.yaml` 1 node × 2 GPUs and `train.sh --nodes=1`; `.gitignore` no longer ignores `scripts/`. Configs compose for both sweep runs and the smoke run (1 × 2, `model.epochs` 10, `max_epochs` 3). A11 pruned. Commit, push and the M2 launch: see the next entry.
  - 2026-10-05 ~00:55: A11 recorded (the user's P4 and P3 responses, then "do not act on P3 P4 yet. just record my inputs"). Before the hold arrived I had launched a P4 subagent and cancelled the end-to-end run 262320; the subagent was stopped with no file changed (hashes equal the checker's), and the run was resubmitted unchanged as 263082. P3 findings written under A11: refresh memory is not the constraint, the training peak at 4,096 per GPU is estimated at 45-50 GiB, per-run time doubles while GPU-hours stay equal, and S8 would need an Amendment 2. Docs prepared for a pause.
  - 2026-10-05 ~00:45: M1 checker done (`audit/2026-10-05_M1_checker.md`, 0 blockers, 1 should-fix, 3 notes, 9 observations). P1 fixed: the S13 warning record missed the memory-efficient-attention kernel; regex fixed, `s13_result.json` regenerated (pass unchanged), docs reworded. O8 fixed (the compare script now requires 100 steps per run). O9 fixed (stamps from `date` and sacct; USAGE lines 28, 43, 62, 101). Decisions P2-P4, O2, O4 listed for the user. Floor merged at 00:30; end-to-end run 262320 queued. Response: `audit/2026-10-05_M1_checker_response.md`.
  - 2026-10-05 ~00:00: resumed. S13 PASS, bit-identical in both pairs (`s13_result.json`); grad_1v4 passes on A100s. Floor parts 0 and 2 hit TIMEOUT at 18:42/18:45, cancelling the merge and the end-to-end run; chain resubmitted as 262306 → 262307 → 262320. Timing run 257654 hit TIMEOUT at step 44 with 18 stalls; the S13 runs show none. Fixed `s13_compare.py` (the seed lives in `.hydra/config.yaml`, not `run_info.json`). Epoch numbering confirmed 1-based for checkpoints and `decisions.json`, with "epoch 0" the pre-training reference (S27); ACTIVE and USAGE reworded. Checker launched.
  - 2026-10-04 ~18:00: D1 checked line by line against the files before a pause. Every number under "Passed on A100" and in the eval-gate line matches its `result.json` or audit; the spec is still a byte prefix of 90a7d38 plus Amendment 1 (sha256 `ab593020…f33eb`). Corrected: the two newest log entries' clock times (they ran about 15 min fast), the step-time line (measured, not 3-5 s), and the job states. New risk: floor parts 0 and 2 share a node and may time out at 18:42-18:45; fallback written. A2 pruned as DONE.
  - 2026-10-04 ~17:40: eval gate finished. Part A: 84 of 84 cells exact (job 257672). Part B: 6 of 600,000 rows fall below 0.9999 with batch 512 and benchmark mode off; all rows are bit-identical with the caches' settings (job 257703). The cause is cuDNN's choice of convolution algorithm. The user's decisions, verbatim: Part B, "Accept the bit-identical run (Recommended)"; end-to-end run, "Submit it now (Recommended)" (job 257785). Earlier, when asked about the H100 jobs, the user chose "Move all S13 runs to A100s". The A100 reruns of the single-GPU smoke tests all pass.
  - 2026-10-04 ~17:30: code review back (0 blockers, 4 should-fixes, 15 notes; `audit/2026-10-04_M1_code_review.md`). All fixed or answered (`audit/2026-10-04_M1_code_review_response.md`), plus a checkpoint deadlock I found while fixing (epoch-end metrics were logged on rank 0 only). All smoke tests rerun on A100 on the final code (jobs 257773, 257774); S13 runs 257775 and 257664-257668.
  - 2026-10-04 ~17:10: passed so far (each with `result.json` under `logs/smoke/<test>/`): adapter, cache, one_step, penalty_twice, zero_sum, grad_1v4 (1 vs 4 GPUs: cosine 0.999997, norms within 0.02%). A 100-step control run on 4 H100s completed (`logs/smoke/lam0/c_run1_h100/`, record only). Per A10 the user moved all six S13 runs to 2×2 A100s: jobs 257663-257668; the S29 timing run is job 257654; the eval gate runs on A100 (subagent). Prep chain done except the drift floor (jobs 257594/257595). Code review subagent running (report due `audit/2026-10-04_M1_code_review.md`).
  - 2026-10-04 ~17:00: store built (jobs 257377/257378); all S52 counts match. After S9 the train tree has 366,439 species, 18,322 held out, and 11 kingdoms / 77 phyla / 281 classes (line 181's 12 / 79 / 283 are pre-S9 counts; no rule uses them).
  - 2026-10-04 ~15:05 to ~16:00: repo moved to `/u/liv/repos/taxa_maze` (symlink at the old path); the S9 drop set rebuilt to 8,084 keys (73,679 train, 3,929 val) under the "real epithet" rule; Amendment 1 appended (spec sha256 `ab593020…f33eb`); B2 and B3 resolved. Details in `agent_legacy/active/2026-10-04_ask-A7_spec-decisions-and-repo-move.md` and `…_asks-A4-A6_blocks-B2-B3.md`.
  - 2026-10-02: intake and preflight; the user set the freeze baseline to `90a7d38` (A1 revision); S9 evidence finished (A6; BioCLIP 1 trained on the pseudo-species rows). Details in `agent_legacy/active/2026-10-04_asks-A4-A6_blocks-B2-B3.md`.

## Blocked
None (B2 and B3 resolved 2026-10-04; see the prune log).

## Parking lot
- (A3, paper 1) Park et al., arXiv 2406.01506, hierarchical orthogonality in LLM representations. Question: which whitener should stage 2 use (within, between, or total), and does their theorem predict our step-perpendicular-to-position penalty exactly, or only under their estimator? Becomes load-bearing when W stops being I.
- (A3, paper 2) arXiv 2602.11448, the frozen-CLIP port of paper 1 via hierarchical sparse coding with SAEs. Question: how does our claim differ (a training-time regularizer on image prototypes versus post-hoc coding of frozen CLIP)? Do they measure step-versus-position orthogonality on BioCLIP-like models, and with what numbers? Load-bearing for novelty and for the "The problem" section of DESIGN.md.
- (A3, paper 3) Sastry 2027, TaxaWalk (lab preprint; unpublished, so keep it local). Read the user's Markdown conversion, `agent/litreview/Sastry2027-TaxaWalk.md` (2026-10-02). The PDF is next to it. Question: what does it contribute that pilot 1 should cite, compare against, or reuse?
- (A5) Flat decoding. Pilot 1 regularizes the geometry, but it still classifies by a flat nearest-neighbor lookup over all labels at a rank. TaxaWalk (sections 1-2) argues for moving from rank to rank, using the possibly well-structured taxonomic text space of BioCLIP 1/2. Limitation of pilot 1, and a direction after it.
- (A5) Image-side constraints only. Pilot 1 imposes the geometric constraints on image prototypes only. The text side could be constrained too, either separately or jointly with the image side.
- (A5) "TaxaWalk++". A future framework could fold parts of pilot 1's design into TaxaWalk's predictive representation-learning framework.
- (A9) Top-down evaluation. Classify from kingdom down to species, with the candidates at rank t restricted to taxa that share the ranks already predicted. The user expects this to match BFL's "top-down constrained inference"; verify against the BFL paper before relying on that. It would be reported next to the per-rank protocol of spec section 7, not instead of it.
- (A9) Rank-balanced penalty. Replace the plain sum over (species, ancestor) terms in each species' penalty with LogSumExp (Li et al., 2020), so each term's gradient is weighted by a softmax over the term values. TaxaWalk uses this for its next-step MSE loss, a different objective. Question: does it balance the six ranks better than the uniform weighting of spec line 116, and how does it change λ calibration?

## Prune log
| date | block | verdict | destination |
|---|---|---|---|
| 2026-10-04 | A4, A5, A6 (asks) | DONE; no live direction references them | `agent_legacy/active/2026-10-04_asks-A4-A6_blocks-B2-B3.md` |
| 2026-10-04 | B2 (spec decisions) | RESOLVED: Amendment 1 appended | same file |
| 2026-10-04 | A7 (spec decisions, repo move) | DONE: Amendment 1 appended; repo on /u/liv | `agent_legacy/active/2026-10-04_ask-A7_spec-decisions-and-repo-move.md` |
| 2026-10-04 | B3 (file quota) | RESOLVED for pilot 1: outputs and repo on /u/liv; tol_embed stays (read-only) | same file |
| 2026-10-04 | A9 (parking-lot ideas) | DONE: both ideas are Parking-lot lines | `agent_legacy/active/2026-10-04_ask-A9_parking-lot-ideas.md` |
| 2026-10-04 | A2 (loggers, run naming) | DONE: wandb for official runs, csv for smoke runs, naming in USAGE "Run naming"; the first wandb run comes with M2 | `agent_legacy/active/2026-10-04_ask-A2_loggers-and-run-naming.md` |
| 2026-10-05 | A11 (first P3/P4 answers) | DONE: decided by A12 (P2, P3 approved; P4 held); findings fed Amendment 2 | `agent_legacy/active/2026-10-05_ask-A11_P3-P4-responses.md` |
