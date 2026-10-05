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

## Directions

### D1: Pilot 1, hierarchical orthogonality penalty on BioCLIP 1
- Asks served: A1.
- Hypothesis (spec section 1): adding the cross-level orthogonality penalty to BioCLIP 1 fine-tuning makes zero-shot classification on iNat21-val competitive across all seven ranks, against BioCLIP 1, BioCLIP 1 + RCME, and BioCLIP 1 + BFL.
- Success bar (quoted from spec section 1): "competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons."
- Failure mode (quoted from spec section 7): "Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification."
- Who decides: the user gives the verdict. I report numbers and interpretations only.
- True now (2026-10-05 ~14:15): M1 is DONE. The checker (`audit/2026-10-05_M1_checker.md`: 0 blockers, 1 should-fix, 3 notes; all 9 smoke-test claims verified; 0 spec deviations; freeze PREFIX_OK), my response (`audit/2026-10-05_M1_checker_response.md`) and the user's decisions (A12, now in `agent_legacy/active/2026-10-05_ask-A12_M1-decisions-amendment-2.md`) close it. Amendment 2 (M1) is appended to the spec. Code since the checker: fp64 gradient cosine with autocast off (P2), `model.train()` at fit start (O6), 0-based checkpoint names and `decisions.json` keys with `init` for the untouched model (O7), 1 node × 2 GPUs at 4,096 images per GPU (A2.1). M2 started 2026-10-05 14:03 on `gpu_a100`, 1 × 2 each. Sweep 264726 (1e-4, rails10) finished epoch 0's 647 steps at 15:48 and is in its refresh: I/O-bound, re-reading evicted pages at its 220 GiB limit, but progressing. Sweep 264727 (3e-4, rails14) has been stalled at step 298 since 15:54, and the smoke run 264728 (rails07) has been stuck in its refresh since about 14:27. In both, every data-loader worker spins one CPU core in the kernel while reading nothing, and the GPUs sit idle (`audit/2026-10-05_lr3e-4_stall.md`; decision pending, Next action 7). The epoch-end path completed once on the pre-commit code at 2 × 2 (job 263082, 02:11). No M2 epoch has ended yet.
  - Spec: `specs/pilot1.md` with Amendments 1 and 2 at the bottom (47,559 bytes, sha256 `5315446f…`), which govern on any conflict, the later one first. The S9 list is `specs/pilot1_s9_missing_species_keys.txt`. Do not edit the preflight report.
  - Paths: the repo is `/u/liv/repos/taxa_maze` (`~/bdbk/repos/taxa_maze` is a symlink to it). Data, the BioCLIP 1 checkpoint and tol_embed stay on `/projects`, read only. Every write goes to `/u/liv`, never under `/u/liv/bdbk` (S1). The store is `/u/liv/data/taxa_maze/store/` and the prep outputs are in `/u/liv/data/taxa_maze/p1/` (USAGE "Data").
  - Code map: DESIGN.md "Idea → code". How to run each test: USAGE "Quickstart (pilot 1)".
  - M1 evidence (archived 2026-10-05): every smoke test, the eval gate, the drift floor and the timing run, with numbers and job ids, is in `agent_legacy/active/2026-10-05_D1_M1-smoke-evidence.md`; the checker's verification is `audit/2026-10-05_M1_checker.md`.
  - Git: the M1 code and docs are committed as `7c1832f` and pushed to `origin/main` (2026-10-05 14:03), then `7ab4cbd` (O9 fixes) and `cbd95ac` (scripts retired); `.gitignore` no longer ignores `scripts/`. `agent/` (except the six docs tracked since the template), `audit/`, `logs/` and `data/` stay untracked, so every path a result depends on is written down here.
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
  5. Bug, found by the paper-1 lit review and verified on 2026-10-05 against job 263082's `metrics.csv` and the code: `L_star/penalized` and `L_star/penalized/cos2_genus` are NaN. `Bank.canonical_penalty` (`src/models/bank.py`) divides by `|step|²·|pos|²` with no floor, and two genera (Cerambycidae: Hexamitodera, Stenandra) each hold two single-image species with identical prototypes, so their genus steps are exactly zero (0/0). The sweep runs will log NaN for these two monitors at every epoch end. No decision uses L*, and the batch penalty floors its denominator at δ (line 112), so it is unaffected. Fix decided by the user on 2026-10-05 at about 16:17 ("Add the floor now and fix the L_star bug"): the same δ floor in `canonical_penalty`, so a zero step counts 0. Applied the same afternoon (log); runs already started keep the old code, so the two sweep runs log NaN for these two monitors until they finish.
  6. Refresh time at 1 × 2: the user chose (c), "report refresh time from 264728 when ready". 264728 entered its refresh at about 14:27; report `time/refresh_s` from its `csv/version_0/metrics.csv` when it lands, then decide on the loader. Update 16:35: 264728's refresh is stuck in the same loader stall as 264727 (item 7), so its time would measure the stall, not a refresh.
  7. Loader stall (`audit/2026-10-05_lr3e-4_stall.md`), decision pending. Options: (1) wait; (2) restart 264727 on the same code; (3) read store rows with `os.pread` instead of per-worker `np.memmap` maps (identical bytes and batches; about 30 lines in `src/data/tol_store.py` plus a 15-min CPU benchmark), then restart the stalled runs; (4) ask the admins for a stuck worker's kernel stack. My recommendation: 3 (with 4 if the mechanism should be confirmed), and cancel 264728, since 264726 is about to cover the same epoch-end path on the current code. Decided 16:54 (user): switch to `os.pread`, benchmark, then restart the stalled run; then "wait" when 264727 moved 298→306. Done: reader switched (`src/data/tol_store.py`, bit-identical bytes, uncommitted); benchmark 1.6-1.7× faster with ~15% less kernel time per image, stall not reproducible offline (audit report §6). 264727 has logged nothing since 16:41; restart awaits the user's go. Decided 17:05 (user): "restart 264727 as v2 with pread plus (a), with watchdog (b), and let 264728 keep running (want to see how long the refresh actually takes)". (a) = `POSIX_FADV_DONTNEED` after each row, so reads leave no page cache; (b) = `audit/2026-10-05_stall_diag/watchdog.sh`, which probes the jobs every 10 min and reports a job that shows the stall signature twice in a row.
  8. Geometry, under consideration by the user: back to 2 × 2 for refreshes under 50 min, despite longer queue waits. Facts at 17:06: no `gpu_a100` node idle (all 18 A100s allocated; 3 of 9 nodes ours; 36 jobs pending, 32 of them one user's dependent arrays), so a 2-node job needs two whole nodes to free at once. With pread the refresh may already fit: reading the store's ~800 GB at the benchmark's ~500 MiB/s per node is about 27 min at 1 × 2 (the GPU floor is about 18 min), about 13 min at 2 × 2. The v2 run's first refresh measures it at 1 × 2, about 70 min after the run starts. A switch back needs an Amendment 3 to A2.1.
  - Timing facts kept for M3 planning (no projection is owed, A2.2). At 2 × 2: 3.1-3.5 s per ordinary step and 6.1 s per monitor step (the S13 runs; half their steps were monitor steps, so their 6.1 s median overstated the ordinary step), 3.06 s per ordinary step, 82 s per ToL-val pass, 2,467 s (41 min) per refresh and 24.8 GiB peak in the completed run 263082; the 349 s eval came from the stalled run 257654. At 1 × 2 (264726, first 60 steps): 6.0 s per ordinary step, 10.5 s per monitor step, 47.2 GiB peak. One fp16 floor pass over a quarter of the train images took 845 s on one GPU alone on a node and did not finish in 2 h with two parts sharing a node, so the loader, not the GPU, bounds the refresh.
  - M3 (decided, A2.4): the iNat21 eval records its encode setting and re-evaluates all six baselines under it; the floor and the S31 cells are same-setting counts.
  - M3 reading caveat (TaxaWalk review, `agent/litreview/taxawalk-and-pilot1.md`; the preprint's §4.1 says per-rank scoring is "unfavourable" to models trained on full lineages): at coarse ranks the per-rank protocol may favour the prefix-trained arms over BioCLIP 1 and RCME for reasons unrelated to geometry. The controlled comparisons, (d) vs (c) and (b) vs (a), are unaffected.
- Artifacts: M2 runs write under `logs/p1/<run_name>/` (Hydra run dir with `steps.jsonl`, `decisions.json`, `run_info.json`; `checkpoints/epoch_NN.ckpt` and `last.ckpt`) and to wandb project `multimodal_lab/taxa_maze`. Everything through M1 (audits, preflight and S9 reports, smoke outputs, store and prep paths) is listed in `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md`.
- Log:
  - 2026-10-05 ~17:00: `ImageStore` switched from per-worker `np.memmap` to `os.preadv` (user decision 16:54); bytes bit-identical on 300 rows across all 63 shards. Benchmark on rails03 (16 workers, 65,536 fresh rows per run): pread 3,297-3,592 img/s against memmap 1,979-2,187, first batch 5-6 s against 8-12 s, ~15% less kernel CPU per image; runs 3-4 most likely under the job's 24 GB limit (the script printed the step cgroup's "max"), and no reader stalled in ~30 s runs, so the benchmark says nothing about preventing the stall. 264727 moved 298→306 at 16:37-16:41, then stopped again; 264726 still refreshing (since 15:48); 264728 still stuck. Restart pending (Next action 7).
  - 2026-10-05 ~16:35: 264727's stall diagnosed (`audit/2026-10-05_lr3e-4_stall.md`; probes and outputs in `audit/2026-10-05_stall_diag/`). It is not read contention, as I said at 16:17: the stuck workers in 264727 and 264728 read nothing and spin one core each in the kernel, with GPUs idle, no OOM and no NCCL error. 264726 is I/O-bound but progressing. Same signature as the rails12 floor parts and the 257654 stalls. Likely cause: every worker memory-maps the whole 850 GB NFS store. Kernel stacks are unreadable for users (`ptrace_scope=2`, `perf_event_paranoid=2`). Decision pending (Next action 7). A3 parking-lot lines pruned at the user's request (all content is in the three topic files).
  - 2026-10-05 ~16:25: L* bug fixed (user decision, Next action 5): `Bank.canonical_penalty` now floors |step|² at δ = 1e-5 (`DEN_FLOOR`, imported from `losses.py`) and skips terms with a zero position, as `penalty_local` does. CPU test on the seed bank: all 14 `L_star/*` values finite (`L_star/penalized` 0.1001, genus 0.0313; the four zero-step genus terms count 0). Runs 264726, 264727 and 264728 keep the old code. Committed with the litreview index rows and these ACTIVE updates (hash in `git log`).
  - 2026-10-05 ~16:17: the user asked why 264727 (3e-4) sits at step 298 while 264726 (1e-4) finished epoch 0 (647 steps at 15:48, now refreshing with GPUs at 82-100%), and why logit_scale is flat at 4.6. Found: 264727 is stalled on data, not compute. Both runs take the same 9.6-9.8 s per monitor step and 6.0 s per ordinary step on 1 × 2 (world 2, rails10 and rails14, nodes not shared); 264727 lost its time to eight stalls of 62-185 s between 15:00 and 15:51 and has logged nothing since 15:54, with GPUs at 0% and idle power, loader workers running, no NCCL error. The long stall began 6 min after 264726's refresh started; the smoke run's refresh (since 14:27) and 264726's each read the whole 800 GB store, so read contention on the store is the likely cause. logit_scale: clamped at ln 100 = 4.605170 (S20), exactly at the ceiling on 356 of 647 and 144 of 299 logged steps; where it falls, Adam moves a scalar by at most about the LR per step, so the drift so far is at most 0.06-0.08 and the minimum seen is 4.596. Decisions: fix the L* bug now; report 264728's refresh time when ready (Next action 5, 6).
  - 2026-10-05 ~16:04: the smoke run 264728 has been in its first refresh for about 96 min (since about 14:27). Both GPUs sit at 0%, and the loader workers each run at about 81% CPU: the loader-bound pattern of the two floor parts that shared rails12. At 2 × 2 (job 263082) the refresh took 41 min. The sweeps reach the same refresh at their first epoch end, about 16:30-17:00. Not stuck; how long it takes at 1 × 2 is still unknown.
  - 2026-10-05 ~16:05: the three A3 lit reviews are in, validated by me under A13 (scripted checks of structure, write scope, key quotes against the saved sources, numbers against the scripts' outputs, repo claims against spec and code; no corrections needed) and indexed in `agent/litreview/README.md`. Outcomes are in the Parking lot (A3 papers 1-3, A5, A9) and D1's M3 reading caveat. A3 pruned as DONE. The paper-1 review found a NaN bug in `Bank.canonical_penalty` (Next action 5). Rules A13 (write scope; evidence scripts stay in place) recorded during the reviews.
  - 2026-10-05 ~15:10: status. Sweep 264726 at step 341 (loss/con 4.75), 264727 at step 257 (5.11); 47.4 GiB peak at 4,096 per GPU; 264727 averages 13.8 s per step against 10.6 s for 264726 on another node, cause not looked at. Smoke run 264728 in its refresh. Scripts retirement committed as `cbd95ac` and pushed.
  - 2026-10-05 ~14:45: `scripts/` pruned at the user's request, keeping what is reusable: 52 one-off preflight helpers moved to `agent_legacy/scripts/preflight/` (verdict README there; all also in git at `7c1832f`). Kept: `env.sh`, `p1/{train.sh,prepare.py,prepare.sh}`, `store/build_store.*`, `smoke/{p1_smoke.*,s13_compare.py,eval_gate.*}` (regression tests and the harness gate), `preflight/{s9_final_keys.py,s9_epithet_census.py,eval_anova_tol.*}`. The deletions are on disk, not yet committed.
  - 2026-10-05 ~14:30: O9 review in (`audit/2026-10-05_O9_doc_review.md`: 21 findings, 3 wrong, 5 stale, 13 minor), all fixed. The three that mattered: (1) job 263082 was never cancelled, it COMPLETED at 02:11 and ran the whole epoch-end path on the pre-commit code at 2 × 2 (refresh 2,467 s, ToL-val 83 s, `epoch_01.ckpt`); my `scancel` at 13:56 hit a finished job and I misreported it; records corrected here, in USAGE and in the archived evidence file. (2) `ckpt_path=` was not a defined key; `configs/train.yaml` now sets `ckpt_path: null`, so the documented resume command composes. (3) USAGE's 4-GPU `grad_1v4` recipe asked for 4 GPUs per node; now `--nodes=2 --ntasks-per-node=2 --gres=gpu:2`. Also: timing facts restated from measured runs (3.1-3.5 s per ordinary step at 2 × 2; 6.0 s and 10.5 s per ordinary and monitor step at 1 × 2, 47.2 GiB peak); B3 pointers; drift-floor job ids; two code comments and the smoke_timing header. Committed as the second commit on `origin/main` (hash in `git log`).
  - 2026-10-05 ~14:15: A12 recorded: the M1 go and the decisions on P2, P3, P4, O1-O4, O6, O7 and O9. Amendment 2 (M1) appended to the spec (47,559 bytes, sha256 `5315446f…`; byte-prefix test PREFIX_OK against 90a7d38 and against the pre-append file). Code: fp64 gradient cosine with autocast off; `model.train()` at fit start; checkpoints `epoch_{epoch:02d}` and `decisions.json` keys 0-based with `init`; `configs/trainer/p1_ddp.yaml` 1 node × 2 GPUs and `train.sh --nodes=1`; `.gitignore` no longer ignores `scripts/`. Configs compose for both sweep runs and the smoke run (1 × 2, `model.epochs` 10; `max_epochs` 3 for the sweep, 1 with 60 batches for the smoke run). A11 pruned. Committed as `7c1832f` (130 files, 2.9 MB) and pushed to `origin/main` at 14:03; then M2 launched at 14:03: sweep (c) at 1e-4 = job 264726, at 3e-4 = job 264727 (`trainer.max_epochs=3`, 1 × 2 A100s, excluding rails11-12), end-to-end smoke run = job 264728. O9 doc review launched afterwards; its fixes go into a second commit.
  - Earlier entries, 2026-10-02 intake through 2026-10-05 ~00:55 (the M1 checker), are in `agent_legacy/active/2026-10-05_D1_log-through-M1.md`.

## Blocked
None (B2 and B3 resolved 2026-10-04; see the prune log).

## Parking lot
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
