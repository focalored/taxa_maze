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
- "262320" is the end-to-end smoke run. Its history, corrected by the O9 review: 262320 started at 00:45 and was cancelled by me at 00:50 after 31 steps (the P3 hold); its resubmission 263082 was not cancelled as I first recorded — it COMPLETED at 02:11 on the pre-commit code at 2 × 2 (60 steps, refresh 2,467 s, ToL-val 83 s, `epoch_01.ckpt` and `last.ckpt` under the old 1-based naming; `logs/smoke/timing/2026-10-05_01-22-57_j263082/`), and my `scancel 263082` at 13:56 acted on a finished job. 264728 repeats the path on the current code at 1 × 2 alongside the sweep; its memory and time numbers are a by-product that nothing waits for.
- The schedule: `_lr(t)` uses `model.epochs × 647 = 6,470` total steps and warmup 65, and `on_fit_start` refuses `model.epochs != 10`; `trainer.max_epochs=3` only stops the sweep run after `epoch_02.ckpt`. At step 1,941 the LR is 0.80 × base; at the last step, 6,469, it is about 6e-8 × base. Checked numerically on 2026-10-05.
- O9 is done by a read-only Opus review of `agent/` after this turn's edits, then my fixes, in a second commit.

### A13 (2026-10-05) Who writes where: agent docs and lit review. Status: ACTIVE (standing rule)
Verbatim ask (2026-10-05, about 15:20-15:26, four messages in this order):
> for the litreview subagents and yourself, should only write to agent/litreview/. read-only other files in taxa_maze. also, i see that subagents are doing litreview. what are you doing in-line?
>
> only you can write to docs in agent/. subagents write to agent/litreview/ only and report outputs that you validate and write as a verifier and orchestrator.
>
> log these instructions to ACTIVE.md
>
> or to litreview/README.md under the MODIFY line. your call.

My interpretation (not the user's words):
- The second message refines the first for me: I am the only writer of the docs in `agent/` (ACTIVE, DESIGN, USAGE, SOTA, the READMEs, the litreview index). No subagent edits them, including code and checker subagents.
- Lit-review subagents write only under `agent/litreview/`: their topic file, `<slug>.notes.md`, and the sources they cite under `agent/litreview/sources/<slug>/`. Everything else in taxa_maze is read-only to them; scratch goes to `/tmp`. They reply with the paths and a short outcome.
- Other subagents keep the README rule: full output to `audit/`, a pointer as the reply, never an `agent/` doc.
- I am verifier and orchestrator: I check each subagent's files with quick scripted checks (format against `agent/litreview/README.md`, quotes and numbers against the saved sources, claims about our repo against the spec and the code), then write the index row and any `agent/` doc change myself.
- Logged here (the verbatim ask) and, as two operational lines, below the MODIFY line of `agent/litreview/README.md`, the file every lit-review subagent reads first.
- What prompted it: my briefs of about 15:05 told two lit-review subagents to save sources under `audit/litreview/`; they wrote 228 files (81 MB) there before the rule reached them at 15:25. They now copy only the sources they cite into `agent/litreview/sources/<slug>/`. Earlier, my P4 brief (00:45) had told a code subagent to edit USAGE.md; under A13 that edit is mine. Whether to delete `audit/litreview/` is the user's call; it holds two evidence scripts (below), so any deletion must spare them.

Verbatim addition (2026-10-05, about 15:28):
> are the outputs im seeing in-line coming from subagents or you? also, keep subagents' scripts in tmp/ alive after the litreview is over. do not delete or migrate them unless upon request, as they are the only source of the evidence provided by the litreview.

My interpretation (not the user's words):
- Every script a lit-review subagent writes, and its output, stays where it was written after the review: nobody deletes, moves or copies it unless the user asks. Today that means the session scratchpad `/tmp/claude-96290/-u-liv/d0251fdc-1166-43e7-9f3f-e7a957853073/scratchpad/` (`hcep_review/`, `tw/`, and whatever the three subagents add) and, written before the rule, `audit/litreview/frozen-clip-hierarchical-geometry-novelty/{digitize_ortho_hist.py,toy_null_hcep_orthogonality.py}` with their outputs. Each notes file names every script by absolute path next to the number it produced. Relayed to the three subagents at 15:29.
- Risk, not acted on: `/tmp` is local to the login node `railsl2`, and its systemd-tmpfiles rule (`q /tmp 1777 root root 10d`) deletes files left untouched for 10 days. Copying the scripts somewhere durable is the user's call.

## Directions

### D1: Pilot 1, hierarchical orthogonality penalty on BioCLIP 1
- Asks served: A1.
- Hypothesis (spec section 1): adding the cross-level orthogonality penalty to BioCLIP 1 fine-tuning makes zero-shot classification on iNat21-val competitive across all seven ranks, against BioCLIP 1, BioCLIP 1 + RCME, and BioCLIP 1 + BFL.
- Success bar (quoted from spec section 1): "competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons."
- Failure mode (quoted from spec section 7): "Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification."
- Who decides: the user gives the verdict. I report numbers and interpretations only.
- True now (2026-10-05 ~14:15): M1 is DONE. The checker (`audit/2026-10-05_M1_checker.md`: 0 blockers, 1 should-fix, 3 notes; all 9 smoke-test claims verified; 0 spec deviations; freeze PREFIX_OK), my response (`audit/2026-10-05_M1_checker_response.md`) and the user's decisions (A12) close it. Amendment 2 (M1) is appended to the spec. Code since the checker: fp64 gradient cosine with autocast off (P2), `model.train()` at fit start (O6), 0-based checkpoint names and `decisions.json` keys with `init` for the untouched model (O7), 1 node × 2 GPUs at 4,096 images per GPU (A2.1). M2 started 2026-10-05 14:03 on `gpu_a100`, 1 × 2 each: sweep 264726 (1e-4, rails10) at step 341 of 647 at 15:09 with loss/con 12.39 → 4.75; sweep 264727 (3e-4, rails14) at step 257, 12.39 → 5.11; both peak at 47.4 GiB. Smoke run 264728 (rails07) finished its 60 steps at 6.0 s each and has been in its refresh since about 14:27. That refresh is loader-bound at 1 × 2 (log, ~16:04), and its end time is unknown. The epoch-end path already completed once on the pre-commit code at 2 × 2 (job 263082, 02:11); 264728 repeats it on the current code. No M2 epoch has ended yet (first ends expected 16:10-16:30).
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
  6. Refresh time at 1 × 2: the user chose (c), "report refresh time from 264728 when ready". 264728 entered its refresh at about 14:27; report `time/refresh_s` from its `csv/version_0/metrics.csv` when it lands, then decide on the loader.
  - Timing facts kept for M3 planning (no projection is owed, A2.2). At 2 × 2: 3.1-3.5 s per ordinary step and 6.1 s per monitor step (the S13 runs; half their steps were monitor steps, so their 6.1 s median overstated the ordinary step), 3.06 s per ordinary step, 82 s per ToL-val pass, 2,467 s (41 min) per refresh and 24.8 GiB peak in the completed run 263082; the 349 s eval came from the stalled run 257654. At 1 × 2 (264726, first 60 steps): 6.0 s per ordinary step, 10.5 s per monitor step, 47.2 GiB peak. One fp16 floor pass over a quarter of the train images took 845 s on one GPU alone on a node and did not finish in 2 h with two parts sharing a node, so the loader, not the GPU, bounds the refresh.
  - M3 (decided, A2.4): the iNat21 eval records its encode setting and re-evaluates all six baselines under it; the floor and the S31 cells are same-setting counts.
  - M3 reading caveat (TaxaWalk review, `agent/litreview/taxawalk-and-pilot1.md`; the preprint's §4.1 says per-rank scoring is "unfavourable" to models trained on full lineages): at coarse ranks the per-rank protocol may favour the prefix-trained arms over BioCLIP 1 and RCME for reasons unrelated to geometry. The controlled comparisons, (d) vs (c) and (b) vs (a), are unaffected.
- Artifacts: M2 runs write under `logs/p1/<run_name>/` (Hydra run dir with `steps.jsonl`, `decisions.json`, `run_info.json`; `checkpoints/epoch_NN.ckpt` and `last.ckpt`) and to wandb project `multimodal_lab/taxa_maze`. Everything through M1 (audits, preflight and S9 reports, smoke outputs, store and prep paths) is listed in `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md`.
- Log:
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
- (A3, paper 1) Reviewed 2026-10-05: `agent/litreview/whitener-for-hierarchical-orthogonality.md` (Park et al., arXiv 2406.01506, ICLR 2025). Σ_B, the species-uniform covariance of the bank's prototypes, is the analogue of Park's whitener: spec line 138 is right about the matrix, but its "Fisher/LDA" label for Σ_W is not Park's LDA. With W = I the theorem predicts nothing about our penalty; with Σ_B it predicts zero only in an ideal world that the step-0 bank is far from, so the penalty is our choice, not Park's theorem used as a loss. Stage-2 suggestion: try Σ_B first (all species, root-centred, fp64, recomputed at each refresh; 23.5% of its trace is sampling noise at step 0, treatment open), recalibrate λ for each W, and report each rank against a shuffled-taxonomy null through the same W. Measured at step 0 with W = I: the null is about 0.0086, not 1/512; kingdom to class sit at it (1.0-1.25×), order and family at 2-2.4×. Becomes load-bearing when W stops being I.
- (A3, paper 2) Reviewed 2026-10-05: `agent/litreview/frozen-clip-hierarchical-geometry-novelty.md`. arXiv 2602.11448 (HCEP, CVPR 2026) decodes a frozen OpenAI CLIP ViT-L/14 by hierarchical sparse coding (no SAE) and plots the same step-versus-ancestor cosine on class-mean image prototypes (ImageNet, CIFAR-100; no BioCLIP, no biological taxonomy; mean cos² about 0.009 and 0.006, digitized). A meaningless random tree reproduces both features of that plot, so it shows no orthogonality beyond chance. Our novelty has to be the training and its effect on per-rank zero-shot accuracy, measured against a null that keeps the prototype construction. Open for DESIGN.md "The problem" (not yet written): HCEP's caption says the condition "holds even in pre-trained models", the published counter-claim to the spec's "geometrically unconstrained" premise.
- (A3, paper 3) Reviewed 2026-10-05, local files only: `agent/litreview/taxawalk-and-pilot1.md` (TaxaWalk, Sastry 2027, unpublished lab preprint; keep it local). Cite, as an internal reference until it is public: its flat-decoding limitation, top-down pruning lowering BioCLIP-2's species accuracy from 77.87 to 74.00 (RnR), and its remark that per-rank scoring is unfavourable to full-lineage models (D1, M3 reading caveat). Compare: nothing; it has no per-rank iNat21 numbers for BioCLIP 1, RCME or BFL, and it changes the decoder, not the representation. The fair test is frozen BioCLIP 1, arm (c) and arm (d), each crossed with a flat, a top-down and a TaxaWalk decoder.
- (A5) Flat decoding. Pilot 1 regularizes the geometry, but it still classifies by a flat nearest-neighbor lookup over all labels at a rank. TaxaWalk (sections 1-2) argues for moving from rank to rank, using the possibly well-structured taxonomic text space of BioCLIP 1/2. Limitation of pilot 1, and a direction after it.
- (A5) Image-side constraints only. Pilot 1 imposes the geometric constraints on image prototypes only. The text side could be constrained too, either separately or jointly with the image side.
- (A5) "TaxaWalk++". A future framework could fold parts of pilot 1's design into TaxaWalk's predictive representation-learning framework. Cheapest concrete form (TaxaWalk review): the decoder crossing in the paper-3 line above.
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
