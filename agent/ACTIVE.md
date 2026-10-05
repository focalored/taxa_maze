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
- Paper 4 (RCME) is DONE (2026-10-02): `agent/litreview/bioclip1-finetune-recipes.md`, verified against the paper and code (`audit/2026-10-02_rcme_recipe_verification.md`; 20 recipe rows: 17 confirmed, 3 contradicted on one clause each, plus one wording fix; all fixed in place).
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
- "262320" is the end-to-end smoke run. Its history, corrected by the O9 review: 262320 started at 00:45 and was cancelled by me at 00:50 after 31 steps (the P3 hold); its resubmission 263082 was not cancelled as I first recorded — it COMPLETED at 02:11 on the pre-commit code at 2 × 2 (60 steps, refresh 2,467 s, ToL-val 83 s, `epoch_01.ckpt` and `last.ckpt` under the old 1-based naming; `logs/smoke/timing/2026-10-05_01-22-57_j263082/`), and my `scancel 263082` at 13:56 acted on a finished job. 264728 repeats the path on the current code at 1 × 2 alongside the sweep; its memory and time numbers are a by-product that nothing waits for.
- The schedule: `_lr(t)` uses `model.epochs × 647 = 6,470` total steps and warmup 65, and `on_fit_start` refuses `model.epochs != 10`; `trainer.max_epochs=3` only stops the sweep run after `epoch_02.ckpt`. At step 1,941 the LR is 0.80 × base; at the last step, 6,469, it is about 6e-8 × base. Checked numerically on 2026-10-05.
- O9 is done by a read-only Opus review of `agent/` after this turn's edits, then my fixes, in a second commit.

## Directions

### D1: Pilot 1, hierarchical orthogonality penalty on BioCLIP 1
- Asks served: A1.
- Hypothesis (spec section 1): adding the cross-level orthogonality penalty to BioCLIP 1 fine-tuning makes zero-shot classification on iNat21-val competitive across all seven ranks, against BioCLIP 1, BioCLIP 1 + RCME, and BioCLIP 1 + BFL.
- Success bar (quoted from spec section 1): "competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons."
- Failure mode (quoted from spec section 7): "Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification."
- Who decides: the user gives the verdict. I report numbers and interpretations only.
- True now (2026-10-05 ~14:15): M1 is DONE. The checker (`audit/2026-10-05_M1_checker.md`: 0 blockers, 1 should-fix, 3 notes; all 9 smoke-test claims verified; 0 spec deviations; freeze PREFIX_OK), my response (`audit/2026-10-05_M1_checker_response.md`) and the user's decisions (A12) close it. Amendment 2 (M1) is appended to the spec. Code since the checker: fp64 gradient cosine with autocast off (P2), `model.train()` at fit start (O6), 0-based checkpoint names and `decisions.json` keys with `init` for the untouched model (O7), 1 node × 2 GPUs at 4,096 images per GPU (A2.1). M2 started 2026-10-05 14:03: sweep runs 264726 (1e-4, rails10, running since 14:08) and 264727 (3e-4, rails14, since 14:10), end-to-end smoke run 264728 (rails07, since 14:16), all on `gpu_a100`. The epoch-end path (refresh, ToL-val, checkpoint) already completed once on the pre-commit code at 2 × 2 (job 263082, 02:11); 264728 repeats it on the current code. No M2 epoch has ended yet.
  - Spec: `specs/pilot1.md` with Amendments 1 and 2 at the bottom (47,559 bytes, sha256 `5315446f…`), which govern on any conflict, the later one first. The S9 list is `specs/pilot1_s9_missing_species_keys.txt`. Do not edit the preflight report.
  - Paths: the repo is `/u/liv/repos/taxa_maze` (`~/bdbk/repos/taxa_maze` is a symlink to it). Data, the BioCLIP 1 checkpoint and tol_embed stay on `/projects`, read only. Every write goes to `/u/liv`, never under `/u/liv/bdbk` (S1). The store is `/u/liv/data/taxa_maze/store/` and the prep outputs are in `/u/liv/data/taxa_maze/p1/` (USAGE "Data").
  - Code map: DESIGN.md "Idea → code". How to run each test: USAGE "Quickstart (pilot 1)".
  - M1 evidence (archived 2026-10-05): every smoke test, the eval gate, the drift floor and the timing run, with numbers and job ids, is in `agent_legacy/active/2026-10-05_D1_M1-smoke-evidence.md`; the checker's verification is `audit/2026-10-05_M1_checker.md`.
  - Git: the M1 code and docs are committed as `7c1832f` and pushed to `origin/main` (2026-10-05 14:03); `.gitignore` no longer ignores `scripts/`. `agent/` (except the six docs tracked since the template), `audit/`, `logs/` and `data/` stay untracked, so every path a result depends on is written down here.
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
  - Timing facts kept for M3 planning (no projection is owed, A2.2). At 2 × 2: 3.1-3.5 s per ordinary step and 6.1 s per monitor step (the S13 runs; half their steps were monitor steps, so their 6.1 s median overstated the ordinary step), 3.06 s per ordinary step, 82 s per ToL-val pass, 2,467 s (41 min) per refresh and 24.8 GiB peak in the completed run 263082; the 349 s eval came from the stalled run 257654. At 1 × 2 (264726, first 60 steps): 6.0 s per ordinary step, 10.5 s per monitor step, 47.2 GiB peak. One fp16 floor pass over a quarter of the train images took 845 s on one GPU alone on a node and did not finish in 2 h with two parts sharing a node, so the loader, not the GPU, bounds the refresh.
  - M3 (decided, A2.4): the iNat21 eval records its encode setting and re-evaluates all six baselines under it; the floor and the S31 cells are same-setting counts.
- Artifacts: M2 runs write under `logs/p1/<run_name>/` (Hydra run dir with `steps.jsonl`, `decisions.json`, `run_info.json`; `checkpoints/epoch_NN.ckpt` and `last.ckpt`) and to wandb project `multimodal_lab/taxa_maze`. Everything through M1 (audits, preflight and S9 reports, smoke outputs, store and prep paths) is listed in `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md`.
- Log:
  - 2026-10-05 ~14:45: `scripts/` pruned at the user's request, keeping what is reusable: 52 one-off preflight helpers moved to `agent_legacy/scripts/preflight/` (verdict README there; all also in git at `7c1832f`). Kept: `env.sh`, `p1/{train.sh,prepare.py,prepare.sh}`, `store/build_store.*`, `smoke/{p1_smoke.*,s13_compare.py,eval_gate.*}` (regression tests and the harness gate), `preflight/{s9_final_keys.py,s9_epithet_census.py,eval_anova_tol.*}`. The deletions are on disk, not yet committed.
  - 2026-10-05 ~14:30: O9 review in (`audit/2026-10-05_O9_doc_review.md`: 21 findings, 3 wrong, 5 stale, 13 minor), all fixed. The three that mattered: (1) job 263082 was never cancelled, it COMPLETED at 02:11 and ran the whole epoch-end path on the pre-commit code at 2 × 2 (refresh 2,467 s, ToL-val 83 s, `epoch_01.ckpt`); my `scancel` at 13:56 hit a finished job and I misreported it; records corrected here, in USAGE and in the archived evidence file. (2) `ckpt_path=` was not a defined key; `configs/train.yaml` now sets `ckpt_path: null`, so the documented resume command composes. (3) USAGE's 4-GPU `grad_1v4` recipe asked for 4 GPUs per node; now `--nodes=2 --ntasks-per-node=2 --gres=gpu:2`. Also: timing facts restated from measured runs (3.1-3.5 s per ordinary step at 2 × 2; 6.0 s and 10.5 s per ordinary and monitor step at 1 × 2, 47.2 GiB peak); B3 pointers; drift-floor job ids; two code comments and the smoke_timing header. Committed as the second commit on `origin/main` (hash in `git log`).
  - 2026-10-05 ~14:15: A12 recorded: the M1 go and the decisions on P2, P3, P4, O1-O4, O6, O7 and O9. Amendment 2 (M1) appended to the spec (47,559 bytes, sha256 `5315446f…`; byte-prefix test PREFIX_OK against 90a7d38 and against the pre-append file). Code: fp64 gradient cosine with autocast off; `model.train()` at fit start; checkpoints `epoch_{epoch:02d}` and `decisions.json` keys 0-based with `init`; `configs/trainer/p1_ddp.yaml` 1 node × 2 GPUs and `train.sh --nodes=1`; `.gitignore` no longer ignores `scripts/`. Configs compose for both sweep runs and the smoke run (1 × 2, `model.epochs` 10; `max_epochs` 3 for the sweep, 1 with 60 batches for the smoke run). A11 pruned. Committed as `7c1832f` (130 files, 2.9 MB) and pushed to `origin/main` at 14:03; then M2 launched at 14:03: sweep (c) at 1e-4 = job 264726, at 3e-4 = job 264727 (`trainer.max_epochs=3`, 1 × 2 A100s, excluding rails11-12), end-to-end smoke run = job 264728. O9 doc review launched afterwards; its fixes go into a second commit.
  - Earlier entries, 2026-10-02 intake through 2026-10-05 ~00:55 (the M1 checker), are in `agent_legacy/active/2026-10-05_D1_log-through-M1.md`.

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
| 2026-10-04 | B3 (file quota) | RESOLVED for pilot 1: outputs and repo on /u/liv; tol_embed stays (read-only) | `agent_legacy/active/2026-10-04_asks-A4-A6_blocks-B2-B3.md` |
| 2026-10-04 | A9 (parking-lot ideas) | DONE: both ideas are Parking-lot lines | `agent_legacy/active/2026-10-04_ask-A9_parking-lot-ideas.md` |
| 2026-10-04 | A2 (loggers, run naming) | DONE: wandb for official runs, csv for smoke runs, naming in USAGE "Run naming"; the first wandb run comes with M2 | `agent_legacy/active/2026-10-04_ask-A2_loggers-and-run-naming.md` |
| 2026-10-05 | A11 (first P3/P4 answers) | DONE: decided by A12 (P2, P3 approved; P4 held); findings fed Amendment 2 | `agent_legacy/active/2026-10-05_ask-A11_P3-P4-responses.md` |
| 2026-10-05 | D1 M1 evidence (smoke tests, eval gate, drift floor, timing run) | DONE: M1 closed by the checker and A12 | `agent_legacy/active/2026-10-05_D1_M1-smoke-evidence.md` |
| 2026-10-05 | D1 artifacts through M1 | DONE: listed and verified by the M1 checker; M2 writes elsewhere | `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md` |
| 2026-10-05 | D1 log, intake through the M1 checker | DONE: outcomes summarized in D1 "True now" | `agent_legacy/active/2026-10-05_D1_log-through-M1.md` |
| 2026-10-05 | `scripts/preflight/` one-off helpers (52 files) | DONE: preflight closed by Amendment 1; superseded by `src/` and `scripts/smoke/` | `agent_legacy/scripts/preflight/` (README: `agent_legacy/scripts/README.md`) |
