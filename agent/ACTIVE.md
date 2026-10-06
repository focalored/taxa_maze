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

### A15 (2026-10-06) Diagnose and fix the (c) recipe before M3. Status: ACTIVE
Verbatim ask (2026-10-06, after the A14 control landed):
> can think of more things to try:
> * alpha = r (instead of 2r)
> * level-restricted contrastive loss without the unique label + group of positive images setup of BFL (although this is expected to perform worse, given BFL introduces it as a design feature)
> * evaluate all runs so far on inat21 val (to get a sense of how these runs are performing compared to baeslines), instead of only looking at ToL-val performance
> * bigger batch size for more contrastive signal, especially if we use the custom grouped sampler
> * lineage f"{lineage}" template instead of f"a photo of {lineage}.": motivated by the fact that val/tol/lineage curves deteriorate significantly
> * image-side LoRA only -> contrastive + penalty move the images only while preserving the text embedding space. i'm betting on this the most, because i believe there is hierarchical structure encoded in the text tower.
> * switch equal level weights for LogSumExp (used in TaxaWalk for their MSE loss), which balances each level by its gradient. but this should be deferred until we fix the equal weighting version, because BFL achieves good training outcomes with equal weighting while we currently do not.
>
> my expectation before we move on to M3 is to achieve matched or better performance than BioCLIP 1 on inat21 after 10 epochs of LoRA fine-tuning on the level-restricted contrastive loss. this may violate the spec's decision not to look at eval until M3 (we've landed on 1e-4 for now, and are currently trying to diagnose/fix the apparently bad training outcome of the (c) arm), and you may push back. however, matched or better than BioCLIP 1 on inat21 after the (c) arm is still lax compared to BFL, which significantly surpasses BioCLIP 1 on their harness after 30 epochs of full fine-tuning on a similar dataset (full ToL-10M rather than our ToL-EOL) with the level-restricted contrastive loss (assuming we implemented it correctly). nonetheless i think matched or surpassed BioCLIP 1 on our (c) arm would be enough to go to M3 and test the penalty. i wouldn't be surprised if further LoRA fine-tuning of the backbone pretrained on the same dataset would move it far even with a slightly "better" contrastive objective (ie level restricted rather than flat).
>
> update docs in agent/ to reflect my hypothesis, assumptions, planned experiments, expectations, and the current state of M2.

Verbatim addition (2026-10-06): "add 266465 interpretations to agent/ docs where relevant, synthesizing with the existing documentation of the previous two runs and keeping things concise. my quick interpretation is that the random sampler helped it achieve slightly better performance than the grouped sampler as expected, but at the species level (photo template) the performance still dropped on ToL-val (while climbing at other ranks)."

Verbatim addition (2026-10-06, 14:44): "another possible test is using a whitener. again, every one of these knobs need to be turned one at a time on the grouped sampler."

My interpretation (not the user's words): the hypothesis, assumptions, gate and the planned probes are under D1 "Working hypothesis and plan for the (c) recipe". Two of the items change what the spec fixes before M3: iNat21 evaluations before M3 (S27, line 172: one iNat21 eval per arm) and any recipe change adopted into the arms (lines 51, 61, 96-97, 154, S8). Both need an amendment the user decides on; a draft is under D1. Nothing is submitted or appended until then.

## Directions

### D1: Pilot 1, hierarchical orthogonality penalty on BioCLIP 1
- Asks served: A1, A15 (A14 done, pruned).
- Hypothesis (spec section 1): adding the cross-level orthogonality penalty to BioCLIP 1 fine-tuning makes zero-shot classification on iNat21-val competitive across all seven ranks, against BioCLIP 1, BioCLIP 1 + RCME, and BioCLIP 1 + BFL.
- Success bar (quoted from spec section 1): "competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons."
- Failure mode (quoted from spec section 7): "Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification."
- Working hypothesis and plan for the (c) recipe (user, 2026-10-06, A15; the spec's hypothesis above is unchanged):
  - Bet: BioCLIP 1's text tower already encodes the hierarchy; moving it (text-side LoRA) costs species names first. Candidate fix: image-side LoRA only, so the contrastive loss and the penalty move images against a fixed text space.
  - Assumptions, with what is known. (1) The level-restricted loss is BFL's: unverified against the paper (BFL released no training code; `src/models/losses.py` is our construction from its equations; a hand-computed tiny batch, about an hour, would check it). (2) "BFL significantly surpasses BioCLIP 1" holds on BFL's own harness (reported 63.00 vs 50.79 at species); on ours the released BFL-Euc is below BioCLIP 1 at species (68.78 vs 70.19) and far above at the other ranks (spec section 1 table). (3) The grouped sampler is load-bearing for the penalty (line 145): under random batches nearly every group is one image, so a sampler change that helps (c) does not carry to (d) as is; K = 8 (line 142's other value, fixed to 16 by S2) keeps groups with about 1,780 species per batch.
  - Expectation and gate before M3 (user): arm (c) after 10 epochs matches or beats BioCLIP 1 on iNat21. Still to pin: the rank(s) and the bar (S31's ±1 point on integer counts). At species the bar is stricter than what the released BFL achieves in our harness; at every other rank BioCLIP 1 is far behind BFL-Euc.
  - Planned probes. Rule (user): one knob per probe, every probe on the grouped sampler, each read against the grouped lr 1e-4 run 264726 (111,727 → 97,244 → 85,586 species-correct) and the untouched model (108,062). 3 epochs at 1 × 2 A100s on the 10-epoch schedule; the user orders them, main bet first. Each line: what changes, the spec line it touches (an amendment if adopted into the arms), code, cost.
    1. Image-side LoRA only (main bet): lines 61, 154; flag in `src/models/bioclip_lora.py:add_qv_lora` plus the text forward without gradient, about 30 lines; about 4.5 h (no text gradients).
    2. alpha = r = 16: lines 61, 154; no code, `model.lora_alpha=16`; about 5 h.
    3. Training text `{lineage}` without the wrapper: line 51(a); flag at `src/models/p1_module.py:156`; about 5 h. The verdict's eval template stays "a photo of" (lines 171, 174).
    4. B = 16,384: S8; no code, 2 nodes × 2 GPUs at 4,096 per GPU; about 3.5 h on 4 GPUs plus a longer queue.
    5. Level loss without dedup and group positives: lines 96-97; flag in `src/models/losses.py`; about 5 h; the user expects it worse.
    6. iNat21 evals of the nine sweep and control checkpoints plus init: S27 and line 172 allow one iNat21 eval per arm, so this needs Amendment 3 (draft below) or a development set instead (iNat21 train-mini, packed at `~/bdbk/data/inat21/train_mini.tar.gz`; extraction plus a harness pointer, a few hours of setup); the eval script exists; about 5 min per checkpoint, under 1 h for all.
    7. LogSumExp level weights (line 178): deferred by the user until the equal-weight recipe trains.
    8. Whitener in the penalty: spec lines 124-139 plan W = Σ^{-1/2} as a second stage, with W = I now (line 133); the lit review reads Σ_B as Park's whitener (`agent/litreview/whitener-for-hierarchical-orthogonality.md`). It acts inside the penalty's cos², so it changes arms (b) and (d) only and does not bear on (c)'s decline; it is a probe for once (c) is settled, with λ from the chosen (c) run. Code: W from the bank's Σ_B (512 × 512, recomputed at each refresh) applied to steps and positions in `src/models/losses.py:penalty_local` and the bank monitor; about 40 lines; cost one (d)-style 3-epoch probe, about 5 h.
  - Draft Amendment 3 (M2), not appended; the user decides the wording and I append it: A3.1 Diagnostic iNat21 evaluations during M2 inform the recipe revision only; checkpoint selection stays on deduplicated ToL-val (S27) and the LR decision on line 161; the M3 verdict evaluations remain one per arm, on the S27 checkpoint of each arm's final recipe, and the M3 report lists every diagnostic evaluation made before them. A3.2 Recipe revision of arm (c): the adopted changes with their lines and evidence. A3.3 Gate before M3: arm (c) at 10 epochs must match BioCLIP 1 within S31's bar at the pinned rank(s) on iNat21 under A2.4's setting. Reason: the sweep's (c) runs lose ToL-val species accuracy; the pilot's question needs a (c) arm that trains. Cost to state: recipe choices made with iNat21 in view make the verdict's baseline comparison optimistic by the amount of looking; few looks and recipe-level decisions keep it small, and the listing keeps it honest.
- Who decides: the user gives the verdict. I report numbers and interpretations only.
- True now (2026-10-06 ~14:25): M1 is DONE (checker `audit/2026-10-05_M1_checker.md`, response `audit/2026-10-05_M1_checker_response.md`; Amendment 2 appended). M2 is open. Under line 161 the sweep picks 1e-4 (85,586 vs 42,516 species-correct on deduplicated ToL-val after 3 epochs; `audit/2026-10-05_M2_sweep_results.md`), but both grouped runs deteriorate at species after their first epoch (1e-4: 111,727 → 97,244 → 85,586 against 108,062 untouched) while kingdom to family climb: line 174's failure mode in the unpenalized arm (c). The A14 control (random batches, lr 1e-4, job 266465, 00:05-06:38; `audit/2026-10-06_A14_sampler_control.md`) lifts the level by 13,700-21,800 species-correct (125,427 → 116,710 → 107,365) and still declines about 9,000 per epoch, ending 697 images below untouched at species with genus +17 and family to kingdom +42 to +87 points above it; mean7 stays near 72. So batch composition explains part of the loss and something shared by both runs drives the rest (the level balance, the moving text tower, or the learning rate); the user's bet is the text tower (A15). The (c) recipe is under revision (A15): no probe is submitted and nothing is appended to the spec until the user decides on Amendment 3 and the probe order. The M2 checker waits for the revised recipe. No iNat21 eval of any M2 checkpoint has been run.
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
  - **M1 (smoke tests pass and are verified).** Done 2026-10-05: Amendment 1, the store build, the code, the section 9 smoke tests, the checker, the user's go (prune log; `agent_legacy/active/2026-10-05_D1_M1-smoke-evidence.md`).
  - **M2 (the LR sweep has a decision).**
    - Run the two 3-epoch sweep runs on arm (c), at 1e-4 and 3e-4, roughly 1.5 hours each if A100s are free. (At 1 × 2 A100s, Amendment 2, the two ran in parallel on two nodes. Measured: 5 h 05 min per run with the pread reader, 7 h 21 min with the old one; `audit/2026-10-05_M2_sweep_results.md` §4.)
    - Decide on species top-1 on deduplicated ToL-val (line 161, S4).
    - Then the checker, then the user's go.
  - **M3 (all arms finish, no verdict).**
    - The winning (c) run continues to 10 epochs, then (d), (a) and (b) in the spec's order (line 153), each with its own λ (S6).
    - No projected hours (S29 as amended by A2.2): the runs report their own times. Do not stall for approval.
    - Then the checker. The user gives the verdict.
- Next action (M2, the LR sweep):
  1. Done: the sweep (1e-4 under line 161) and the A14 control; both audits are written.
  2. Awaiting the user: (a) Amendment 3, yes or no, and its wording (draft above); (b) the gate's rank(s) and bar; (c) which probes to submit and in what order (1 and 2 can run on two nodes at once; 1 needs its code flag first, with a scripted check that the default path is unchanged); (d) the line 161 gap (a lower learning rate was not swept) stays open; (e) whether M2 closes only once the (c) recipe is settled (my reading of A1's milestone: yes, the checker then runs once).
  3. After those decisions: add the chosen flags, check them, submit the probes with fresh run names (`p1c_lvl_lr1e-4_s42_<tag>_v1`), and report each epoch end against the control's and the grouped run's counts. iNat21 evaluations only after an amendment.
  4. M3 path once (c) is settled: its 10-epoch run resumes from `last.ckpt` or starts fresh under the revised recipe (the user's call); λ for (d) = 0.2 / mean r_t over steps 50-250 of that grouped (c) run (currently 27.787 for 264726; the random-batch control gives 9.911 and is not used); S27 selects its checkpoint on ToL-val.
  5. The end-to-end smoke run is not resubmitted after 264728's timeout (user, 2026-10-05); the sweep and control runs exercised the same path nine times.
  - Timing facts kept for M3 planning (no projection is owed, A2.2). At 2 × 2: 3.1-3.5 s per ordinary step, 6.1 s per monitor step, ToL-val 82 s, refresh 41 min with the old reader (job 263082, at night). At 1 × 2: 5.8-6.0 s per ordinary step (median, the same with either reader), 10.5 s per monitor step, 47.4 GiB peak; refresh 25.0-33.5 min with the pread reader (265500: 2,007.6 / 1,708.5 / 1,497.8 s) against 59.1-73.8 min with the old one (264726: 4,425.6 / 3,737.1 / 3,546.2 s) or stuck (264728, unfinished after 2 h 49 min); ToL-val 144-149 s for both forms. An epoch after the first takes 1 h 32-35 min at 1 × 2 with the pread reader (2 h 12 min with the old one). Reader benchmarks on CPU nodes (16 workers, 65,536 fresh rows per run): pread 1,850-3,592 img/s against memmap 1,626-2,187, with less kernel CPU; DONTNEED leaves 0.6 GiB of page cache per 9.2 GiB read against 9.3 GiB without it, at no throughput cost (`audit/2026-10-05_stall_diag/bench_*.txt`). Random batches (A14 control): 8.5 s per step, 60.1 GiB peak, refresh 28-31 min, epochs 2 h 03-18 min, run 6 h 33 min.
  - M3 (decided, A2.4): the iNat21 eval records its encode setting and re-evaluates all six baselines under it; the floor and the S31 cells are same-setting counts.
  - M3 reading caveat (TaxaWalk review, `agent/litreview/taxawalk-and-pilot1.md`; the preprint's §4.1 says per-rank scoring is "unfavourable" to models trained on full lineages): at coarse ranks the per-rank protocol may favour the prefix-trained arms over BioCLIP 1 and RCME for reasons unrelated to geometry. The controlled comparisons, (d) vs (c) and (b) vs (a), are unaffected.
- Artifacts: M2 runs write under `logs/p1/<run_name>/` (Hydra run dir with `steps.jsonl`, `decisions.json`, `run_info.json`; `checkpoints/epoch_NN.ckpt` and `last.ckpt`) and to wandb project `multimodal_lab/taxa_maze`. Everything through M1 (audits, preflight and S9 reports, smoke outputs, store and prep paths) is listed in `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md`.
- Log:
  - 2026-10-06 ~14:25: A14 control 266465 finished 06:38; results and reading in `audit/2026-10-06_A14_sampler_control.md` (level up, decline persists). User's A15 recorded: hypothesis, assumptions, gate and seven planned probes under D1; Amendment 3 drafted, not appended. A14 marked DONE and pruned; the 2026-10-05 log entries moved to `agent_legacy/active/2026-10-05_D1_log-sweep-first-round.md`.
  - Entries from 2026-10-05 ~17:20 (the reader switch) to ~23:09 (the sweep's first-round results) are in `agent_legacy/active/2026-10-05_D1_log-sweep-first-round.md`.
  - 2026-10-06 ~00:22: the sampler control 266465 started at 00:05 on rails11 (steps from 00:08). First 65 steps: 8.45 s median (12.6 max), 60.1 GiB peak, `loss/con` 12.14 → 5.81, species image-to-text term 0.72 → 0.46; `run_info.json` records `sampler: random`. Docs updated for the first-round sweep results, the sampler flag and the control (ACTIVE.md, DESIGN.md, USAGE.md).
  - Entries from 2026-10-05 ~14:15 (M2 launch) to ~17:00 (the reader switch) are in `agent_legacy/active/2026-10-05_D1_log-M2-launch-to-reader-switch.md`.
  - Earlier entries, 2026-10-02 intake through 2026-10-05 ~00:55 (the M1 checker), are in `agent_legacy/active/2026-10-05_D1_log-through-M1.md`.

## Blocked
None (B2 and B3 resolved 2026-10-04; see the prune log).

## Parking lot
- (user, 2026-10-06) Text-side penalty, no bank: "ditch the image bank and the prototype setup, keep image tower frozen, and apply the penalty to the text side- taxonomic label embeddings truncated at each corresponding rank." Each node's position is then the text embedding of its truncated label, computed fresh for the in-batch nodes, so the EMA bank and the species prototypes are not needed. Related to the A5 line on constraining the text side; this is the text-only form with the image tower frozen.
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
| 2026-10-06 | A14 (sampler flag and the lr 1e-4 random-batch control) | DONE: flag added (`41380d3`); control ran (266465); random batches lift ToL-val species by 13,700-21,800 images, the decline persists | `agent_legacy/active/2026-10-06_ask-A14_sampler-control.md`, `audit/2026-10-06_A14_sampler_control.md` |
| 2026-10-06 | D1 log, the reader switch to the sweep's first-round results | DONE: outcomes in D1 "True now" and `audit/2026-10-05_M2_sweep_results.md` | `agent_legacy/active/2026-10-05_D1_log-sweep-first-round.md` |
