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

Verbatim addition (2026-10-07):
> D1 image-side LoRA: the bet about this experiment sits on the idea that moving the text tower even with just the level-restricted contrastive objective costs fine-grained species discrimination, while coarse ranks improve. one probe that should precede / run in parallel is (a) the zero-mean parent test (is each genus close to the average of its species, and so on for every internal node wrt its species-level labels), (b) the penalty terms applied to the text space, with internal nodes being the taxonomic labels truncated at each rank. if the output is zero-mean parents and near-orthogonality satisfied by text labels, then image-side LoRA becomes transferring text-side geometry to the image tower. otherwise training outcome of image-side LoRA is in a vacuum -- if we're keeping the text embedding space frozen as an anchor we better diagnose that anchor.
>
> perhaps the most important next experiment though is: run arm (a) on the same configs and track species accuracy on ToL-val. if it climbs steadily over three epochs (or maybe let's set it to five; three is a little short), then the level-restricted objective can be isolated as the culprit and we can maybe experiment with LogSumExp from TaxaWalk to see if the failure mode is rank-balancing. if species-level accuracy still deteriorates like the (c) runs, then something is wrong with the fine-tuning recipe. that's when we can turn knobs recorded in D1 such as batch size, alpha, training text template, image-side LoRA, a different dataset (that BioCLIP 1 wasn't pretrained on).
>
> immediate next actions:
> * for the next few experiments, change max_epoch from 3 to one of {5, 8, 10}, whichever makes more sense. we want to see more of the curves to make better judgment
> * allow and conduct inat21 val split evaluation for future runs to inform revising the fine-tuning recipe, so long as it is "legal" (unlike for checkpoint selection).
>
> make updates to agent/ docs. update draft of amendment 3 before we enter the "second standing phase" of M2. i think assumptions and expectations as written in D1 can be kept. other parts of the working hypothesis/plan for (c) need revision.

Verbatim addition (2026-10-07, 12:03): "submit arm (a) and resubmit arm (c) at lr=1e-4 and trainer.max_epochs=5, with eval on inat21 val queued. now"

Verbatim addition (2026-10-08, after the five-epoch runs and their iNat21 diagnostics):
> Update: runs with 5 epochs and their evals finished. i will first give you explicit directions for what to update in ACTIVE, then answer your previous open questions awaiting my response, then give my interpretations of the runs, then give you my next steps.
>
> A.
> ACTIVE.md:133: replace run 264726 with run 281124 as the reference run for M2 phase 2's (c) runs, which is the same training config but with 5 epochs rather than 3. confirm that run 281124's first three species-correct on ToL-val are close to "111,727 → 97,244 → 85,586".
>
> ACTIVE.md:177: this is the last sentence in the line, which lists which lines in specs/pilot1.md are being subject to change based on M2 phase 2 runs. mark this line itself as subject to change, i.e. we may come up with new knobs to turn as part of M2 phase 2, that may affect additional lines in the original spec pilot1.md.
>
> ACTIVE.md:178: qualify the stated gate before M3 by adding that it is possible that LoRA fine-tuning alone on ToL-EOL (already used for pretraining) may not bring gains to classification accuracy. part of M2 phase 2's goal is to determine the "ceiling" of fine-tuning with a contrastive objective alone. i can think of several possible reasons arm (c) never beats BioCLIP 1 at every rank.
>
> B.
> skip iNat21 eval for earlier runs; as per Amendment 3, keep eval runs 281238/281239/281240. approve Amendment 3 after changes in A have been made.
>
> C. interpretation
> observations on val/tol/photo/: on arm (a) species jumped to ~54% after the first epoch and stayed there for the rest, dropping to 52~53 after the fifth epoch. the other ranks accuracy mostly dropped below val_init and stayed at the dropped value for the remaining epochs, which is within expectation given the species-level only contrastive objective. on arm (c), all curves reproduced p1c_lvl_lr1e-4_s42_v1's first three values; species continued to deteriorate past the third epoch all the way till the fifth epoch to ~18.
>
> hypothesis: the flat curves of arm (a) across ranks suggest training saturation with LoRA fine-tuning on ToL-EOL. in my opinion, there is no confident way of testing this hypothesis, but also no urgent need to.
>
> hypothesis: the jump in species to 54% on ToL-val and rough maintenance across five epochs supports the argument that the level-restricted contrastive objective of arm (c) costs species-level semantics while improving other ranks, likely because of the supervision at coarse ranks competing with species-level supervision.
> * the outcome that would support this hypothesis: with a more rank-balanced weighting scheme for the level-restricted contrastive objective, such as LogSumExp (used in TaxaWalk), species on arm (c) should at least not deteriorate, while accuracy improves across coarse ranks during training.
> * higher-level hypothesis/argument: the level-restricted objective is the culprit.
>
> D. next steps
> 0. general directions: these steps do not need to be gated; instead they can be queued after one another. you may otherwise raise concerns or suggest a gate if something irreversible or important depends on a particular experiment. leave Branch B options as written in ACTIVE.md D1 as they are.
> 1. implement and run text-space probe. in parallel, run image-side only LoRA with configs matching: 281183 (a), 281184 (c-uniform).
> 2. closely read TaxaWalk.md, implement LogSumExp, choose reasonable temperature based on TaxaWalk, then run on arm (c) with otherwise matched config to 281184 (c-uniform). nickname this c-balanced.
> 3. iNat21 eval on all of these runs' checkpoints under pilot1.md:A2.4 setting.
> 4. start these runs on arm (d), with matched config to: 281184 (c-uniform), c-balanced, and one of {image-side c-balanced, image-side c-uniform} due to compute constraints and depending on which weighting wins. rationale: no need to start (d) on a perfect baseline, we can afford comparing (d) vs. (c) on different objectives/recipes to a certain extent.
>
> E. finally, amend to pilot1.md, update DESIGN, USAGE, and update ACTIVE.md last (e.g., M2 plan, true now, next action, prune draft and dated content).

Verbatim addition (2026-10-08, mid-turn): "before you submit the first job, write a commit and show me"

My interpretation (not the user's words): "281124" in A is read as 281184, the five-epoch (c) run (281124 is another user's job); its first three counts are the sweep run's exactly. A's three edits were made to the draft and Amendment 3 was appended on 2026-10-08 (B). C's interpretation is quoted in `audit/2026-10-08_M2_phase2_round1.md` §4. D is the plan under D1: steps 1, 3 and the first run of 4 were queued at 16:27; step 2 (c-balanced) waits on one decision I raise, the direction LogSumExp weights the levels in (audit §5.5). Branch B stays as written. The commit request arrived after the 16:27 submissions; the jobs were still pending, so the commit made right after holds exactly the tree they run.

## Directions

### D1: Pilot 1, hierarchical orthogonality penalty on BioCLIP 1
- Asks served: A1, A15 (A14 done, pruned).
- Hypothesis (spec section 1): adding the cross-level orthogonality penalty to BioCLIP 1 fine-tuning makes zero-shot classification on iNat21-val competitive across all seven ranks, against BioCLIP 1, BioCLIP 1 + RCME, and BioCLIP 1 + BFL.
- Success bar (quoted from spec section 1): "competitive performance at coarse levels (kingdom to order) (which means beating RCME and competitive with BFL-Euc), and matched or better performance at the genus and species level than both BFL and BioCLIP 1. Allow margin of +- 1 point for all comparisons."
- Failure mode (quoted from spec section 7): "Species-level classification falls below BioCLIP 1's 70.19% (under "a photo of [label]." convention) in return for better coarse-level classification."
- Working hypothesis and plan for the (c) recipe (user, 2026-10-06, A15; the spec's hypothesis above is unchanged):
  - Bet: BioCLIP 1's text tower already encodes the hierarchy; moving it (text-side LoRA) costs species names first. Candidate fix: image-side LoRA only, so the contrastive loss and the penalty move images against a fixed text space. Precondition (user, 2026-10-07): the text space must first show zero-mean parents and near-orthogonality (plan step 0); if it does, image-side LoRA is a transfer of text-side geometry to the image tower; if not, it trains in a vacuum and the anchor needs diagnosing first. Tests queued 2026-10-08: the text-space probe (job 283683) and image-side LoRA under both objectives (283677 arm a, 283678 arm c).
  - Assumptions, with what is known. (1) The level-restricted loss is BFL's: unverified against the paper (BFL released no training code; `src/models/losses.py` is our construction from its equations; a hand-computed tiny batch, about an hour, would check it). (2) "BFL significantly surpasses BioCLIP 1" holds on BFL's own harness (reported 63.00 vs 50.79 at species); on ours the released BFL-Euc is below BioCLIP 1 at species (68.78 vs 70.19) and far above at the other ranks (spec section 1 table). (3) The grouped sampler is load-bearing for the penalty (line 145): under random batches nearly every group is one image, so a sampler change that helps (c) does not carry to (d) as is; K = 8 (line 142's other value, fixed to 16 by S2) keeps groups with about 1,780 species per batch.
  - Expectation and gate before M3 (user): arm (c) after 10 epochs matches or beats BioCLIP 1 on iNat21. Still to pin: the rank(s) and the bar (S31's ±1 point on integer counts). At species the bar is stricter than what the released BFL achieves in our harness; at every other rank BioCLIP 1 is far behind BFL-Euc. Now Amendment 3 A3.4, with the owner's qualification: LoRA fine-tuning alone on ToL-EOL may bring no gain, and finding that ceiling is part of the second phase; the rank(s) and the checkpoint it is read on (S27-selected or final) are pinned before M3.
  - Plan for M2's second phase (user, 2026-10-07, revised 2026-10-08; Amendment 3 A3.2). Rules: one knob per run; every run on the grouped sampler; `trainer.max_epochs=5` on the 10-epoch schedule; steps are queued one after another, not gated, unless something irreversible or important depends on one; Branch B stays as written. Reference run for the (c) comparisons: 281184 (c-uniform, `p1c_lvl_lr1e-4_s42_v2`, five epochs; ToL-val species-correct 111,727 → 97,244 → 85,586 → 72,523 → 49,210, the sweep run's values exactly for the first three) and untouched BioCLIP 1 (108,062; iNat21 70,193). Costs at 1 × 2 A100s: about 7.5-8 h per five-epoch run plus queue; iNat21 diagnostics 8 min per run.
    - Round 1, done 2026-10-07 (`audit/2026-10-08_M2_phase2_round1.md`): arm (a) 281183 and arm (c) 281184, with iNat21 diagnostics 281239 and 281240. Reading: the objective is the main driver (same recipe: (a) keeps +33,200 ToL-val species-correct after five epochs and +3.4 points over BioCLIP 1 on iNat21; (c) ends −58,900 and −19.3); the decline transfers to iNat21; after one epoch c-uniform beats BioCLIP 1 at every rank on iNat21 (species +2,043 images, average 79.55 against 41.54).
    - Round 2, queued 2026-10-08 16:27: step 0, the text-space probe (job 283683, `scripts/p1/text_geometry.py`, one GPU, result to `audit/2026-10-08_text_geometry/`); step 1, image-side LoRA only (`model.lora_towers=image`, lines 61 and 154) on arm (a), job 283677 (`p1a_flat_lr1e-4_s42_imglora_v1`), and on arm (c), job 283678 (`p1c_lvl_lr1e-4_s42_imglora_v1`); step 4a, arm (d) matched to c-uniform with λ = 27.787 (S6), job 283679 (`p1d_lvl-pen_lr1e-4_s42_v1`). iNat21 diagnostics queued behind each: 283680, 283681, 283682 (step 3, standing under A3.1).
    - Step 2, c-balanced (LogSumExp over the seven level terms, TaxaWalk's form, line 178), on arm (c) matched to 281184: waits on one decision. TaxaWalk's τ·log Σ exp(ē_t/τ) weights the rank with the largest loss most; in (c) that is family (3.0) and order (2.5), with species among the smallest (0.7-1.2), so as written it moves weight away from species (audit §5.5). Options for the owner: TaxaWalk's form as written (predicts a worse species curve); the same over each level's loss relative to its own starting value; per-level gradient-norm balancing; a direct species up-weight. Any form needs its own λ (S22) and a log of the seven weights. Then step 4b, arm (d) matched to c-balanced, with λ from that run.
    - Step 4c, one of arm (d) on image-side c-balanced or image-side c-uniform, by which weighting wins; after steps 1 and 2.
    - Branch B, as written: batch size 16,384 (S8; 2 × 2 A100s), alpha = r = 16 (lines 61, 154), training text `{lineage}` (line 51(a); the verdict's template stays "a photo of"), a dataset BioCLIP 1 was not pretrained on (no candidate chosen); later, the level loss without dedup and group positives (lines 96-97) and the whitener in the penalty (lines 124-139; arms (b) and (d) only).
  - Amendment 3 (A3.1-A3.4) was appended to `specs/pilot1.md` on 2026-10-08 after the owner's approval: diagnostic iNat21 evaluations during M2 (recipe only, never selection; the sweep runs and the control are not evaluated), the second phase's rules, the recipe-revision record (its list of lines itself subject to change), and the qualified expectation before M3 (rank and checkpoint pinned before M3). The spec is now 51,801 bytes, sha256 `eaec4f73…`, a byte-prefix extension of `90a7d38`.
- Who decides: the user gives the verdict. I report numbers and interpretations only.
- True now (2026-10-08 ~16:31): M1 is DONE (checker `audit/2026-10-05_M1_checker.md`, response `audit/2026-10-05_M1_checker_response.md`). M2 is in its second phase under Amendment 3 (appended 2026-10-08). The sweep picks lr 1e-4 (line 161; `audit/2026-10-05_M2_sweep_results.md`); the random-batch control showed batch composition is a part, not the cause (`audit/2026-10-06_A14_sampler_control.md`); round 1 of the second phase showed the objective is the main driver: under one recipe arm (a) holds species (ToL-val 53.95% → 52.58%, iNat21 75.49% → 73.59% against untouched 40.23% and 70.19%) while c-uniform loses it (41.59% → 18.32%; 72.24% → 50.89%) and gains every coarse rank (`audit/2026-10-08_M2_phase2_round1.md`). Round 2 is queued (16:27): the text-space probe 283683, image-side LoRA on (a) 283677 and (c) 283678, arm (d) matched to c-uniform 283679 (λ 27.787), with iNat21 diagnostics 283680-283682 behind the training jobs. c-balanced waits on the owner's choice of balancing form (the LogSumExp direction, audit §5.5). A3.4's rank and checkpoint are unpinned. The M2 checker waits for the revised recipe.
  - Spec: `specs/pilot1.md` with Amendments 1, 2 and 3 at the bottom (51,801 bytes, sha256 `eaec4f73…`), which govern on any conflict, the later one first. The S9 list is `specs/pilot1_s9_missing_species_keys.txt`. Do not edit the preflight report.
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
    - Then the checker, then the user's go. (2026-10-07: M2 has a second phase, the recipe revision; its plan is under the working hypothesis above.)
  - **M3 (all arms finish, no verdict).**
    - The winning (c) run continues to 10 epochs, then (d), (a) and (b) in the spec's order (line 153), each with its own λ (S6).
    - No projected hours (S29 as amended by A2.2): the runs report their own times. Do not stall for approval.
    - Then the checker. The user gives the verdict.
- Next action (M2, the LR sweep):
  1. Round 2 running or queued: 283677 (a, image-side LoRA), 283678 (c, image-side LoRA), 283679 (d, λ 27.787), each about 8 h once started, with iNat21 diagnostics 283680-283682 behind them; the text-space probe 283683 (about 15 min once started) reports to `audit/2026-10-08_text_geometry/`. Read each run's epoch ends against 281184 and 281183.
  2. Decision for the owner before step 2: which quantity the rank balancing acts on (audit §5.5: TaxaWalk's LogSumExp over the level losses as written weights family and order most and species least). Then: read `agent/litreview/Sastry2027-TaxaWalk.md` closely for τ, implement the chosen form with its seven weights logged, run c-balanced matched to 281184, then d-balanced with λ from it.
  3. After round 2 and the balanced runs: step 4c (one arm (d) image-side variant), then Branch B knobs as the readings direct, one per run.
  4. Still open: A3.4's rank(s) and checkpoint (S27-selected or final), pinned by a dated amendment before M3; the line 161 lower-LR gap; the M2 checker runs once the (c) recipe is settled; the end-to-end smoke run stays unsubmitted (user, 2026-10-05).
  - Timing facts kept for M3 planning (no projection is owed, A2.2). At 2 × 2: 3.1-3.5 s per ordinary step, 6.1 s per monitor step, ToL-val 82 s, refresh 41 min with the old reader (job 263082, at night). At 1 × 2: 5.8-6.0 s per ordinary step (median, the same with either reader), 10.5 s per monitor step, 47.4 GiB peak; refresh 25.0-33.5 min with the pread reader (265500: 2,007.6 / 1,708.5 / 1,497.8 s) against 59.1-73.8 min with the old one (264726: 4,425.6 / 3,737.1 / 3,546.2 s) or stuck (264728, unfinished after 2 h 49 min); ToL-val 144-149 s for both forms. An epoch after the first takes 1 h 32-35 min at 1 × 2 with the pread reader (2 h 12 min with the old one). Reader benchmarks on CPU nodes (16 workers, 65,536 fresh rows per run): pread 1,850-3,592 img/s against memmap 1,626-2,187, with less kernel CPU; DONTNEED leaves 0.6 GiB of page cache per 9.2 GiB read against 9.3 GiB without it, at no throughput cost (`audit/2026-10-05_stall_diag/bench_*.txt`). Random batches (A14 control): 8.5 s per step, 60.1 GiB peak, refresh 28-31 min, epochs 2 h 03-18 min, run 6 h 33 min. Five-epoch runs at 1 × 2 with the pread reader: 7 h 24 min (arm a, 5.5 s per step, 45.6 GiB) and 7 h 52 min (arm c, 5.9 s, 47.4 GiB). iNat21 diagnostic: 73-103 s per checkpoint, 8 min per five-checkpoint run on one A100.
  - M3 (decided, A2.4): the iNat21 eval records its encode setting and re-evaluates all six baselines under it; the floor and the S31 cells are same-setting counts.
  - M3 reading caveat (TaxaWalk review, `agent/litreview/taxawalk-and-pilot1.md`; the preprint's §4.1 says per-rank scoring is "unfavourable" to models trained on full lineages): at coarse ranks the per-rank protocol may favour the prefix-trained arms over BioCLIP 1 and RCME for reasons unrelated to geometry. The controlled comparisons, (d) vs (c) and (b) vs (a), are unaffected.
- Artifacts: M2 runs write under `logs/p1/<run_name>/` (Hydra run dir with `steps.jsonl`, `decisions.json`, `run_info.json`; `checkpoints/epoch_NN.ckpt` and `last.ckpt`) and to wandb project `multimodal_lab/taxa_maze`. Everything through M1 (audits, preflight and S9 reports, smoke outputs, store and prep paths) is listed in `agent_legacy/active/2026-10-05_D1_artifacts-through-M1.md`.
- Log:
  - 2026-10-08 ~16:31: round 1 of the second phase read (`audit/2026-10-08_M2_phase2_round1.md`): (a) 281183 and (c) 281184 for five epochs with iNat21 diagnostics; (c) reproduces 264726 exactly for three epochs. Owner's directions A-E applied: reference run 281184; A3.3's list marked subject to change; A3.4 qualified; Amendment 3 appended (51,801 bytes); earlier runs not evaluated on iNat21. Added `model.lora_towers` (image adapters initialise identically; text LoRA count 0 under `image`) and `scripts/p1/text_geometry.{py,sh}`. Queued 283677-283683 at 16:27; committed the tree they run right after. Raised the LogSumExp direction before c-balanced. 2026-10-06/07 entries moved to `agent_legacy/active/2026-10-08_D1_log-phase2-preparation.md`.
  - Entries from 2026-10-06 (the control) to 2026-10-07 (round 1 submitted and validated) are in `agent_legacy/active/2026-10-08_D1_log-phase2-preparation.md`.
  - Entries from 2026-10-05 ~17:20 (the reader switch) to ~23:09 (the sweep's first-round results) are in `agent_legacy/active/2026-10-05_D1_log-sweep-first-round.md`.
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
| 2026-10-07 | prune-log rows dated 2026-10-02 to 2026-10-04 (6 rows: M1-era asks, blocked items and directions) | moved out to keep this file near 200 lines | `agent_legacy/active/2026-10-07_prune-log_through-2026-10-04.md` |
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
