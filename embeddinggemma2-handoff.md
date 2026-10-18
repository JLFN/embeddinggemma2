---
project: embeddinggemma2
plan_start_commit: 0b12d1be6e01368be910af96b0de4fe482641e4f
last_updated_commit: 7ec9b56dfc0aa00ad5279d1f7b4c79a1a36eebfc
branch: dev
remote: 
handoff_written_at_context_usage: 18%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-18T20:53:20+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: dirty
---

Handoff + executable plan — EmbeddingGemma-2 VQ (2026-10-18)

Note on the frontmatter: it MUST be the first thing in the file, with
bare values (no inline comments after the value) — the handoff-hooks
parser reads it verbatim, and a commented or misplaced frontmatter
silently breaks the SessionStart probe readout. The scaffold writes
this exact shape; hand-written files must match it. <COMMIT> means the
full commit hash (plan_start_commit = the commit the current plan
began from); <PERCENT> = the context usage read from the session's
signals.json (target 25-35, hard cap 40).

0. Rules for writing THIS document (read before editing, not just before reading)

- Only write or replace this file at a clean, verified, committed boundary. Never checkpoint mid-work.
- Start drafting once context usage crosses ~25%. Finalize and commit before the auto-compact threshold — do not wait for it to begin writing. The writing is a summarization task and degrades like any other task under context pressure. Treat 25-35 as the drafting target and the threshold as the hard cap. The threshold is the harness's own cut-over: [session] auto_compact_threshold_percent in ~/.opengrok/config.toml (this machine: 60), which makes it compact this session in place — no new session is opened.
- Every claim in "Current state" must be re-verified by running the relevant command in this session, not copied or paraphrased from the previous handoff. If you cannot re-run it, mark it unverified rather than asserting it.
- Hard caps: "Current state" max 15 lines. "Guardrails and open items" max 15 lines. "Next unit plan" has no cap — it is the actual work spec. If a section would exceed its cap, cut detail; do not compress into denser prose (denser prose is where drift hides).
- Do NOT: replay prior units' logs (git log is the record), re-explain global rules already in the loaded guardrails, restate anything the staleness check in section 6 already covers, or record acceptance evidence you did not personally observe this session.

1. Session protocol (read first)

This project is a sequence of units; each unit ends with a verified, locally
committed step of the plan, and a push is offered, never taken (no remote is
configured yet).

1. Read context usage from the session's signals.json (contextTokensUsed /
   contextWindowTokens) under ~/.opengrok/sessions/<percent-encoded-cwd>/
   <session-id>/, newest session dir. If signals.json is absent, fall back to
   chat_history.jsonl bytes / 4 and say in the session report that the number
   is a fallback, not the gauge.
2. If usage is below the threshold (~/.opengrok/config.toml
   [session].auto_compact_threshold_percent, this machine 60): continue
   immediately — write the next unit's plan into section 3, execute it,
   verify, commit, re-check usage, repeat.
3. At the threshold the harness compacts this session in place; nothing is
   launched. This handoff must be written and committed BEFORE the threshold,
   because a compaction carries a summary of the conversation, not unwritten
   plan prose.
4. Preferred cadence: once the finished unit's handoff, the project memory
   entry and the commits are in place, call the compact_session tool as the
   last action of that turn.

2. Current state (2026-10-18 — COMPLETE, verified)

- Unit 2 outcome: the dense-matrix VQ encoder exists and its known-answer test
  passes. tools/eg2_vq.c includes the reference encoder verbatim
  (vq_qc.h:vq_encode_full, X=NULL nocal path, plus vq_pack and
  vq_unpack_dequant) and reuses src/common/ds4_st41.h as the safetensors index;
  bin/eg2_vq builds with the command in the file header.
- Round trip is exact: for every matrix measured, pack size equals
  vq_payload_bytes and decode-vs-encode maxabs is 0.
- Selftest (deterministic 512x512): hot 4/512 cos 0.9946 at 2.41 effective bpw,
  cold 8/256 cos 0.9685 at 1.16 effective bpw.
- First real measurements (docs/ladder-embeddinggemma2.txt): hot tier on text
  tower cos 0.9568-0.9629 (relL2 27.0-29.1%); cold tier on audio and vision FFN
  cos 0.8552 and 0.8001 (relL2 51.8% and 60.0%).
- FINDING that supersedes the unit 1 estimate: the cold tier MISSES the
  DeepSeek quality bar (cos 0.910-0.925) on audio and vision FFN, so the
  two-tier split as chosen does not hold; 283.1 MB is not reachable at the
  agreed quality, while an all-hot allocation costs about 345.2 MB (4.31x).
- Encoder cost measured on the CPU path: 1.5-2.4 s per 0.5-4.2M-parameter
  matrix, so the full 553M-parameter VQ set is tens of minutes and the GPU path
  (untrusted on this card, see section 4) is not needed for the first pass.
- Context-usage note: signals.json is absent for this session; the frontmatter
  percent is the bytes/4 fallback (582 KB transcript) against an unverified
  window (the active model id has no explicit context_window block).

3. Next unit plan — tier selection from a measured ladder, then the full encode
   and RECIPE.md (2026-10-18)

Status: not yet started.

Goal: pick the cheapest per-class tier that still meets the quality bar, with
the choice resting on measured rows only, then run the full encode and write
RECIPE.md from the measured numbers.

Step 1 — complete the ladder (acceptance: docs/ladder-embeddinggemma2.txt has a
measured row for every class x tier, and the chosen tier per class is the
cheapest passing row, named as such).
- Add the intermediate tiers to the tool's CLI call: dim 4 / nc 1024 (2.5 bpw),
  dim 8 / nc 512 (1.125 bpw), and dim 8 / nc 1024 (1.25 bpw) if the first two
  still miss. Existing measured rows: 4/512 and 8/256.
- Sample per class, 2-4 tensors each, using the exact names in
  docs/tensor-inventory.tsv: text-tower q/k/v/o and mlp gate/up/down, text-ple,
  audio attention and FFN, vision attention and FFN.
- The bar is the DeepSeek V4.1 recipe's recorded quality (cos 0.925 at 13 bit,
  0.910-0.911 at 12 bit) and it applies per class. A class that cannot reach
  the bar at any tier is a finding, not a silent downgrade: it moves to the
  cheapest passing tier and the deviation is recorded with its measurement, or
  the class is left at source precision and the size consequence is stated.

Step 2 — rewrite the allocation from the measured ladder (acceptance:
docs/allocation-plan.txt carries the chosen tier per class, the per-class
cosine, and a byte estimate computed from the ladder, with the previous
283.1 MB figure marked superseded).
- Known reference points: all-hot is 345.2 MB (4.31x); all-cold is about 253 MB
  but fails the bar on FFN.

Step 3 — the container and the full encode (acceptance: the artifact exists,
its size matches the account, SHA256SUMS is written, and a decode of every
payload agrees with the encoder's own dequant at maxabs 0).
- Container: versioned header with a recipe string, a slot table mapping tensor
  name hash to payload offset and size, then the DQVQ payloads verbatim. Do not
  invent payload fields; the payload contract is vq_qc_bytes.h and
  /data/ds4-dfm-rs/ds41_vq_fmt.h, and an unknown container version must be a
  hard stop (the version whitelist precedent is ds4vq_blob_ok).
- Carries the skeleton per the unit 1 allocation: token table at q4_K (the
  reference recipe's choice for embeddings), tower frontends and tiny tensors
  at source precision.

Step 4 — RECIPE.md in the format of RECIPE-deepseek41-vq.md (pinned inputs
table with the checkpoint revision and the YoungAi revision a4e6d31, settings
table with the chosen tiers, exact commands, verification table with the
measured cosines, build notes). Replace the DeepSeek text currently in
RECIPE.md; RECIPE-deepseek41-vq.md keeps it.

Alternative worth naming in the recipe: the reference toolchain solves this
allocation problem without calibration in v41_nc_alloc.c (per-tensor energy
Σw², rate-distortion greedy with relative distortion 2^(-b/4) per index bit,
byte account from the same formulas). If per-class tiering leaves the artifact
larger than the owner accepts, that allocator is the next unit, and it is the
owner's own stated preference ("quantize from the weights, no corpora").

Execution order: CPU path only for this unit. The earlier phases showed the CPU
cost is minutes, not hours, and the GPU kernels are untrusted here.

Verify, package, handoff: run the ladder, rewrite the allocation, run the full
encode, write RECIPE.md, then commit with the trailer "Unit: 3 complete" as the
last body line, refresh section 2, rotate this section, update the memory entry,
and offer the push (no remote exists, so that offer is to create one).

4. Project facts a fresh session cannot re-derive

- What exists: .models/EmbeddingGemma-2-914f7f8 (pinned checkpoint, 1.5 GB,
  gitignored); tools/eg2_vq.c and bin/eg2_vq (the encoder, build command in the
  file header); docs/tensor-inventory.tsv; docs/allocation-plan.txt;
  docs/ladder-embeddinggemma2.txt; RECIPE-deepseek41-vq.md (style and quality
  bar reference); RECIPE.md (still the DeepSeek text, replaced in step 4).
- The encoder TU compiles with go2b_qc.h (g2_inv), onebit_quant.c (f16), a
  local dq_matmul definition, and vq_qc.h included verbatim; it emits the
  reference header's known harmless warnings (unused static plan-table paths,
  "vq_w2_dim used but never defined" from those unused paths).
- Reference toolchain pinned at /data/YoungAi revision a4e6d31
  (gguf-tools/quantize/vq_qc.h, vq_qc_bytes.h, go2b_qc.h, onebit_quant.c;
  src/common/ds4_st41.h, ds4_float.h). The future loader is /data/ds4-dfm-rs,
  which this session verified has no gemma and no embeddings support at all
  (no "gemma" string in crates/ or ds4.c; family list deepseek4|motif3|
  solar-open2|exaone-moe|dots3-note|qwen4exp|glm5-next|k2-horizon|iquest_q1; no
  embeddings endpoint), so it is not on the critical path for this unit.
- Environment: RTX 4070 SUPER, 12282 MiB, sm_89; CUDA 13.3; max dynamic smem
  per block opt-in 101376; pageableMemoryAccess 0, concurrentManagedAccess 1,
  unifiedAddressing 1. The VQ GPU kernels (vq_gpu.cu) assume unified memory
  ("malloc pointers passed directly, no explicit transfer") and their error
  path warns at most four times and continues, so a faulting kernel can leave a
  silently wrong index stream: add explicit copies and re-run the known-answer
  test before trusting any GPU output. The smem gate nc*(dim+1)*4 rules out
  dim 8 / nc 4096 here (147456 B); 4/512 and 8/256 are 10240 B and 9216 B.
- transformers 5.14.1 has no embedding_gemma2 model (the checkpoint config pins
  5.18.0.dev0), which is why the nocal path is the one this plan uses; no
  reference forward pass is required by the owner's chosen gate.
- Guardrails mount: /data/embeddinggemma2/.opengrok/rules/project-work-rules.md
  (link=yes exclude=yes listed=yes), excluded via .git/info/exclude.

5. Guardrails and open items

- Global guardrails in effect: the always-on core is ~/.opengrok/AGENTS.md
  (rules 0, 0.1, 1, 2, 4, 8, 15, 16, 17, 18, 20, 21, 22). The project-workflow
  rules 3, 5, 6, 7, 9, 10, 12, 13, 14, 19 and 23 live in
  ~/.opengrok/project-work-rules.md and load through the mount above.
- Repo hygiene: .gitignore excludes .models/ and graphify-out/; no remote is
  configured, so rule 23's push offer is an offer to create one.
- LIVE DECISION: the cold tier missed the bar on audio and vision FFN. Step 1
  settles it by measurement; if no cheap tier passes, the artifact lands near
  the all-hot 345.2 MB (4.31x) and the owner is told the size consequence rather
  than the quality being quietly lowered.
- OPEN: the owner's scope answer came as a quality statement, not an explicit
  list, so the reading used here (all three towers' dense matrices VQ'd, token
  table and frontends as skeleton) still needs confirmation.
- OPEN: rule 12 graphify graph and rule 19 QA evidence before any push; rule 10
  has no Cargo.lock to scan because this repo is not Rust.
- OPEN: whether to pursue the v41_nc_alloc-style per-tensor allocator after step
  4 (see section 3's alternative) is the owner's call, and depends on the size
  step 2 actually reaches.

6. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline 0b12d1be6e01368be910af96b0de4fe482641e4f..HEAD
    git diff --stat 0b12d1be6e01368be910af96b0de4fe482641e4f..HEAD
    git log --grep "Unit: .* complete" 0b12d1be6e01368be910af96b0de4fe482641e4f..HEAD

Trust git over this document's headings: continue with the first unit whose
work is not in the diff.

Staleness check:

    cd /data/embeddinggemma2 && git log --oneline 5e510a73fb167ddfc015b33b3f0a482a874d044d..HEAD

A single docs commit carrying this handoff is expected; anything else must be
read before sections 2-5 are trusted.

This document is COMPLETE and self-contained: a new session can execute the
project end to end from this file alone. It carries the plan ahead, the facts
git cannot provide, and the guardrails. History is deliberately absent — git
log and the project memory entry are the historical record.
