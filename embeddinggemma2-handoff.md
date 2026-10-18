---
project: EmbeddingGemma-2 VQ
plan_start_commit: 0b12d1be6e01368be910af96b0de4fe482641e4f
last_updated_commit: 0b12d1be6e01368be910af96b0de4fe482641e4f
branch: dev
remote: none
handoff_written_at_context_usage: 11
handoff_written_at: 2026-10-18T18:48:18Z
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
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

- Unit 1 outcome: repo initialised on dev at 0b12d1b; project-work guardrails
  mounted (open-grok-mount-rules --check: link=yes exclude=yes listed=yes);
  checkpoint pinned and fetched; measured inputs committed.
- Checkpoint: google/embeddinggemma-2 revision 914f7f89142e33e77833254d9c9b90c3cef7303b
  at .models/EmbeddingGemma-2-914f7f8 (1.5 GB, gitignored). model.safetensors
  1.49 GB, 1376 tensors, 1488.7 MB by header byte sum.
- Class sizes (MB): audio-ffn 402.7, token table 268.4, text tower 235.7,
  vision-ffn 226.5, audio-attn 128.0, audio frontend 75.6, vision-attn 75.5,
  text-ple 37.7, vision frontend 32.6, tiny 5.5.
- Allocation (docs/allocation-plan.txt): VQ cold 8/256 (1.0 bpw) for audio and
  vision attn+ffn, VQ hot 4/512 (2.25 bpw) for the text tower and text-ple,
  token table q4_K, tower frontends kept at source precision. 477 target
  matrices, all with cols divisible by the tier dim (0 exceptions).
- Estimate: 283.1 MB against 1488.7 MB = 5.26x smaller.
- Context-usage note: signals.json is absent for this session, the frontmatter
  percent is the bytes/4 fallback (449 KB transcript ~ 112k tokens), and the
  active model id has no explicit context_window block in config.toml.

3. Next unit plan — dense-matrix VQ encoder plus known-answer test (2026-10-18)

Status: not yet started.

Goal: an encoder that turns each allocated matrix into the YoungAi VQ payload
byte-for-byte, and a known-answer test that proves encode/decode before any
full run. Nothing in the payload format is invented: it is copied from the
reference toolchain, cited below.

Ground truth to read first (rule 0.2 — read the artifact, never a summary):

- /data/YoungAi/gguf-tools/quantize/vq_qc.h
  - vq_encode_full (line 86): kmeans with strided subsample (<= 200k vectors,
    8 Lloyd rounds, codebook re-round-tripped through f16 before assignment),
    GPTQ over column groups of 128 with 1% diagonal damping, channel weighting
    from 1/diag(H^-1), per-row gains by closed-form ridge clamped to [0.25,4.0].
    Tolerates X = NULL: that is the "nocal" path (no GPTQ feedback, no row
    gains) the DeepSeek recipe records as "Calibration data: None".
  - vq_pack (line 241) and vq_unpack_dequant (line 268): payload bytes.
  - vq_qc_bytes.h: vq_idx_bytes / vq_payload_bytes — the single byte-accounting
    header shared by writer and allocator. Payload = 16 B header
    ('DQVQ', u16 dim, u16 nc, u32 rows, u32 cols) + nc*dim f16 codebook +
    rows f16 gains + index stream. Index width = ceil(log2 nc); exactly 8 bits
    is stored as one byte per index, wider widths as a LE bitstream plus one
    guard byte. Value rule: value[row][col] = codebook[idx][col % dim] * gain[row].
- /data/ds4-dfm-rs/ds41_vq_fmt.h: the DQVL/DQVQ container contract, the slot
  table at offset 16, and the version whitelist (an unknown version must be a
  hard load-time stop, never a mis-decode).
- /data/ds4-dfm-rs/crates/ds4-core/src/vq.rs: the Rust CPU oracle for decode;
  the encoder's round-trip must agree with it.

Phase A — safetensors manifest reader (C, in this repo).
- Mirror the toolchain's own style for config parsing: strstr-based minimal
  key search over the header JSON (v41_cfg.h is the precedent), no JSON library.
- Emit per-tensor name, shard, absolute data offset, dtype, shape.
- Acceptance: the byte sum equals 1488.7 MB and the tensor count equals 1376,
  cross-checked against docs/tensor-inventory.tsv.

Phase B — encoder eq2_vq plus container.
- Encode exactly the allocation table: hot dim 4 / nc 512 for language_model.*
  and ple.*, cold dim 8 / nc 256 for audio_tower.* and vision_tower.* attn/ffn.
- Call vq_encode_full with the X = NULL (nocal) path, pack with vq_pack,
  write a container whose header carries a version, a slot table (tensor name
  hash -> payload offset and size) and the recipe string. Reuse the DQVQ
  payload bytes verbatim; only the dense header/slot table is new, and it
  mirrors DQVL's shape.
- Acceptance: for a fixed deterministic PRNG matrix and for one real matrix
  (language_model.layers.0.self_attn.q_proj.weight), encode, decode with
  vq_unpack_dequant, and report cosine and max-abs error; the Rust oracle in
  /data/ds4-dfm-rs/crates/ds4-core/src/vq.rs must produce identical values on
  the same payload.

Phase C — measured ladder (the ladder must be measured, never estimated;
ladder_vq4.txt states this rule for the DeepSeek plan table).
- For every class, encode a representative sample at both tiers and record
  cosine and relative L2 per matrix in docs/ladder-embeddinggemma2.txt.
- Bar to match: the DeepSeek V4.1 numbers this recipe must equal — cosine
  0.925 at 13 bit and 0.910-0.911 at 12 bit for routed experts, 17.06%
  relative error for layer 20 of V4.1 (per RECIPE-deepseek41-vq.md).
- A class that misses the bar at the cold tier moves to hot and the file is
  updated; vq_qc.h records the same failure mode for DeepSeek w2, where
  codebooks were measured ineffective and a different scheme was used.
- Acceptance: the ladder file exists with measured values and a clear
  pass/fail against the bar per class.

Phase D — full encode, artifact, checksums, and RECIPE.md.
- Run the full encode, write the artifact plus SHA256SUMS, and write
  RECIPE.md in the format of RECIPE-deepseek41-vq.md (pinned inputs table,
  settings table, exact commands, verification table, build notes), replacing
  the DeepSeek text that currently sits there.
- Acceptance: artifact bytes equal the estimate within the accounted
  overhead; every matrix's cosine is in the ladder file; RECIPE.md states the
  measured numbers, not expectations.

Execution order and the GPU question: run the CPU path first. The GPU kernels
in vq_gpu.cu are written for the Spark's unified memory ("malloc pointers
passed directly, no explicit transfer"), and this machine measured
pageableMemoryAccess = 0, so that path needs explicit copies before it can be
trusted; vqg_last_err also warns at most four times and continues, so a
faulting kernel can leave a silently wrong index stream. If the CPU path is
too slow at Phase D scale, add the copies and re-run the Phase B known-answer
test before using the GPU path for any real matrix. Second constraint to
respect: the GPU gate is nc*(dim+1)*4 bytes against a 101376-byte opt-in
limit, so dim 8 / nc 4096 (147456 B) can never run on the GPU here; the
chosen tiers are 10240 B and 9216 B and fit.

Verify, package, handoff: run Phase A/B acceptance checks, the Phase C ladder,
then the full encode; record measured times per class; commit with the trailer
"Unit: 2 complete" as the last body line; refresh section 2, rotate this
section to the next unit, write the memory entry, and offer the push (no
remote exists yet, so the offer is to create one, and rule 6 visibility rules
apply then).

Notes/decisions: nocal is a deliberate choice — the project owner chose "just
like the recipe says", and the DeepSeek recipe records no calibration data.
The largest remaining size lever after this unit is the 114.2 MB kept at
source precision (audio frontend 75.6 MB + vision frontend 32.6 MB), followed
by the token table, which sits at q4_K (75.5 MB) per the reference allocation;
both are candidates for a later unit with a measured ladder, not for this one.

4. Project facts a fresh session cannot re-derive

- What exists: .models/EmbeddingGemma-2-914f7f8 (pinned checkpoint, 1.5 GB,
  gitignored); docs/tensor-inventory.tsv (1376 tensors, name/dtype/shape/bytes/
  class); docs/allocation-plan.txt (tier allocation and byte math);
  RECIPE-deepseek41-vq.md (style and quality-bar reference); RECIPE.md (still
  the DeepSeek text, replaced in Phase D).
- Toolchain outside this repo: /data/YoungAi/gguf-tools/quantize (the VQ
  encoder, GPU kernels and byte accounting); /data/ds4-dfm-rs (the future
  loader — verified this session to have no gemma support at all: no gemma
  reference in crates/ or ds4.c, family list deepseek4|motif3|solar-open2|
  exaone-moe|dots3-note|qwen4exp|glm5-next|k2-horizon|iquest_q1, and no
  embeddings endpoint, so its VQ decoders are expert- and KV-shaped).
- Environment: RTX 4070 SUPER, 12282 MiB, sm_89; CUDA 13.3 driver and nvcc;
  max dynamic smem per block opt-in 101376; pageableMemoryAccess 0,
  concurrentManagedAccess 1, unifiedAddressing 1 (measured with a local probe).
  torch 2.13.0+cu130 sees the GPU; transformers 5.14.1 has no embedding_gemma2
  model (not in CONFIG_MAPPING; the model config pins 5.18.0.dev0), which is
  why the nocal path is the one this plan uses.
- Guardrails mount: /data/embeddinggemma2/.opengrok/rules/project-work-rules.md
  (link=yes exclude=yes listed=yes), excluded via .git/info/exclude.

5. Guardrails and open items

- Global guardrails in effect: the always-on core is ~/.opengrok/AGENTS.md
  (rules 0, 0.1, 1, 2, 4, 8, 15, 16, 17, 18, 20, 21, 22). The project-workflow
  rules 3, 5, 6, 7, 9, 10, 12, 13, 14, 19 and 23 live in
  ~/.opengrok/project-work-rules.md and load through the mount above.
- Repo hygiene: .gitignore excludes .models/ and graphify-out/; no remote is
  configured, so rule 23's push offer is an offer to create one.
- OPEN: measure the ladder (Phase C) before the full encode; no estimate may
  enter ladder-embeddinggemma2.txt.
- OPEN: decide the encode path (CPU first vs explicit-copy GPU) after the
  Phase B known-answer test — the GPU path is untrusted until that test passes
  on this card.
- OPEN: rule 12 graphify graph and rule 19 QA evidence before any push; rule
  10 has no Cargo.lock to scan because this repo is not Rust.
- OPEN: the model owner should confirm the scope reading used here (all three
  towers' dense matrices VQ'd, token table and frontends as skeleton), since
  that reading came from the quality answer, not an explicit scope choice.

6. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline 0b12d1be6e01368be910af96b0de4fe482641e4f..HEAD
    git diff --stat 0b12d1be6e01368be910af96b0de4fe482641e4f..HEAD
    git log --grep "Unit: .* complete" 0b12d1be6e01368be910af96b0de4fe482641e4f..HEAD

Trust git over this document's headings: continue with the first unit whose
work is not in the diff.

Staleness check:

    cd /data/embeddinggemma2 && git log --oneline 0b12d1be6e01368be910af96b0de4fe482641e4f..HEAD

A single docs commit carrying this handoff is expected; anything else must be
read before sections 2-5 are trusted.

This document is COMPLETE and self-contained: a new session can execute the
project end to end from this file alone. It carries the plan ahead, the facts
git cannot provide, and the guardrails. History is deliberately absent — git
log and the project memory entry are the historical record.
