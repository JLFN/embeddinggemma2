---
project: embeddinggemma2
plan_start_commit: d6da36680643c48c144536caf658e360b6d30d82
last_updated_commit: d591b9c7bdee07f998c52d041b0c40c01e68fafc
branch: dev
remote: 
handoff_written_at_context_usage: 20%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-19T18:22:43+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: dirty
---

Handoff + executable plan — EmbeddingGemma-2 VQ in ds4-rs (rewritten 2026-10-19: the plan exact, the dropped routes named)

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
- Do NOT: replay prior units' logs (git log is the record), re-explain global rules already in the loaded guardrails, restate anything the staleness check in section 8 already covers, or record acceptance evidence you did not personally observe this session.
- This file is the plan of record. docs/ds4-gemma-port-phases.md holds the row-by-row work list behind it; where the two disagree, THIS file wins and the phase doc is corrected in the same commit.

1. Session protocol (read first)

This project is a sequence of units; each unit ends with a verified, locally
committed step of the plan, and a push is offered, never taken (no remote is
configured on the outer repo).

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

2. Current state (2026-10-19, verified in this session; unit 17 has landed phase 7's P3.1 and P3.2, phase 5 is STOPPED by the owner's directive, and the CUDA unblocking plus this card's build wiring are DONE and committed)

- Workspace: /data/embeddinggemma2, branch dev, HEAD d591b9c (unit 14), tree DIRTY BY DESIGN — unit 15's and unit 17's files are uncommitted, and the exact stale inventory is section 8. Nested ds4-rs is CLEAN and FIVE local commits ahead of 5c11fc64: 9a6498da (the five sm_90 guards), ef79b5c2 (the server's missing GemmaVq arm), c7dd3ebb (this card's sm_89 arch recorded), bc353351 (P3.1, the dim-4 VQ device GEMM), 669a8777 (P3.2, the symmetric-mask attention kernel). Nothing is pushed.
- Unit 15 (phase 5, row X.7) is UNFINISHED and was STOPPED, not cancelled, by the owner: "stop the CPU we fix the GPU now" (2026-10-19 16:28Z) cut both legs of the restarted gate at 255 of 583 items so the machine went to the device path. Evidence and state: out/phase5-gate-stop-note-1628Z.txt, with the killed log kept as out/phase5-gate-stopped-1628Z.log and both partial raws renamed *-raw-stopped-1628Z.txt. A partial raw is not a leg (reuse needs 583 lines plus a matching .meta), so a restart is the full run — measured at ~1.4 s/item per leg on an idle host, i.e. about 15 minutes, against the ~2 hours the earlier contended run implied. The tables arrive with the `gate 5:` line.
- Unit 17 (phase 7, rows P3.1/P3.2) is the unit this section describes: two device kernels, each with its own gate, each committed in ds4-rs, with the evidence in docs/eg2vq-device-gate7.txt and the logs out/phase7-p31-gate.log / out/phase7-p32-gate.log. It also found and measured a defect of the CPU path's own decoder (the subnormal halving above, tools/eg2_subnorm.c) and two defects of its own kernels, which the gates caught.
- Unit 15's gate has now been interrupted twice: the first run started 15:08:12Z and was cancelled by a session interrupt at 16:03:34Z (3322 s) with each leg at 409 of 583 items and about 15 tokens/s under two-leg contention; the restart began 18:21Z and was stopped by the owner's directive at 18:28Z with each leg at 255 of 583 items, at ~1.4 s/item because the host was idle by then. Neither partial raw is reusable (a leg needs 583 lines plus a matching .meta), so the next restart is the full run. The restart's log is out/phase5-gate-stopped-1628Z.log and the note beside it records the state, the reason and the corrected duration.
- The frozen set (new this unit, committed only at unit 15's commit): tests/retrieval/scifact-test-{queries,pool,qrels}.tsv — the BEIR SciFact test split taken whole, 300 queries / 283 documents / 339 labelled pairs, digests 1247b1f4.. / caf498d9.. / 864141f9... Gate 5 re-runs the builder and refuses unless all three come back byte for byte.
- The harness (new this unit): tools/eg2_retrieval_set.sh (the builder), tools/eg2_phase5_gate.sh (gate 5: one ids build, two weight legs over one ids file, conversion, scoring, hashes), tools/eg2_oracle_to_jsonl.sh, tools/eg2_retrieval_score.sh (agreement plus recall@k/MRR with invariants), tools/eg2_retrieval_score_selftest.sh (8 hand-checked cases, passes).
- Already measured, reproducible in about a second (log out/phase5-phase4pair.log): the codec on the phase-4 pair — 768 mean cosine 0.960069342 / min 0.829288170 / p5 0.936426481 / p25 0.951163299 / p50 0.961872843 / p75 0.973416818, with 24 items below 0.95 and 1 below 0.90 at 768, and 0.976135232 mean at 128. docs/retrieval-cost-gate5.txt holds sections 1-5a and 8; 5b/5c and the result paragraph are marked PENDING until the legs land.
- The CUDA unblocking is DONE and committed (ds4-rs 9a6498da): the blocker was never a missing backend. It is FIVE sm_90-only instruction sites — four `cp.async.bulk.prefetch.L2.global` and one `mbarrier.try_wait.parity` — in three headers (cuda/glm_vq_q4p.cuh 1, cuda/glm_vq_flash.cuh 2, cuda/glm_vq_gemv.cuh 2), each arch-guarded and each keeping the sm_90+/Blackwell instruction verbatim. Counted in the artifacts rather than from prose: `grep -c '!defined(__CUDA_ARCH__)'` gives 1/2/2, and 9a6498da is 42 insertions and 0 deletions over those three files. Two corrections to what an earlier version of this handoff said: the count is five sites, not six instructions; and the guard spelling is not a deviation from both precedents — ds41_vq_prefill_mma.cuh:294 already spells it with the !defined clause (my sites take the spelling from it) while glm_vq_mma.cuh:85 uses the bare form, and ds41 has no fallback arm at all, so my sites take the fallback from glm_vq_mma.cuh:89's #else instead.
- Measured with nvcc 13.3 this session: `mbarrier.try_wait.parity` is REFUSED for sm_89 ("Feature 'mbarrier.try_wait.parity' requires .target sm_90 or higher") and compiles for sm_121a; `mbarrier.test_wait.parity` plus `prefetch.global.L2` compile for both.
- This card's build wiring is DONE and committed (ds4-rs c7dd3ebb): `make cuda CUDA_ARCH=sm_89` recorded `CUDA_ARCH := sm_89` in the gitignored .ds4-cuda-config.mk, replacing `native` (which resolves to the same card here — measured — but does not name it when the file is read on its own). A rebuild with no arch anywhere on the command line therefore compiles `-arch=sm_89` and yields ds4_cuda.1.sm_89.cubin, and the card is named in the tree's tracked files (README.md:91 and the Makefile help block). The whole target list builds from the record — `make cuda` exit 0, with ds4, ds4-server, ds4-bench and ds4-agent each carrying ten sm_89 cubins and each exiting 0 on `--help` — and the three device tests pass on this card (out/cuda-card-wiring.log, out/cuda-card-wiring-build.log). One object covering both this card and the Spark was measured impossible, so the arch stays a per-build choice (out/cuda-multiarch-probe.log). Building the whole list is also what surfaced ef79b5c2: this working copy could not build its server at all, because worker_run.rs's family match never got the GemmaVq arm its CLI twin received in 467af3a4. ds4_cuda.o is GITIGNORED, so the prebuilt objects were copied to out/cuda-objects-backup-sm121a before any of this (94 MB; `cuobjdump --list-elf` confirms the sm_121a cubin) — that copy is the rollback, not git.
- The gate-5 inputs, all verified this session: artifact sha256 5bcf2216.., baseline 0d84f1a9.., ids out/phase5-ids.tsv 9464c272.. (583 items / 107,029 tokens), items 66b956ad.., oracle d214f217.., and the phase-4 jsonl pair scored above is 9aefc2eb.. (baseline) / 0b5557cb.. (artifact).
- Nothing measured so far claims: an END-TO-END device result (phase 7's P3.1 and P3.2 kernels are gated one at a time; nothing composes them into a tower yet, so there is no device tokens/s and no VRAM figure — P3.3 owns both), a serving route (phase 6), the vision/audio towers (phase 8), a latency number for the phase-5 legs (both ran concurrently, so their secs= are contended), or an MTEB-comparable score (the pool is the 283 labelled documents, not the 5183-document corpus).
- Guardrails re-verified this session: all five P-1 checks return zero and `sha256sum -c SHA256SUMS` passes on all five entries. The frontmatter usage figure is the gauge here, not a fallback: signals.json reads contextTokensUsed 126471 / contextWindowTokens 1000000 = 12%.


3. Next unit plan — run EmbeddingGemma-2 in ds4-rs, phase by phase (2026-10-19)

Status: 1a, 2, 1b, 3 and 4 done (all 2026-10-19); 5 STOPPED by the owner's directive (the CPU legs were cut at 16:28Z to free the machine for the GPU; a restart is a full run, measured now at ~15 minutes on an idle host); 7's first two rows DONE this unit (P3.1 and P3.2, each gated and committed in ds4-rs), P3.3 OPEN; 6 and 8 ahead. Goal: the ds4 runtime loads our GGUF artifact (a GGUF carrying a
`gemma_vq` payload, per the owner's container decision) and emits the 768-dim
embeddings this model exists for, on the GPU, with the measurements that show the
port matches the pinned original. Runtime is Rust + C + CUDA; no Python anywhere,
no ONNX, no PyTorch (section 5 lists every route removed).

One rule for the whole plan: every mechanism below is READ from the cited file
before it is written, per project rule 0.2. Where the plan says "mirror", it means
open both files side by side and copy the shape, not paraphrase it.

Commit convention: each unit's final commit carries the trailer
"Unit: <N> complete" as its last body line, so unit boundaries are recoverable
with `git log --grep`.

PHASE 0 — the `gemma_vq` payload contract (phase-doc row X.8). DONE 2026-10-19, reading only: no reader, writer or encoder code was written for it.

- Landed: docs/ds4-gemma-payload-contract.md — the metadata keys, their GGUF types, the tensor naming, the payload byte layout, the recipe string, where shape and rank live, and the host/device split, each with a file:line citation.
- Read (all paths relative to /data/embeddinggemma2/ds4-rs unless another root is given):
  - crates/ds4-core/src/glm_vq.rs:276 (`glm_vq.codebooks`), :353 and :366 (`glm_vq.logical_shapes`, the mechanism that carries rank), :435 (`glm_vq.format_version`), :466 (`glm_vq.native_policy`).
  - crates/ds4-core/src/glm_vq/layout.rs and crates/ds4-core/src/glm_vq/release.rs (how a VQ release is described and validated).
  - crates/ds4-core/src/ling31_vq.rs:23-29 (ARCH / VARIANT / SOURCE_MODEL / RECIPE / RECIPE_V2) and its release module.
  - crates/ds4-core/src/identify.rs:183-186 (the metadata gate that makes a family self-identifying in GGUF).
  - ds4_glmvq_bind.inc:1-3 (the stated host/device split) and ds4_glmvq.h (the C-side types).
  - The payload carrier itself: /data/YoungAi/vq_fmt.h, our vendored copy ds41_vq_fmt.h:4-12 and its dim-generic decoder :73-105, and /data/YoungAi/gguf-tools/quantize/v41_to_gguf.c (the only shipping example of a ds4 VQ family being written into a GGUF, and the mapping of VQ bytes to GGUF tensor types).
  - The writer helper that existing tool uses: /data/YoungAi/src/common/ds4_gguf_write.h.
- Gate, both determinations settled by measurement, not assumed:
  1. Our kind-0 payload bytes ARE a DQVQ payload byte for byte. The 16 bytes at file offset 139664 of the artifact read `44 51 56 51 | 04 00 | 00 10 | 00 10 00 00 | 00 04 00 00` (`DQVQ`, dim 4, nc 4096, rows 4096, cols 1024), the length closes `vq_payload_bytes` exactly (1613841 B), and a fresh pack of the same tensor by the same binary produced the same payload digest, sha256 d9c02751e1757834af34a8235ea00a1a9af7db92ee7266541d62746068ed62bd. The artifact's own verify run reports all 1376 slots bit-identical to a fresh encode (docs/artifact-verify-summary.txt).
  2. Therefore phase 1b's artifact step is a re-wrap (I/O bound, byte-preserving), not a re-encode (measured at ~14.5 min of shard wall time plus the merge). A re-encode remains the fallback and was not needed.
- One correction found here and fixed in the same commit: the decoder is at ds41_vq_fmt.h:73-105, not :72-104 (this file and the phase doc cited it one line early).

PHASE 1a — the writer and the baseline GGUF (phase-doc rows X.5/X.5a). DONE 2026-10-19.

- Landed: tools/eg2_gguf.c (+ bin/eg2_gguf), tools/eg2_gguf_check.sh, the vendored
  tools/ds4_gguf_write.h (byte-identical to /data/YoungAi/src/common/ds4_gguf_write.h,
  sha256 5bfc5dd4fc8646709d521ad9a212aa0c6cba83cce2f0016baf0ff698364f1fff), the port's
  own reader dump ds4-rs/crates/ds4-core/examples/gguf_inventory.rs, RECIPE.md section 4b,
  docs/baseline-gguf-gate1a.txt (the evidence).
- Artifact: out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf, re-emitted in phase 2 with the
  tokenizer correction (557,949,600 B, sha256 0d84f1a9…; the pre-correction file is at
  out/baseline-noeos-variant.gguf, sha256 a982ca41…); 413 tensors, 219 bf16 copied
  verbatim + 194 f32 widened, 41 metadata keys (38 before the correction).
- Gate 1a PASSED, both halves: (a) one tensor round-tripped through the port's own
  reader first (output_norm.weight at abs 15765504, byte-identical), then all 413 —
  the writer's manifest and the ds4-rs reader's dump agree on every tensor's name,
  type, rank, dims, bytes and offset, 219 tensors are byte-for-byte the checkpoint's,
  194 are its exact BF16->F32 widening; (b) llama.cpp at the pin loads the file and
  prints a 768-dim embedding (built at /data/llama-anchor-build, 24 s wall).
- Read for it: conversion/gemma.py (the whole Gemma4/EmbeddingGemma2 chain: :131 the
  Gemma3 norm shift, :637-639 Gemma4's zero shift, :641-676 set_vocab, :676-733
  set_gguf_parameters, :813-832 the EmbeddingGemma2 overrides), gguf-py/gguf/vocab.py
  (:191-330 the special-token and merges extraction, :563-684 LlamaHfVocab, :84-110
  SpecialVocab.add_to_gguf), gguf-py/gguf/gguf_writer.py and constants.py,
  conversion/base.py:1220-1370 (the per-tensor dtype decision), src/llama-model.cpp
  (:1370-1400, :1455-1545, :3902-3912), src/llama-hparams.cpp:8-17 and .h:524-533,
  src/llama-vocab.cpp (:2065-2300 the tokenizer dispatch and merges, :2595-2640 the
  token/scores/type arrays), src/models/gemma-embedding2.cpp in full.
- FOUR corrections to docs/ds4-gemma-payload-contract.md section 7, each measured:
  1. 1-D tensors and *_norm.weight ship F32, not BF16 (conversion/base.py:1244-1246).
     An all-BF16 file loads and then aborts in the graph:
     binary_op: unsupported types: dst: f32, src0: f32, src1: bf16.
  2. attention.sliding_window is written DOUBLED: 2 x 512 = 1024
     (conversion/gemma.py:824-825), so with llama-hparams.h:524-533 the half-window
     is 512 (a 1025-wide span), not 256/513. Phase 4 and 7 must use the doubled
     value; the earlier "n_swa 512 -> +/-256" note in this file was wrong.
  3. attention.sliding_window_pattern is a BOOL ARRAY from layer_types
     (conversion/gemma.py:681-682), not the scalar 6; the loader prefers the array.
  4. The key set also carries attention.causal=false, rope.dimension_count 512 /
     _swa 256, gemma-embedding2.pooling_type 1 (MEAN) and general.file_type 32;
     tokenizer.ggml.pre is NOT written (the loader pins "gemma4" itself).
  Also settled: norm weights ship VERBATIM (Gemma4Model.norm_shift = 0.0, so no +1
  shift, unlike Gemma3), and the tokenizer is gemma4 BPE whose merges are REQUIRED
  (514,906, pair form, U+0101 escaping inert here), tokens = model.vocab keys by id,
  scores -1000.0, type BYTE/CONTROL/NORMAL.
- Baseline deviation, named: the baseline's token table ships BF16, not q8_0, so the
  baseline is the unquantized original and phase 5's delta against it is the whole
  cost of the recipe. q8_0 belongs to the artifact.

PHASE 1b — the artifact GGUF (phase-doc row X.5b): a re-wrap, per phase 0. DONE 2026-10-19 (unit 12). Evidence: docs/artifact-gguf-gate1b.txt.

- Landed: tools/eg2_gguf.c gained the `artifact` mode (a container reader, the plan, the re-wrap, the manifest), with the metadata path factored into emit_hparams/emit_tokenizer so both files share it — re-emitting the baseline after the refactor produced the identical sha256, which is the proof the change is byte-transparent. tools/eg2_artifact_check.sh is gate 1b. ds4-rs gained GgufFile::dump_kv_values and the example's --kv-values, so the metadata can be read back with values.
- Artifact: out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf, 437,508,416 B, sha256 5bcf2216…: arch gemma_vq, format_version 1, the recipe verbatim, 1376 tensors (461 vqblob type 42 at nd 1 ne={payload bytes}; 1 q8_0 type 8 ne={512,262144}; 914 bf16 type 30 at the checkpoint's rank), 42 KV, alignment 32. Tensor data 421,604,557 B; header, KV, table and padding 15,881,824 B.
- Gate 1b: every tensor present and the writer's manifest agreeing with the port's own reader on name/type/rank/dims/bytes/offset; all 1376 byte-identical to the container's payloads; all 914 verbatim tensors equal to the checkpoint's own bytes; all 461 VQ tensors beginning with the DQVQ magic; the metadata read back value by value, differing from the baseline's in exactly gemma_vq.format_version/.recipe (and carrying no general.file_type, per the precedent); bit-reproducible on a fresh run.
- Contract correction six, measured: the file's rank is never 0 — the 480 checkpoint scalars ship rank 1, ne {1}, because both readers reject nd 0 (ds4.c:3353, crates/ds4-core/src/tensors.rs:266, ggml.c:1786).
- A named property, not a defect: ds4's private type 42 is llama.cpp's q2_0, so the reference refuses the artifact at its first VQ tensor (and gguf-py identically); the anchor leg stays on the baseline, and the artifact's acceptance leg is the port's own reader.

PHASE 2 — the external anchor (phase-doc row X.5). DONE 2026-10-19. Evidence: docs/anchor-gate2.txt.

- Landed: tests/anchor/anchor-items.tsv (113 items, the checkpoint's own task prompts), tests/expected/anchor-embeddinggemma2.jsonl (the reference's vectors, sha256 7561254c…, bit-reproducible), tools/eg2_anchor.sh, tools/eg2_anchor_cmp.sh, tools/ref_vocab_probe.py.
- The sentence set is authored test prose, TAB-separated id / prompt_name / text, with 91 items carrying a prefix from config_sentence_transformers.json, 22 unprefixed ("Raw"), and 14 edge cases (empty input, one space, one character, digits, URL, four scripts, doubled spaces). Three items are the checkpoint README's own prose (README:48-100, :124-202, :255-335) at 1,175 / 1,221 / 1,398 tokens — they exist because every other item is far under the file's 512-token mask half-window, so an implementation that ignored the window would agree on all the short ones.
- The anchor is one model invocation per item (measured: the same input in a 113-item batch differs by up to 3.4e-4 per element from the same input alone, so a batched anchor would make every vector depend on the set). Per item it stores the exact input string, the reference's own token count, its sha256, and the 768 + 512 + 256 + 128 vectors, the shorter ones truncated and L2-renormalized per the model card (README:166-185).
- Gate: 113 items, all four widths, provenance in the file's first line (llama.cpp 31288c9c6, the model's sha256, the items' sha256, the exact command), and a from-scratch re-run reproducing the digest bit for bit. Invariants checked: all four widths are unit vectors to 1.1e-7 (the reference's print precision), two byte-identical inputs give bit-identical vectors, and every stored truncated vector equals the renormalized leading part of the stored 768 vector to 2e-7.
- The fifth contract correction came out of this unit, before the anchor could be built honestly: the file's tokenizer block was missing the three keys the reference's own SpecialVocab writes. Established by RUNNING that reference code (tools/ref_vocab_probe.py over the checkpoint → add_special_token {bos: true, eos: true, sep: false}, ids bos 2 / eos 1 / unk 3 / pad 0 / mask 4), not by re-reading it: this checkpoint's tokenizer.json post_processor is TemplateProcessing<single> = [<bos>, A, <eos>], so transformers appends <eos> and the file must say so (gguf-py/gguf/vocab.py:191-330; src/llama-vocab.cpp:3625 for what the flag does). The baseline was re-emitted with it (sha256 now 0d84f1a9…), gate 1a re-run and passing, and the load no longer warns.
- Both open decisions settled by measurement, both ways, over the same items: with the eos token the anchor's vectors are BIT-IDENTICAL to a run where the flag is absent and "<eos>" is typed into the text (max element delta 0.000e+00 at all four widths — so the flag is exactly the literal token), while simply dropping eos moves them by mean cosine 0.96 (min 0.84) and dropping the prefix by 0.87 (min 0.74). Those five lines are the reason neither choice was left to taste.
- Task-level sanity, so the numbers mean something: over the 12 retrieval pairs in the set, the anchor, the no-eos run and the no-prefix run all rank the paired document first for 12 of 12 queries (MRR 1.0000) at 768 and at 128. Twelve easy pairs are not a hard enough test to separate them; the labelled set that can is phase 5's.
- Limits, stated so phase 4 does not over-claim: the anchor is a second opinion, never the oracle (the port's own C reference path stays that), and the reference's own output moves by 3.4e-4 per element with batch shape, so the declared C-vs-anchor mean-cosine gate of 0.9999 has about three orders of margin; a claim tighter than 1e-3 per element would be a claim about llama.cpp's batching, not about the port.

PHASE 3 — the loader: identity, payload reader, layouts, registration. DONE 2026-10-19 (unit 13).

- Landed: crates/ds4-core/src/gemma_vq.rs (the arch gate and the `gemma_vq.format_version` gate, `validate_part` running from `tensors.rs::parse_one`, the per-payload validator, `validate_metadata`) and crates/ds4-core/src/gemma_vq/layout.rs (all 1376 specs, the VQ byte length computed from the tower architecture); `TensorError::GemmaVq` and the dispatch; `ModelFamily::GemmaVq` 16, `Variant::GemmaVqFlash` 19, `SHAPE_GEMMA_VQ`, the arch route, and the identify/layout/bind/validate/serving/serving_host/tok arms; `gguf_inventory --family`; the C identity (`DS4_MODEL_FAMILY_GEMMA_VQ` 16, `DS4_VARIANT_GEMMA_VQ` 19, `DS4_SHAPE_GEMMA_VQ`, the shape switch arm, the `weights_bind` arm) with ds4_eg2vq.h + ds4_eg2vq_bind.inc and the Makefile deps; tools/eg2_loader_check.sh as gate 3; and the shape's third copy in tests/parity/shape_c_oracle.c so `test-catalog-parity` compares all three.
- Gate passed: the reader's 1376-tensor dump equals the writer's manifest line for line with the family validator in the path, two loads are identical, and 8 tampered copies are refused by name (payload magic, dim, rows, zero row gain, nonfinite codebook, nonzero padding, truncation, metadata value). `cargo test -q -p ds4-core` passes (443 unit tests plus 29 integration targets) and `cargo check --workspace --all-targets` is clean. Evidence: docs/loader-gate3.txt.
- Two refusals by design, not stubs: `Vocab::load_specials` refuses this family by name until the tokenizer port lands (phase 4), and `eg2vq_bind` stops by name because an unhandled family falls through to the DeepSeek binding. Nothing can run the artifact through another family's mathematics.
- What this unit settled for phase 4 — read these before writing the tower:
  - The file's tensor names are the checkpoint's own, in ascending order, and the layout module is the name and shape authority; it already equals the file tensor for tensor.
  - `gemma_vq.attention.sliding_window` is 1024 (the converter's doubling of the checkpoint's one-sided 512, conversion/gemma.py:824-825), so the symmetric mask spans ±512, and the family's `n_swa_period` is 6 (5 sliding then 1 full, from the checkpoint's `layer_types`).
  - The tier is dim 4 / nc 4096 with 12-bit indices; `ds4vq_dequant_f32` (ds41_vq_fmt.h:73-105) is dim- and width-generic and its magic whitelist accepts exactly our `DQVQ` payloads, so the CPU decode needs no new decoder.
  - A VQ tensor is rank 1 with `ne` = payload bytes and its geometry lives in its own header, which Rust has already validated; the C side reads the record table (ds4_eg2vq.h) and never the payload.
- Named deviations from this document's own Land line, each justified in docs/loader-gate3.txt: crates/ds4-cli/src/lib.rs needed no family arm (the sites named here are the MTP speculation lists and this family has no MTP; its entry point is phase 6); the phase doc's `ds4_gemma_vq_*` file names are wrong and the real files are `ds4_eg2vq.{h,inc}`; and the layout landed as the family's shape authority rather than as a bind-only table.

PHASE 4 — the text tower on the CPU reference path (phase-doc rows P1.1-P1.20). DONE 2026-10-19 (unit 14). Evidence: docs/text-tower-gate4.txt.

- Landed: the tokenizer `ds4-rs/crates/ds4-core/src/tok/gemma_vq.rs` plus its driver `ds4-rs/crates/ds4-core/examples/gemma_vq_tok.rs`; the tower as the C reference forward `ds4-rs/ds4_eg2vq_graph.inc` with its bind `ds4-rs/ds4_eg2vq_bind.inc` (every type-42 payload decoded once into a 520 MiB f32 cache) and its oracle `ds4-rs/tests/parity/eg2vq_embd_c_oracle.c`; the Rust host's own leg as `ds4-rs/crates/ds4-core/tests/eg2vq_ref.rs`. Gate script: `tools/eg2_phase4_gate.sh` (its first leg is `tools/eg2_tok_check.sh`).
- The gate, as run (all four numbers in docs/text-tower-gate4.txt): the host's ids equal llama.cpp's own on all 113 items, id for id (7088 tokens, 0 mismatches); the C reference on the BASELINE weights reproduces the anchor at 768 mean cosine 0.99999681 / min 0.99998727 (declared floor 0.9999, met; 6.3e-04 worst element); and the same graph on the VQ artifact agrees with the anchor at mean 0.96007 / min 0.82945, which is the quantization's own gap (RECIPE.md section 5 measures a 0.9859 text-tower weight cosine) and phase 5's subject.
- The one thing to read before touching this: the anchor's model is the BASELINE GGUF, not the artifact — llama.cpp refuses the artifact at its type-42 tensors (ds4's private 42 is `vqblob`, llama.cpp's is `q2_0`). So a C-vs-anchor cosine is only meaningful on baseline weights, and the C bind deliberately carries both weight legs (artifact names and llama.cpp names) so that the declared floor and the container report are separable numbers. Judging the port by the artifact leg would blame the tower for the codec.
- Two corrections of the plan this unit made. (1) The phase doc's P1.8 said the symmetric half-window is 256 and the span 513; `n_swa` is the file's 1024, so the half is 512 and the span 1025 (corrected in the same commit). (2) The tokenizer is host-side only: no C gemma4 splitter was added, because the C oracle consumes ids and the tokenizer's oracle is the pinned reference itself (`tools/eg2_tok_check.sh`); the three `tests/tokenizer.rs` arms that used to say "no tokenizer yet" now say that.
- What this unit does NOT claim: no device path (the CUDA object does not build in this working copy), no route (phase 6), no tower encoders (phase 8), and no quality verdict — the artifact's 0.96 is what the codec costs on this task, which phase 5 measures properly against the baseline.

PHASE 5 — what VQ costs (phase-doc row X.7). UNFINISHED 2026-10-19 (unit 15): the harness is built and self-tested, but its two legs were cancelled mid-run at 409 of 583 items each and are NOT running; the retrieval numbers are NOT in hand. Restarting costs the full run, not the remainder.

- Act: the same sentence set and a frozen labelled retrieval set through our own engine, once with the baseline GGUF and once with the VQ artifact; report the cosine agreement distribution (mean, min, quantiles) and recall@k / MRR at 768 and at each MRL width.
- Gate: the numbers are committed with the artifact hashes they were measured on. This is the owner's "not less smart" answer, and it needs no third party.
- Landed this unit, uncommitted until its commit: tools/eg2_retrieval_set.sh (the set builder, byte-reproducible), tests/retrieval/scifact-test-{queries,pool,qrels}.tsv (300 queries / 283 documents / 339 labelled pairs; the digests are pinned inside the gate), tools/eg2_phase5_gate.sh (gate 5 itself), tools/eg2_oracle_to_jsonl.sh, tools/eg2_retrieval_score.sh (the scorer: agreement, recall@k, MRR@10, top-1-changed count, top1-top2 margin, and the invariants that make a wrong number fail loudly), tools/eg2_retrieval_score_selftest.sh (8 hand-checked cases) and docs/retrieval-cost-gate5.txt (sections 1-5a and 8 written; 5b/5c and the result paragraph PENDING).
- Resume it with `tools/eg2_phase5_gate.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf`. A leg is REUSED when its raw file exists AND its .meta records the same ids sha and the same oracle sha AND the raw still has 583 items, so a re-run after the first completion costs minutes instead of 78.
- Still to do in this unit, in order: (1) add the phase-4-pair agreement-only step to tools/eg2_phase5_gate.sh — never edit that script while it is running, bash re-reads it; the same scoring is already run and logged at out/phase5-phase4pair.log; (2) re-run the gate so the final log carries every table; (3) fill sections 5b/5c of docs/retrieval-cost-gate5.txt plus its result paragraph; (4) RECIPE.md section 5 (the codec's cost on a task), its section 6 item 3 and its section 7 retrieval row; (5) phase-doc row X.7 to DONE and section 9's sequence; (6) commit the outer repo with the trailer "Unit: 15 complete".
- The one number already in hand, for orientation only: on the phase-4 sentence pair the artifact's vectors agree with the baseline's at 768 mean cosine 0.960069342 / min 0.829288170 (out/phase5-phase4pair.log, which reproduces phase 4's own table and adds the quantiles). The retrieval delta — whether the ranking changed — is what this phase exists to produce and it is NOT in hand.

PHASE 6 — the entry point (phase-doc rows P2.1-P2.4)

- Land: a verb that emits one text's vector, then a `v1/embeddings` route beside the
  server's existing arms, plus the task-prompt selection from
  config_sentence_transformers.json and MRL truncation on request.
- Where, each site re-verified in this session (unit 15) rather than carried over:
  - CLI: `crates/ds4-cli/src/lib.rs:184` `parse_args` (the flag parser). The family
    match arms at `:441-452` and `:489-492` are the MTP/speculation lists — this
    family has no MTP, so those are NOT the sites. The crate has exactly three
    binaries, all `required-features = ["native"]` (`crates/ds4-cli/Cargo.toml:12-25`),
    so the verb is either a fourth `[[bin]]` (e.g. `src/bin/ds4-embed-rs.rs`) or a
    flag on the existing `ds4-rs` binary; the CLI's own build script compiles only
    linenoise (`crates/ds4-cli/build.rs`).
  - Route: `crates/ds4-server/src/serve.rs:1143`
    `fn prepare_client(cfg, inner, stream) -> Option<PreparedJob>` holds the whole
    dispatch — GET `/v1/models` `:1169`, `/v1/models/*` `:1183`, `/metrics` `:1194`,
    `/v1/stats` `:1212`, and the POST arm at `:1240` (first line:
    `let inference = shed_surface_for_path(&req.path).is_some();`). An embeddings
    route is a new POST arm beside it plus a surface:
    `crates/ds4-server/src/http.rs:283` `shed_surface_for_path`, `:293`
    `parse_surface_for_path`, over `enum WireSurface`
    (`crates/ds4-server/src/route/mod.rs:74`).
  - Capability arms: `crates/ds4-core/src/serving.rs:664` `serving_caps(family,
    variant)`, whose GemmaVq arm at `:791` already declares this encoder's caps
    (banks None, disk None, snapshot None, mtp None, host Any); and
    `crates/ds4-core/src/serving_host.rs:1100` `family_native_limit`, whose GemmaVq
    arm at `:1116` is FAMILY_NATIVE_MAX with a comment naming this phase.
  - Verified absent today: no pooling implementation and no embeddings route anywhere
    under `crates/` (grep: only `gemma_vq.rs:128-130` checking the metadata's
    `pooling_type`, and the caps comment at `serving.rs:788-789` naming this phase).
    The mean-pool + L2 composition exists only on the C reference path
    (`ds4_eg2vq_ref_embed`).
  - BLOCKER — CLOSED and committed this unit (row X.9; ds4-rs 9a6498da, c7dd3ebb, ef79b5c2). The tree
    links every Rust binary against `DS4_RS_LINK_OBJS = native/bridge/ds4_bridge.o
    $(CORE_OBJS)` (`Makefile:332`, used at `:355-357`), and on this platform `CORE_OBJS`
    carries the CUDA objects (`Makefile:78-80`: `ds4_cuda.o`, the eight `cuda/mmq/*.o`,
    `cuda/qwen38_ple.o`), so the host cannot be built without a CUDA object set this card can
    execute. The blocker was five sm_90-only instruction sites — not six, and not a missing
    backend: four `cp.async.bulk.prefetch.L2.global` (`cuda/glm_vq_flash.cuh:313` and `:351`,
    `cuda/glm_vq_gemv.cuh:50` and `:133`) and one `mbarrier.try_wait.parity.shared.b64`
    (`cuda/glm_vq_q4p.cuh:130`). The `ds41_vq_prefill_mma.cuh` and `glm_vq_mma.cuh` sites named
    in the earlier version of this bullet were ALREADY guarded by the tree itself and never
    belonged in the count; `grep -c '!defined(__CUDA_ARCH__)'` on the three edited headers gives
    1/2/2.
    - The owner's directive — "we need to get cuda support, it should be much faster" — settled
      this phase's choice in favour of (a); the guards are committed (9a6498da, 42 insertions, 0
      deletions over three headers), taking their spelling from
      `cuda/ds41_vq_prefill_mma.cuh:294` and their fallback arm from `cuda/glm_vq_mma.cuh:89`.
    - Measured with nvcc 13.3: the sm_90 forms are REFUSED for sm_89 ("requires .target sm_90 or
      higher") and compile for sm_121a; the fallbacks compile for both.
    - The arch is recorded now (c7dd3ebb): `.ds4-cuda-config.mk` reads `CUDA_ARCH := sm_89` in
      place of `native`, so a build with no flag on the command line is still card-native, and
      the whole target list builds from the record — `make cuda` exit 0, four Rust hosts, ten
      sm_89 cubins each (`out/cuda-card-wiring-build.log`).
    - One object covering both this card and the Spark is NOT available at the cost of a flag:
      ptxas refuses `.kind::mxf4` on sm_89 (`out/cuda-multiarch-probe.log`), so the arch stays a
      per-build choice and a sync to a Spark must run `make cuda-spark`.
    - Building the whole list surfaced a regression of ours and fixed it (ef79b5c2):
      `crates/ds4-server/src/worker_run.rs:16` never got the `GemmaVq` arm its CLI twin received
      in 467af3a4, so the server crate did not compile at all.
  - What (b) still gives, and why it is only the fallback now: `CPU_CORE_OBJS =
    ds4_cpu.o ds4_ple.o ds4_distributed.o` (`Makefile:81`) plus `ds4_bridge.o` is what the C
    oracles already build and run, so it stays the short path to a working CLI verb over phase
    5's CPU-only engine. With the arch fixed no CUDA object resists a sm_89 build, so phase 6
    does not need it; it is the documented fallback if one ever does again.
  - Rollback: the guarded headers are committed (9a6498da), so their rollback is the commit
    rather than a checkout. The object files are NOT tracked (`ds4_cuda.o` is gitignored): the
    prebuilt set is kept in `out/cuda-objects-backup-sm121a` (94 MB, sm_121a cubin), which is the
    rollback, not git. The arch record reverts with one `make cuda-spark` or `make cuda-generic`.
  - Next for phase 6: no CUDA work remains. The host links, all four Rust binaries build from the
    recorded arch, and `./ds4 --help` and `./ds4-server --help` exit 0. The verb and the route
    below are the work.
- Gate: the route returns the same vector as the CLI byte for byte for the same input
  and the same prompt name.

PHASE 7 — the device path (phase-doc rows P3.1-P3.3). Unit 17 (2026-10-19): P3.1 and P3.2 are DONE, P3.3 is what remains and it is the phase's gate. The owner's directive ("we need to get cuda support, it should be much faster") is why this phase runs before 6.

- DONE P3.1, the dim-4 VQ device GEMM: ds4-rs/cuda/eg2vq_vq.cuh (committed bc353351). Payload opened in-kernel with its own geometry re-checked, codebook in shared, 12-bit stream decoded per slab, 64-token x 64-row tile on 16 x 16 threads, 40 KB static shared, f32 in and f32 accumulation because the CPU reference path is f32. Gate: tests/test_eg2vq_vq.cu, `make test-eg2vq-vq`, out/phase7-p31-gate.log — rel RMS 4.01e-07 to 8.22e-07 against an FP64 dot of the payload's own decoded values at all six tower geometries, min cosine 1.000000000, wrong-geometry payload refused, 3.14 TFLOP/s at 2048x512x184 and 3.13 at 512x2048x1414 (single kernel). Two bugs of its own were caught by that gate (a slab-stride formula and a block-local unpack row index) and are recorded in docs/eg2vq-device-gate7.txt section 2b.
- DONE P3.2, the attention kernel: ds4-rs/cuda/eg2vq_attn.cuh (committed 669a8777). One block per query token, one warp per query head, the key range split across lanes, three passes (scores + max, expf + sum, weighted sum over output dims), scale exactly 1.0f, no causal mask, the SYMMETRIC ±512 mask (the tree's existing windowed kernel is trailing and refuses rows > 256, so this is new code). Gate: tests/test_eg2vq_attn.cu, `make test-eg2vq-attn`, out/phase7-p32-gate.log — rel RMS 5.77e-07 to 1.78e-06 and min per-(query,head) cosine 1.000000000 at all seven tower shapes, plus a measured mask edge (distance 512 sees the value row, 513 and 1100 blind, device output exactly zero). An item beyond 2048 tokens is refused by the launcher, not truncated.
- OPEN P3.3, the tower end to end on the device — the gate of this phase, and the one the owner's premise is settled by. STEP 1 DONE: ds4-rs/cuda/eg2vq_ops.cuh (commit 467f5431) carries the tower's remaining primitives, each mirroring ds4_eg2vq_graph.inc's own expression — the q8_0 token-row gather scaled by sqrt(n_embd), row and per-head RMS with the sum accumulated in double, NeoX RoPE over the whole head from a caller-built table, gelu-mul, the 768 projection with output_norm, the mean pool and L2. The RoPE table exists because --use_fast_math compiles sinf/cosf to the SFU approximations, whose fp32 argument reduction loses about 8e-05 rad at position 1414; the table uses the graph's own f32 angle accumulation with double trig, and the rotation is measured BIT-IDENTICAL to the reference. Gate: `make test-eg2vq-ops`, out/phase7-p33-ops-gate.log — RMS at f32 rounding (5.6e-08 to 6.0e-08), RoPE and the gather bit-identical, proj768+pool+L2 4.9e-07 with |v| = 1.000000, gelu-mul 2.39e-06 with its bound set to 1e-05 for that measured reason (fast-math tanhf is not libm's). STILL OPEN, in order: (2) the family adapter ds4_eg2vq_gpu.cuh on the ds4_glmvq_gpu.cuh pattern — payloads copied to the device ONCE per process (437 MB, through the ds4_eg2vq_record table the CPU binder already builds), plus the per-layer f32 norm vectors, the 24 layer scalars, the bf16 768 projection and the per-layer RoPE tables (~1.4 MB an item on the full layers); (3) the tower sequence, exactly the CPU graph's op order (ds4_eg2vq_graph.inc:169-337); (4) tests/test_eg2vq_gpu.cu — the artifact's real payloads and real item ids through the device tower against the CPU path's COMMITTED vectors (gate 5's own jsonl inputs, so no new reference is needed), with the parity bound stated before the run and tokens/s plus VRAM printed by the run; (5) the numbers recorded and the row closed.
- Measured constraint the P3.3 design inherits: the GEMM re-reads each payload once per 64-token tile, so a 1414-token document reads the whole 437 MB about 23 times; the mitigation is a larger token tile (one constant) or a K-split, and nothing has been measured about it at tower scale yet.
- Finding recorded in P3.1 that P3.3's parity gate must expect: the tree's vendored half converter `ds4vq_f16` (ds41_vq_fmt.h:38-43) returns exactly HALF the IEEE value for every subnormal f16 input, so our CPU reference path decodes subnormal codebook entries at half magnitude while the device kernel follows the file (tools/eg2_subnorm.c measures it on the real artifact: 8,893 of 7,553,024 codebook entries, 6.09756e-05 the largest, 2.28e-09 of the decoded weight L1 mass). The device follows the file's own reader; fixing the shared converter is a separate decision, not a silent edit.
- Not started: the DS4_CUDA_HAVE_MXF4-free device adapter's registration in ds4.c's family path (the glue that makes the *runtime* pick the device tower rather than the CPU one). Phase 6's entry point and this are the two integration steps left after P3.3.

- Land: a dim-4 VQ device kernel, a production bidirectional attention kernel with the symmetric ±n_swa/2 mask, and the tower end to end on the GPU with payloads consumed in-kernel (never expanded to dense — the 520 MiB f32 decode per leg is exactly what must not happen on the device path).
- Gate: GPU vectors equal the CPU path within a stated tolerance, plus a latency and a VRAM number printed by the run.
- The trap this project has already measured, which the design must avoid: the encoder's kernels were 146x faster than the CPU scalar loop on one isolated stage and still 1.25 to 3x SLOWER per tensor end to end, because the data path was PCIe (RECIPE.md section 8). The tower is a few hundred MB of weights on a 12,282 MiB card, so residency is not the constraint; per-operation host-to-device copies would be. So the measurement that settles "faster" is a weights-resident device run over the same ids as the CPU oracle, reporting cosine parity (against the CPU vectors, and against the frozen anchor on baseline weights) together with tokens/s and VRAM.

PHASE 8 — vision and audio towers (phase-doc rows P4.1-P4.4), only if the owner wants them

- Their graphs live in /data/llama.cpp/tools/mtmd/models/gemma4v.cpp and gemma4a.cpp (wired at tools/mtmd/clip.cpp:973-979 and :1115-1121), the tie-in projections are already verbatim BF16 in our container, and the projector variant for this checkpoint is not stated by the conversion (conversion/gemma.py:978, :1057) and must be read out before porting. Gate: the phase 4 comparison, per tower.

Order of work, and the first command of the next session:

0 (done) → 1a (done) → 2 (done) → 1b (done, the artifact GGUF exists) → 3 (done, the loader) → 4 (done, the text tower and the tokenizer) → 5 (STOPPED by the owner's directive of 16:28Z, see the item below) → 6 (the entry point; its CUDA blocker is CLOSED) → 7 (P3.1 and P3.2 DONE 2026-10-19, P3.3 the remaining gate) → 8 if the owner wants it.

The owner cut phase 5's CPU run mid-flight on 2026-10-19 16:28Z ("stop the CPU we fix the GPU now") and unit 17 has been the device path since. Phase 5 is therefore UNFINISHED, not failed: both legs stood at 255 of 583 items, no .meta was written, so neither raw is reusable and a restart is the whole run — out/phase5-gate-stop-note-1628Z.txt records the state, and the stopped artifacts are preserved as out/phase5-*-raw-stopped-1628Z.txt. The same note corrects the earlier "~2 hours" estimate: with the host idle the legs ran at ~1.4 s/item, so a full restart is ~15 minutes, not two hours (the earlier figure was measured while the CUDA objects were being built).

First commands of the next session, in this order:

1. Continue phase 7 at row P3.3, which is now the critical path: the device path's
   P3.1 (VQ GEMM) and P3.2 (attention) are done, gated and committed
   (`cd ds4-rs && make test-eg2vq-vq && make test-eg2vq-attn`, both exit 0), so the
   next artifact is cuda/eg2vq_ops.cuh plus ds4_eg2vq_gpu.cuh plus
   tests/test_eg2vq_gpu.cu, exactly as PHASE 7 lists them. Re-run the two existing
   gates first: they are cheap (seconds) and they are the floor P3.3 builds on.
2. Restart gate 5 when the machine is free (the owner deferred it, it is not
   cancelled): `tools/eg2_phase5_gate.sh <artifact.gguf> <baseline.gguf>` in the
   background — about 15 minutes idle, and it holds ~16 cores, so run it when the
   device work is not being measured. Legs are only reused when their .meta matches
   and the raw holds 583 items; the stopped raws do not qualify.
3. Re-confirm the standing gates before trusting any number: `sha256sum -c SHA256SUMS` (all five entries),
   `tools/eg2_loader_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf
   out/artifact-manifest.tsv` (1376 tensors, 8 refusals by name). Gate 4 is the third and costs about 20
   minutes; its result stands from unit 14 and was re-run then: `gate 4: PASSED (baseline C reference vs
   anchor, 768 mean cosine 0.999996809 >= 0.9999)`.
4. Then phase 6, whose first question is the one phase 6 records and which must be answered by attempting
   the link rather than by reasoning about it: does the Rust host resolve against the sm_89 CUDA objects, or
   does it need `CPU_CORE_OBJS` plus `ds4_bridge.o`? Its CUDA blocker is closed (row X.9), and it is also
   where the device tower's registration in ds4.c's family path belongs, so the *runtime* picks the device
   path rather than the CPU one.
5. Phase 5's remaining work (the tables, the evidence doc, RECIPE, row X.7) once its gate has run.

4. Reference map — the exact repos and files (pinned; verified in this session)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (the work lands here) | /data/embeddinggemma2/ds4-rs @ e235e0f8 plus two units of reader work (GgufFile::dump_kv_values + `gguf_inventory --kv-values`, then the `gemma_vq` loader: crates/ds4-core/src/gemma_vq.rs + gemma_vq/layout.rs, the family arms, ds4.c's identity, ds4_eg2vq.h/.inc), branch feat/ds41-dsml-carriage, gitignored, own git | every crates/ds4-core/src path named in section 3, ds4.c, ds4_glmvq*.inc/.h, cuda/*.cuh, tests/parity/, the Makefile |
| Form and metadata authority | /data/llama.cpp @ 31288c9c (b11723, 2026-10-18), MIT, read-only, source only (no build/ present) | src/models/gemma-embedding2.cpp (232 lines), src/llama-arch.cpp, src/llama-hparams.*, src/llama-model.cpp, src/llama-graph.cpp, src/llama-vocab.cpp, ggml/include/gguf.h, ggml/src/gguf.cpp, examples/embedding/embedding.cpp, conversion/gemma.py, conversion/base.py, gguf-py/gguf/** (vocab.py, gguf_writer.py, constants.py), tools/mtmd/** (phase 8) |
| The anchor build (phase 2; a build of the pin, never inside its tree) | /data/llama-anchor-build (CPU, built 2026-10-19) | bin/llama-embedding; the frozen vectors it produces are phase 2's deliverable |
| Writer and payload precedent | /data/YoungAi (reference toolchain, revision a4e6d31) | src/common/ds4_gguf_write.h, gguf-tools/quantize/v41_to_gguf.c, vq_fmt.h |
| Verification method | /data/ds4-dfm-rs (the live ds4 project; not modified from here) | AGENTS.md (retained `-c` oracles), tests/parity/, tests/ds4_proof.py, Makefile (the false-green guard) |
| The level to hit (phase 2) | our own committed data | tests/anchor/anchor-items.tsv (113 items, the checkpoint's task prompts, three of them long enough to exercise the mask) and tests/expected/anchor-embeddinggemma2.jsonl (the reference's vectors at 768/512/256/128, sha256 7561254c…); tools/eg2_anchor.sh regenerates it, tools/eg2_anchor_cmp.sh compares any two |
| The artifact to load (phase 3's input, and phase 4's) | our own committed recipe | out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf (sha256 5bcf2216…, 1376 tensors per docs/artifact-gguf-gate1b.txt), out/artifact-manifest.tsv (name/type/rank/dims/bytes/offset/role per tensor — the loader's gate joins against it), out/artifact-kv.txt (the metadata with values), and the container out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin (sha256 8902fb8a…) as the byte-verified record behind it |
| The loader and its gate (phase 3, landed) | our own code | ds4-rs/crates/ds4-core/src/gemma_vq.rs + gemma_vq/layout.rs, the family arms, ds4.c's `DS4_SHAPE_GEMMA_VQ`, ds4_eg2vq.h/.inc; tools/eg2_loader_check.sh (gate 3) and docs/loader-gate3.txt (the evidence) |
| The tower and its gate (phase 4, landed) | our own code | ds4-rs/crates/ds4-core/src/tok/gemma_vq.rs + examples/gemma_vq_tok.rs + tests/eg2vq_ref.rs; ds4-rs/ds4_eg2vq_graph.inc + ds4_eg2vq_bind.inc + tests/parity/eg2vq_embd_c_oracle.c (the CPU reference forward and its oracle); tools/eg2_tok_check.sh, tools/eg2_phase4_gate.sh, docs/text-tower-gate4.txt |
| The tokenizer's own oracle (phase 4; the family has no C splitter by design) | /data/llama.cpp at the pin, run through the anchor build | `llama-embedding ... --verbose-prompt` prints the reference's per-item token ids; tools/eg2_tok_check.sh regenerates that dump (out/tok-anchor-verbose.log) and diffs it against the Rust host's, id for id |
| The task-level cost of the codec (phase 5, in flight) | our own code and data | tools/eg2_retrieval_set.sh (the builder), tests/retrieval/scifact-test-{queries,pool,qrels}.tsv (300 queries / 283 documents / 339 pairs; digests pinned inside the gate), tools/eg2_phase5_gate.sh (gate 5), tools/eg2_oracle_to_jsonl.sh, tools/eg2_retrieval_score.sh (agreement, recall@k, MRR@10, top-1-changed, margin, invariants), tools/eg2_retrieval_score_selftest.sh (8 hand-checked cases), docs/retrieval-cost-gate5.txt (the evidence), out/phase5-* (gitignored) |
| The labelled data, and its pinned origin | BEIR SciFact (Thakur et al., NeurIPS 2021 Datasets and Benchmarks track), fetched once | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip, pinned in tools/eg2_retrieval_set.sh as ZIP_SHA 536e1444.. ; the extracted out/beir/scifact is gitignored, and the gate re-runs the builder from it whenever it is present |
| The CUDA portability pattern (row X.9, DONE: ds4-rs 9a6498da, c7dd3ebb) | the tree's own precedents, mirrored rather than invented | cuda/glm_vq_mma.cuh:85 (the bare `#if __CUDA_ARCH__ >= 900` with a `prefetch.global.L2` #else arm) and cuda/ds41_vq_prefill_mma.cuh:294 (the same guard spelled with the !defined clause and NO fallback arm) — neither precedent has both halves, so the five new sites take the spelling from one and the fallback from the other; the three headers now committed (cuda/glm_vq_q4p.cuh, cuda/glm_vq_flash.cuh, cuda/glm_vq_gemv.cuh; 42 insertions, 0 deletions); the arch record ds4-rs/.ds4-cuda-config.mk (`CUDA_ARCH := sm_89`, gitignored); the prebuilt-object backup out/cuda-objects-backup-sm121a (94 MB, the sm_121a cubin) |
| The model | /data/embeddinggemma2/.models/EmbeddingGemma-2-914f7f8 (1.5 GB, 15 files, gitignored) | config.json, per_layer_config, config_sentence_transformers.json, 1_Pooling/config.json, modules.json, tokenizer.{json,model}, tokenizer_config.json, README.md |
| Counter-example only | /data/llama-prism @ b10743 | not needed; read only to avoid copying its 1/sqrt(head_dim) attention scale |

5. Routes removed — do not resurrect any of them

- ONNX and onnxruntime, in any role (runtime, export, instrument). Owner, 2026-10-19: "remove this onnx i dont want it and no pytorch".
- PyTorch, transformers, sentence-transformers, and any Python ML stack: forbidden as a compute path by AGENTS.md P-1, and unusable here anyway — the installed transformers has no embedding_gemma module, and llama.cpp's converter imports torch at convert_hf_to_gguf.py:12. The isolated-venv item is dead.
- The DQVD container as the target format: retired by the owner's container decision. The 421 MB DQVD file remains only as the input to phase 1 (the re-wrap or re-encode source) and as the byte-verified record of the quantization itself.
- The `eg2_pack reconstruct` fallback (rebuild a safetensors directory from the DQVD container): DQVD-only and out of date the moment the GGUF exists; the GGUF plus its declared layout is the loadable form now.
- candle, burn, tch, and any Rust tensor framework: AGENTS.md P-1 item 5; the arithmetic is our own C/CUDA kernels.
- /data/llama-prism: a counter-example, never a source. Metal and macOS: not a target here (Linux, NVIDIA only).
- Not in scope unless the owner says otherwise: key-value checkpoints (phase-doc X.3), the distributed/agent surfaces of ds4, and the vision/audio towers (phase 8).

6. Project facts a fresh session cannot re-derive

- Workspace layout: RECIPE.md (the artifact recipe: sections 4 commands, 5 quality, 6 what the port must implement, 7 identity); RECIPE-deepseek41-vq.md (the ds4 VQ recipe we mirror); docs/ds4-gemma-port-phases.md (row list); docs/ds4-gemma-port-reference.md (the port map and the four-leg verification); docs/tensor-inventory.tsv and docs/encode-plan.tsv (the per-tensor truth); docs/artifact-verify-summary.txt; docs/ladder-embeddinggemma2.txt (the measured tier ladder); docs/{baseline-gguf-gate1a,anchor-gate2,artifact-gguf-gate1b,loader-gate3,text-tower-gate4,retrieval-cost-gate5}.txt (the evidence, one per unit); tests/retrieval/ (the frozen labelled set, phase 5); tools/{eg2_vq.c,eg2_pack.c,eg2_gpuprobe.cu,eg2_gguf.c} and tools/{eg2_gguf_check,eg2_artifact_check,eg2_loader_check,eg2_tok_check,eg2_phase4_gate,eg2_retrieval_set,eg2_phase5_gate,eg2_oracle_to_jsonl,eg2_retrieval_score,eg2_retrieval_score_selftest,eg2_anchor,eg2_anchor_cmp}.sh with bin/ (their binaries); out/ (all artifacts and logs, gitignored, including cuda-objects-backup-sm121a); AGENTS.md (project guardrails).
- The loader (phase 3): crates/ds4-core/src/gemma_vq.rs is the family (the arch and `gemma_vq.format_version` gate, `validate_part` from `tensors.rs::parse_one`, the per-payload validator, `validate_metadata`), gemma_vq/layout.rs is the shape authority for all 1376 tensors, and ds4-rs/ds4.c holds `DS4_SHAPE_GEMMA_VQ` with its third copy in tests/parity/shape_c_oracle.c. The file's rank census is 1354 rank 1 / 7 rank 2 / 13 rank 3 / 2 rank 4; a VQ payload's single dimension is its byte length, and the layout computes the same number from the tower architecture through `vq_payload_bytes`.
- The tower (phase 4): the tokenizer lives host-side in crates/ds4-core/src/tok/gemma_vq.rs (a `gemma4` pre-tokenizer: partition on the CONTROL/USER_DEFINED/UNKNOWN token set plus `</s>`, escape spaces to U+2581, split on newline runs, then the merge queue with the `<0xXX>` byte fallback; two named limits: invalid UTF-8 and `ignore_merges`). The arithmetic lives in ds4-rs/ds4_eg2vq_graph.inc with ds4_eg2vq_bind.inc: every type-42 payload is decoded once into a 520 MiB f32 cache (`ds4vq_dequant_f32`, geometry from the payload's own header, checked against the architecture), and `ds4_eg2vq_ref_embed(m, w, ids, n_ids, out768)` is the pool-and-normalize entry. The bind carries TWO weight legs by name — the artifact's HF names and the baseline's llama.cpp names — because the anchor's vectors came from the baseline; that is what makes the declared floor and the container report separable numbers. The oracle is CPU-only: `tests/parity/eg2vq_embd_c_oracle <gguf> <ids.tsv>` (its binary is gitignored, built by `make tests/parity/eg2vq_embd_c_oracle`), and it consumes the host's id dump from `cargo run -p ds4-core --release --example gemma_vq_tok -- <gguf> <items.tsv> <prompt-map.json>`.
- The artifact's quantizer: nocal VQ (codebooks solved on weights alone), dim 4, nc 4096, text-tower mean cosine 0.9859; the ladder's measured comparison is dim 4/nc 4096 at cos 0.9869 versus dim 8/nc 4096 at 0.9218 on the same representative tensor (docs/ladder-embeddinggemma2.txt).
- Model shape to load: 24 layers, hidden 512, 4 heads, 2 KV heads, head_dim 256; layers 5/11/17/23 full attention with head_dim 512 and 1 KV head; intermediate 2048; rms eps 1e-6; vocab 262,144; embedding dim 768; sliding window 512 in the checkpoint's one-sided sense — the FILE carries 1024 (`gemma_vq.attention.sliding_window`, the converter's doubling), and the symmetric mask is ±512; mean pool then L2 normalize.
- Machine: RTX 4070 SUPER (12,282 MiB, sm_89), CUDA 13.3, 28 CPUs. The Rust side is built with `cargo` plus `--features native` (every ds4-cli binary needs that feature) and the C side through the tree's Makefile — measured this unit: `/data/build/linux/build.sh -p /data/embeddinggemma2/ds4-rs` refuses the binaries (`target ds4-agent-rs in package ds4-cli requires the features: native`), so rule 5's documented iterative-development exception is what applies here. `make ds4.o` takes 39 s and `make ds4_cpu.o` 11 s. The CUDA objects are per-arch and the tree records its arch in .ds4-cuda-config.mk, which now names this card: `CUDA_ARCH := sm_89`, written by `make cuda CUDA_ARCH=sm_89` (ds4-rs c7dd3ebb), so a build here is card-native with no flag and the whole target list is green (out/cuda-card-wiring-build.log). The set that shipped was sm_121a-only (a Blackwell cubin, unusable here); the sm_121a regression object and the rollback copy of that set are in out/, and each fallback pair is legal on both archs (nvcc 13.3, measured).
- ds4-rs remotes are shared with the live ds4 project (origin Baekpica/ds4-dfm-rs, fork JLFN/ds4-dfm-rs): a push from here is shared state and needs confirmation. /data/ds4-dfm-rs itself is not to be modified from this workspace.
- Guardrails: global ~/.opengrok/AGENTS.md (core) + the mount at .opengrok/rules/project-work-rules.md (symlink; do not edit for project rules) + this repo's AGENTS.md. .gitignore excludes .models/, ds4-rs/, graphify-out/, graphify-rs-out/, out/ and *.o.

7. Guardrails and open items

- P-1 (no Python in the runtime, ever) and P-2 (references are citations or one-shot instruments, never dependencies; no ONNX, no PyTorch) are in this repo's AGENTS.md with their gate commands. All five checks return zero today, including ldd against the real ds4-server binary.
- Owner decisions in force: the container is GGUF + `gemma_vq` payload; the external anchor run was approved once the baseline GGUF existed (done, phase 2); quality is paramount — VQ is for compactness, never a licence to lose accuracy; and (2026-10-19) CUDA must actually be used — "we need to get cuda support, it should be much faster" — which settles phase 6's (a)/(b) choice as (a), the arch guard, and makes phase 7 a committed direction rather than an option.
- CLOSED (CUDA, row X.9, unit 16): the five guards are committed (9a6498da), the sm_89 object set and the host link are measured, the arch is recorded so every build here is card-native (c7dd3ebb), the whole target list builds green, the three device tests pass on this card, and the family arm the full build surfaced is fixed (ef79b5c2). What the row does NOT claim: any device-path speed, because no gemma device kernel exists at all — the tokens/s comparison is P3.3. Evidence: docs/cuda-sm89-port.txt.
- OPEN (owner): whether the vision/audio towers (phase 8) are wanted. OPEN: whether KV checkpoints mean anything for an embeddings family.
- OPEN (environment, (a) now CLOSED): (a) the CUDA object could not be rebuilt for sm_89 because of five sm_90-only instruction sites in the GLM-VQ headers; all five are guarded now, the object builds and the whole tree builds green, so this is no longer a blocker. (b) STILL OPEN: the canonical builder cannot build this tree at all until it can pass `--features native` — `/data/build/linux/build.sh -p /data/embeddinggemma2/ds4-rs` refuses the binaries (measured), so rule 5's documented iterative-development exception is what applies here. Fixing it means changing a builder shared with every other project on this machine, so it is proposed rather than done.
- OPEN (phase 5, IN FLIGHT and NOT RUNNING): the two legs were cancelled with the session interrupt at 16:03:34Z, each at 409 of 583 items. Then: a restart (the full run), the tables, the evidence doc, RECIPE, the phase-doc row and the commit. Residual note: a reused leg is trusted on its .meta plus the raw line count, so deleting a .meta forces an honest re-run — and a partial raw is never a leg.
- CLOSED this unit: the dim-4 VQ device kernel's scope — the owner's directive put it in scope (phase 7).
- OPEN (phases 1b, 3, 4): none — all three done, each with a reproducible gate (`tools/eg2_artifact_check.sh`, `tools/eg2_loader_check.sh`, `tools/eg2_phase4_gate.sh`). Residual notes rather than work: the tokenizer has no C counterpart by design; the C reference is not a second independent implementation, since the anchor is the only foreign one and that is why the declared floor is measured on baseline weights; and an out-of-range VQ index cannot be encoded at this tier (12-bit indices exactly cover nc 4096), so only a future non-power-of-two codebook would need an index scan.
- OPEN before any push: rule 12 graphify, rule 19 QA evidence and rule 10 OSV scan for ds4-rs (it carries a Cargo.lock); the outer repo has no remote, so publishing is an offer. The three guard edits in ds4-rs should be committed locally as their own unit before phase 7 builds on them.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git diff --stat d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git log --grep "Unit: .* complete" d6da36680643c48c144536caf658e360b6d30d82..HEAD
    ./bin/eg2_pack_cpu check out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin
    tools/eg2_artifact_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
        out/artifact-manifest.tsv out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin \
        .models/EmbeddingGemma-2-914f7f8 out/artifact-inventory.txt
    tools/eg2_loader_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
        out/artifact-manifest.tsv
    sha256sum -c SHA256SUMS
    jq -e . tests/expected/anchor-embeddinggemma2.jsonl > /dev/null && echo "anchor parses"
    # gate 4 (about 20 minutes; the token dump is reused, not regenerated)
    cd ds4-rs && make ds4_cpu.o tests/parity/eg2vq_embd_c_oracle >/dev/null && cd ..
    EG2_TOK_REF_LOG=out/tok-anchor-verbose.log tools/eg2_phase4_gate.sh \
        out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
        out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf \
        tests/anchor/anchor-items.tsv .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json \
        | tail -25
    git -C ds4-rs log --oneline -1 && git -C ds4-rs status --short
    # phase 5 (gate 5): the harness's own self-test, then where the legs stand
    tools/eg2_retrieval_score_selftest.sh
    tail -3 out/phase5-gate.log
    awk -F'\t' 'NF==3 && $2 ~ /^[0-9]+$/ {n++; s+=$2} END{print FILENAME, n, s}' \
        out/phase5-{baseline,artifact}-raw.txt        # 583 items / 107029 tokens per leg when done
    # the sm_89 CUDA object and the rollback copy of the prebuilt set
    tail -3 out/cuda-sm89-build.log
    cuobjdump --list-elf ds4-rs/ds4_cuda.o            # ds4_cuda.sm_89.cubin = the build succeeded
    ls -l out/cuda-objects-backup-sm121a/ds4_cuda.o   # the untracked rollback, 94 MB tree
    # the card's build wiring: the record, the guards, and the green full build
    cat ds4-rs/.ds4-cuda-config.mk                    # CUDA_ARCH := sm_89 = the record is in place
    grep -c '!defined(__CUDA_ARCH__)' ds4-rs/cuda/glm_vq_{q4p,flash,gemv}.cuh    # 1 2 2 = five sites
    tail -4 out/cuda-card-wiring-build.log.summary    # make cuda exit 0, four hosts, ten cubins each
    git -C ds4-rs log --oneline -3                    # 9a6498da, ef79b5c2, c7dd3ebb = the three local commits
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'

Trust git over this document's headings: continue with the first phase whose work is
not in the diff. Where an in-flight unit's work is not committed yet the diff is empty
by definition, so for unit 15 the sections above ARE the inventory: at the time of
writing the outer repo is still at d591b9c (unit 14) and carries unit 15 as untracked
plus modified — docs/retrieval-cost-gate5.txt, docs/cuda-sm89-port.txt, tests/retrieval/,
tools/eg2_{retrieval_set,phase5_gate,oracle_to_jsonl,retrieval_score,
retrieval_score_selftest}.sh, and edits to RECIPE.md, AGENTS.md,
docs/ds4-gemma-port-phases.md and this file. ds4-rs is CLEAN: the CUDA guards, the server
family arm and the build wiring are its three newest commits (9a6498da, ef79b5c2,
c7dd3ebb), local only, nothing pushed. Anything else in either diff must be read before
sections 2-6 are trusted. Note that the artifacts, the retrieval source, every
out/phase5-* file and the CUDA objects live under gitignored paths (out/, .models/,
ds4-rs/), so they never appear in a diff: they are identified by the digests in
SHA256SUMS, by the pins inside the gate scripts, and by section 2.

This document is COMPLETE and self-contained: a new session can execute the project
end to end from this file alone. It carries the plan ahead, the exact reference
files, the facts git cannot provide, and the guardrails. History is deliberately
absent — git log and the project memory entry are the historical record.
