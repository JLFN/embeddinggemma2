---
project: embeddinggemma2
plan_start_commit: d6da36680643c48c144536caf658e360b6d30d82
last_updated_commit: 07e0ccd786efe60be8f99620f4b3a31d650c4cfd
branch: dev
remote: https://github.com/JLFN/embeddinggemma2
handoff_written_at_context_usage: 9%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-19T21:14:02+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
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

2. Current state (2026-10-19, re-verified at the 07e0ccd boundary: every phase through 7 is landed and phase 5 closed in this unit, so both repositories are clean and the only unpushed work is this unit's two outer-repo commits)

- Workspace: /data/embeddinggemma2, branch dev, HEAD 4849880, tree CLEAN. The outer repo has no origin remote, only publish = https://github.com/JLFN/embeddinggemma2, whose dev was at 46741d5 before this unit and now carries this unit's three commits: the push ran 2026-10-19 with the owner's approval ("run the QA gate, then push"), fast-forward 46741d5..4849880, and the remote tip was verified afterwards with `git ls-remote --heads publish`. Nothing is unpushed. (An earlier version of this card said publish/dev stood at e16c820; the measured tip before this push was 46741d5.) Rule 19's QA gate did NOT run for the PREVIOUS push (that waiver is recorded in section 7) and DID run for this unit: deepseek-v4.1-CC-flash, qa-evidence/qa-report.md, verdict overall PASS, pushed with the handoff. Nested ds4-rs is CLEAN at 29873445 on feat/ds41-dsml-carriage, EIGHTEEN commits ahead of fork/feat/ds41-dsml-carriage, and its publish = https://github.com/JLFN/embeddinggemma2-ds4-rs carries that same tip at refs/heads/dev (measured this session), so publish it with `git push publish feat/ds41-dsml-carriage:refs/heads/dev` — plain `git push publish dev` fails, because the working copy's local dev tracks fork/dev and is behind. origin and the fork are untouched, and unit 15 changed NOTHING in ds4-rs.
- Unit 15 (phase 5, row X.7) is DONE, and it is this unit's subject: gate 5 ran to completion in 1000 s wall with both legs concurrent, and the codec's task-level cost is measured on labels this project did not write. Same CPU reference tower, both weight sources, one ids file: the vectors agree at mean cosine 0.956039571 / min 0.898186175 at 768 over the 583 items (0.972493258 mean at 128), and the rankings move by MRR@10 0.951778 -> 0.935819 and recall@1 0.888278 -> 0.861611, with recall@10 0.983333 -> 0.990000 and 23 of 300 queries changing their top-1 document against a baseline mean top1-top2 margin of 0.085722; at 128 dims truncation and quantization compound (MRR -0.029414, recall@5 0.954667 -> 0.915889). Evidence docs/retrieval-cost-gate5.txt (5b, 5c and the result paragraph filled in this unit), log out/phase5-gate.log.
- The two legs ran with the -O3 build of the oracle, not the Makefile's -O0 entry: EG2_PHASE5_ORACLE is new in tools/eg2_phase5_gate.sh, both legs use the SAME binary so the level cannot bias the artifact-versus-baseline delta, and the two builds were measured equivalent before use (min cos 1.000000000, max rel_rms 9.0e-07 over 24 items / 3297 tokens) at 21.87 s against 149.17 s — about 12 minutes per leg instead of ~80, which is what made the restart affordable (out/phase5-oracle-o0-vs-o3.log). Two false claims in that gate's header were found and fixed while doing it: the oracle is NOT single-threaded (716% CPU on a four-item run; the tower uses the tree's persistent pool, ds4.c:2238-2241), and the two legs share this host, so their secs= are contended wall time.
- What is left, in the order the plan sets: (1) phase 6's P2.3 remainder — an embedding-shaped serving cap and batching several items into one tower call; (2) the C CLI's tokenizer gap, which unit 19 recorded as by-design for now (the family's gemma4 BPE is Rust-side, so ds4-c refuses the artifact by name rather than duplicating the encoder); (3) phase 8, the vision and audio towers, only on the owner's word; (4) the owner's reading of the quality numbers above, for which no threshold was ever declared and none is implied. Nothing is in flight.
- Machine-facing facts, all re-verified this session and unchanged by this unit: artifact sha256 5bcf2216.., baseline 0d84f1a9.., ids out/phase5-ids.tsv 9464c272.. (583 items / 107,029 tokens), items 66b956ad.., the -O3 oracle a9f74d0b.., the phase-4 jsonl pair 9aefc2eb.. (baseline) / 0b5557cb.. (artifact), ds4-rs at 29873445, the CUDA arch record `CUDA_ARCH := sm_89` in the gitignored .ds4-cuda-config.mk, and the five sm_90-only guard sites (ds4-rs 9a6498da, 42 insertions over three headers).
- Unit 15's first slice (8ddd057) is what this unit completed: the frozen set tests/retrieval/scifact-test-{queries,pool,qrels}.tsv (300 queries / 283 documents / 339 labelled pairs; digests 1247b1f4.. / caf498d9.. / 864141f9.., re-derived from out/beir/scifact by the gate on every run and refused if they differ) and the harness tools/eg2_{retrieval_set,phase5_gate,oracle_to_jsonl,retrieval_score,retrieval_score_selftest}.sh, whose self-test's 8 hand-checked cases pass.
- Unit 19 (phase 6: the crash guard, the C embedding entry, the --embed verb and the /v1/embeddings route, all ds4-rs fa9d96a0, each gated — docs/eg2vq-embed-entry-gate6.txt), phase 7's device tower (units 17-18) and the CUDA arch work (row X.9, unit 16) are the units before this one, all landed and published. This unit touched none of them.
- Nothing measured to date claims: an MTEB-comparable score (the pool is the 283 labelled documents, not the split's 5183-document corpus, and no nDCG is computed); a serving-shaped latency or concurrency number (the tower runs one item at a time); a serving-cap arm for this family (P2.3); or anything about the vision and audio towers (phase 8). SIGNALS: a serving route and an embedding entry point are NO LONGER on this list — unit 19 closed both, and the codec's quality cost left it in this unit.
- Guardrails re-verified this session: all five P-1 checks return zero and `sha256sum -c SHA256SUMS` passes on all five entries. The frontmatter usage figure is the gauge here, not a fallback: signals.json read contextTokensUsed 82166 / contextWindowTokens 1000000 = 8% when this section was written.


3. Next unit plan — run EmbeddingGemma-2 in ds4-rs, phase by phase (2026-10-19)

Status: ALL EIGHT PHASES ARE NOW SETTLED OR CLOSED — 0 through 7 done (all 2026-10-19), with phase 6's P2.3 carrying one open item (an embedding-shaped serving cap and batching several items into one tower call) and its C-CLI tokenizer gap recorded as by-design for now. Phase 5 (gate 5, row X.7) CLOSED in unit 15: the quality answer exists (mean vector agreement 0.956039571 at 768, MRR@10 0.951778 -> 0.935819, recall@10 0.983333 -> 0.990000 on the frozen SciFact set — section 2 and PHASE 5 below). Phase 8 (the vision and audio towers) awaits only the owner's word. Goal: the ds4 runtime loads our GGUF artifact (a GGUF carrying a
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
- What this unit does NOT claim, as of that unit (each of the three was closed later and none is still open): no device path — the CUDA object did not build in this working copy then (row X.9, units 17-18 since), no route (phase 6, unit 19 since), no tower encoders (phase 8, still ahead), and no quality verdict — the artifact's 0.96 is what the codec costs on this task, which phase 5 has since measured against the baseline (PHASE 5 above).

PHASE 5 — what VQ costs (phase-doc row X.7). DONE 2026-10-19 (unit 15, whose first slice was 8ddd057 and whose completion is 07e0ccd): the harness, the frozen set and the numbers are all in hand, and the numbers are the row's deliverable.

- Act: the same CPU reference tower over both container paths on a frozen labelled retrieval set, reporting the cosine agreement distribution (mean, min, quantiles) and recall@k / MRR at 768 and at each MRL width. Ran in 1000 s wall for both legs concurrent, with the -O3 oracle (see the next bullet).
- Gate: the numbers are committed with the artifact hashes they were measured on. This is the owner's "not less smart" answer, and it needs no third party.
- THE NUMBERS, on BEIR SciFact's test split taken whole (300 queries / 283 documents / 339 labelled pairs; the gate re-derives the three frozen files from out/beir/scifact and refuses unless they come back byte for byte): vectors agree at mean cosine 0.956039571 / min 0.898186175 at 768 over the 583 items (p1 0.922665454, p5 0.937516751, p25 0.951416471, p50 0.957530714, p75 0.962775245; worst item 6421792), 0.972493258 at 128; 583 below 0.99, 124 below 0.95 and 1 below 0.90 at 768, and at 128 seven below 0.95 and none below 0.90. Retrieval: MRR@10 0.951778 -> 0.935819 and recall@1 0.888278 -> 0.861611 at 768, with recall@5 0.979333 -> 0.975333 and recall@10 0.983333 -> 0.990000 (the artifact finds the labelled evidence inside ten slightly more often); 23 of 300 queries changed top-1 against a baseline mean top1-top2 margin of 0.085722. At 512 the deltas shrink (-0.012917 MRR), at 256 and 128 they grow (-0.020496, -0.029414 MRR; recall@5 0.954667 -> 0.915889 at 128), so MRL truncation and the codec compound. No threshold is declared anywhere — the numbers and their hashes are the answer, and the reading is the owner's.
- Landed with it: tools/eg2_retrieval_set.sh (the set builder, byte-reproducible), tests/retrieval/scifact-test-{queries,pool,qrels}.tsv (digests pinned inside the gate), tools/eg2_phase5_gate.sh (gate 5 itself, now with the EG2_PHASE5_ORACLE selector), tools/eg2_oracle_to_jsonl.sh, tools/eg2_retrieval_score.sh (the scorer: agreement, recall@k, MRR@10, top-1-changed count, top1-top2 margin, and the invariants that make a wrong number fail loudly), tools/eg2_retrieval_score_selftest.sh (8 hand-checked cases), docs/retrieval-cost-gate5.txt (sections 1-8 complete; 5b/5c and the result paragraph filled in this unit), and the RECIPE sections 4e/5/5b/6 item 3/7 rows.
- How to re-run it: `EG2_PHASE5_ORACLE=$PWD/out/eg2vq_embd_c_oracle_o3 tools/eg2_phase5_gate.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf`. A leg is REUSED when its raw file exists AND its .meta records the same ids sha and the same oracle sha AND the raw still has 583 items, so a re-run after the first completion costs minutes instead of ~12. The -O3 oracle is built by the cc line in RECIPE command 24 and was measured equivalent to the Makefile's -O0 entry before use (min cos 1.000000000, max rel_rms 9.0e-07 over 24 items / 3297 tokens, 21.87 s against 149.17 s; out/phase5-oracle-o0-vs-o3.log) — the equivalence matters because the -O0 entry costs ~80 minutes per leg.
- One number from the phase-4 sentence pair, unchanged and still reproducible in about a second: 768 mean cosine 0.960069342 / min 0.829288170 with 24 items below 0.95 and 1 below 0.90 (out/phase5-phase4pair.log). The retrieval-set numbers above are lower (0.9560), and nothing here attributes that to document length specifically: the two sides of this set agree almost equally (query 0.958537658, document 0.953391422 at 768).

PHASE 6 — the entry point (phase-doc rows P2.1-P2.4).  Unit 19 (2026-10-19):
DONE for P2.1, P2.2 and P2.4, PARTIAL for P2.3.  ds4-rs fa9d96a0 is the whole
change (1138 insertions, 0 deletions over 16 files) and the evidence is
docs/eg2vq-embed-entry-gate6.txt.

- The crash is gone, measured.  `./ds4 -m <artifact> -p hello -n 1` exited 139
  with a core dump on BOTH backends (twice each); it now prints one refusal --
  "family gemma_vq (EmbeddingGemma-2) is an embedding tower ... there is no
  next-token distribution to decode.  Use the embedding path: POST
  /v1/embeddings on the server, or --embed on the CLI." -- and exits 1, no core
  dump (out/phase6-guard-rs-cuda.log, out/phase6-guard-rs-cpu.log).  The guard
  is one function (ds4_report_embedding_family_refusal) called from three sites:
  ds4_engine_generate_argmax, ds4_session_create and ds4_engine_boot_prewarm,
  plus the same refusal at the Rust CLI's own command line.
- The entry point: ds4_engine_embed_gemma_vq (ds4.c, declared ds4.h) beside
  ds4_engine_vision_encode_memory -- the tree's own template for a
  non-generation engine operation.  It binds the device plan and its workspace
  ONCE per engine (eg2vq_device_ensure; the workspace grows in 256-token steps
  and never shrinks) and uses the device tower on a graph backend, the same
  ds4_eg2vq_ref_embed the CPU paths gate, otherwise.  Refuses by name: an empty
  item, an item past DS4_EG2VQ_MAX_TOKENS (2048, static-asserted equal to the
  attention kernel's EG2VQ_ATTN_KEYS_MAX), a token id outside the embedding
  table (whose row count is read from the table), a non-gemma_vq engine.  Bridge
  slot: ds4_bridge_eg2vq_has_embed / _embed (the v41 has_*/run pair's shape).
- The CLI verb: `ds4 -m <artifact> --embed TEXT [--task NAME] [--dims N]`, with
  --embed-file / --embed-stdin / --prompt-map, JSON on stdout.  Gated by
  tools/eg2_embed_cli_gate.sh, which takes the frozen set's RAW TEXT through the
  verb and compares it against the device parity raw the phase-7 harness produced
  from the set's IDS -- two independent drivers: cos 1.000000000 on all 12 items
  (6 queries, 6 documents of 803-1414 tokens), rel_rms <= 5.745e-07 against a
  declared 1e-03, and token counts identical to the harness's own ids file.
- The route: GET /health and POST /v1/embeddings (input string or array, model,
  task default Retrieval-query, dimensions 768/512/256/128, encoding_format
  float), OpenAI-compatible, carried to the owner thread through PreparedJob::
  embed + ParsedRequest::embedding_carrier rather than a new ReqKind (that would
  have rippled through eight match sites).  Gate tools/eg2_embed_route_gate.sh:
  /health as above, 768 floats, dimensions 256 with L2 norm 1.000000045, a bad
  task as a 400 listing the checkpoint's names, and route-vs-CLI cos
  1.000000000000; checked once more against the device raw directly (cos
  1.000000000).
- What is NOT done: P2.3's remainder (no embedding-shaped serving-cap row, no
  batching several items into one tower call), the C CLI's tokenizer gap (ds4-c
  needs DeepSeek's special tokens in its legacy vocab loader and the family's
  gemma4 BPE is Rust-side -- porting it to C would duplicate the encoder), and
  QUALITY, which that unit did not touch at all: gate 5 / row X.7 was still open
  then and is CLOSED NOW by unit 15 (the numbers are in PHASE 5 above).

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

PHASE 7 — the device path (phase-doc rows P3.1-P3.3). Units 17-18 (2026-10-19): COMPLETE. P3.1 and P3.2 landed in unit 17; P3.3's steps 2-4 landed in unit 18 (ds4-rs 4892fb4b) and closed the phase. The owner's directive ("we need to get cuda support, it should be much faster") is why this phase ran before 6, and it is now answered with a measurement rather than an argument: 9.9x per item on the frozen set's short queries and 18.1x on retrieval-length documents, against the same CPU source compiled with the tree's own -O3 flags.

- DONE P3.1, the dim-4 VQ device GEMM: ds4-rs/cuda/eg2vq_vq.cuh (committed bc353351). Payload opened in-kernel with its own geometry re-checked, codebook in shared, 12-bit stream decoded per slab, 64-token x 64-row tile on 16 x 16 threads, 40 KB static shared, f32 in and f32 accumulation because the CPU reference path is f32. Gate: tests/test_eg2vq_vq.cu, `make test-eg2vq-vq`, out/phase7-p31-gate.log — rel RMS 4.01e-07 to 8.22e-07 against an FP64 dot of the payload's own decoded values at all six tower geometries, min cosine 1.000000000, wrong-geometry payload refused, 3.14 TFLOP/s at 2048x512x184 and 3.13 at 512x2048x1414 (single kernel). Two bugs of its own were caught by that gate (a slab-stride formula and a block-local unpack row index) and are recorded in docs/eg2vq-device-gate7.txt section 2b.
- DONE P3.2, the attention kernel: ds4-rs/cuda/eg2vq_attn.cuh (committed 669a8777). One block per query token, one warp per query head, the key range split across lanes, three passes (scores + max, expf + sum, weighted sum over output dims), scale exactly 1.0f, no causal mask, the SYMMETRIC ±512 mask (the tree's existing windowed kernel is trailing and refuses rows > 256, so this is new code). Gate: tests/test_eg2vq_attn.cu, `make test-eg2vq-attn`, out/phase7-p32-gate.log — rel RMS 5.77e-07 to 1.78e-06 and min per-(query,head) cosine 1.000000000 at all seven tower shapes, plus a measured mask edge (distance 512 sees the value row, 513 and 1100 blind, device output exactly zero). An item beyond 2048 tokens is refused by the launcher, not truncated.
- DONE P3.3, the tower end to end on the device (unit 18, ds4-rs 4892fb4b), the phase's gate and the one the owner's premise is settled by. STEP 1 (unit 17): ds4-rs/cuda/eg2vq_ops.cuh (commit 467f5431) carries the tower's remaining primitives, each mirroring ds4_eg2vq_graph.inc's own expression — the q8_0 token-row gather scaled by sqrt(n_embd), row and per-head RMS with the sum accumulated in double, NeoX RoPE over the whole head from a caller-built table, gelu-mul, the 768 projection with output_norm, the mean pool and L2. The RoPE table exists because --use_fast_math compiles sinf/cosf to the SFU approximations, whose fp32 argument reduction loses about 8e-05 rad at position 1414; the table uses the graph's own f32 angle accumulation with double trig, and the rotation is measured BIT-IDENTICAL to the reference. Gate: `make test-eg2vq-ops`, out/phase7-p33-ops-gate.log — RMS at f32 rounding (5.6e-08 to 6.0e-08), RoPE and the gather bit-identical, proj768+pool+L2 4.9e-07 with |v| = 1.000000, gelu-mul 2.39e-06 with its bound set to 1e-05 for that measured reason (fast-math tanhf is not libm's). STEPS 2-4 DONE (unit 18), in the order they were needed: (2) the family adapter ds4-rs/ds4_eg2vq_gpu.cuh, included once by ds4_cuda.cu on the ds4_glmvq_gpu.cuh pattern — the text tower's 217 payloads (58,658,409 B / 55.94 MiB, NOT 437 MB: that is the whole artifact file, and the tower uses 217 of the artifact's 461 payloads, the rest being the audio and vision towers; all 461 are 237,289,117 B) and its 196 aux slots bound into two device arenas ONCE per process through eg2vq_device_tables(), which fills the record/aux contract from the CPU binder's own abs_offset and dim[0], with the payload header, the file span and the GEMM tile divisibility (rows%64, cols%256) all re-checked at bind, the q8_0 token table (136.02 MiB) and the bf16 768 projection read in-kernel, the norm vectors and scalars widened to f32 on the host (the CPU path reads f32), a 17-plane workspace sized by max_tokens, and the RoPE table pair built with the graph's own f32 angle accumulation plus double trig; (3) the tower sequence, ds4_eg2vq_graph.inc:169-370's op order verbatim — the adapter adds no arithmetic, only the order, the buffers and the binding; (4) tests/parity/eg2vq_embd_gpu_oracle.c, the artifact's real payloads and real frozen-set ids through the device tower (compiled WITHOUT -DDS4_NO_GPU and linked against the CUDA objects), with the bounds cos >= 0.999999 / rel_rms <= 1e-03 declared before the run. Run A, the 300-item query partition / 8,059 tokens: min cos(device,cpu) 0.999999950, max rel_rms 3.149e-04, the CPU leg reproducing the committed oracle raw at cos 1.000000000 / rel_rms 0.000e+00 over 255 comparable items, device 5.39 s = 0.018 s/item = 1494.8 tokens/s (steady state 1499.0), and 414.0 MiB by cudaMemGetInfo delta = payload 55.94 + weights 137.04 + workspace 212.01 MiB (plus a 6.29 MiB RoPE pair) against the CPU leg's own 520.0 MiB dense f32 decode, which the device leg never builds. Run B, the twelve longest documents (803-1414 tokens): min cos 0.999999876, max rel_rms 4.978e-04, device 3.68 s = 2733.1 tokens/s, the 1414-token item 564.94 ms = 2503 tokens/s. THE SPEEDUP IS QUOTED AGAINST THE FAIR LEG: the gate prints 65.1x and 125.2x against its own -O0 reference entry, while the same CPU source at the tree's CFLAGS (-O3 -ffast-math -march=native, built to out/eg2vq_embd_c_oracle_o3) takes 53.53 s and 66.54 s and agrees with the -O0 leg at cos 1.000000000 / rel_rms 9.5e-07 — so the honest ratios are 9.9x and 18.1x. Both runs are cross-checked by an independent instrument (tools/eg2_parity_check.awk, -F'\t' mandatory; run A exact, run B to the last printed digit). (5) recorded in docs/eg2vq-device-gate7.txt sections 5-6, with the two corrections this unit made: the 437 MB payload figure, and the cross-check tool's own whitespace-vs-tab bug.
- Measured constraint the P3.3 design inherited, now measured AT TOWER SCALE: the GEMM re-reads each payload once per 64-token tile, so a 1414-token document reads the tower's 55.9 MiB of payload (58,658,409 B — NOT 437 MB, which is the whole artifact file) about 23 times, roughly 1.35 GB, in 564.94 ms on the device — an achieved payload rate of 2.4 GB/s, which is arithmetic on the payload size and the item's wall time rather than a traffic counter, and enough to say the item is not bandwidth-bound. Nothing has needed the mitigation (a larger token tile, one constant).
- Finding recorded in P3.1 that P3.3's parity gate must expect: the tree's vendored half converter `ds4vq_f16` (ds41_vq_fmt.h:38-43) returns exactly HALF the IEEE value for every subnormal f16 input, so our CPU reference path decodes subnormal codebook entries at half magnitude while the device kernel follows the file (tools/eg2_subnorm.c measures it on the real artifact: 8,893 of 7,553,024 codebook entries, 6.09756e-05 the largest, 2.28e-09 of the decoded weight L1 mass). The device follows the file's own reader; fixing the shared converter is a separate decision, not a silent edit.
- Still not started, and now phase 6's first two jobs: the device tower's registration in ds4.c's family path (the glue that makes the *runtime* pick the device tower rather than nothing at all — today nothing calls it), and the refusal guard that makes the decode path say "this checkpoint has no generation head" instead of segfaulting. Both measured this unit: the family binds at ds4.c:10287 but no runtime function calls ds4_gpu_eg2vq_embed; and `./ds4 -m <artifact> -p hello -n 1` dies with SIGSEGV (exit 139, core dumped), twice, identically with --cpu, because family 16 has a shape case (ds4.c:2881) and a bind arm but no forward/session arm, so the generic prefill reaches embed_token_f16 and dereferences w->token_embd — a slot gemma_vq never populates, since its token table is bound under w->eg2vq (ds4_eg2vq_bind.inc:233-239).

- Land: a dim-4 VQ device kernel, a production bidirectional attention kernel with the symmetric ±n_swa/2 mask, and the tower end to end on the GPU with payloads consumed in-kernel (never expanded to dense — the 520 MiB f32 decode per leg is exactly what must not happen on the device path).
- Gate: GPU vectors equal the CPU path within a stated tolerance, plus a latency and a VRAM number printed by the run.
- The trap this project has already measured, which the design must avoid: the encoder's kernels were 146x faster than the CPU scalar loop on one isolated stage and still 1.25 to 3x SLOWER per tensor end to end, because the data path was PCIe (RECIPE.md section 8). The tower is a few hundred MB of weights on a 12,282 MiB card, so residency is not the constraint; per-operation host-to-device copies would be. So the measurement that settles "faster" is a weights-resident device run over the same ids as the CPU oracle, reporting cosine parity (against the CPU vectors, and against the frozen anchor on baseline weights) together with tokens/s and VRAM.

PHASE 8 — vision and audio towers (phase-doc rows P4.1-P4.4), only if the owner wants them

- Their graphs live in /data/llama.cpp/tools/mtmd/models/gemma4v.cpp and gemma4a.cpp (wired at tools/mtmd/clip.cpp:973-979 and :1115-1121), the tie-in projections are already verbatim BF16 in our container, and the projector variant for this checkpoint is not stated by the conversion (conversion/gemma.py:978, :1057) and must be read out before porting. Gate: the phase 4 comparison, per tower.

Order of work, and the first command of the next session:

0 (done) → 1a (done) → 2 (done) → 1b (done, the artifact GGUF exists) → 3 (done, the loader) → 4 (done, the text tower and the tokenizer) → 5 (DONE 2026-10-19, unit 15: the gate ran with the -O3 oracle and the quality answer is committed — MRR@10 0.951778 -> 0.935819 and recall@10 0.983333 -> 0.990000 at 768) → 6 (DONE 2026-10-19, unit 19: the crash guard, the C embedding entry, the --embed verb and the /v1/embeddings route, each gated; P2.3 partial) → 7 (COMPLETE 2026-10-19: units 17 and 18, P3.1, P3.2 and P3.3 all closed) → 8 if the owner wants it.

Phase 5's history, kept because it explains the two artefacts on disk: the owner cut its CPU run mid-flight on 2026-10-19 16:28Z ("stop the CPU we fix the GPU now") and the device path ran instead. Both legs stood at 255 of 583 items and no .meta was written, so neither raw was reusable — out/phase5-gate-stop-note-1628Z.txt records that state and the stopped partials are preserved as out/phase5-*-raw-stopped-1628Z.txt (they are NOT the raws the numbers came from; those are out/phase5-{baseline,artifact}-raw.txt with .meta files, oracle sha a9f74d0b..). The restart cost ~12 minutes per leg rather than the ~80 the -O0 entry needs, which is why it was affordable at all.

First commands of the next session, in this order:

1. NOTHING IS IN FLIGHT: verify that before doing anything — `git status --short` in
   both repositories (outer at 07e0ccd plus this handoff commit, ds4-rs at 29873445),
   then the standing gates below. The next work item is a decision, not a task.
2. The owner's decision on the quality numbers (unit 15's, section 2 and PHASE 5
   above): MRR@10 falls 1.6 points at 768 while recall@10 does not, and no threshold
   is declared. Nothing in the plan proceeds past this until the owner reads them.
3. Phase 6's remainder, only if the owner asks: the C CLI's tokenizer gap
   (ds4-c needs an embedding-side tokenizer; the family's gemma4 BPE is Rust-side
   today, so porting it duplicates the encoder — the shipped binaries are Rust),
   and P2.3's serving-cap row plus batching several items into one tower call.
4. Re-confirm the standing gates before trusting any number: `sha256sum -c SHA256SUMS` (all five entries),
   `tools/eg2_loader_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf
   out/artifact-manifest.tsv` (1376 tensors, 8 refusals by name). Gate 4 is the third and costs about 20
   minutes; its result stands from unit 14 and was re-run then: `gate 4: PASSED (baseline C reference vs
   anchor, 768 mean cosine 0.999996809 >= 0.9999)`.
5. Phase 5 may be re-run cheaply if anything about the numbers is questioned: the two
   legs exist with matching .meta files, so a re-run reuses them (minutes), and the
   -O3 oracle's equivalence to the -O0 entry is itself measured in
   out/phase5-oracle-o0-vs-o3.log (or re-measure it on the 24-item subset first if
   the compiler has changed since).

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
| The task-level cost of the codec (phase 5, DONE in unit 15) | our own code and data | tools/eg2_retrieval_set.sh (the builder), tests/retrieval/scifact-test-{queries,pool,qrels}.tsv (300 queries / 283 documents / 339 pairs; digests pinned inside the gate), tools/eg2_phase5_gate.sh (gate 5, with the EG2_PHASE5_ORACLE selector), tools/eg2_oracle_to_jsonl.sh, tools/eg2_retrieval_score.sh (agreement, recall@k, MRR@10, top-1-changed, margin, invariants), tools/eg2_retrieval_score_selftest.sh (8 hand-checked cases), docs/retrieval-cost-gate5.txt (the evidence, sections 1-8 complete), out/phase5-* and out/eg2vq_embd_c_oracle_o3 (gitignored) |
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
- CLOSED in unit 15 (phase 5, row X.7): the codec's task-level cost is measured and committed with its hashes — 0.956039571 mean vector agreement and MRR@10 0.951778 -> 0.935819 at 768 on BEIR SciFact's test split, recall@10 0.983333 -> 0.990000, and the compounding at 128 dims (-0.029414 MRR). Two residual notes rather than work: a reused leg is trusted on its .meta plus the raw line count, so deleting a .meta forces an honest re-run and a partial raw is never a leg; and the numbers were produced with the -O3 build of the oracle, whose equivalence to the tree's -O0 entry is measured in out/phase5-oracle-o0-vs-o3.log (re-measure it if the compiler changes). OPEN (owner, NEW): what that trade is worth — no threshold was declared, so the reading of those numbers is the owner's call.
- CLOSED this unit: the dim-4 VQ device kernel's scope — the owner's directive put it in scope (phase 7).
- OPEN (phases 1b, 3, 4): none — all three done, each with a reproducible gate (`tools/eg2_artifact_check.sh`, `tools/eg2_loader_check.sh`, `tools/eg2_phase4_gate.sh`). Residual notes rather than work: the tokenizer has no C counterpart by design; the C reference is not a second independent implementation, since the anchor is the only foreign one and that is why the declared floor is measured on baseline weights; and an out-of-range VQ index cannot be encoded at this tier (12-bit indices exactly cover nc 4096), so only a future non-power-of-two codebook would need an index scan.
- PUBLISHED (2026-10-19, owner's instruction, both NEW public repos, origin and fork
  untouched):
    outer repo   https://github.com/JLFN/embeddinggemma2        branch dev, 26 commits
    ds4-rs copy  https://github.com/JLFN/embeddinggemma2-ds4-rs branch dev, the working
                 copy's whole history (2334 commits), including this project's work and the
                 live ds4 project's five local ds41 commits (trace instrument, Responses
                 suffix, two fixes, docs) — the owner was asked and chose the whole line.
  Both are public with dev as the default branch (verified with gh repo view immediately
  before the push).  What did NOT go: any artifact.  out/, .models/ and ds4-rs/ are
  gitignored, so the published outer repo has no GGUF, no payload, no index and no CUDA
  object — 45 tracked files at push time.
  Pre-push checks, all run in this unit, in the order the rules demand them:
    rule 10 OSV: 140 packages mapped, 0 vulnerable, twice (ds4-rs and the outer repo, whose
      scan resolves the same Cargo.lock through the nested copy);
    rule 12 graphify: a graph exists for both trees (session-start check);
    rule 6 provenance/visibility: Baekpica/ds4-dfm-rs and JLFN/ds4-dfm-rs are PUBLIC, as is
      JLFN/YoungAi, and the one verbatim third-party file in the outer repo
      (tools/ds4_gguf_write.h) is byte-identical to that public YoungAi file, so no private
      content was published.  The frozen retrieval set redistributes BEIR SciFact, which the
      dataset owner's own card states is CC BY-NC 2.0; tests/retrieval/PROVENANCE.md records
      the source, digest and that non-commercial term.
  WAIVED for the PREVIOUS push only: rule 19's QA-tester gate did not run for the push of
  2026-10-19 that published 46741d5.  The owner was asked for the QA model and answered
  "no QA", and for the gate itself chose "push now, with your explicit waiver recorded".
  That waiver covered that push and nothing else; for THIS unit the owner chose "run the
  QA gate, then push", and the per-project QA model is now recorded (below).
- RULE 19, THIS UNIT (2026-10-19): the QA-tester ran on deepseek-v4.1-CC-flash — the model
  the owner selected for this project, registered in ~/.opengrok/config.toml as
  [model."deepseek-v4.1-CC-flash"], which is now this project's qa model.  Report
  qa-evidence/qa-report.md, committed with this handoff, verdict `overall PASS`.  It
  re-derived every number of the evidence doc's sections 5b/5c from the two raws with the
  project's own scorer, cross-checked them with the parity instrument, rebuilt both oracle
  builds from source, and drove the gate in a /tmp sandbox to watch its reuse and refusal
  paths (a 580-line raw is not reused, a changed oracle sha forces a re-run, a tampered
  qrels file exits 2, the real out/ raws untouched).  It found no wrong number and three
  wrong sentences, all corrected in the QA commit: the jsonl helper's "byte-identical awk
  body / the gate diffs the two bodies" claim (neither was true; the rows ARE identical),
  the threading figure (a single 716% sample where six later runs read 558-671%), and the
  .meta's oracle binding, which is by sha256 and not by path.  What the pass could NOT
  test: a full 583-item re-run (it re-ran three items and reused the committed legs).
- OPEN (owner, NEW): the rule 19 item-4 mount.  This repo ships gate scripts but has no
  tests/qa-gate.sh and no tests/run.sh, so nothing MECHANICALLY blocks a push without fresh
  QA evidence — the gate is only as good as the agent honoring it.  The owner was asked and
  chose to record it rather than add it now: the next unit that ships code should copy
  ~/.opengrok/templates/qa-gate.sh into tests/qa-gate.sh, parameterize it for this repo
  (QA_BASE must be the PUBLISHED dev tip, since this repo has no origin; QA_MODEL=deepseek-v4.1-CC-flash;
  QA_SURFACE_PATHS must cover tools/, tests/retrieval/ and ds4-rs's crates, which the
  generic defaults do not), wire it into tests/run.sh, and re-run it before the next push.
- PUSHED 2026-10-19 (unit 15, after the QA pass and with the owner's approval, "run the
  QA gate, then push"): `git push publish dev`, fast-forward 46741d5..4849880, the remote
  tip verified after the push with `git ls-remote --heads publish` and JLFN/embeddinggemma2
  read as PUBLIC with dev as its default branch (`gh repo view` immediately before the
  push). What went: this unit's three commits only — RECIPE.md, docs/ds4-gemma-port-phases.md,
  docs/retrieval-cost-gate5.txt, tools/eg2_phase5_gate.sh, tools/eg2_oracle_to_jsonl.sh, the
  handoff and qa-evidence/qa-report.md. What did NOT go, as before: any artifact (out/,
  .models/ and ds4-rs/ are gitignored), and ds4-rs itself needed no push. Pre-push checks
  run immediately before it: clean tree, `git branch -vv` on dev, the range printed by
  `git log --oneline 46741d5..HEAD` exactly the three commits, rule 10 OSV (140 packages,
  0 vulnerable), rule 12 graph (present), rule 19 QA (fresh, overall PASS), rule 6
  visibility (PUBLIC). Rollback if it must be undone: `git push --force-with-lease publish
  46741d5:refs/heads/dev` — a force-push, so it is the owner's call, and the local history
  is kept either way.
- The three guard edits in ds4-rs are committed (9a6498da, ef79b5c2, c7dd3ebb) and phase 7
  was built on them; nothing is left open there.

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
    # phase 5 (gate 5): the harness's own self-test, then the legs and their .meta
    tools/eg2_retrieval_score_selftest.sh
    tail -8 out/phase5-gate.log                       # the agreement + retrieval tables
    cat out/phase5-{baseline,artifact}.meta           # oracle_sha a9f74d0b.. on both (the -O3 build)
    awk -F'\t' 'NF==3 && $2 ~ /^[0-9]+$/ {n++; s+=$2} END{print FILENAME, n, s}' \
        out/phase5-{baseline,artifact}-raw.txt        # 583 items / 107029 tokens per leg
    awk -F'\t' -f tools/eg2_parity_check.awk out/phase5-baseline-raw.txt \
        out/phase5-artifact-raw.txt                   # min cos 0.898186179, the instrument's own check
    # the sm_89 CUDA object and the rollback copy of the prebuilt set
    tail -3 out/cuda-sm89-build.log
    cuobjdump --list-elf ds4-rs/ds4_cuda.o            # ds4_cuda.sm_89.cubin = the build succeeded
    ls -l out/cuda-objects-backup-sm121a/ds4_cuda.o   # the untracked rollback, 94 MB tree
    # the card's build wiring: the record, the guards, and the green full build
    cat ds4-rs/.ds4-cuda-config.mk                    # CUDA_ARCH := sm_89 = the record is in place
    grep -c '!defined(__CUDA_ARCH__)' ds4-rs/cuda/glm_vq_{q4p,flash,gemv}.cuh    # 1 2 2 = five sites
    tail -4 out/cuda-card-wiring-build.log.summary    # make cuda exit 0, four hosts, ten cubins each
    git -C ds4-rs log --oneline -3                    # 29873445, fa9d96a0, 4892fb4b = the newest three, all published
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'

Trust git over this document's headings: continue with the first phase whose work is
not in the diff. Every phase through 7 is landed as of this boundary, so nothing is in
flight and the diff d6da366..HEAD is the whole record: the outer repo's HEAD is 4849880
and its published dev is the same commit (this unit's three commits were pushed
2026-10-19, 46741d5..4849880, fast-forward; the tip before the push was 46741d5,
measured with `git ls-remote --heads publish`): 07e0ccd carries tools/eg2_phase5_gate.sh
(the EG2_PHASE5_ORACLE selector and the corrected header), docs/retrieval-cost-gate5.txt
(5b, 5c, 7a, 7b and the result paragraph), RECIPE.md (4e/5/5b/6 item 3/7) and
docs/ds4-gemma-port-phases.md (row X.7 DONE, section 9, section 10); 99efb30 carries the
state card; 4849880 carries qa-evidence/qa-report.md, the three claims the QA pass
corrected (tools/eg2_oracle_to_jsonl.sh, tools/eg2_phase5_gate.sh,
docs/retrieval-cost-gate5.txt) and this file. 46741d5 was the previous unit's
handoff correction and 8ddd057 was unit 15's first slice
(the frozen set and the harness). ds4-rs is CLEAN at 29873445 on
feat/ds41-dsml-carriage — THIS UNIT TOUCHED NOTHING THERE — and its publish/dev is that
same tip (measured); the CUDA guards, the server family arm and the build wiring are older local
commits (9a6498da, ef79b5c2, c7dd3ebb) on that branch. Anything else in either diff must be read before
sections 2-6 are trusted. Note that the artifacts, the retrieval source, every
out/phase5-* file, out/eg2vq_embd_c_oracle_o3 and the CUDA objects live under gitignored
paths (out/, .models/, ds4-rs/), so they never appear in a diff: they are identified by
the digests in SHA256SUMS, by the pins inside the gate scripts, and by the sessions'
evidence docs.

This document is COMPLETE and self-contained: a new session can execute the project
end to end from this file alone. It carries the plan ahead, the exact reference
files, the facts git cannot provide, and the guardrails. History is deliberately
absent — git log and the project memory entry are the historical record.
