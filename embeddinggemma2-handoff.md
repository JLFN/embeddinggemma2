---
project: embeddinggemma2
plan_start_commit: d6da36680643c48c144536caf658e360b6d30d82
last_updated_commit: fca7ba6c537a5d15cec9fb1aa31d05391104fa51
branch: dev
remote: 
handoff_written_at_context_usage: 27%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-19T14:55:00+02:00
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

2. Current state (2026-10-19, verified in this session, unit 13)

- Workspace: /data/embeddinggemma2, branch dev, HEAD fca7ba6 plus this unit's commit, tree clean but for this handoff. Nested ds4-rs carries this unit's loader — the `gemma_vq` family and its layout, the family arms, the C identity, ds4_eg2vq.{h,inc} — on top of 125e902e, branch feat/ds41-dsml-carriage.
- Artifact (phase 1b, THE RUNNABLE FORM): out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf, 437,508,416 B, sha256 5bcf22161dda46bccbd5123b26d519efccad981cbfe11cce7227415b3373b33f — arch gemma_vq, 1376 tensors (461 type-42 vqblob, 1 type-8 q8_0 token table, 914 type-30 bf16), 42 KV, DATA_POS 15,881,824. Gate 1b still passes with the family validator in the path (re-run this unit).
- THE LOADER (phase 3, new): `cargo run -q -p ds4-core --release --example gguf_inventory -- <artifact> --no-kv --family` reports `IDENTIFY EmbeddingGemma-2 VQ family=16 variant=19`, `META ok`, `BIND ok slots=1376`, `LAYOUT ok specs=1376`. Gate 3 (`tools/eg2_loader_check.sh`) passes: the reader's 1376-tensor dump equals the writer's manifest line for line, two loads are identical, and 8 tampered copies are refused by name. Evidence: docs/loader-gate3.txt.
- Baseline (phase 1a/2): out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf, 557,949,600 B, sha256 0d84f1a9…, unchanged. Gate 1a passes; llama.cpp at the pin loads it.
- Anchor (phase 2, frozen, unchanged this unit): tests/anchor/anchor-items.tsv (113 items, sha256 0e746c6c…) and tests/expected/anchor-embeddinggemma2.jsonl (sha256 7561254c…). What phase 4 is measured against.
- Container (the artifact's input, still the byte-verified record of the quantization): out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin, 421,758,160 B, `bin/eg2_pack_cpu check` says kind=0 slots=1376, structure OK.
- Tools: bin/{eg2_vq,eg2_pack,eg2_pack_cpu,eg2_gpuprobe,eg2_gguf}; tools/{eg2_gguf_check.sh,eg2_artifact_check.sh,eg2_loader_check.sh (new: gate 3),eg2_anchor.sh,eg2_anchor_cmp.sh,ref_vocab_probe.py}.
- New this unit, measured: the family layout matched the artifact tensor for tensor on its first run, so the hand-written 1376-entry table is verified against the file rather than assumed; an out-of-range index cannot be encoded at this tier (12-bit indices exactly cover nc 4096), so the reachable payload tamper is a geometry/byte-account disagreement, which the loader refuses; and the C copies of the shape are compared by the existing parity test.
- Environment findings, pre-existing and both measured (section 7): ds4_cuda.o is stale and its rebuild fails in ptxas while targeting sm_89, so the C binaries cannot be relinked from this working copy; and /data/build/linux/build.sh cannot build this tree's binaries at all (they require `--features native`).
- Guardrails: all five P-1 checks return zero on 2026-10-19; `sha256sum -c SHA256SUMS` passes on all five entries.

3. Next unit plan — run EmbeddingGemma-2 in ds4-rs, phase by phase (2026-10-19)

Status: 1a, 2, 1b and 3 done (all 2026-10-19); 4..8 ahead. Goal: the ds4 runtime loads our GGUF artifact (a GGUF carrying a
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

PHASE 4 — the text tower on the CPU reference path (phase-doc rows P1.1-P1.20). NEXT UNIT.

- Authority, every line from /data/llama.cpp/src/models/gemma-embedding2.cpp unless named: embedding lookup scaled by sqrt(512) :86; no-cache, non-causal inputs :90-92 (causal_attn = false at :7); per-layer input block :97 with :215-232; attn_norm :109; q/k/v and heads :113-123; weighted RMS on Q and K, weightless on V :125-127; per-layer RoPE base (:132-137, base selection src/llama-model.cpp:2748-2750, the SWA split rule src/llama-hparams.cpp:8-20) ; attention scale 1.0 :139-141; attn_post_norm + residual :149-152; gated tanh-GELU FFN :156-170; the per-layer embedding block :172-193; layer_scalar :195-196; the control-vector hook is a no-op without a cvec file :199 (implement nothing, record the decision); output_norm + the 768 projection :206-211; mean pool src/llama-graph.cpp:4049-4053; L2 normalize in the caller, examples/embedding/embedding.cpp:70.
- The one mechanism to get exactly right, and the reason: the sliding layers use a SYMMETRIC window — declared at gemma-embedding2.cpp:4 (LLAMA_SWA_TYPE_SYMMETRIC) and masked at |p1 - p0| > n_swa/2 (src/llama-hparams.h:524-533), and the file's n_swa is 1024, because the converter doubles HF's one-sided 512 (conversion/gemma.py:824-825: "HF sliding_window is bidirectional, llama.cpp expects the full window size"). So a token sees ±512, a 1025-wide span. With causal_attn = false every query sees every row, so a trailing cut (what ds4's kernel does at ds4_cuda.cu:42214) is the wrong mask — and any implementation that reads the checkpoint's 512 straight into n_swa would get the window half the width it should be.
- Tokenizer: .models/EmbeddingGemma-2-914f7f8/tokenizer.json + tokenizer_config.json + tokenizer.model (262,144 entries, byte fallback), ported in the pattern of crates/ds4-core/src/tok/ling31.rs.
- Land: the family's C graph (ds4_eg2vq_graph.inc) plus its CPU reference entry, and the tokenizer module.
- Gate: the frozen sentence set (phase 2) through three things — the C reference path, the Rust host, and the anchor vectors. Report mean and minimum cosine and max element difference for C-vs-Rust and C-vs-anchor, at 768 and at 512/256/128 with re-normalization. Declared before the run: C-vs-anchor mean cosine ≥ 0.9999 (if it is not, the port is wrong, not the anchor), and Rust-vs-C within a stated element tolerance.

PHASE 5 — what VQ costs (phase-doc row X.7)

- Act: the same sentence set and a frozen labelled retrieval set through our own engine, once with the baseline GGUF and once with the VQ artifact; report the cosine agreement distribution (mean, min, quantiles) and recall@k / MRR at 768 and at each MRL width.
- Gate: the numbers are committed with the artifact hashes they were measured on. This is the owner's "not less smart" answer, and it needs no third party.

PHASE 6 — the entry point (phase-doc rows P2.1-P2.4)

- Land: a CLI verb (crates/ds4-cli/src/lib.rs:184 parse_args, family arms :451, :491), then /v1/embeddings in crates/ds4-server/src/serve.rs beside the existing arms (:1169, :1183, :1212 — measured today: no embeddings route and no pooling anywhere in crates/), plus the serving capability and native-limit arms (crates/ds4-core/src/serving.rs:715, serving_host.rs:1111), the task-prompt selection from config_sentence_transformers.json, and MRL truncation on request.
- Gate: the route returns the same vector as the CLI byte for byte for the same input and prompt.

PHASE 7 — the device path (phase-doc rows P3.1-P3.3)

- Read: cuda/ds41_vq_row.cuh:55 (the kernel refuses any dim but 8) and crates/ds4-core/src/vq.rs:42 (VQ_DIM = 8) against our dim-4 tier; the attention kernel ds4_cuda.cu:42201 with the row cap :42208 (`rows > 256u`), the `float score[256]` at :42211 and the trailing-window line :42214, its public API ds4_gpu.h:3645 (no window) and :3658 (window), and the VQ-in-kernel execution pattern ds4_glmvq_gpu.cuh.
- Land: a dim-4 VQ device kernel, a production bidirectional attention kernel with the symmetric ±n_swa/2 mask, and the tower end to end on the GPU with payloads consumed in-kernel (never expanded to dense).
- Gate: GPU vectors equal the CPU path within a stated tolerance, plus a latency and a VRAM number printed by the run.

PHASE 8 — vision and audio towers (phase-doc rows P4.1-P4.4), only if the owner wants them

- Their graphs live in /data/llama.cpp/tools/mtmd/models/gemma4v.cpp and gemma4a.cpp (wired at tools/mtmd/clip.cpp:973-979 and :1115-1121), the tie-in projections are already verbatim BF16 in our container, and the projector variant for this checkpoint is not stated by the conversion (conversion/gemma.py:978, :1057) and must be read out before porting. Gate: the phase 4 comparison, per tower.

Order of work, and the first command of the next session:

0 (done) → 1a (done) → 2 (done) → 1b (done, the artifact GGUF exists) → 3 (done, the loader) → 4 (the text tower, the tokenizer) → 5 → 6 → 7 → 8 if wanted.

First commands of the next unit (phase 4, the text tower): confirm both gates still
hold before writing any arithmetic — `tools/eg2_loader_check.sh
out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/artifact-manifest.tsv`
must print its line counts with 1376 tensors and 8 refusals, `tools/eg2_artifact_check.sh
out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/artifact-manifest.tsv
out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin .models/EmbeddingGemma-2-914f7f8
out/artifact-inventory.txt` must print its three counts, and `sha256sum -c SHA256SUMS`
must pass. Then build the tower exactly as the PHASE 4 block above specifies, with the
tokenizer first: the port of the `gemma4` pre-tokenizer and the family's specials, so
the anchor's per-item token counts can be reproduced before any matmul exists. Its gate
is the frozen 113-item set through the C reference path, the Rust host and the anchor,
at all four MRL widths, with the C-vs-anchor mean cosine ≥ 0.9999 declared before the
run. The re-encode fallback for the artifact remains available and is not expected to
be needed.

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

- Workspace layout: RECIPE.md (the artifact recipe: sections 4 commands, 5 quality, 6 what the port must implement, 7 identity); RECIPE-deepseek41-vq.md (the ds4 VQ recipe we mirror); docs/ds4-gemma-port-phases.md (row list); docs/ds4-gemma-port-reference.md (the port map and the four-leg verification); docs/tensor-inventory.tsv and docs/encode-plan.tsv (the per-tensor truth); docs/artifact-verify-summary.txt; docs/ladder-embeddinggemma2.txt (the measured tier ladder); docs/{baseline-gguf-gate1a,anchor-gate2,artifact-gguf-gate1b,loader-gate3}.txt (the evidence, one per unit); tools/{eg2_vq.c,eg2_pack.c,eg2_gpuprobe.cu,eg2_gguf.c} and tools/{eg2_gguf_check,eg2_artifact_check,eg2_loader_check,eg2_anchor,eg2_anchor_cmp}.sh with bin/ (their binaries); out/ (all artifacts and logs, gitignored); AGENTS.md (project guardrails).
- The loader (phase 3): crates/ds4-core/src/gemma_vq.rs is the family (the arch and `gemma_vq.format_version` gate, `validate_part` from `tensors.rs::parse_one`, the per-payload validator, `validate_metadata`), gemma_vq/layout.rs is the shape authority for all 1376 tensors, and ds4-rs/ds4.c holds `DS4_SHAPE_GEMMA_VQ` with its third copy in tests/parity/shape_c_oracle.c. The file's rank census is 1354 rank 1 / 7 rank 2 / 13 rank 3 / 2 rank 4; a VQ payload's single dimension is its byte length, and the layout computes the same number from the tower architecture through `vq_payload_bytes`.
- The artifact's quantizer: nocal VQ (codebooks solved on weights alone), dim 4, nc 4096, text-tower mean cosine 0.9859; the ladder's measured comparison is dim 4/nc 4096 at cos 0.9869 versus dim 8/nc 4096 at 0.9218 on the same representative tensor (docs/ladder-embeddinggemma2.txt).
- Model shape to load: 24 layers, hidden 512, 4 heads, 2 KV heads, head_dim 256; layers 5/11/17/23 full attention with head_dim 512 and 1 KV head; intermediate 2048; rms eps 1e-6; vocab 262,144; embedding dim 768; sliding window 512 in the checkpoint's one-sided sense — the FILE carries 1024 (`gemma_vq.attention.sliding_window`, the converter's doubling), and the symmetric mask is ±512; mean pool then L2 normalize.
- Machine: RTX 4070 SUPER (12,282 MiB, sm_89), CUDA 13.3, 28 CPUs. The Rust side is built with `cargo` plus `--features native` (every ds4-cli binary needs that feature) and the C side through the tree's Makefile — measured this unit: `/data/build/linux/build.sh -p /data/embeddinggemma2/ds4-rs` refuses the binaries (`target ds4-agent-rs in package ds4-cli requires the features: native`), so rule 5's documented iterative-development exception is what applies here. `make ds4.o` takes 39 s and `make ds4_cpu.o` 11 s; the CUDA object and the C binaries are a separate story (section 7).
- ds4-rs remotes are shared with the live ds4 project (origin Baekpica/ds4-dfm-rs, fork JLFN/ds4-dfm-rs): a push from here is shared state and needs confirmation. /data/ds4-dfm-rs itself is not to be modified from this workspace.
- Guardrails: global ~/.opengrok/AGENTS.md (core) + the mount at .opengrok/rules/project-work-rules.md (symlink; do not edit for project rules) + this repo's AGENTS.md. .gitignore excludes .models/, ds4-rs/, graphify-out/, graphify-rs-out/, out/ and *.o.

7. Guardrails and open items

- P-1 (no Python in the runtime, ever) and P-2 (references are citations or one-shot instruments, never dependencies; no ONNX, no PyTorch) are in this repo's AGENTS.md with their gate commands. All five checks return zero today, including ldd against the real ds4-server binary.
- Owner decisions in force: the container is GGUF + `gemma_vq` payload (2026-10-19); the external anchor run is approved once the baseline GGUF exists; quality is paramount — VQ is for compactness, never a licence to lose accuracy.
- OPEN (owner): whether the dim-4 VQ device kernel (phase 7) is in scope now or the device path waits for a dim-8 sibling artifact. OPEN (owner): whether the vision/audio towers (phase 8) are wanted. OPEN: whether KV checkpoints mean anything for an embeddings family.
- OPEN (phase 1b): none — done, and its gate is reproducible (`tools/eg2_artifact_check.sh`). If a payload ever fails it, the fallback is a re-encode, never a hand-repaired byte.
- OPEN (phase 3): none — done, and its gate is reproducible (`tools/eg2_loader_check.sh`). Residual notes rather than open work: an out-of-range index cannot be encoded at this tier (12-bit indices exactly cover nc 4096), so a future tier whose codebook is not a power of two is the only case that would need an index scan; and the C shape is verified as a table by `test-catalog-parity`, but nothing consumes it until phase 4 builds the graph.
- OPEN (environment, both pre-existing and measured this unit, neither caused by this project's changes): (a) `make ds4-c ds4-server` cannot relink here — ds4_cuda.o (2026-10-14) is older than the CUDA sources it depends on (2026-10-18), and rebuilding it fails in ptxas with `Feature 'mbarrier.try_wait.parity' requires .target sm_90 or higher` because the tree's `.ds4-cuda-config.mk` pins `CUDA_ARCH := native` (sm_89 here); ds4.o and ds4_cpu.o build fine; (b) the canonical builder cannot build this tree at all until it can pass `--features native`. Both matter to phases 4 and 7.
- OPEN (phase 4): the anchor's token counts are per item in the anchor file, so the tokenizer port can be checked before any matmul is written; use the three long items for the mask, and hold the declared C-vs-anchor mean-cosine gate at 0.9999 (the anchor's own batch-shape spread is 3.4e-4 per element, three orders inside that). The tokenizer is the first thing to port: `llama-vocab.cpp:528-536` pins the `gemma4` pre-tokenizer (newline split over raw UTF-8 with escaped spaces and no GPT-2 byte encoding), and `:2787` forces `add_bos` true for it.
- OPEN before any push: rule 12 graphify, rule 19 QA evidence, rule 10 OSV scan for ds4-rs (it carries a Cargo.lock); the outer repo has no remote, so publishing is an offer.

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
    git -C ds4-rs log --oneline -1 && git -C ds4-rs status --short
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'

Trust git over this document's headings: continue with the first phase whose work is
not in the diff. Staleness check at the time of writing: after this unit's commit
only plan and evidence documents are expected in the outer repo — this handoff, the
gate-3 evidence, and the phase-doc, contract, RECIPE and AGENTS corrections committed
with it, plus the new tools/eg2_loader_check.sh. In ds4-rs, this unit's commit is the
loader: crates/ds4-core/src/gemma_vq.rs and gemma_vq/layout.rs, the family arms in
identify/shape/layout/bind/lib/validate/serving/serving_host/tok/tensors, the
`gguf_inventory --family` leg, tests/parity/shape_c_oracle.c, ds4.c's family identity,
ds4_eg2vq.h and ds4_eg2vq_bind.inc, and the Makefile deps. Anything else in the diff
must be read before sections 2-6 are trusted. Note that the artifact and its manifest
live in the gitignored out/, so they are not in the diff: they are identified by the
digests in SHA256SUMS and section 2.

This document is COMPLETE and self-contained: a new session can execute the project
end to end from this file alone. It carries the plan ahead, the exact reference
files, the facts git cannot provide, and the guardrails. History is deliberately
absent — git log and the project memory entry are the historical record.
