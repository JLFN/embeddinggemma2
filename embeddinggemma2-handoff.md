---
project: embeddinggemma2
plan_start_commit: d6da36680643c48c144536caf658e360b6d30d82
last_updated_commit: d3ba929297f1de181f93a20d79f81cf7210d630a
branch: dev
remote: 
handoff_written_at_context_usage: 21%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-19T09:35:46+02:00
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

2. Current state (2026-10-19, verified in this session)

- Workspace: /data/embeddinggemma2, branch dev, HEAD d3ba929 plus this unit's commit, tree clean but for this handoff. Nested ds4-rs at eb67d493 on feat/ds41-dsml-carriage with this unit's new example (crates/ds4-core/examples/gguf_inventory.rs) committed locally there.
- Artifact (the DQVD container being retired in 1b): out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin, 421,758,160 B; `bin/eg2_pack_cpu check` says kind=0 slots=1376, recipe `towers=vq4x4096 sharedcb=0 cbfp16 token=q8_0 frontend=vq other=keep nocal src=914f7f89142e`, structure OK.
- Baseline (phase 1a, new): out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf, 557,949,472 B, sha256 a982ca419f2c302c2d38a5a6004e9ae0a3bb582bd893795d152b56559955c725; `bin/eg2_gguf` wrote it in 3.0 s, the ds4-rs dump and the writer's manifest agree on all 413 tensors, and llama.cpp at the pin loads it and prints a 768-dim embedding. SHA256SUMS carries both files.
- Tools built: bin/eg2_vq, bin/eg2_pack, bin/eg2_pack_cpu, bin/eg2_gpuprobe, bin/eg2_gguf (new this unit), plus tools/eg2_gguf_check.sh.
- Checkpoint: .models/EmbeddingGemma-2-914f7f8, 1.5 GB, pinned revision 914f7f8.
- References verified today: /data/llama.cpp at 31288c9c (clean tree), built for the anchor at /data/llama-anchor-build (llama-embedding, CPU backend, 24 s wall); /data/YoungAi carries the native GGUF writer precedent (its header vendored this unit) and vq_fmt.h; /data/YoungAi/src/common/ds4_gguf_write.h sha256 5bfc5dd4…, unchanged.
- Guardrails: all five P-1 checks return zero, and the ldd check is now non-vacuous — ds4-rs/ds4-server is a built 76 MB CUDA-linked ELF (libcudart.so.13, libcublas.so.13) with 0 python/torch entries.
- Behavior corrections recorded this unit, each measured: the file's attention.sliding_window is 1024 (the converter doubles HF's 512, conversion/gemma.py:824-825) so the symmetric mask spans ±512; 1-D and *_norm.weight tensors must ship F32 (conversion/base.py:1244-1246) or the reference's CPU graph aborts; and this model's norm weights ship verbatim (Gemma4Model.norm_shift = 0.0).

3. Next unit plan — run EmbeddingGemma-2 in ds4-rs, phase by phase (2026-10-19)

Status: 1a done (2026-10-19), 2/1b/3..8 ahead. Goal: the ds4 runtime loads our GGUF artifact (a GGUF carrying a
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
- Artifact: out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf, 557,949,472 B, sha256
  a982ca419f2c302c2d38a5a6004e9ae0a3bb582bd893795d152b56559955c725; 413 tensors,
  219 bf16 copied verbatim + 194 f32 widened, 39 metadata keys.
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

PHASE 1b — the artifact GGUF (phase-doc row X.5b): a re-wrap, per phase 0.

- Metadata and tensor set to write: the artifact GGUF per contract section 4 (arch
  `gemma_vq`, alignment 32, `gemma_vq.format_version` 1, `gemma_vq.recipe` verbatim,
  hparams with the `gemma_vq.` prefix): 461 VQ payloads as type 42 `vqblob` with
  nd=1 ne=[payload_bytes], the token table as type 8 `q8_0` ne=[512,262144], 914
  verbatim BF16 at their true rank.
- Our own inputs: the DQVD container's slot table (bin/eg2_pack_cpu check re-prints it),
  docs/tensor-inventory.tsv and docs/encode-plan.tsv; the DQVD writer's guarantees are
  at tools/eg2_pack.c:980-1077 and must hold in GGUF terms.
- Gate 1b (artifact): all 1376 slots present in one GGUF; a ds4-side dump of every
  tensor equals what `bin/eg2_pack_cpu check` plus the per-slot re-encode comparison
  already produce; SHA256SUMS regenerated for the new file.
- Cost: a re-wrap, minutes, because phase 0 proved the payloads byte-identical.

PHASE 2 — the external anchor, once the baseline GGUF exists (phase-doc row X.5; owner-approved 2026-10-19)

- Act: build /data/llama.cpp at 31288c9c in a scratch directory (never in that tree), run examples/embedding/embedding.cpp over the frozen sentence set with the baseline GGUF from phase 1a, and commit the resulting vectors as data (tests/expected/anchor-embeddinggemma2.jsonl) with the revision, the exact command line, the prompt strings used (from the checkpoint's config_sentence_transformers.json) and digests.
- Read before trusting any of it: examples/embedding/embedding.cpp:70 (`common_embd_normalize`) — the normalization is the application's, not the graph's; the graph output is the raw pooled vector.
- Gate: at least 100 sentences, all four MRL widths (768 / 512 / 256 / 128, each re-normalized after truncation per the checkpoint README:166-185), file committed with its provenance, and a re-run reproducing the same digests.
- Why here and not later: this freezes the denominator before any port code is written, and it is a second opinion beside our own chain, never it — the port's C reference path stays the oracle for Rust and CUDA, exactly as /data/ds4-dfm-rs/AGENTS.md states for its retained `-c` oracles.

PHASE 3 — the loader: identity, payload reader, layouts, registration (phase-doc rows P0.4-P0.10; P0.1-P0.3 are retired with DQVD)

- Read: the two shipping families end to end, as the pattern — crates/ds4-core/src/identify.rs:169-207 (routing) and :183-186 (the family gate), crates/ds4-core/src/shape.rs (ModelFamily values through Glm53Vq = 14 at :35 and Ling31Vq = 15, Ling31FlashVq = 18 at :104, name arms :49 and :71), crates/ds4-core/src/layout.rs (LayoutSpec :119, family layouts() :3791), crates/ds4-core/src/bind.rs:993, crates/ds4-core/src/lib.rs (the Glm53Vq arms :1248, :1274, :1528), crates/ds4-core/src/tensors.rs (TensorInfo / ShardPlan / TensorInventory and the error variants), C side ds4.c:382-383 (family enum), :657-660 (shape catalog), :10240-10250 (bind dispatch), with ds4_glmvq_bind.inc / ds4_glmvq.h as the `.inc` shape to mirror.
- Land: crates/ds4-core/src/gemma_vq.rs (payload reader + validator + TensorInfo fill, the contract from phase 0), its error variant in tensors.rs, the gemma arms in identify.rs / shape.rs / layout.rs / bind.rs / lib.rs, the C identity + new ds4_eg2vq_bind.inc with ds4_eg2vq.h, and the family arms in crates/ds4-cli/src/lib.rs (:184 parse_args, family arms :451, :491).
- Host decode for the CPU path is mostly free: `ds4vq_dequant_f32` (ds41_vq_fmt.h:73-105) is dim- and width-generic and already handles dim 4; check its magic whitelist against what our encoder writes before relying on it.
- Gate: the ds4-side dump of all 1376 tensors equals the encoder's verify output line for line, and loading the artifact twice is deterministic (same dump, same hashes).

PHASE 4 — the text tower on the CPU reference path (phase-doc rows P1.1-P1.20)

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

0 (done) → 1a (done) → 2 (anchor freeze) → 1b (artifact GGUF, re-wrap) → 3 → 4 → 5 → 6 → 7 → 8 if wanted.
First command of the next unit (phase 2): the llama.cpp build already exists at
/data/llama-anchor-build (llama-embedding, CPU, built from the pin this session), so
phase 2 starts with the frozen sentence set — the
at-least-100-sentence file with the checkpoint's own task prompts from
config_sentence_transformers.json — then
`LD_LIBRARY_PATH=/data/llama-anchor-build/bin /data/llama-anchor-build/bin/llama-embedding
-m out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf --pooling mean -p "<sentence>"`
per sentence at all four MRL widths (768 / 512 / 256 / 128, each re-normalized after
truncation per the checkpoint README:155-185), and committing
tests/expected/anchor-embeddinggemma2.jsonl with the revision, the command line and
the digests. Open decision for it: whether tokenizer.ggml.add_eos_token should be
true (the converter does not write it; llama.cpp warns) and how the task prompts are
applied — decide by measurement, both ways, and record which one the frozen vectors
carry.

4. Reference map — the exact repos and files (pinned; verified in this session)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (the work lands here) | /data/embeddinggemma2/ds4-rs @ eb67d493, branch feat/ds41-dsml-carriage, gitignored, own git | every crates/ds4-core/src path named in section 3, ds4.c, ds4_glmvq*.inc/.h, cuda/*.cuh, tests/parity/, the Makefile |
| Form and metadata authority | /data/llama.cpp @ 31288c9c (b11723, 2026-10-18), MIT, read-only, source only (no build/ present) | src/models/gemma-embedding2.cpp (232 lines), src/llama-arch.cpp, src/llama-hparams.*, src/llama-model.cpp, src/llama-graph.cpp, src/llama-vocab.cpp, ggml/include/gguf.h, ggml/src/gguf.cpp, examples/embedding/embedding.cpp, conversion/gemma.py, conversion/base.py, gguf-py/gguf/** (vocab.py, gguf_writer.py, constants.py), tools/mtmd/** (phase 8) |
| The anchor build (phase 2; a build of the pin, never inside its tree) | /data/llama-anchor-build (CPU, built 2026-10-19) | bin/llama-embedding; the frozen vectors it produces are phase 2's deliverable |
| Writer and payload precedent | /data/YoungAi (reference toolchain, revision a4e6d31) | src/common/ds4_gguf_write.h, gguf-tools/quantize/v41_to_gguf.c, vq_fmt.h |
| Verification method | /data/ds4-dfm-rs (the live ds4 project; not modified from here) | AGENTS.md (retained `-c` oracles), tests/parity/, tests/ds4_proof.py, Makefile (the false-green guard) |
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

- Workspace layout: RECIPE.md (the artifact recipe: sections 4 commands, 5 quality, 6 what the port must implement, 7 identity); RECIPE-deepseek41-vq.md (the ds4 VQ recipe we mirror); docs/ds4-gemma-port-phases.md (row list); docs/ds4-gemma-port-reference.md (the port map and the four-leg verification); docs/tensor-inventory.tsv and docs/encode-plan.tsv (the per-tensor truth); docs/artifact-verify-summary.txt; docs/ladder-embeddinggemma2.txt (the measured tier ladder); tools/{eg2_vq.c,eg2_pack.c,eg2_gpuprobe.cu} and bin/ (their binaries); out/ (all artifacts and logs, gitignored); AGENTS.md (project guardrails).
- The artifact's quantizer: nocal VQ (codebooks solved on weights alone), dim 4, nc 4096, text-tower mean cosine 0.9859; the ladder's measured comparison is dim 4/nc 4096 at cos 0.9869 versus dim 8/nc 4096 at 0.9218 on the same representative tensor (docs/ladder-embeddinggemma2.txt).
- Model shape to load: 24 layers, hidden 512, 4 heads, 2 KV heads, head_dim 256; layers 5/11/17/23 full attention with head_dim 512 and 1 KV head; intermediate 2048; rms eps 1e-6; vocab 262,144; embedding dim 768; sliding window 512; mean pool then L2 normalize.
- Machine: RTX 4070 SUPER (12,282 MiB, sm_89), CUDA 13.3, 28 CPUs; Rust binds through the canonical builder (/data/build/linux/build.sh -p /data/embeddinggemma2/ds4-rs, rule 5); the C/CUDA side builds through the tree's Makefile, and the carried-over objects and the 76 MB ds4-server binary make the first build incremental.
- ds4-rs remotes are shared with the live ds4 project (origin Baekpica/ds4-dfm-rs, fork JLFN/ds4-dfm-rs): a push from here is shared state and needs confirmation. /data/ds4-dfm-rs itself is not to be modified from this workspace.
- Guardrails: global ~/.opengrok/AGENTS.md (core) + the mount at .opengrok/rules/project-work-rules.md (symlink; do not edit for project rules) + this repo's AGENTS.md. .gitignore excludes .models/, ds4-rs/, graphify-out/, graphify-rs-out/, out/ and *.o.

7. Guardrails and open items

- P-1 (no Python in the runtime, ever) and P-2 (references are citations or one-shot instruments, never dependencies; no ONNX, no PyTorch) are in this repo's AGENTS.md with their gate commands. All five checks return zero today, including ldd against the real ds4-server binary.
- Owner decisions in force: the container is GGUF + `gemma_vq` payload (2026-10-19); the external anchor run is approved once the baseline GGUF exists; quality is paramount — VQ is for compactness, never a licence to lose accuracy.
- OPEN (owner): whether the dim-4 VQ device kernel (phase 7) is in scope now or the device path waits for a dim-8 sibling artifact. OPEN (owner): whether the vision/audio towers (phase 8) are wanted. OPEN: whether KV checkpoints mean anything for an embeddings family.
- OPEN (phase 2): tokenizer.ggml.add_eos_token (the converter does not write it; llama.cpp warns) and how the checkpoint's task prompts (config_sentence_transformers.json) enter the anchor's sentences. Phase 1a needed neither.
- OPEN before any push: rule 12 graphify, rule 19 QA evidence, rule 10 OSV scan for ds4-rs (it carries a Cargo.lock); the outer repo has no remote, so publishing is an offer.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git diff --stat d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git log --grep "Unit: .* complete" d6da36680643c48c144536caf658e360b6d30d82..HEAD
    ./bin/eg2_pack_cpu check out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin
    sha256sum -c SHA256SUMS
    git -C ds4-rs log --oneline -1 && git -C ds4-rs status --short
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'

Trust git over this document's headings: continue with the first phase whose work is
not in the diff. Staleness check at the time of writing: after cada4bd only plan
documents are expected — this handoff and any phase-doc correction committed with
it. Anything else in the diff must be read before sections 2-6 are trusted.

This document is COMPLETE and self-contained: a new session can execute the project
end to end from this file alone. It carries the plan ahead, the exact reference
files, the facts git cannot provide, and the guardrails. History is deliberately
absent — git log and the project memory entry are the historical record.
