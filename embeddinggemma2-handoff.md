---
project: embeddinggemma2
plan_start_commit: d42fc8fdf8cef8dc0f6806d249160ae5472abe77
last_updated_commit: 2450b9a64495aa11eb8360240458ae7f6320ef4c
branch: dev
remote: 
handoff_written_at_context_usage: 20%   (read from signals.json at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-20T08:35:00+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ in ds4-rs (rewritten 2026-10-20 at the llama-free-boundary: the landed phases are rotated out into git and the evidence docs, and section 3 is the ONE unit that is left)

Note on the frontmatter: it MUST be the first thing in the file, with
bare values (no inline comments after the value) — the handoff-hooks
parser reads it verbatim, and a commented or misplaced frontmatter
silently breaks the SessionStart probe readout. The scaffold writes this
exact shape; hand-written files must match it. <COMMIT> means the full
commit hash (plan_start_commit = the commit the current plan began from);
<PERCENT> = the context usage read from the session's signals.json
(target 25-35, hard cap 40).

0. Rules for writing THIS document (read before editing, not just before reading)

- Only write or replace this file at a clean, verified, committed boundary. Never checkpoint mid-work.
- Start drafting once context usage crosses ~25%. Finalize and commit before the auto-compact threshold — do not wait for it to begin writing.
- Every claim in "Current state" must be re-verified by running the relevant command in this session, not copied from the previous handoff. If you cannot re-run it, mark it unverified.
- Hard caps: "Current state" max 15 lines. "Guardrails and open items" max 15 lines. "Next unit plan" has no cap — it is the actual work spec.
- Do NOT: replay prior units' logs (git log is the record), re-explain global rules already in the loaded guardrails, restate anything the staleness check covers, or record acceptance evidence you did not observe this session.
- Section 3 holds ONLY the next unit. When a unit completes, fold its outcome into section 2 and REPLACE section 3 with the next unit's plan — history is not appended, it is rotated out into git and the evidence docs.
- This file is the plan of record. docs/ds4-gemma-port-phases.md holds the row-by-row work list behind it; where the two disagree, THIS file wins and the phase doc is corrected in the same commit.

1. Session protocol (read first)

STANDING OWNER INSTRUCTION (2026-10-19, verbatim in intent): the owner is AFK and
wants the session to CONTINUE AUTOMATICALLY after each handoff and each automatic
compaction. So: never stop to ask a question, never wait for a prompt, and never
end a turn with work left that the plan already specifies. Work unit after unit:
finish it, verify it, commit it locally, rotate this handoff, and continue. The one
thing that stays an offer is the PUSH (rule 23): record the offer in section 7 and
keep working; do not push on your own initiative.

SECOND OWNER DIRECTIVE (2026-10-20, verbatim): "We will not use llama it must be
ported all". Read as a boundary rule: the pin is the AUTHORITY the port reads and
cites (rule 0.2) and the one-shot producer of the frozen numbers, but no build, no
run and no default gate of this project executes it. Measured 2026-10-20 and closed
by commit 3729ec4: ds4-rs contains no llama call and links no llama library, gate 22
reads the mmproj back with the Rust host's own reader, and the two remaining
default-path runs of the pin (the vision harness, the tokenizer dump's regeneration)
are opt-in — tools/eg2_vision_anchor.sh and EG2_TOK_ANCHOR=1. Everything the pin
produced that matters is committed data.

1. Read context usage from the session's signals.json (contextTokensUsed /
   contextWindowTokens) under ~/.opengrok/sessions/<percent-encoded-cwd>/
   <session-id>/, newest session dir. If signals.json is absent, fall back to
   chat_history.jsonl bytes / 4 and say so in the session report.
2. If usage is below the threshold (~/.opengrok/config.toml
   [session].auto_compact_threshold_percent, this machine 60): continue
   immediately — write the next unit's plan into section 3 if it is not there,
   execute it, verify, commit, re-check usage, repeat.
3. At the threshold the harness compacts this session in place; nothing is
   launched. This handoff must be written and committed BEFORE the threshold.
4. Preferred cadence: once the finished unit's handoff, the project memory entry
   and the commits are in place, call the compact_session tool as the last action
   of that turn. The handoff write itself also parks a compaction.

2. Current state (2026-10-20, at the unit-23 close: the vision tower runs on the CPU reference path and gate 23 is green in the fast path, so the next work is the DEVICE tower)

- OWNER DIRECTIVE, 2026-10-19 (verbatim intent): "we need to add vision also see how the other model variant handle it". This answers the long-open phase-8 decision in favour of vision (the audio tower is NOT in scope; the audio decision stays open). It defers, and does not cancel, unit 21 (batching): its complete executable spec is parked at docs/eg2vq-batch21-plan.txt, together with the reference reading already done for it, so the resumed unit does not repeat either.
- WHAT THE ARTIFACT ACTUALLY CARRIES (measured this session, docs/tensor-inventory.tsv + out/artifact-kv.txt): of 1376 tensors, 208 are vision_tower.encoder.layers.* (16 layers x 13), 2 are vision_tower.patch_embedder.* (input_proj.weight 768x768, position_embedding_table 2x10240x768), and 744 are audio_tower.* with its frontend; the text tower we ported is 217 of the 461 vqblob payloads, and the VISION TOWER IS DENSE BF16, not VQ. The two tie-in projections are in-file: embed_vision.embedding_projection.weight 512x768 and embed_audio.embedding_projection.weight 512x1536.
- The image path, read from the reference at the pin: image -> vision tower (config.json vision_config: hidden 768, 16 layers, 12/12 heads, head_dim 64, ffn 3072, patch 16, pooling 3, rope theta 100 axial, rms eps 1e-6, standardize false) -> 3x3 average pool -> scale sqrt(768) -> weightless RMS -> 768-to-512 projection -> N soft tokens injected at the image-token spans of the text tower we already have. The graph is /data/llama.cpp @ 31288c9c tools/mtmd/models/gemma4v.cpp, 133 lines, and the projector variant is ANSWERED from conversion/gemma.py:1001 (GEMMA4V; our checkpoint maps to EmbeddingGemma2VisionAudioModel at :1055, which subclasses Gemma4VisionAudioModel) — phase-doc row P4.4 is no longer open.
- Workspace: /data/embeddinggemma2, branch dev, HEAD = 2450b9a (the unit-23 vision gate and evidence), tree CLEAN after this handoff's own commit. The outer repo has no origin, only publish = https://github.com/JLFN/embeddinggemma2, whose dev is at d45394f — so TWELVE commits are unpushed (the ten recorded at the previous boundary, plus 2450b9a and the handoff). The push is an OFFER, recorded in section 7.
- Nested ds4-rs: CLEAN at f459e924 on feat/ds41-dsml-carriage, TWENTY-ONE commits ahead of fork/feat/ds41-dsml-carriage; its publish = https://github.com/JLFN/embeddinggemma2-ds4-rs stands at 29873445 (refs/heads/dev), so THREE ds4-rs commits are unpushed: 68837a53 (the caps), 70278c19 (the QA corrections) and f459e924 (the vision tower). Push with `git push publish feat/ds41-dsml-carriage:refs/heads/dev`; plain `git push publish dev` fails (the local dev tracks fork/dev and is behind).
- The owner's declared quality floor for this tier is RECORDED and MET (docs/retrieval-cost-gate5.txt section 9; RECIPE.md 5b): recall@10 >= 0.983333, MRR@10 >= 98% of baseline (0.932742), recall@1 >= 0.855, mean cosine >= 0.95 — margins 0.006667 / 0.003077 / 0.006611 / 0.006040, all PASS. In percentages at 768: vectors keep 95.60% agreement, MRR@10 98.32%, recall@1 97.00%, recall@5 99.59%, recall@10 100.68% (better), and 7.67% of queries change top-1. The floor covers the 768 width only; 512/256/128 have none.
- Rule 19 is now MECHANICAL: tests/qa-gate.sh (a copy of the canonical template with three recorded deviations: QA_BASE = publish/dev because there is no origin; the three surface patterns retargeted at C/Rust/shell; it scans BOTH repositories, because ds4-rs is gitignored in the outer tree and the first version therefore passed a unit with no surfaces visible) and tests/run.sh (the fast path: digests, container, scorer self-test, loader gate 3, gate 22 the mmproj read-back, QA gate; `--deep` adds gate 1b, cargo test -p ds4-core and gate 4). Verified: red on the committed unit, green after the QA report; `bash tests/qa-gate.sh` exit 0 as of this boundary (25.3 s for the whole fast path: gate 23 builds its oracle at -O3 and runs both fixtures, ~10 s of it the build).
- UNIT 23 IS DONE (phase 8, row P4.1): the vision tower runs on the CPU reference path in ds4-rs (f459e924) with gate 23 green in the fast path. Files: ds4_eg2vq_vision.h + ds4_eg2vq_vision_bind.inc + ds4_eg2vq_vision_graph.inc + ds4_resize_pillow.inc (the Pillow resampler MOVED verbatim out of ds4.c's DS4_NO_GPU block, which is why the CPU-only oracle can now resize) and tests/parity/eg2vq_vision_c_oracle.c; the outer repo carries tools/eg2_vision_gate23.sh (fast-path step 6), tests/vision/reference/preproc-facts.txt (14th digest) and docs/eg2vq-vision-tower-gate23.txt. MEASURED against the frozen dumps: pattern max-abs 0.10275560, relrms 0.008235; gradient 0.06645030, 0.005285; min per-row cosine 0.99978457 / 0.99988406, against a bound declared and printed before the run (0.30 / 0.025 / 0.998). TWO MECHANISMS THE GRAPH NEVER STATES were found in the kernels and are what makes the port agree: every bf16-weight matmul quantizes its ACTIVATION to bf16 (ggml-cpu/ggml-cpu.c:395-400, :1328-1340) EXCEPT the patch conv, whose (im2col, weight) argument order leaves it f32 (ggml.c:4785-4789), and the FFN activation is GELU_QUICK computed through ggml's f16 table (clip.cpp:1414-1429, vec.h:47, :1047-1057). The gate kills the discrimination question: every wrong mechanism is 5x to 100x further away (7.60% tanh GELU, 47.20% swapped position slices, 103.50% swapped rope axes, 114.75% kq_scale=1/sqrt(d), 58.27% whole-width V RMS, 134.64% exchanged gate/up, 3.11% bf16-rounding the conv). The port's own noise floor is measured too: a 1e-7 kick moves it 0.068% and SATURATES, and the same source at -O2 differs by 0.064%.
- What is left of the plan, in order: (1) the VISION phase (section 3: the oracle first, then the CPU tower, the device tower and the serving splice); (2) unit 21, the batching of several items into one tower call, DEFERRED with its spec at docs/eg2vq-batch21-plan.txt; (3) the C CLI's tokenizer gap (ds4-c refuses the artifact by name; the family's gemma4 BPE is Rust-side, so porting it duplicates the encoder) — owner's call; (4) the AUDIO tower — owner's call; (5) the two pushes, when the owner answers.
- Nothing measured to date claims: an MTEB-comparable score; a serving-shaped latency or concurrency number for a batch (the tower still runs one item at a time); anything about the vision and audio towers; or any quality verdict past the four declared floors.
- Machine-facing facts RE-RUN this session (2026-10-20): `sha256sum -c SHA256SUMS` all FOURTEEN entries OK (the twelfth and thirteenth from the previous boundary are tok-anchor-ids.tsv 9e6ed555.. and the mmproj 9ffb86e3..; the fourteenth is preproc-facts.txt c9aa56a3..); tests/run.sh fast path GREEN in 25.3 s (digests, DQVD container, scorer self-test, loader gate 3 = 1376 tensors agree with the manifest + 8 tampered copies refused by name, gate 22 native `EG2VQ_VISION_MMPROJ_PASS`, gate 23 `EG2VQ_VISION_GATE23_PASS`, QA gate overall PASS); `make ds4.o` (the CUDA host build) also compiles clean with the new files; tools/eg2_vision_anchor.sh GREEN after the rename (pin accepts the mmproj at 211/211 and reproduces all four frozen dumps byte-identically); tools/eg2_tok_check.sh GREEN against the committed ids (113 items, 0 count mismatches, 0 id mismatches, pin not run). Carried over from the previous boundary and NOT re-run here: the artifact sha256 5bcf2216.., the -O3 oracle a9f74d0b.., CUDA_ARCH sm_89, rule 10 OSV (140 packages / 0 vulnerable in both repos), and gate 4/5/20's numbers; they are recorded in those gates' own out/ logs and evidence docs.
- The unit-20 QA findings are all closed (four corrections committed, one recorded as informational); the unit-23 QA pass ran under the same model (deepseek-v4.1-CC-flash) and its findings are in qa-evidence/qa-report.md.


3. Next unit plan — unit 24: the vision tower on the DEVICE (phase 8, row P4.2)

Goal: the same 16 blocks, the same weights and the same numbers as the CPU leg
unit 23 landed, run by CUDA kernels, gated against THE CPU LEG (not against the
frozen dumps directly: the CPU leg is the port's own definition of correct, and
it is already within 0.82%/0.53% of the reference).

STEP 0 — the facts unit 23 established, which unit 24 must carry verbatim. Each
is a mechanism the reference's KERNELS impose, not the graph; each was measured,
and the numbers are in docs/eg2vq-vision-tower-gate23.txt section 4:

 1. EVERY bf16-weight matmul quantizes its ACTIVATION to bf16 (round-to-nearest-
    even, ggml-impl.h:623-635), because this revision's BF16 vec_dot_type IS
    BF16 (ggml-cpu/ggml-cpu.c:395-400, conversion :1328-1340). The tower's
    matmuls are the 7 per-block matrices and the 768->512 projection.
 2. THE PATCH CONV IS THE ONE EXCEPTION: ggml_conv_2d passes (im2col, weight) as
    (src0, src1), so the f32 im2col drives vec_dot_type and the bf16 weight is
    merely widened (ggml.c:4785-4789). Its activation is NOT quantized. Rounding
    it moved both fixtures away (1.5% -> 3.1% on gradient).
 3. The FFN activation is gelu_quick computed through ggml's F16 TABLE: the input
    is quantized to f16, the table entry (itself f16) is widened back
    (vec.h:47, :1047-1057, ggml-cpu.c:3884-3891). On the device, the honest
    equivalent is to build the same 65536-entry f16 table (or the same four
    conversions) rather than calling an intrinsic gelu.
 4. The attention is kq_scale EXACTLY 1.0, NO mask, NO sliding window, 12 query
    heads over 12 KV heads, head_dim 64, and the reference's executed flash path
    rounded K and V to f16 (clip.cpp:787-790) — so eg2vq_attn.cuh is NOT
    reusable: its whole shape (window, causal mask, scale, one KV head) is the
    text tower's.
 5. The pool row order is q = x' + (grid_w/3)*y' and the 3x3 average divides by
    9; the embedder's RMS is weightless with eps 1e-6.

What unit 24 has to build, in the order the change has to land:

 1. THE WEIGHTS ON THE DEVICE. The tower's matrices are already bf16 in the
    mmproj — the device path needs NO widening, which is a simplification the
    CPU leg could not take (it widens once into a 576.0 MiB f32 cache). The
    adapter should take the mmproj as the engine's second model
    (e->vision_model, ds4.c:72689's pattern) and bind by name, exactly as
    ds4_eg2vq_vision_bind.inc does, with a device plan in the shape of
    ds4_eg2vq_aux/ds4_eg2vq_record: the four per-block norms and the two head
    norms as f32, the 7 matrices + the patch conv + the projection as bf16, the
    position table f32 (only rows < grid_w and < grid_h are ever read).
 2. THE OPS: a 16x16 stride-16 patch conv as a GEMM (792 x 768 x 768, the
    im2col layout is im2col[c*256 + h*16 + w]); the two axial position tables
    added by row lookup; per-head RMS with the [64] weights; the axial NeoX rope
    (theta 100, first 32 dims on pos_x, second 32 on pos_y); the weightless V
    RMS; attention; GEGLU; the 3x3 average pool; the weighted/weightless norms.
 3. THE GATE: parity against the CPU leg on both fixtures (the CPU oracle's
    `soft` mode output), with the SAME discipline the device path already uses:
    equality first, then tolerance, then speed. Record the device's own
    deviation from the CPU leg separately from the CPU leg's deviation from the
    frozen dumps — the two must not be conflated in the evidence.
 4. WIRE IT: an engine entry shaped like ds4_engine_embed_gemma_vq (unit 20's
    pattern: readiness checks, refusals by name), and the caps question: the
    device plan's workspace is a new size class, so the embedding-shaped cap
    entry (ds4_core::embed_caps(GemmaVq)) may need a vision slot or a second
    entry — decide by reading how unit 20's caps are enforced, not by inventing
    a second mechanism.

WHAT UNIT 24 MUST NOT DO: re-derive the graph (it is read and cited in
ds4_eg2vq_vision_graph.inc), re-tune the preprocessing (gate 23 pins it), or
chase the CPU leg's 0.5-0.8% residual — that residual is the CPU leg's own
op-level rounding, already named and measured, and it is not a device target.
Do not "fix" the bf16 activation quantization away because it looks like a
precision loss: it is what the reference does, and removing it costs 1.5%.

Note on the row the phase doc gives this unit: P4.2 ("the device tower"), with
P4.3 (the serving splice) as unit 25 after it. Nothing in unit 23's work changes
that order.

4. Reference map — the exact repos and files (pinned; verified at this boundary)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (all the work) | /data/embeddinggemma2/ds4-rs @ 70278c19, branch feat/ds41-dsml-carriage, gitignored by the outer repo, own git | ds4.c, ds4.h, ds4_eg2vq.{h,inc,graph.inc,gpu.cuh}, ds4_glmvq*.{h,inc}, cuda/eg2vq_{attn,ops,vq}.cuh, cuda/*.cuh, crates/ds4-core/src/**, crates/ds4-server/src/**, crates/ds4-sys/src/lib.rs, tests/parity/, the Makefile |
| Form and metadata authority | /data/llama.cpp @ 31288c9c (b11723, MIT, read-only, source only) | src/models/gemma-embedding2.cpp, src/llama-{arch,hparams,model,graph,vocab,batch}.cpp, ggml/include/gguf.h, conversion/{gemma,base}.py, gguf-py/gguf/**, examples/embedding/embedding.cpp, tools/mtmd/** (phase 8) |
| The VISION reference (phase 8) | /data/llama.cpp @ 31288c9c, the same pin | tools/mtmd/models/gemma4v.cpp (the graph, 133 lines: :5-117), tools/mtmd/clip.cpp (:337-450 build_vit's per-layer order and :144-147 the weightless V RMS, :254-256 the patch grid, :973-979 the builder dispatch, :1719-1736 the GEMMA4V hparams, :4419-4428 the image-token count, :5562-5568 pos_x/pos_y, :1342-1352 the mmproj keys), tools/mtmd/clip-impl.h:112-157 (the mmproj tensor names), tools/mtmd/mtmd.cpp:953-961 (dyn_size is the gemma4v preprocessor), mtmd-image.cpp:57-96 (the PAD_CEIL branch), :122-156 (the size solver), :204-390 (resize_pillow), :7-14 (the u8 -> f32 conversion), clip-model.h:66-68 (pad and pad-colour defaults), :204-211 (the token-to-pixel limits), mtmd-helper.cpp:404 (the decode, no EXIF), conversion/gemma.py:978-1063 (the projector variant and the tensor mapping) |
| The VISION weights file, and the frozen image reference | our own committed data | out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf (sha256 9ffb86e3.., 211 tensors, written by tools/eg2_mmproj.c from the checkpoint) and tests/vision/{eg2-vision-gradient-512x384.png,eg2-vision-pattern-512x384.png} with tests/vision/reference/*.{soft,text}.tsv (the pin's own numbers, produced once by tools/eg2_vision_anchor.sh) |
| The anchor build (phase 2) | /data/llama-anchor-build (CPU, plus an mtmd target built for the vision anchor) | bin/llama-embedding; the frozen text vectors it produced are phase 2's deliverable, and since 2026-10-20 nothing but tools/eg2_vision_anchor.sh touches this tree |
| Writer and payload precedent | /data/YoungAi (a4e6d31) | src/common/ds4_gguf_write.h, gguf-tools/quantize/v41_to_gguf.c, vq_fmt.h |
| Verification method | /data/ds4-dfm-rs (the live ds4 project; never modified from here) | AGENTS.md (the retained `-c` oracles), tests/parity/, Makefile (the false-green guard) |
| The anchor vectors | our own committed data | tests/anchor/anchor-items.tsv (113 items; three long enough to exercise the mask) and tests/expected/anchor-embeddinggemma2.jsonl (768/512/256/128 vectors, sha256 7561254c..); tools/eg2_anchor.sh / eg2_anchor_cmp.sh |
| The artifact, and the container behind it | our own committed recipe | out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf (sha256 5bcf2216.., 1376 tensors), out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf (0d84f1a9..), out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin (8902fb8a..), out/artifact-manifest.tsv, out/artifact-kv.txt |
| The family's contract, loader, tower | our own code | crates/ds4-core/src/gemma_vq.rs (+ the caps constants) and gemma_vq/layout.rs; tok/gemma_vq.rs; ds4_eg2vq_graph.inc + ds4_eg2vq_bind.inc; tools/eg2_{loader_check,phase4_gate,tok_check}.sh; docs/{loader-gate3,text-tower-gate4}.txt |
| The gate-5 harness and the frozen set | our own code and data | tools/eg2_retrieval_set.sh, tests/retrieval/scifact-test-{queries,pool,qrels}.tsv, tools/eg2_{phase5_gate,oracle_to_jsonl,retrieval_score,retrieval_score_selftest}.sh, docs/retrieval-cost-gate5.txt (sections 1-9) |
| The embedding entry and its caps (phase 6) | our own code | ds4.c's ds4_engine_embed_gemma_vq, ds4_bridge_eg2vq_*, crates/ds4-core/src/serving.rs (EmbedCaps/embed_caps), crates/ds4-core/tests/eg2vq_serving_caps.rs, fair route gate tools/eg2_embed_route_gate.sh (sections 1-5), docs/eg2vq-serving-caps-gate20.txt, docs/eg2vq-embed-entry-gate6.txt |
| The device path (phase 7) | our own code | cuda/eg2vq_*.cuh, ds4_eg2vq_gpu.cuh, tests/parity/eg2vq_embd_gpu_oracle.c, docs/eg2vq-device-gate7.txt, out/cuda-objects-backup-sm121a |
| The labelled data, and its origin | BEIR SciFact (Thakur et al., NeurIPS 2021 D&B) | the zip URL pinned in tools/eg2_retrieval_set.sh (ZIP_SHA 536e1444..); tests/retrieval/PROVENANCE.md records CC BY-NC 2.0 |
| The model | /data/embeddinggemma2/.models/EmbeddingGemma-2-914f7f8 (gitignored) | config.json, config_sentence_transformers.json (the 20 task prompts), 1_Pooling/config.json, tokenizer.json, README.md (the truncation contract) |
| Counter-example only | /data/llama-prism @ b10743 | never a source; read only to avoid its 1/sqrt(head_dim) scale |

5. Routes removed — do not resurrect any of them

- ONNX and onnxruntime in any role; PyTorch/transformers/sentence-transformers and
  any Python ML stack (owner, 2026-10-19: "remove this onnx i dont want it and no
  pytorch"; AGENTS.md P-1). An isolated venv is dead too.
- The DQVD container as the target format (owner's container decision); it remains
  only as phase 1's input and the byte-verified record of the quantization.
- The `eg2_pack reconstruct` fallback (DQVD-only, superseded by the GGUF).
- candle, burn, tch or any Rust tensor framework: the arithmetic is our own C/CUDA.
- Executing the pinned llama.cpp in any default path (owner directive 2026-10-20): no
  build, run or gate of this project links it. Its numbers are committed data and
  tools/eg2_vision_anchor.sh (opt-in) is the only thing that runs it.
- Metal and macOS: not a target here (Linux, NVIDIA only).
- Not in scope unless the owner says so: KV checkpoints (row X.3), ds4's
  distributed/agent surfaces, and the vision/audio towers (phase 8).

6. Project facts a fresh session cannot re-derive

- Workspace layout: RECIPE.md (recipe, quality, what the port implements, identity);
  RECIPE-deepseek41-vq.md (the ds4 VQ recipe mirrored); docs/ds4-gemma-port-phases.md
  (the row list — the authoritative work log); docs/ds4-gemma-port-reference.md;
  docs/{tensor-inventory.tsv,encode-plan.tsv,artifact-verify-summary.txt,
  ladder-embeddinggemma2.txt}; docs/{baseline-gguf-gate1a,anchor-gate2,
  artifact-gguf-gate1b,loader-gate3,text-tower-gate4,retrieval-cost-gate5,
  eg2vq-embed-entry-gate6,eg2vq-device-gate7,eg2vq-serving-caps-gate20}.txt;
  qa-evidence/qa-report.md (the latest rule 19 pass); tests/{anchor,expected,
  retrieval}/ + tests/{run.sh,qa-gate.sh}; tools/eg2_*.{c,sh} + bin/; out/ (all
  artifacts and logs, gitignored).
- The caps (unit 20): crates/ds4-core/src/gemma_vq.rs holds ITEM_TOKENS_MAX 2048
  (the device attention key plane) and BATCH_ROWS_MAX 8192 (a PORT policy: 108,548 B
  of workspace per row, 848.03 MiB at the cap), reached through
  `ds4_core::embed_caps(family)`; /health reports them under `embed`; the route
  refuses past either by name as 400; the CPU reference's rows are packed, so
  BATCH_ROWS_MAX is also the batch's row count.
- The tower (phase 4/7): the tokenizer is host-side (crates/ds4-core/src/tok/gemma_vq.rs,
  a gemma4 pre-tokenizer with the `<0xXX>` byte fallback; limits: invalid UTF-8,
  ignore_merges); the CPU reference is ds4_eg2vq_graph.inc (+ ds4_eg2vq_bind.inc,
  which decodes all 217 tower payloads into a 520 MiB f32 cache and carries BOTH
  weight legs by name); the device path is ds4_eg2vq_gpu.cuh over cuda/eg2vq_*.cuh,
  payloads consumed in the kernel (55.9 MiB plan, 137 MiB weights, workspace sized by
  max_tokens: 212.01 MiB at 2048 rows); the oracle is
  `tests/parity/eg2vq_embd_c_oracle <gguf> <ids.tsv>` (built by make).
- Shape to load: 24 layers, hidden 512, 4 q heads, 2 KV heads, head_dim 256; layers
  5/11/17/23 full (head_dim 512, 1 KV head); ffn 2048; rms eps 1e-6; vocab 262,144;
  out 768; the file's sliding_window is 1024 (the converter's doubling of the
  checkpoint's 512), so the symmetric mask is ±512; mean pool then L2 normalize.
- The vision tower (phase 8, new): in the SAME checkpoint as the text tower and DENSE
  BF16 (208 tensors under vision_tower.encoder.layers.*, 2 under
  vision_tower.patch_embedder.*), with config.json's vision_config as the shape
  authority: hidden 768, 16 layers, 12/12 heads, head_dim 64, ffn 3072, patch 16,
  pooling_kernel_size 3, position_embedding_size 10240, rope theta 100 with type
  axial, rms eps 1e-6, standardize false, use_clipped_linears false. The port does
  not read it from the checkpoint: `tools/eg2_mmproj.c` writes it to
  out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf (211 tensors, sha256 9ffb86e3..),
  and the engine opens THAT as its second model (`e->vision_model`, the companion
  GGUF pattern at ds4.c:72689, bound by name with required_tensor), exactly as the
  other families' vision GGUFs are bound (glm53_vision_weights_bind, ds4.c:9155+).
  The tie-in is embed_vision.embedding_projection.weight 512x768 (in the mmproj as
  mm.input_projection.weight [768,512]). The image markers are "<|image>" and
  "<image|>" (mtmd.cpp:955-956), image_token_id is 258880 with boi 255999 and eoi
  258882, and the audio side (audio_tower.*, audio_token_id 258881, embed_audio
  512x1536) is OUT of scope: the owner asked for vision.
- Machine: RTX 4070 SUPER (12,282 MiB, sm_89), CUDA 13.3, 28 CPUs. Rust is built with
  cargo + `--features native`; the C side via the tree's Makefile (`make ds4 ds4-server`
  rebuilds the Rust binaries in ~13 s incrementally); the canonical /data/build builder
  refuses this tree (recorded; rule 5's iterative exception applies). The arch record
  is ds4-rs/.ds4-cuda-config.mk (`CUDA_ARCH := sm_89`), so builds are card-native.
- Quality (gate 5): mean vector agreement 0.956039571 at 768 (min 0.898186175),
  MRR@10 0.951778 -> 0.935819, recall@1 0.888278 -> 0.861611, recall@10 0.983333 ->
  0.990000 on the frozen SciFact set; the owner's declared floor and the verdict are
  section 9 of docs/retrieval-cost-gate5.txt.
- ds4-rs remotes are shared with the live ds4 project (origin Baekpica/ds4-dfm-rs, fork
  JLFN/ds4-dfm-rs, publish JLFN/embeddinggemma2-ds4-rs); /data/ds4-dfm-rs is never
  modified from this workspace.
- Guardrails: global ~/.opengrok/AGENTS.md + this repo's mount at
  .opengrok/rules/project-work-rules.md + AGENTS.md (project P-1/P-2). .gitignore
  excludes .models/, ds4-rs/, out/, graphify-{out,rs-out}/ and *.o.

7. Guardrails and open items

- P-1 (no Python in the runtime, ever) and P-2 (references are citations or one-shot instruments; no ONNX/PyTorch) hold; all five P-1 checks return zero and `sha256sum -c SHA256SUMS` passes (re-verified at this boundary).
- Owner decisions in force: the container is GGUF + `gemma_vq` payload; the anchor run was approved; quality is paramount, "not less smart, just compact"; CUDA must actually be used (phase 7 did, 9.9x/18.1x per item); and (2026-10-19) the tier-tolerant quality floor recorded in section 9 of the gate-5 doc.
- CLOSED (rule 19 item 4, unit 20): the project now has tests/qa-gate.sh and tests/run.sh, so a push without fresh QA evidence fails MECHANICALLY. What it does NOT do: assert the REPORT's content beyond the surface names and the verdict line; the QA-tester's own judgement is still what makes the evidence worth anything.
- CLOSED (owner directive 2026-10-20, "We will not use llama it must be ported all"): measured that ds4-rs never called or linked the pin, and the two default-path executions were moved to opt-in — commit 3729ec4. Gate 22 is native (our reader), the anchor is `tools/eg2_vision_anchor.sh` (run once, green, 211/211 + all four frozen dumps reproduced), the tokenizer reference is committed data (`tests/expected/tok-anchor-ids.tsv`), and the whole fast path is 5.1 s. What is NOT claimed: that the pin is gone from the machine (it is still the authority we read, and the anchor still links it when asked) — only that nothing the port builds, runs or gates by default touches it.
- CLOSED (row P2.3's cap half, unit 20): the embedding-shaped caps exist, are advertised, are enforced by name and are parity-tested across the three files that carry the item ceiling. CLOSED (quality): the floor is declared and met.
- OPEN (owner, NEW): the C CLI's tokenizer gap — ds4-c refuses the artifact by name because the family's gemma4 BPE lives Rust-side, so porting it duplicates the encoder. The shipped binaries are Rust.
- DECIDED (owner, 2026-10-19): the vision tower (phase 8) IS wanted and is now the
  next work (section 3). DECIDED in the same message: the reference for it is the
  pattern the OTHER model variants use, which is the engine-level vision contract
  (probe/encode_memory over a vision weights table, ds4.h:297-310 with the GLM-5.3
  adapter as the worked example) — NOT the glmvq family, whose vision does not
  exist.
- STILL OPEN (owner): whether the AUDIO tower is wanted; whether KV checkpoints mean
  anything for an embeddings family.
- DEFERRED (owner's redirect, 2026-10-19): unit 21, batching several items into one
  tower call (phase-doc row P2.3's last slice). Its complete spec, plus the
  reference reading already done for it, is parked at docs/eg2vq-batch21-plan.txt.
  No code was written for it.
- OPEN (environment): the canonical builder cannot build this tree until it can pass `--features native`; fixing it changes a builder shared by every project on this machine, so it is PROPOSED, not done.
- OPEN (owner, NEW): the two pushes. Outer: `git push publish dev`, range d45394f..HEAD (FOURTEEN commits as of c923818; read the log for the exact list: docs/tests/tools/C plus the vision fixtures, their frozen reference, the llama-free boundary, the vision tower and gate 23 — no artifact: out/, .models/ and ds4-rs/ are gitignored; the tracked binaries in bin/ are rebuilt by the gates, and the anchor binary is deliberately NOT tracked because it links the pin). Target publish = https://github.com/JLFN/embeddinggemma2 (rule 6 wants a fresh `gh repo view` immediately before the push). Nested: `git push publish feat/ds41-dsml-carriage:refs/heads/dev`, range 29873445..HEAD (FOUR commits as of 0dd35319: the caps, the unit-20 QA corrections, the vision tower, the citation corrections). Gates already run for both: rule 10 OSV 140 packages / 0 vulnerable (both roots, at the previous boundary — re-run before an actual push), rule 19 QA fresh and PASS for unit 23 (the report covers this unit's surfaces), rule 5/12 as recorded. Rollback if a push must be undone: `git push --force-with-lease publish d45394f:refs/heads/dev` (outer) — a force-push, so the owner's call; both local histories are kept either way.
- STANDING (owner, 2026-10-19): the owner is AFK and wants automatic continuation after each handoff and compaction; pushes stay offers. Recorded in section 1.
- The unit-20 QA findings are all closed (four corrections committed, one recorded as informational); the unit-23 QA pass ran under the same recorded model (deepseek-v4.1-CC-flash) and its findings are in qa-evidence/qa-report.md.
- Nothing is in flight in either repository. The outer repo's changes since the
  unit-22 boundary are the boundary unit alone (gate 22 rewritten native, the anchor
  script added and the harness renamed, the tokenizer gate's frozen ids, SHA256SUMS,
  tests/run.sh and the four docs), committed as 3729ec4. The nested ds4-rs checkout
  is untouched: everything through unit 22, and the boundary change, is outer-repo
  work.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git log --grep "Unit: .* complete" d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git status --short && git branch -vv                 # clean, on dev, 10 unpushed commits
    tests/run.sh                                         # the fast path (5.1 s), ending in the rule 19 gate
    sha256sum -c SHA256SUMS                              # THIRTEEN entries
    tools/eg2_loader_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/artifact-manifest.tsv
    bash tests/qa-gate.sh                                # must be green before any push
    git -C ds4-rs log --oneline -3 && git -C ds4-rs status --short   # 70278c19, clean
    cuobjdump --list-elf ds4-rs/ds4_cuda.o | tail -2     # ds4_cuda.sm_89.cubin = the arch record is real
    cat ds4-rs/.ds4-cuda-config.mk                       # CUDA_ARCH := sm_89
    tail -8 out/phase5-gate.log                          # the agreement + retrieval tables
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'
    # the deep gates, when a number is questioned (each log is in out/):
    #   gate 4   (~20 min, the token dump is reused): tools/eg2_phase4_gate.sh <artifact> <baseline> tests/anchor/anchor-items.tsv .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
    #   gate 5   (minutes, the legs are reused):      tools/eg2_phase5_gate.sh
    #   gate 20  (~15 s, needs ds4-rs/ds4-server built): tools/eg2_embed_route_gate.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf
    #   gate 22  (seconds, NATIVE since 2026-10-20; ALSO step 5 of tests/run.sh):
    #            tools/eg2_mmproj_check.sh   -> EG2VQ_VISION_MMPROJ_PASS
    #            tests/vision/ holds the two fixtures and their four frozen dumps
    #   the anchor, OPT-IN and the only thing that runs the pin (~10 s, needs
    #   /data/llama-anchor-build): tools/eg2_vision_anchor.sh
    #   the tokenizer gate against the committed ids (no pin):
    #            tools/eg2_tok_check.sh <artifact> tests/anchor/anchor-items.tsv .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
    #   gate 23  (unit 23, in the fast path as step 6): tools/eg2_vision_gate23.sh

Trust git over this document's headings: continue with the first unit whose work is
not in the diff, which at this boundary is section 3 (unit 24, the vision tower on
the DEVICE — the CPU leg of unit 23 runs and gate 23 is green, and the port
executes no pin code). Every phase through 7 is landed; unit 21 (batching) is
DEFERRED with its spec parked at docs/eg2vq-batch21-plan.txt. The outer repo's HEAD
is HEAD with its published dev at d45394f (FOURTEEN unpushed as of c923818); ds4-rs is at
HEAD with its published dev at 29873445. Anything else in either unpushed range
must be read before sections 2-6 are trusted. The artifacts, the retrieval source,
every out/ log and the CUDA objects live under gitignored paths (out/, .models/,
ds4-rs/), so they never appear in a diff: they are identified by the digests in
SHA256SUMS, by the pins inside the gate scripts, and by the evidence docs' own
digests.

This document is self-contained: a new session can execute the remaining unit from
this file alone. History is deliberately absent — git log, the phase doc and the
evidence docs are the historical record.
