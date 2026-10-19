---
project: embeddinggemma2
plan_start_commit: d42fc8fdf8cef8dc0f6806d249160ae5472abe77
last_updated_commit: 8fde076
branch: dev
remote: 
handoff_written_at_context_usage: 14%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-19T23:25:00+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ in ds4-rs (rewritten 2026-10-19 at the unit-20 boundary: the landed phases are rotated out into git and the evidence docs, and section 3 is the ONE unit that is left)

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

2. Current state (2026-10-19, at the unit-22 boundary: the vision oracle landed and is frozen, both repositories clean, and the next work is the vision tower's port)

- OWNER DIRECTIVE, 2026-10-19 (verbatim intent): "we need to add vision also see how the other model variant handle it". This answers the long-open phase-8 decision in favour of vision (the audio tower is NOT in scope; the audio decision stays open). It defers, and does not cancel, unit 21 (batching): its complete executable spec is parked at docs/eg2vq-batch21-plan.txt, together with the reference reading already done for it, so the resumed unit does not repeat either.
- WHAT THE ARTIFACT ACTUALLY CARRIES (measured this session, docs/tensor-inventory.tsv + out/artifact-kv.txt): of 1376 tensors, 208 are vision_tower.encoder.layers.* (16 layers x 13), 2 are vision_tower.patch_embedder.* (input_proj.weight 768x768, position_embedding_table 2x10240x768), and 744 are audio_tower.* with its frontend; the text tower we ported is 217 of the 461 vqblob payloads, and the VISION TOWER IS DENSE BF16, not VQ. The two tie-in projections are in-file: embed_vision.embedding_projection.weight 512x768 and embed_audio.embedding_projection.weight 512x1536.
- The image path, read from the reference at the pin: image -> vision tower (config.json vision_config: hidden 768, 16 layers, 12/12 heads, head_dim 64, ffn 3072, patch 16, pooling 3, rope theta 100 axial, rms eps 1e-6, standardize false) -> 3x3 average pool -> scale sqrt(768) -> weightless RMS -> 768-to-512 projection -> N soft tokens injected at the image-token spans of the text tower we already have. The graph is /data/llama.cpp @ 31288c9c tools/mtmd/models/gemma4v.cpp, 133 lines, and the projector variant is ANSWERED from conversion/gemma.py:1001 (GEMMA4V; our checkpoint maps to EmbeddingGemma2VisionAudioModel at :1055, which subclasses Gemma4VisionAudioModel) — phase-doc row P4.4 is no longer open.
- THE OBSTACLE, stated plainly: no shipped binary at the pin embeds an image (examples/embedding/embedding.cpp has no mtmd/mmproj/image support; tools/mtmd/mtmd-cli.cpp has no embedding or pooling mode) and the Python ML stack was removed by owner instruction, so the pin's own converter cannot run here either. The reference must therefore be an instrument we build, exactly as the baseline text GGUF was: our own writer emits an mmproj, and a small harness linked against the anchor tree's mtmd dumps mtmd_get_output_embd for a frozen image. Section 3 specifies it.
- UNIT 22 IS DONE (phase 8): the vision oracle exists, is reference-accepted and is frozen. `tools/eg2_mmproj.c` writes an mmproj from the checkpoint (211 tensors: the patch conv with the converter's own permute, the two axial position tables, the 16 blocks, `mm.input_projection`); the norms and the position table are WIDENED to f32 because the reference's own graph refuses the mixed-type elementwise op (measured: `binary_op: unsupported types: dst: f32, src0: f32, src1: bf16`, ggml-cpu/binary-ops.cpp:135), everything else keeps the checkpoint's bf16. The pin's mtmd ACCEPTS it and reads back all 211 tensors in agreement with the writer's manifest; the preprocessing it performs is 528x384 -> 88 soft tokens (33x24 patches / 9), and its own dump of `mtmd_get_output_embd` is 88 x 512. `tests/parity/eg2vq_vision_oracle.cpp` dumps that plus the pooled 768-dim embedding of the spliced sequence (92 positions: 2 text + 88 image + 2 text) by ONE `llama_process` over a `llama_batch_ext` — and both defects it hit on the way are recorded because they are the unit-25 design constraints: `mtmd_helper_eval_chunks` pools PER CHUNK (two different images gave byte-identical pooled vectors), and the public `llama_batch` cannot carry a mixed token+embedding batch at all (it segfaults). Gate: `tools/eg2_mmproj_check.sh`, 6.0 s, `EG2VQ_VISION_ORACLE_PASS`, now step 5 of tests/run.sh's fast path, log out/eg2vq-vision-oracle-gate22.log, evidence docs/eg2vq-vision-oracle-gate22.txt. Frozen: two ffmpeg-generated fixtures plus their four dumps in tests/vision/reference/, all six digests in SHA256SUMS (twelve entries now).
- LEARNED THE HARD WAY, for whoever writes the next llama.cpp-linking instrument: the library is `libllama`, so the link flag is `-lllama` and NOT `-llama` (the latter means `liblama.so`); link the anchor build's .so files by path. The mtmd library was not built in the anchor tree and had to be (`cmake --build /data/llama-anchor-build --target mtmd -j`); that is additive to the phase-2 reference build.
- Workspace: /data/embeddinggemma2, branch dev, HEAD = the commit that carries this handoff (unit 22's writer, harness, gate, fixtures and frozen reference), tree CLEAN after it. The outer repo has no origin, only publish = https://github.com/JLFN/embeddinggemma2, whose dev is at d45394f — so EIGHT commits are unpushed (the five through 2f52c12, then d42fc8f, 8fde076 and the unit-22 commit). The push is an OFFER, recorded in section 7.
- Nested ds4-rs: CLEAN at 70278c19 on feat/ds41-dsml-carriage, TWENTY commits ahead of fork/feat/ds41-dsml-carriage; its publish = https://github.com/JLFN/embeddinggemma2-ds4-rs stands at 29873445 (refs/heads/dev), so TWO ds4-rs commits are unpushed: 68837a53 (the caps) and 70278c19 (the QA corrections). Push with `git push publish feat/ds41-dsml-carriage:refs/heads/dev`; plain `git push publish dev` fails (the local dev tracks fork/dev and is behind).
- Unit 20 (phase 6, row P2.3, slice 1) is DONE and merged: the family now has an EMBEDDING-SHAPED cap entry instead of a decode-shaped one. `ds4_core::embed_caps(ModelFamily::GemmaVq) = Some(EmbedCaps { item_tokens_max: 2048, batch_rows_max: 8192 })`, `None` for every decode family; reported as `controls.embed` in the plan JSON and as `embed` by GET /health; refused by name as 400 in `run_embed` and before the FFI call in `Model::embed_gemma_vq`; the item ceiling is held equal across Rust, ds4_eg2vq.h and cuda/eg2vq_attn.cuh by the new crates/ds4-core/tests/eg2vq_serving_caps.rs; `family_native_limit(GemmaVq)` is that ceiling and DS4_GEMMAVQ_PREFILL_CHUNK cannot lift it. Live: tools/eg2_embed_route_gate.sh sections 4b/4c, log out/eg2vq-route-gate-unit20b.log, exit 0, EG2VQ_EMBED_ROUTE_PASS. Evidence: docs/eg2vq-serving-caps-gate20.txt.
- The owner's declared quality floor for this tier is RECORDED and MET (docs/retrieval-cost-gate5.txt section 9; RECIPE.md 5b): recall@10 >= 0.983333, MRR@10 >= 98% of baseline (0.932742), recall@1 >= 0.855, mean cosine >= 0.95 — margins 0.006667 / 0.003077 / 0.006611 / 0.006040, all PASS. In percentages at 768: vectors keep 95.60% agreement, MRR@10 98.32%, recall@1 97.00%, recall@5 99.59%, recall@10 100.68% (better), and 7.67% of queries change top-1. The floor covers the 768 width only; 512/256/128 have none.
- Rule 19 is now MECHANICAL: tests/qa-gate.sh (a copy of the canonical template with three recorded deviations: QA_BASE = publish/dev because there is no origin; the three surface patterns retargeted at C/Rust/shell; it scans BOTH repositories, because ds4-rs is gitignored in the outer tree and the first version therefore passed a unit with no surfaces visible) and tests/run.sh (the fast path: digests, container, scorer self-test, loader gate 3, gate 22 the vision oracle, QA gate; `--deep` adds gate 1b, cargo test -p ds4-core and gate 4). Verified: red on the committed unit, green after the QA report; `bash tests/qa-gate.sh` exit 0 as of this boundary.
- The unit-20 QA pass (deepseek-v4.1-CC-flash) returned `verdict: overall PASS`, qa-evidence/qa-report.md, and found five things: four wrong/misleading sentences (a 1400-word item is 1410 tokens, not ~1057; 212.0078 MiB not 212.0086; two "section 7" pointers that meant 5c/5b; and the row-budget message's "the 8 inputs add up to 8460") and one informational run-to-run difference in the device path's last printed digit. All four corrections are committed (ds4-rs 70278c19, outer 2f52c12) and recorded in docs/eg2vq-serving-caps-gate20.txt section 6.
- What is left of the plan, in order: (1) the VISION phase (section 3: the oracle first, then the CPU tower, the device tower and the serving splice); (2) unit 21, the batching of several items into one tower call, DEFERRED with its spec at docs/eg2vq-batch21-plan.txt; (3) the C CLI's tokenizer gap (ds4-c refuses the artifact by name; the family's gemma4 BPE is Rust-side, so porting it duplicates the encoder) — owner's call; (4) the AUDIO tower — owner's call; (5) the two pushes, when the owner answers.
- Nothing measured to date claims: an MTEB-comparable score; a serving-shaped latency or concurrency number for a batch (the tower still runs one item at a time); anything about the vision and audio towers; or any quality verdict past the four declared floors.
- Machine-facing facts re-verified this session: artifact sha256 5bcf2216.., baseline 0d84f1a9.., container 8902fb8a.., ids out/phase5-ids.tsv, the -O3 oracle a9f74d0b.., ds4-rs 70278c19, `sha256sum -c SHA256SUMS` all five OK, loader gate 3 1376 tensors + 8 refusals, scorer self-test 8 cases, `CUDA_ARCH := sm_89` with the sm_89 cubin present, and rule 10 OSV 140 packages / 0 vulnerable in BOTH repos (run at this boundary).



3. Next unit plan — unit 23: the vision tower on the CPU reference path (phase 8, row P4.1)

Goal: the vision forward in ds4-rs, op for op from the reference graph, gated
against the frozen oracle unit 22 produced (tests/vision/reference/, two fixtures,
88 soft tokens x 512 dims each plus the pooled 768-dim vector of the spliced
sequence). The reference itself is read, not remembered:
tools/mtmd/models/gemma4v.cpp:5-117 is the graph, and unit 22's evidence doc
(docs/eg2vq-vision-oracle-gate22.txt) records the preprocessing contract it sits
behind (align 48 bicubic, 1/255 rescale with no normalize per the checkpoint's
processor_config.json, patch grid from the aligned size, tokens = patches / 9).

STEP 0, before any code: read how THIS tree already decodes an image. The shipping
vision families take encoded image bytes (ds4_engine_vision_probe /
vision_encode_memory take `const uint8_t *encoded`), so the decoder exists
somewhere in ds4-rs — ds4_glm53_image.inc is the GLM-5.3 one, and Ling3VL and
Step-3.7 have their own. Mirror that path; do not write a second image decoder and
do not add a dependency the tree does not already have. If the sibling decoders
live outside this repo (vendored from /data/ds4-dfm-rs), say so in the evidence and
read the vendored source before using it (rule 0.2).

Then, in the order the change has to land:

 1. ds4_eg2vq_vision.h (new): the tower's shape constants and the entry points, in
    the grain of ds4_eg2vq.h — vision hidden 768, 16 layers, 12 heads / 12 KV,
    head_dim 64, ffn 3072, patch 16, pooling 3, position_embedding_size 10240,
    rope theta 100 axial, rms eps 1e-6, output 768 -> 512 — every number read from
    the checkpoint's config.json, and the mmproj's key set
    (docs/eg2vq-vision-oracle-gate22.txt) is the cross-check that the names mean
    what they say.
 2. the preprocessing: image bytes -> an aligned patch grid -> per-patch rows, with
    the two axial position indices per patch, reproducing the reference exactly
    (the alignment, the bicubic kernel, the 1/255 rescale, the pad). This is the
    part that decides whether the port can match the oracle at all, so it lands
    FIRST and gets its own comparison: for the frozen fixtures, our patch rows and
    position indices against the reference's own readout (mtmd prints the grid;
    the patch values are checkable through the harness by dumping its
    preprocessing — extend tests/parity/eg2vq_vision_oracle.cpp with a
    `--dump-patches` mode rather than guessing).
 3. the tower: patch conv (the permuted [16,16,3,768] weight — the analyzer must
    undo unit 22's permute or read the artifact's own [768][768] layout; state
    which), the two axial position tables added before the blocks, 16 blocks of
    RMS norms (the four per layer, in the graph's order: ln_1 pre-attention,
    attn_post_norm, ln_2 pre-FFN, ff_post_norm), q/k/v/o, per-head q/k RMS on the
    64-wide heads, weightless V RMS, axial NeoX RoPE (first half on pos_x, second
    half on pos_y at offset n_dim/2, theta 100), attention at kq_scale exactly
    1.0 with NO sliding window and NO causal mask, the 3x3 average pooler, the
    sqrt(768) scale, the weightless RMS and the 768->512 projection.
 4. the gate: our soft tokens against tests/vision/reference/*.soft.tsv and our
    pooled vector against *.text.tsv, with the tolerance DECLARED BEFORE the run.
    The tolerance has to contain one measured, known divergence: the oracle's
    norms and position table are f32 (the reference graph refuses bf16 there) while
    the artifact our port reads carries them bf16, so the two are not bit-identical
    by construction. Measure the difference and state it; do not widen the bound
    until it passes silently.
 5. then wire it to the engine (a ds4_engine_embed_gemma_vq-shaped image entry,
    unit 25 covers the bridge/route side) and rotate this handoff.

WHAT UNIT 23 MUST NOT DO: invent a preprocessing, a norm order, or an image
decoder. Every one of those has a reference already — mtmd's preprocessor, the
graph's own sequence in gemma4v.cpp + clip.cpp, and the tree's existing decoders.

4. Reference map — the exact repos and files (pinned; verified at this boundary)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (all the work) | /data/embeddinggemma2/ds4-rs @ 70278c19, branch feat/ds41-dsml-carriage, gitignored by the outer repo, own git | ds4.c, ds4.h, ds4_eg2vq.{h,inc,graph.inc,gpu.cuh}, ds4_glmvq*.{h,inc}, cuda/eg2vq_{attn,ops,vq}.cuh, cuda/*.cuh, crates/ds4-core/src/**, crates/ds4-server/src/**, crates/ds4-sys/src/lib.rs, tests/parity/, the Makefile |
| Form and metadata authority | /data/llama.cpp @ 31288c9c (b11723, MIT, read-only, source only) | src/models/gemma-embedding2.cpp, src/llama-{arch,hparams,model,graph,vocab,batch}.cpp, ggml/include/gguf.h, conversion/{gemma,base}.py, gguf-py/gguf/**, examples/embedding/embedding.cpp, tools/mtmd/** (phase 8) |
| The VISION reference (phase 8) | /data/llama.cpp @ 31288c9c, the same pin | tools/mtmd/models/gemma4v.cpp (the graph, 133 lines), tools/mtmd/clip.cpp (:973-979 the builder dispatch, :1719-1736 the GEMMA4V hparams, :4419-4428 the image-token count, :1342-1352 the mmproj keys), tools/mtmd/clip-impl.h:112-157 (the mmproj tensor names), tools/mtmd/mtmd.cpp:953-959 and mtmd-image.cpp:122-156 (the preprocessing), clip-model.h:204-211 (the token-to-pixel limits), conversion/gemma.py:978-1063 (the projector variant and the tensor mapping) |
| The anchor build (phase 2) | /data/llama-anchor-build (CPU) | bin/llama-embedding; the frozen vectors it produced are phase 2's deliverable |
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
- The vision tower (phase 8, new): in the SAME artifact as the text tower and DENSE
  BF16 (208 tensors under vision_tower.encoder.layers.*, 2 under
  vision_tower.patch_embedder.*), with config.json's vision_config as the shape
  authority: hidden 768, 16 layers, 12/12 heads, head_dim 64, ffn 3072, patch 16,
  pooling_kernel_size 3, position_embedding_size 10240, rope theta 100 with type
  axial, rms eps 1e-6, standardize false, use_clipped_linears false. The tie-in is
  embed_vision.embedding_projection.weight 512x768 (vision 768 -> text 512). The
  image markers are "<|image>" and "<image|>" (mtmd.cpp:956-957), image_token_id is
  258880 with boi 255999 and eoi 258882, and the audio side (audio_tower.*,
  audio_token_id 258881, embed_audio 512x1536) is OUT of scope: the owner asked for
  vision.
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
- OPEN (owner, NEW): the two pushes. Outer: `git push publish dev`, range d45394f..HEAD (EIGHT commits: docs/tests/tools/C plus the vision fixtures and their frozen reference — no artifact: out/, .models/ and ds4-rs/ are gitignored; the tracked binaries in bin/ are rebuilt by the gates), target publish = https://github.com/JLFN/embeddinggemma2 (PUBLIC with dev as default, verified when it was created; rule 6 wants a fresh `gh repo view` immediately before the push). Nested: `git push publish feat/ds41-dsml-carriage:refs/heads/dev`, range 29873445..70278c19. Gates already run for both: rule 10 OSV 140 packages / 0 vulnerable (both roots, this boundary), rule 19 QA fresh and PASS for unit 20, rule 5/12 as recorded. Rollback if a push must be undone: `git push --force-with-lease publish d45394f:refs/heads/dev` (outer) — a force-push, so the owner's call; both local histories are kept either way.
- STANDING (owner, 2026-10-19): the owner is AFK and wants automatic continuation after each handoff and compaction; pushes stay offers. Recorded in section 1.
- The unit-20 QA findings are all closed (four corrections committed, one recorded as informational).
- Nothing is in flight in either repository. The outer repo's only changes since the
  pre-unit-22 boundary are unit 22's own (the writer, the harness, the gate, the two
  fixtures, the four frozen dumps, the evidence doc, SHA256SUMS and tests/run.sh's
  new step 5), all committed together with this handoff. The nested ds4-rs checkout
  is untouched by unit 22: the vision oracle is an outer-repo instrument.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git log --grep "Unit: .* complete" d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git status --short && git branch -vv                 # clean, on dev, 5 unpushed commits
    tests/run.sh                                         # the fast path, ending in the rule 19 gate
    sha256sum -c SHA256SUMS                              # five entries
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
    #   gate 22  (6 s, the vision oracle; ALSO step 5 of tests/run.sh's fast path):
    #            tools/eg2_mmproj_check.sh   -> EG2VQ_VISION_ORACLE_PASS
    #            tests/vision/ holds the two fixtures and their four frozen dumps

Trust git over this document's headings: continue with the first unit whose work is
not in the diff, which at this boundary is section 3 (unit 23, the vision tower on
the CPU reference path — unit 22's oracle is landed and frozen). Every phase through
7 is landed; unit 21 (batching) is DEFERRED with its spec parked at
docs/eg2vq-batch21-plan.txt. The outer repo's HEAD is the commit carrying this file
with its published dev at d45394f; ds4-rs is at 70278c19 with its published dev at
29873445. Anything else in either unpushed range must be read before sections 2-6
are trusted. The artifacts, the retrieval source, every out/ log and the CUDA
objects live under gitignored paths (out/, .models/, ds4-rs/), so they never appear
in a diff: they are identified by the digests in SHA256SUMS, by the pins inside the
gate scripts, and by the evidence docs' own digests.

This document is self-contained: a new session can execute the remaining unit from
this file alone. History is deliberately absent — git log, the phase doc and the
evidence docs are the historical record.
