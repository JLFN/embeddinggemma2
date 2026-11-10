---
project: embeddinggemma2
plan_start_commit: 9f252ef
last_updated_commit: 7a94441f95eabbd35aef4f341d7f9b09b7d94529
branch: feat/eg2vq-attn-perf
remote: 
handoff_written_at_context_usage: 28%
handoff_written_at: 2026-11-10T14:25:02+01:00
session_id: 01a1c513-c770-7f32-943d-da6a4603e4da
boundary_state: clean/unpushed
---
--- handoff probe (auto) ---
=== handoff probe (embeddinggemma2) ===
--- units completed since plan start (9f252ef):
  (none)
--- progress diff stat:
README.md                               | 176 ++++++++++++++++++
 RECIPE.md                               |   4 +
 docs/USAGE.md                           | 296 ++++++++++++++++++++++++++++++
 docs/ds4-gemma-port-phases.md           |   2 +-
 docs/eg2vq-batch-gate27.txt             |  14 ++
 docs/eg2vq-embed-media-gate26.txt       |  18 ++
 embeddinggemma2-handoff.md              | 251 +++++++++++++++----------
 qa-evidence/qa-findings-attn-grouped.md | 176 ++++++++++++++++++
 qa-evidence/qa-findings-attn-perf.md    |  94 ++++++++++
 qa-evidence/qa-findings-vq-r8.md        | 111 +++++++++++
 qa-evidence/qa-report.md                | 313 ++++++++++++--------------------
 tests/run.sh                            |   8 +-
 12 files changed, 1169 insertions(+), 294 deletions(-)
--- staleness (last_updated_commit e5d9e9ce30d5b1223999a0a35bc67d8e722a5e36 .. HEAD):
  (clean: handoff is current)
--- end handoff probe ---


Handoff + executable plan — EmbeddingGemma-2 VQ in ds4-rs (rotated 2026-11-10 after the SIXTH performance unit: the two norm primitives run one warp per row, the forward is 49.2 -> 40.0 ms of kernel time and the corpus gains 1.17x at batch 8; section 3 now points at the 768 projection, the largest kernel left that is not a VQ payload GEMM)

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

THIRD OWNER DIRECTIVE (2026-10-20, verbatim): "lets ctoninue the QA should not run
before we are complete with the entire code". DISCHARGED: the port was completed by
unit 27 (the plan's last code row) and the single accumulated pass ran as unit 28 --
the deferral file is deleted, qa-evidence/qa-report.md reads `verdict: overall PASS`
(fresh, covering all eleven extracted surfaces and the hand-read multi-line entries),
tests/qa-gate.sh is green in the fast path (step 11), and the pass's three findings
were each fixed with the check that confirms it (qa-evidence/qa-findings-u28.md).
Nothing about rule 19 remains open.

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

2. Current state (2026-11-10, re-verified IN THIS SESSION at the SIXTH performance unit's boundary: the tower's norms run one warp per row; feat/eg2vq-attn-perf is the published branch with PR #1 open in each repo, dev untouched in both)

- THE SIXTH PERFORMANCE UNIT, LANDED LOCALLY: engine c7ae8da0 (the reduction, the two harnesses, the Makefile and .gitignore rules, the record); the branch itself is published at c207e97f, and the outer handoff commit is the other half of the local-only set. No push has been made or authorized since the owner's one push.
- WHAT IT IS: cuda/eg2vq_ops.cuh's two norm primitives now run ONE WARP PER ROW — the lane's strided slice read once into registers, the sum of squares folded with a __shfl_xor_sync butterfly (five shuffles, no shared memory, no block barrier), one template instantiation per per-lane count (2/4/8/16/32, i.e. widths to 1,024), the old block kernels kept as eg2vq_ops_rms_block for anything wider. Same arithmetic as the graph (double sum, double mean, cast before the eps, x*scale*w, ds4_eg2vq_graph.inc:44-53); only the summation tree changed.
- THE PROOF THAT NO OTHER LEG MOVED: the two forms are BIT-IDENTICAL at every width the text and vision towers run — 0 of 283,648 row values and 0 of 242,944 per-head values differ, max 0 ULP (tests/test_eg2vq_ops.cu), and the vision gates 23/24 print their recorded digits.
- MEASURED, one window, both forms interleaved in one process: 3.07x a launch at 512 x 2,451 rows (0.0622 -> 0.0203 ms), 2.54x on the 58,824-row PLE norm, 2.63x on the q head norm, 2.52x on k/v; the two kernels 14.08 -> 4.90 ms and the forward 49.2 -> 40.0 ms of kernel time.
- END TO END, both round pairings, old build c207e97f: corpus batch 4 1.148x/1.142x, batch 8 1.171x/1.161x, batch 1 1.025x/1.018x. The sweep's single-item p50 row is NOT this unit's number (24.5 vs 14.7 ms in the sweep; a controlled 12-request re-measurement has both builds at 14.3-15.5 ms).
- GATES: test-eg2vq-ops / -attn / -vq PASS; gate 7 unmoved at min cos 0.999999768 and max rel_rms 6.813e-04; gate 27 PASS at 0.000e+00 and 21.56x; tests/run.sh fast path exit 0 through gates 22-27 and the QA gate; P-1's five checks zero; OSV 140 packages / 0 vulnerable.
- THE DISPATCH DECISION: no row-count threshold. A 27-row call is 0.75x on the warp form (1.4 us a launch) and that is accepted, because a dispatch that picks arithmetic from a call's size is exactly the contract unit 27's byte identity must not have.
- THE RECORD: ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/norm-warp-reduction.md, norm-warp-ab-bench{,-run1}.txt, nsys-kern-sum-b8-norm-warp.txt, gate7-norm-warp.txt, corpus-norm-ab-{new,old}-{1,2}.csv with their health readouts, norm-batch8-request.json.
- NOT RE-RUN: the deep path (tests/run.sh --deep), gate 20's route gate, the loader digest, gate 5, and the engine repo's stale rule-19 mount. The unit is one kernel header, two harnesses, a Makefile rule and documentation.

3. Next unit plan — THE 768 PROJECTION, now the largest kernel left that is not a VQ payload GEMM

(1) THE UNIT, AND WHY IT IS NEXT. MEASURED AT THIS BOUNDARY, not extrapolated:
    the fresh nsys profile of the same 8-item / 2,451-token batch on the build
    that just landed (pages 001-008 of out/live-demo/pages, 42+162+11+104+624+
    481+424+603 tokens; raw: ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/
    nsys-kern-sum-b8-norm-warp.txt) reads 40.0 ms of kernel time, down from
    49.2 ms, and the split is now:
      VQ GEMM     19.39 ms (48.5%): mma<64> 10.78 / 144 launches, mma<128> 8.61
      proj768      7.13 ms (17.8%): ONE launch, 2,451 rows x 512 x 768
      attention    5.41 ms (13.5%): slide 3.83 / 20 launches, full 1.58 / 4
      norms        4.90 ms (12.2%): warp<16> 3.18 / 133, warp<8> 1.72 / 60
      other ops    2.79 ms (7.0%): add 0.76, scale 0.69, gelu 1.08, rope 0.52,
                  embed 0.06, pool 0.04
    proj768 is 1.93 GFLOP in 7.13 ms = 271 GFLOP/s, i.e. 1/100 of what the
    card does on the payload GEMM, and it is NOT launch-bound (one launch).
    Freeze the baseline in the same window before touching anything:
    ./tests/bench_eg2vq_ops 200 and ./tests/bench_eg2vq_vq 7.

(2) WHAT IT IS TODAY, read from the kernel before anything is changed:
    cuda/eg2vq_ops.cuh's eg2vq_ops_proj768_kernel — one block of 256 threads
    per ROW. Prologue: the output_norm RMS of the row (double sum, the same
    shared tree the norms used to have) into a dynamic-shared e_s of n_embd
    floats. Then each thread owns a few of the 768 outputs and walks all 512
    weights of that output's bf16 row, widening each by shift+memcpy and
    accumulating in f32. Every block reads the whole 768x512 bf16 matrix
    (786 KB, L2-resident) and the inner loop is scalar.
    Records that must survive: the result goes to the ROW's own plane
    (acc + row * n_out) so an item's numbers cannot depend on which items
    share the call (unit 27), and the accumulation order the reference uses is
    the CPU binder's f32 dot (ds4_eg2vq_graph.inc:339-356).
    Candidates, cheapest first:
      - A bf16 m16n8k16 GEMM of [rows x 512] x [512 x 768] with f32
        accumulators, the operands EXACTLY as the file stores them (bf16, so
        the widening is exact and only the accumulation order changes), the
        output_norm as the prologue of the row tile, and the row's own plane
        as the epilogue. The pattern to port is this project's own payload
        GEMM (cuda/eg2vq_vq.cuh's mma kernel) and the dense reference's
        (feat/spark2_5-family, cuda/spark2_5_vq_gemm.cuh) — PR 142, the branch
        the owner pointed at for the dense tower.
      - A shared-tile f32 GEMM (4-8 rows per block, the weight tile staged
        once) if the tensor-core leg's per-row-plane epilogue turns out to be
        the constraint; it keeps the dot's order and only amortizes the read.
      - NOT a change to the expression: the norm, the plane and the pool stay
        where the graph puts them. A different accumulation order is allowed
        only if gate 7 measures it inside its declared bound, and the bound is
        reported before the run, not after.

(3) THE METHOD, unchanged from the last four units: one change per unit, A/B'd
    inside one window with both round pairings, the bench harness extended
    first (bench_eg2vq_ops has the norm rows; the projection needs its own
    row at 2,451 / 294 / 27 rows), gate 7 plus the interleaved 143-page corpus
    sweep (batches 1, 4 and 8) as the end-to-end evidence, `make test-eg2vq-*`
    for the kernel gate, tests/run.sh and gate 27 for the contract legs, and
    REG/STACK/LOCAL/SHARED from cuobjdump reported. The bench binary compiles
    the header in, so rebuild it after every header edit.

(4) THE OWNER'S STANDING LIST, unchanged: the C CLI's tokenizer gap; the AUDIO
    tower (out of scope); a MULTI-IMAGE item; retiring the inert QA deferral
    branch in tests/qa-gate.sh; the three branch-standard decisions (default
    branch, the chore/fmt-workspace unit at c595fca3 whose gate is red,
    retiring the duplicate feat/ds41-dsml-carriage); and the engine repo's
    inherited stale rule-19 mount (ds4-rs/tests/qa-gate.sh, red over another
    project's surfaces — the project's mount is the OUTER one). A kernel-level
    QA pass over the tensor-core arithmetic (three units now) is the open
    discipline item; the rule-19 extractor cannot see a kernel header.

(5) PUSH DISCIPLINE: unchanged. feat/eg2vq-attn-perf is published with PR #1
    open in each repo (tip c207e97f engine, 5786e6f outer). The local-only set
    is engine c7ae8da0 plus the outer docs commits. A push is an OFFER: run the
    four checks fresh (rule 6 `gh repo view` on both targets, rule 10 OSV,
    rule 19's gate, rule 23's branch/range checks against publish/dev, since
    this fork has no origin/dev), report branch, range, visibility, gate
    results and the exact rollback, and wait for the owner's answer.

4. Reference map — the exact repos and files (pinned; verified at this boundary)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (all the work) | /data/embeddinggemma2/ds4-rs @ 10e73aa0, branch dev (published as publish/dev), gitignored by the outer repo, own git | ds4.c, ds4.h, ds4_eg2vq.{h,inc,graph.inc,gpu.cuh}, ds4_eg2vq_vision.{h,inc}, ds4_glmvq*.{h,inc}, cuda/eg2vq_{attn,ops,vq,vis}.cuh, native/bridge/ds4_bridge.{c,h}, crates/ds4-core/src/**, crates/ds4-server/src/**, crates/ds4-sys/src/lib.rs, tests/parity/, tests/test_eg2vq_{attn,ops}.cu, the Makefile |
| Form and metadata authority | /data/llama.cpp @ 31288c9c (b11723, MIT, read-only, source only) | src/models/gemma-embedding2.cpp, src/llama-{arch,hparams,model,graph,vocab,batch}.cpp (llama-batch.cpp's ubatch packing and llama-graph.cpp's fill_mask are unit 27's semantics), ggml/include/gguf.h, conversion/{gemma,base}.py, gguf-py/gguf/**, examples/embedding/embedding.cpp, tools/mtmd/** (phase 8) |
| The VISION reference (phase 8) | /data/llama.cpp @ 31288c9c, the same pin | tools/mtmd/models/gemma4v.cpp (:5-117), tools/mtmd/clip.cpp (:337-450, :144-147, :254-256, :973-979, :1719-1736, :4419-4428, :1342-1352), clip-impl.h:112-157, mtmd.cpp:953-961, mtmd-image.cpp:57-96 / :122-156 / :204-390 / :7-14, clip-model.h:66-68 / :204-211, mtmd-helper.cpp:404, conversion/gemma.py:978-1063 |
| The WIRE's own contract | our own committed data | .models/EmbeddingGemma-2-914f7f8/README.md section 5 (the model card: `<|image|>` marks the position, prefixes apply to text only, the 70..1120 budget) and config.json's ids (image 258880, boi 255999, eoi 258882) |
| The VISION weights file, and the frozen image reference | our own committed data | out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf (sha256 9ffb86e3.., 211 tensors, written by tools/eg2_mmproj.c) and tests/vision/{eg2-vision-gradient-512x384.png,eg2-vision-pattern-512x384.png} with tests/vision/reference/*.{soft,text}.tsv (the pin's own numbers, produced once by tools/eg2_vision_anchor.sh) |
| The anchor build (phase 2) | /data/llama-anchor-build (CPU, plus an mtmd target built for the vision anchor) | bin/llama-embedding; the frozen text vectors it produced are phase 2's deliverable, and nothing but tools/eg2_vision_anchor.sh touches this tree |
| Writer and payload precedent | /data/YoungAi (a4e6d31) | src/common/ds4_gguf_write.h, gguf-tools/quantize/v41_to_gguf.c, vq_fmt.h, vq_qc_bytes.h |
| Verification method | /data/ds4-dfm-rs (the live ds4 project; never modified from here) | AGENTS.md, tests/parity/, Makefile (the false-green guard) |
| The anchor vectors | our own committed data | tests/anchor/anchor-items.tsv (113 items) and tests/expected/anchor-embeddinggemma2.jsonl (768/512/256/128 vectors, sha256 7561254c..); tools/eg2_anchor.sh / eg2_anchor_cmp.sh |
| The artifact, and the container behind it | our own committed recipe | out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf (sha256 5bcf2216.., 1376 tensors), out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf (0d84f1a9..), out/...-dqvd.bin (8902fb8a..), out/artifact-manifest.tsv, out/artifact-kv.txt |
| The family's contract, loader, tower | our own code | crates/ds4-core/src/gemma_vq.rs (+ the caps, the cost model and the image constants), gemma_vq/layout.rs, tok/gemma_vq.rs; ds4_eg2vq_graph.inc + ds4_eg2vq_bind.inc; tools/eg2_{loader_check,phase4_gate,tok_check}.sh; docs/{loader-gate3,text-tower-gate4}.txt |
| The gate-5 harness and the frozen set | our own code and data | tools/eg2_retrieval_set.sh, tests/retrieval/scifact-test-{queries,pool,qrels}.tsv (gate 27's two fixtures are cut from the queries and the pool), tools/eg2_{phase5_gate,oracle_to_jsonl,retrieval_score,retrieval_score_selftest}.sh, docs/retrieval-cost-gate5.txt |
| The embedding entries, their caps, the wire and the batch (phases 6-8) | our own code | ds4.c's ds4_engine_embed_gemma_vq{,_image,_batch}, ds4_bridge_eg2vq_*, crates/ds4-core/src/{serving.rs,gemma_vq.rs}, crates/ds4-core/tests/eg2vq_serving_caps.rs, crates/ds4-core/examples/{gemma_vq_tok,gemma_vq_media_ids,gemma_vq_vocab}.rs, crates/ds4-server/src/{serve.rs,parse.rs,generate.rs,serve_embed_media_test.rs,serve_embed_batch_test.rs}, tests/parity/eg2vq_embd_gpu_oracle.c (--batch/--batch-synth/--batch-bad/--no-cpu), tests/parity/eg2vq_splice_cuda_oracle.c, tools/eg2_{embed_route_gate,embed_media_gate26,embed_batch_gate27}.sh, docs/{eg2vq-serving-caps-gate20,eg2vq-embed-entry-gate6,eg2vq-embed-media-gate26,eg2vq-batch-gate27,eg2vq-batch21-plan}.txt |
| The device path (phase 7) | our own code | cuda/eg2vq_*.cuh, ds4_eg2vq_gpu.cuh (the layout and its cost model), tests/parity/eg2vq_embd_gpu_oracle.c, docs/eg2vq-device-gate7.txt, out/cuda-objects-backup-sm121a |
| The labelled data, and its origin | BEIR SciFact (Thakur et al., NeurIPS 2021 D&B) | the zip URL pinned in tools/eg2_retrieval_set.sh (ZIP_SHA 536e1444..); tests/retrieval/PROVENANCE.md records CC BY-NC 2.0 |
| The model | /data/embeddinggemma2/.models/EmbeddingGemma-2-914f7f8 (gitignored) | config.json, config_sentence_transformers.json (the 20 task prompts), processor_config.json (the image processor's own max_soft_tokens 280), 1_Pooling/config.json, tokenizer.json, README.md |
| Counter-example only | /data/llama-prism @ b10743 | never a source; read only to avoid its 1/sqrt(head_dim) scale |

5. Routes removed — do not resurrect any of them

- ONNX and onnxruntime in any role; PyTorch/transformers/sentence-transformers and
  any Python ML stack (owner, 2026-10-19: "remove this onnx i dont want it and no
  pytorch"; AGENTS.md P-1). An isolated venv is dead too.
- The DQVD container as the target format (owner's container decision); it remains
  only as phase 1's input and the byte-verified record of the quantization.
- The `eg2_pack reconstruct` fallback (DQVD-only, superseded by the GGUF).
- candle, burn, tch or any Rust tensor framework: the arithmetic is our own C/CUDA.
- Executing the pinned llama.cpp in any default path (owner directive 2026-10-20).
- Metal and macOS: not a target here (Linux, NVIDIA only).
- A SECOND image decoder, a second caps mechanism, a C-side sequence builder, or a
  second span-shape check: the wire builds ids in Rust, decodes images with the chat
  route's own decoder, renders the contract through EmbedCaps::to_json, and the
  splice's span start is derived from the ids in ONE place (the core), as the CPU
  reference does.
- Not in scope unless the owner says so: KV checkpoints (row X.3), ds4's
  distributed/agent surfaces, the audio tower, and video frames.

6. Project facts a fresh session cannot re-derive

- Workspace layout: RECIPE.md; RECIPE-deepseek41-vq.md; docs/ds4-gemma-port-phases.md
  (the row list — the authoritative work log); docs/ds4-gemma-port-reference.md;
  docs/{tensor-inventory.tsv,encode-plan.tsv,artifact-verify-summary.txt,
  ladder-embeddinggemma2.txt}; docs/{baseline-gguf-gate1a,anchor-gate2,
  artifact-gguf-gate1b,loader-gate3,text-tower-gate4,retrieval-cost-gate5,
  eg2vq-embed-entry-gate6,eg2vq-device-gate7,eg2vq-serving-caps-gate20,
  eg2vq-vision-oracle-gate22,eg2vq-vision-tower-gate23,eg2vq-vision-tower-gate24,
  eg2vq-splice-gate25,eg2vq-embed-media-gate26,eg2vq-batch-gate27,eg2vq-batch21-plan}.txt;
  qa-evidence/; tests/{anchor,expected,retrieval,vision}/ + tests/{run.sh,qa-gate.sh};
  tools/eg2_*.{c,sh} + bin/; out/ (all artifacts and logs, gitignored).
- The caps and the cost model (units 20, 26, 27): crates/ds4-core/src/gemma_vq.rs
  holds ITEM_TOKENS_MAX 2048 (the device attention key plane) and BATCH_ROWS_MAX 8192
  (a PORT policy, now the row budget a batched call actually uses), plus the image
  constants (IMAGE_TOKEN_ID 258880, the two markers, IMAGE_SOFT_TOKENS_MIN/MAX
  70/1120, IMAGE_PIXELS_MIN/MAX 161280/2580480, IMAGE_EMBD 512, IMAGE_PER_ITEM_MAX
  1), reached through `ds4_core::embed_caps(family)`; /health reports them under
  `embed` (the image half under `embed.image`); the route refuses past either
  ceiling by name as 400. THE WORKSPACE IS 116,748 B A ROW (228.03 MiB at 2048 rows,
  912.13 MiB at 8192) and `DS4_EG2VQ_MAX_ROWS` in ds4_eg2vq.h is the C mirror,
  held equal to the Rust constant by crates/ds4-core/tests/eg2vq_serving_caps.rs.
- The batch (unit 27): `Model::embed_gemma_vq_batch(ids, item_off)` ->
  `ds4_bridge_eg2vq_embed_batch` -> `ds4_engine_embed_gemma_vq_batch` ->
  `ds4_gpu_eg2vq_embed_batch` (device) / `ds4_eg2vq_ref_embed_batch` (CPU, which
  loops the gated one-item reference). `ids` are the items packed contiguously and
  `item_off` their n_items + 1 spans; the device core DERIVES pos/len from the spans
  and refuses a table that does not cover the rows, an empty item, zero items, an
  item past the key plane, or rows past the workspace's budget. The route prepares
  every item and batches the TEXT ones; media items keep one call each. A batched
  item's vector IS its single-item vector, byte for byte.
- The wire (unit 26): an input item is a string, or an object with a `content` array
  of the chat route's parts (text/image_url, one image, data URI).  The sequence:
  `ds4_core::embed_media_ids` -> the anchor's 92 positions for task Raw and
  `<|image|>`.  The bridge's ds4_bridge_model_vision_encode_rows fills a CALLER-owned
  buffer (expect_rows = the caller's own probe's token_count).  ServerConfig::vision
  is the boot fact the route refuses an image item on; the CPU backend can carry
  --vision for THIS family only (ds4.c's guard, commented there).
- The tower (phases 4/7): the tokenizer is host-side (crates/ds4-core/src/tok/gemma_vq.rs);
  the CPU reference is ds4_eg2vq_graph.inc (+ ds4_eg2vq_bind.inc, a 520 MiB f32
  cache); the device path is ds4_eg2vq_gpu.cuh over cuda/eg2vq_*.cuh (55.9 MiB plan,
  137 MiB weights, workspace sized by ROWS); the oracle is
  `tests/parity/eg2vq_embd_c_oracle <gguf> <ids.tsv>` (CPU) and
  `tests/parity/eg2vq_embd_gpu_oracle` (device, with --batch/--batch-synth/--batch-bad/--no-cpu).
- Shape to load: 24 layers, hidden 512, 4 q heads, 2 KV heads, head_dim 256; layers
  5/11/17/23 full (head_dim 512, 1 KV head); ffn 2048; rms eps 1e-6; vocab 262,144;
  out 768; the file's sliding_window is 1024, so the symmetric mask is ±512; mean
  pool then L2 normalize.
- The vision tower (phase 8): 16 layers, hidden 768, 12/12 heads, head_dim 64, ffn
  3072, patch 16, merge 3, position table 10240x768x2 f32, axial NeoX RoPE theta 100
  (each of the two calls rotates n_rot = head_dim/2 = 32 dims, pairing (i, i+16)),
  rms eps 1e-6, 70..1120 soft tokens, align 48, 768->512 projection.  The engine opens
  out/...-\vision-mmproj.gguf as `e->vision_model` (the companion-GGUF pattern,
  ds4.c:72689) and binds it by name with required_tensor.
- Machine: RTX 4070 SUPER (12,282 MiB, sm_89), CUDA 13.3, 28 CPUs.  Rust is built
  with cargo + `--features native`; the C side via the tree's Makefile (`make ds4
  ds4-server` rebuilds incrementally); the canonical /data/build builder refuses this
  tree (recorded; rule 5's iterative exception applies).  The arch record is
  ds4-rs/.ds4-cuda-config.mk (`CUDA_ARCH := sm_89`).  The ENGINE holds a
  single-instance lock, so two servers cannot run at once — gates 26 and 27 start and
  stop one per backend.  A batched call with 8192 rows allocates a 912 MiB workspace.
- Quality (gate 5): mean vector agreement 0.956039571 at 768 (min 0.898186175),
  MRR@10 0.951778 -> 0.935819, recall@1 0.888278 -> 0.861611, recall@10 0.983333 ->
  0.990000 on the frozen SciFact set; the owner's floor and the verdict are section 9
  of docs/retrieval-cost-gate5.txt.
- ds4-rs remotes are shared with the live ds4 project (origin Baekpica/ds4-dfm-rs,
  fork JLFN/ds4-dfm-rs, publish JLFN/embeddinggemma2-ds4-rs); /data/ds4-dfm-rs is
  never modified from this workspace.
- Guardrails: global ~/.opengrok/AGENTS.md + this repo's mount at
  .opengrok/rules/project-work-rules.md + AGENTS.md (project P-1/P-2).  .gitignore
  excludes .models/, ds4-rs/, out/, graphify-{out,rs-out}/ and *.o.

7. Guardrails and open items

- P-1 (no Python in the runtime, ever) and P-2 hold; all five P-1 checks return zero and `sha256sum -c SHA256SUMS` passes (14/14, re-verified at this boundary).
- RULE 19 IS SATISFIED for the whole accumulated port AND for the two performance units: qa-evidence/qa-report.md is the VQ GEMM unit's report (fresh at d3b5397) and reads `verdict: overall PASS`, qa-evidence/qa-deferral.txt is deleted, and tests/qa-gate.sh prints `QA GATE: ALL PASS (no new runtime, route or public-API surfaces relative to publish/dev and ds4-rs/publish/dev)` with exit 0. Any NEW commit that adds a surface makes the report stale again and the gate red — which is the rule working, not a problem: a push must therefore happen with no unprepared surface after the report.
- OPEN (owner), added at this boundary: the ENGINE repo carries an inherited stale rule-19 mount — ds4-rs/tests/qa-gate.sh and ds4-rs/qa-evidence/qa-report.md (2026-10-18, from the ds4-dfm-rs lineage, 76 KB, covering that project's PR 67). The QA tester ran it by mistake, got "QA GATE: FAIL (342 check(s) red)" over surfaces belonging to another project, and reported it as this unit's gate. The project's mount is the OUTER tests/qa-gate.sh, which scans both trees. Retiring the inner copy is engine-branch surgery, so it is an owner decision.
- F6 stands: the surface extractor matches single-line definitions, so multi-line C entry points never appear in its list. Unit 27's are ds4_engine_embed_gemma_vq_batch, ds4_gpu_eg2vq_embed_batch, ds4_eg2vq_ref_embed_batch, eg2vq_ops_proj768_pool_batch, ds4_bridge_eg2vq_embed_batch and Model::embed_gemma_vq_batch's Rust checks; unit 26's are listed in section 3 of its evidence doc.
- OPEN FLAKE, RECORDED NOT FIXED: `cargo test -p ds4-server --lib` fails `cache_identity::tests::bounded_sidecar_and_ple` intermittently (3 runs: 1 pass, 2 fail; the two identity arrays come out identical, so the test's own premise — that a rewrite changes the stat — does not hold on this filesystem). cache_identity.rs is not in units 25-27's diffs; the fast path does not run it. Fixing it touches the shared engine branch: a finding for the owner.
- CLOSED: the container, the quality floor, the caps, the llama-free boundary, phase 7's device path, the vision tower on both legs, the splice, the engine entries, the wire, the batch (units 23-27), and the FINAL RULE 19 PASS (unit 28: eleven surfaces covered, three findings fixed and closed).
- LANDED (2026-10-21, on the owner's word): the push, which was an OFFER for three units and is now done. Both targets were verified fresh first (rule 6: JLFN/embeddinggemma2 and JLFN/embeddinggemma2-ds4-rs, both PUBLIC, default branch dev), rule 10 OSV was re-run (140 packages, 0 vulnerable), rule 19's gate and the whole fast path were green at the pushed revision, and the ranges were read from git rather than from this file. Result: `publish/feat/eg2vq-attn-perf` in both repositories, and PR #1 into dev in each (JLFN/embeddinggemma2-ds4-rs#1 and JLFN/embeddinggemma2#1). dev was NOT pushed to directly, per rule 23. The outer PR carries dev's five unpushed doc commits with it. NO FURTHER PUSH IS AUTHORIZED: the next one is an offer again, and it must re-run the same four checks fresh.
- OPEN (owner): the C CLI's tokenizer gap (the family's gemma4 BPE is Rust-side; the C CLI refuses the artifact by name) and the AUDIO tower (out of scope; the owner asked for vision).
- STANDING (owner, 2026-10-19): AFK; automatic continuation after each handoff and compaction; pushes stay offers. Recorded in section 1 with the two later directives.
- Nothing is in flight in either repository. The nested checkout's working tree is clean at e01eba8e; the outer tree carries the unit-27 and unit-28 commits and this handoff.
- The unit-20, -23, -24 and -28 QA findings are closed: the first three were four corrections plus one informational (unit 24's dispositions are docs/eg2vq-vision-tower-gate24.txt section 9), and unit 28's three are qa-evidence/qa-findings-u28.md. The rule 19 pass HAS run since unit 24 -- it IS unit 28, and it covers every surface through unit 27 -- so no pass is outstanding.
- OPEN (owner), added at this boundary: retiring the QA gate's deferral branch (tests/qa-gate.sh:150-162, section 3 item 5) and the MULTI-IMAGE item (section 3 item 4, a faithfulness gap rather than a defect). Also OPEN: the C CLI's tokenizer gap and the AUDIO tower, both recorded above.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git status --short && git branch -vv                 # clean, on feat/eg2vq-attn-perf; read the count with `git rev-list --count publish/dev..HEAD`, never from a comment
    git log --oneline publish/dev..HEAD                  # what the outer branch carries BELOW the push (0406051, d3b5397, 259ce6b, 3aa7f99, 198fbc9 are published; only this handoff's own commit is local)
    git -C ds4-rs status --short && git -C ds4-rs log --oneline -8   # clean; HEAD = 7cafa2a2 (the chart and the error pair) on feat/eg2vq-attn-perf, based on dev 10e73aa0 = publish/dev
    git -C ds4-rs rev-list --count publish/dev..HEAD     # 12 over publish/dev, ALL PUBLISHED (the three units: 31285531, 8d618ecb, 72ef3ac1 and their records/QA commits, then 7cafa2a2 the chart) -- and `publish/dev..dev` must be 0
    make -C ds4-rs test-eg2vq-attn                       # EG2VQ_ATTN_PASS: the kernel THIS unit changed (nine arithmetic cases, mincos 1.000000000), plus the two dispatcher-fallback cases
    make -C ds4-rs bench-eg2vq-attn                      # attention's per-shape table (the shape A/B this unit's record quotes)
    make -C ds4-rs test-eg2vq-vq                         # EG2VQ_VQ_PASS: the previous unit's kernel, unchanged since
    make -C ds4-rs bench-eg2vq-vq                        # the per-shape table the GEMM unit's record quotes (~10% error bar per row)
    tests/run.sh                                         # the fast path (~2 m 36 s this boundary, gates 22-27 included), ending in the rule 19 gate -- RE-RUN and green at this boundary (the first attempt failed only because a stale ds4-server of an earlier session held /tmp/ds4.lock)
    sha256sum -c SHA256SUMS                              # FOURTEEN entries
    tools/eg2_loader_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/artifact-manifest.tsv
    bash tests/qa-gate.sh                                # rule 19: must print the ALL PASS line (the OUTER mount, not ds4-rs/tests/qa-gate.sh)
    bash tools/eg2_embed_route_gate.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf   # gate 20: EG2VQ_EMBED_ROUTE_PASS (NOT in the fast path -- the unit-28 QA pass found it red for two units, so run it by hand)
    cd ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/qa-identity && bash build.sh && ./qa-identity   # QA_IDENTITY_OK: 12,492,416 floats, old vs new, 0 differing
    # the attention unit's own instruments (BOTH probes must be built against the
    # PRE-change header, or they fail their own "V0 is not the shipped kernel"
    # anchor -- see that unit's record section 7):
    #   mkdir -p /tmp/ab-attn/cuda && cp ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/attn-before.cuh.txt /tmp/ab-attn/cuda/eg2vq_attn.cuh
    #   nvcc ... -I/tmp/ab-attn -o /tmp/probe-attn ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/probes/probe-attn.cu && /tmp/probe-attn 20       # PROBE_ATTN_OK
    #   nvcc ... -I/tmp/ab-attn -o /tmp/probe-attn-grp ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/probes/probe-attn-grp.cu && /tmp/probe-attn-grp 20   # PROBE_ATTN_GRP_OK
    cuobjdump --list-elf ds4-rs/ds4_cuda.o | tail -2     # ds4_cuda.sm_89.cubin = the arch record is real
    cat ds4-rs/.ds4-cuda-config.mk                       # CUDA_ARCH := sm_89
    tail -8 out/phase5-gate.log                          # the agreement + retrieval tables
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'
    # the deep path (tests/run.sh --deep: gate 1b, the Rust suite, gate 4) measured
    # 24 m 55 s on 2026-10-20; the deep gates individually, when a number is
    # questioned (the 2026-10-20 run's logs are in out/reverify-d9b3693/, the
    # long-standing ones in out/):
    #   gate 4   (~20 min, the token dump is reused): tools/eg2_phase4_gate.sh <artifact> <baseline> tests/anchor/anchor-items.tsv .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
    #   gate 5   (minutes, the legs are reused):      tools/eg2_phase5_gate.sh
    #   gate 7   (the device tower against the CPU leg, the oracle's own default): ds4-rs/tests/parity/eg2vq_embd_gpu_oracle <artifact> <ids.tsv> [--batch] [--no-cpu]
    #   gate 20  (~15 s, needs ds4-rs/ds4-server built): tools/eg2_embed_route_gate.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf
    #   gate 22  (seconds, NATIVE): tools/eg2_mmproj_check.sh -> EG2VQ_VISION_MMPROJ_PASS
    #   the anchor, OPT-IN and the only thing that runs the pin (~10 s, needs
    #   /data/llama-anchor-build): tools/eg2_vision_anchor.sh
    #   the tokenizer gate against the committed ids (no pin):
    #            tools/eg2_tok_check.sh <artifact> tests/anchor/anchor-items.tsv .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
    #   gate 23  (unit 23, fast-path step 6): tools/eg2_vision_gate23.sh
    #   gate 24  (unit 24, fast-path step 7): bash tools/eg2_vision_gate24.sh
    #   gate 25  (unit 25, fast-path step 8): bash tools/eg2_splice_gate25.sh
    #   gate 26  (unit 26, fast-path step 9; ~65 s: it stops and starts one server
    #            per backend, so the port is busy for its run):
    #            bash tools/eg2_embed_media_gate26.sh
    #   gate 27  (unit 27, fast-path step 10; ~30 s, and its 8192-row legs allocate
    #            a 912 MiB workspace size class):
    #            bash tools/eg2_embed_batch_gate27.sh
    # the sequence driver the wire's leg 0 uses (vocabulary only, no weights):
    #   (cd ds4-rs && cargo run -q -p ds4-core --release --example gemma_vq_media_ids -- <artifact> '<|image|>' 88 Raw)

Trust git over this document's headings: every unit whose work this document
describes is in the diff, both repositories are clean on
feat/eg2vq-attn-perf, and dev is untouched in BOTH (engine dev = publish/dev =
10e73aa0, zero commits ahead; outer dev = a43479d, which the published PR
carries). The port itself is complete through phase 8 — code-complete, gated,
QA'd (rule 19's report reads verdict: overall PASS and the gate is green) — and
section 3's item 1 is the one code unit in flight-by-plan: the payload GEMM,
re-ablated before it is changed. The three performance units are PUBLISHED and
their PRs are open; everything after this handoff's own commit is unpushed and
every push stays an offer.

What is published, and what is not: both repositories'
feat/eg2vq-attn-perf is on the remote through the engine's 7cafa2a2 (the chart)
and the outer's 198fbc9 (this handoff's rotation), with PR #1 open into dev in
each; the only local-only commit is THIS handoff's edit. The engine's branch
carries the attention kernel's vec4 loads, the VQ GEMM's rows-per-thread change
and the attention grouped pass 1 with their records, probes and three QA rounds.
Read `git rev-list --count publish/dev..HEAD` in each repository, and
`git log --oneline @{u}..HEAD` for what is still local, rather than a number
printed here. The artifacts, the retrieval source, every out/ log and the CUDA
objects live under gitignored paths (out/, .models/, ds4-rs/), so they never
appear in a diff: they are identified by the digests in SHA256SUMS, by the pins
inside the gate scripts, and by the evidence docs' own digests.

This document is self-contained: a new session can act on it alone. At this
boundary that means section 3 item 1 — re-ablate the payload GEMM at its shipped
R=8, then optimize what the ablation names — and nothing else without the owner's
word. History is deliberately absent: git log, the phase doc and the evidence
docs are the historical record.
