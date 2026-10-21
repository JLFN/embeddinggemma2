---
project: embeddinggemma2
plan_start_commit: 9f252ef
last_updated_commit: d3b53979914eb9c26e8f099d31b9a2038f6dccbf
branch: feat/eg2vq-attn-perf
remote: 
handoff_written_at_context_usage: 4%
handoff_written_at: 2026-10-21T08:22:23+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ in ds4-rs (rotated 2026-10-21 after the SECOND performance unit: the VQ payload GEMM's rows-per-thread change landed with a measured 1.2x-1.8x, three QA rounds closed on record defects with no code defect, and the next unit's target is named in section 3 — the attention kernel, which is now the co-largest item)

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

2. Current state (2026-10-21, re-verified IN THIS SESSION at the second performance unit's boundary: the port is code-complete, gated and QA'd; this boundary is a PERFORMANCE unit, the owner's own request, on a feature branch and NOT PUSHED)

- THE PERFORMANCE UNIT'S SECOND HALF, LANDED: engine feat/eg2vq-attn-perf = 1c9938ba, FOUR commits over dev 10e73aa0 (8d618ecb the kernel; 793386db, 0f9ab8cf, 1c9938ba the evidence and the QA rounds), and dev is untouched — proved by `git rev-list --count publish/dev..dev` = 0 in the engine repo. The outer repo is on feat/eg2vq-attn-perf: its dev a43479d plus 0406051 (the attention unit) and d3b5397 (this one). Nothing is pushed in either.
- WHAT IT IS: eg2vq_vq_gemm_kernel's block goes 16x16 -> 16x8 and each thread owns 8 output rows instead of 4 (32 accumulators), with the two-pass index unpack a 128-thread block needs. The 64x64 tile, the 8.5 KB slab, the 41,472 B of static shared, the launcher's guard and grid math, and the accumulation order are unchanged, so the kernel is BIT-IDENTICAL.
- WHY: an ablation, because ncu refuses on this machine (ERR_NVGPUCTRPERM). The kernel is not arithmetic-bound (the whole outer product replaced by FADDs with every load kept: 1.00x-1.12x) and not gather-bound (1.08x-1.18x); it is bound by the ACTIVATION LOAD — one load per K step instead of four: 2.0x-3.1x; the same four loads addressed block-uniformly, one 32-byte sector per instruction instead of sixteen: 2.4x-4.2x. Per K step a thread loads 16 B per token and spends T*R*4 scalar multiply-adds, so R (rows per thread) is the only register-blocking lever.
- MEASURED, same window with both binaries interleaved (the only honest form here): kernel 1.26x-1.79x round-1 pairs and 1.23x-1.75x round-2 pairs, mostly ~1.6x, 3,674-4,337 -> 6,519-7,002 GFLOP/s at 4520 tokens; corpus 143 pages batch 8 11.153/11.146 -> 8.893/8.903 s (7,240 -> 9,070 tok/s), batch 1-8 pooled 1.20x-1.25x; device tower 1,499.0 (phase 7's own record) -> 2,060.9 tok/s, both sides cross-window. CUMULATIVE over both performance units: 18.7998 -> 8.903 s at batch 8 = 2.11x, 4,295 -> 9,070 tok/s.
- THE RECORD: ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/vq-gemm-rows.md and every raw file beside it (ab-table.txt, ab-sweep.txt, four speed-bench CSVs, three probe outputs plus their sources, qa-identity/ the byte-identity driver, gate7-embd-gpu-oracle.txt, two nsys kernel summaries, bench-vq-before.txt). REJECTED WITH ITS MEASUREMENT: a 128-row/256-thread tile on 50,176 B of dynamic shared (0.83x at 512x512 x 306 tokens), and T=2/R=8 with 32-token blocks (1.28x-1.38x).
- GATES, ALL RE-RUN ON THE FINAL REVISION: `make test-eg2vq-vq` -> EG2VQ_VQ_PASS (mincos 1.000000000 everywhere, wrong-geometry payload refused with the output untouched); gate 7 `tests/parity/eg2vq_embd_gpu_oracle` -> EG2VQ_GPU_PASS (min cos 0.999999950, max rel_rms 3.150e-04 — numerically identical to the attention unit's record); compute-sanitizer memcheck 0 errors; REG 80 -> 128, STACK 0 LOCAL 0, shared 41,472 B; the caps did not move.
- RULE 19: THREE PASSES, FAIL-FAIL-PASS, AND NO CODE DEFECT IN ANY OF THEM. The extractor names no new surface and exits 0, and the pass ran by hand anyway (deepseek-v4.1-CC-flash, the project's recorded model). Pass 1 caught a probe that had become self-comparing, a probe that crashed when rebuilt from the documented command, an unbacked figure, a range taken from the favourable round, three numbers that traced to nothing, and one false agreement claim; pass 2 caught the same class again in three one-line places (a corpus pairing claim batch 1 contradicts, a citation to a line that does not contain the figure, a bytes-per-multiply-add sentence that contradicted itself by 4x); pass 3 confirmed the corrections by recomputing every cell. Evidence: qa-evidence/qa-report.md and qa-evidence/qa-findings-vq-r8.md (which SUPERSEDES qa-findings-attn-perf.md as the operative findings file).
- THE LESSONS THIS UNIT COST, both now in the kernel's own header: an ablation probe that takes its tile from the header stops being an instrument the moment the header changes (the rebuilt probe crashed with an illegal memory access; it pins PVQ_T/PVQ_R now), and a measurement is only as good as the sentence that reports it — the QA pass failed twice on sentences that claimed more than the numbers support, with every number in them real.
- NOT RE-RUN AT THIS BOUNDARY, named so the next session does not assume them: the fast path (tests/run.sh) and the deep path (tests/run.sh --deep), gates 20-27, the loader digest check, and the OSV scan. THIS UNIT TOUCHES ONE KERNEL plus documentation, benchmark artifacts and the Makefile-free probe sources — no crate, no route, no cap, no artifact — and they were green at 10e73aa0.

3. Next unit plan — the next PERFORMANCE unit (attention is the co-largest item), then the owner's standing list

(1) THE NEXT PERFORMANCE UNIT, AND MEASURE BEFORE CHANGING ANYTHING. Where the
    batch forward's kernel time sits now (nsys, 8 items / 2451 tokens):
    eg2vq_vq_gemm_kernel 139.2 ms (48.4%), eg2vq_attn_kernel 116.7 ms (40.5%,
    of a 287.8 ms total), the small ops ~11%. The two big items are now
    comparable, but their arithmetic intensity is not: the GEMM reaches
    6,519-7,002 GFLOP/s at 4,520 tokens while attention reaches 867-1,497 at any
    size, and in the batch profile attention does about 1.2% of the forward's
    arithmetic. So the attention kernel's headroom is the larger one, and its
    LIMITER IS NOT ESTABLISHED — do not guess at it. Read
    cuda/eg2vq_attn.cuh first-hand before proposing anything (rule 0.2): 4 warps
    per item, a 2048-key score plane (32 KB of static shared, so 2 blocks/SM),
    pass 1 writing the plane and pass 3 reading it back, and the vision tower's
    cuda/eg2vq_vis.cuh as the in-tree precedent for an online softmax.
    Method, in order, exactly as the GEMM unit did it:
      a. Freeze the baseline: `./tests/bench_eg2vq_attn [reps]` on the tower's
         nine shapes, and `./tests/bench_eg2vq_vq [reps]` for the GEMM's own
         post-change numbers. Same window or nothing.
      b. Ablate the SHIPPED attention kernel with the same one-component-at-a-time
         method (probes/probe-vq.cu is the worked example: a template<int V> with
         every variant register-neutral, and the probe pinned to its own tile so
         it cannot drift into the header). Candidate components, none measured
         yet: the shared score plane's write-then-read traffic, the K/V loads'
         sector pattern (the same 16-sectors-per-instruction question the GEMM
         answered), the softmax pass's exponentials and the two barriers per
         slab. NO counters exist here, so the ablation is the instrument.
      c. One change at a time, A/B'd inside one window against (a), gated by
         `make test-eg2vq-attn` (min cosine 1.000000000 against the CPU
         reference) plus gate 7, and — if the attention arithmetic itself moves —
         the oracle's min cos 0.999999950 / max rel_rms 3.150e-04 budget, which
         is the declared bound and must not be weakened.
    Also worth one measurement each before any code: the GEMM's remaining 48.4%
    (re-ablate it at R=8 — the activation pressure halved, so what binds now is
    unmeasured) and the small ops' ~11% (242 rms launches, 144 rms_head, 144 add
    for 2,451 tokens: 434+144+144 launches of tiny grids).

(2) THE OWNER'S STANDING LIST, unchanged by this unit: the C CLI's tokenizer gap;
    the AUDIO tower (out of scope; vision was asked for); a MULTI-IMAGE item (the
    faithfulness gap: the checkpoint and the reference allow several `<|image|>`
    placeholders per item, this port supports one because the splice fills one
    contiguous run); retiring the QA gate's deferral branch (tests/qa-gate.sh's
    deferral block is inert today but still live behaviour — a checked-in
    qa-evidence/qa-deferral.txt would silence rule 19's verdict); and the three
    decisions the branch standard left open (the default branch, the
    chore/fmt-workspace formatting unit at c595fca3 whose rule 19 gate is red,
    and retiring the duplicate feat/ds41-dsml-carriage). NEW at this boundary:
    the engine repo carries an INHERITED STALE rule-19 mount
    (ds4-rs/tests/qa-gate.sh and ds4-rs/qa-evidence/qa-report.md dated
    2026-10-18, from the ds4-dfm-rs lineage) — the QA tester ran it, found it red
    with 342 checks over surfaces that belong to another project, and reported it
    as this unit's gate. The project's mount is the OUTER tests/qa-gate.sh, which
    scans both trees. Retiring the stale inner copy is an owner decision (it is
    engine-branch surgery) and belongs on this list.

(3) PUSH DISCIPLINE, unchanged: every push is an OFFER (rule 23), and no unit
    pushes itself. The two branches (engine and outer, both
    feat/eg2vq-attn-perf) hold everything that is unpushed, with dev untouched in
    both repositories. When the owner says push, re-run the four pre-push checks
    fresh — rule 6 `gh repo view` on both targets, rule 10 OSV, rule 19's gate,
    and the rule 23 branch/range checks — read the counts from
    `git rev-list --count publish/dev..HEAD` rather than from this document, and
    then push the branch and open the dev PR rather than pushing to dev.

(4) WHEN NOTHING IS ASKED: run the next performance unit as (1) describes —
    measure, ablate, change one thing, A/B in one window, gate, record, QA — and
    keep this handoff and the memory entry current at each boundary. Do not
    re-open a closed row to "improve" it, do not weaken a declared bound, do not
    run the pinned llama.cpp in a default path, and do not push without the
    owner's word.
WHAT A SESSION MUST NOT DO AT THIS BOUNDARY: re-open a closed row to "improve" it
(the port's quality floor, the caps and the cost model are pinned by tests and
gates), weaken a declared bound (gate 7's cos >= 0.999999 / rel_rms <= 1e-03 is
the tightest one and it is a measurement, not a preference), run the pin in a
default path (owner directive), or push without the owner's word — the only thing
standing between this state and a release is the owner's answer, not more code.
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
- OPEN (owner): the two pushes, ready to take. Rule 10 is re-run and fresh (osv-mcp: 140 packages, 0 vulnerable, ds4-rs/Cargo.lock, 2026-10-20T12:11Z), rule 19 is satisfied as of the report, and both local histories are clean. Outer: `git push publish dev`, range d45394f..HEAD (read `git rev-list --count publish/dev..HEAD` for the exact number; docs/tools/tests/gates 22-27 and their evidence — no artifact: out/, .models/ and ds4-rs/ are gitignored; the tracked binaries in bin/ are rebuilt by the gates, and the anchor binary is deliberately NOT tracked because it links the pin). Target publish = https://github.com/JLFN/embeddinggemma2 (rule 6 wants a fresh `gh repo view` immediately before the push). Nested: `git push publish feat/ds41-dsml-carriage:refs/heads/dev`, range 29873445..HEAD (TWELVE commits as of e01eba8e). Gates already run for both: rule 10 OSV 140 packages / 0 vulnerable (re-run 2026-10-20, fresh), rule 19 satisfied (report fresh, gate green), rule 5/12 as recorded. Rollback if a push must be undone: `git push --force-with-lease publish d45394f:refs/heads/dev` (outer) — a force-push, so the owner's call; both local histories are kept either way.
- OPEN (owner): the C CLI's tokenizer gap (the family's gemma4 BPE is Rust-side; the C CLI refuses the artifact by name) and the AUDIO tower (out of scope; the owner asked for vision).
- STANDING (owner, 2026-10-19): AFK; automatic continuation after each handoff and compaction; pushes stay offers. Recorded in section 1 with the two later directives.
- Nothing is in flight in either repository. The nested checkout's working tree is clean at e01eba8e; the outer tree carries the unit-27 and unit-28 commits and this handoff.
- The unit-20, -23, -24 and -28 QA findings are closed: the first three were four corrections plus one informational (unit 24's dispositions are docs/eg2vq-vision-tower-gate24.txt section 9), and unit 28's three are qa-evidence/qa-findings-u28.md. The rule 19 pass HAS run since unit 24 -- it IS unit 28, and it covers every surface through unit 27 -- so no pass is outstanding.
- OPEN (owner), added at this boundary: retiring the QA gate's deferral branch (tests/qa-gate.sh:150-162, section 3 item 5) and the MULTI-IMAGE item (section 3 item 4, a faithfulness gap rather than a defect). Also OPEN: the C CLI's tokenizer gap and the AUDIO tower, both recorded above.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git status --short && git branch -vv                 # clean, on feat/eg2vq-attn-perf; read the count with `git rev-list --count publish/dev..HEAD`, never from a comment
    git log --oneline publish/dev..HEAD                  # what the outer branch carries unpushed (0406051, then this unit's d3b5397 + the handoff commit)
    git -C ds4-rs status --short && git -C ds4-rs log --oneline -5   # clean; HEAD = the VQ GEMM unit (1c9938ba) on feat/eg2vq-attn-perf, based on dev 10e73aa0 = publish/dev
    git -C ds4-rs rev-list --count publish/dev..HEAD     # 4 (8d618ecb the kernel, then three evidence/QA commits) -- and `publish/dev..dev` must be 0
    make -C ds4-rs test-eg2vq-vq                         # EG2VQ_VQ_PASS: the kernel this unit changed, gated
    make -C ds4-rs bench-eg2vq-vq                        # the per-shape table the unit's record quotes (~10% error bar per row)
    make -C ds4-rs test-eg2vq-attn                       # EG2VQ_ATTN_PASS: the previous unit's kernel, unchanged since
    make -C ds4-rs bench-eg2vq-attn                      # attention's per-shape table (the next unit's baseline)
    tests/run.sh                                         # the fast path (2 m 51 s at 6cd172f, gates 22-27 included), ending in the rule 19 gate -- NOT re-run at this boundary
    sha256sum -c SHA256SUMS                              # FOURTEEN entries
    tools/eg2_loader_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/artifact-manifest.tsv
    bash tests/qa-gate.sh                                # rule 19: must print the ALL PASS line (the OUTER mount, not ds4-rs/tests/qa-gate.sh)
    bash tools/eg2_embed_route_gate.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf   # gate 20: EG2VQ_EMBED_ROUTE_PASS (NOT in the fast path -- the unit-28 QA pass found it red for two units, so run it by hand)
    cd ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/qa-identity && bash build.sh && ./qa-identity   # QA_IDENTITY_OK: 12,492,416 floats, old vs new, 0 differing
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
10e73aa0, zero commits ahead; outer dev = a43479d). The port itself is complete
through phase 8 — code-complete, gated, QA'd (rule 19's report reads
verdict: overall PASS and the gate is green) — and section 3's item 1 is the one
code unit in flight-by-plan: the VQ GEMM, measured before it is changed. Nothing
in this unit is pushed, and every push stays an offer.

What is unpushed at this boundary, in one line each: the engine's
feat/eg2vq-attn-perf = 31285531 (one commit over dev: the attention
vectorization, the bench harness and its make target, .gitignore, three stale
citations, the gitignored artifacts) and the outer repo's feat/eg2vq-attn-perf
(this handoff, qa-evidence/qa-report.md and qa-evidence/qa-findings-attn-perf.md,
and the pre-existing docs commits the branch inherited from dev). Read
`git rev-list --count publish/dev..HEAD` in each repository rather than a number
printed here. The artifacts, the retrieval source, every out/ log and the CUDA
objects live under gitignored paths (out/, .models/, ds4-rs/), so they never
appear in a diff: they are identified by the digests in SHA256SUMS, by the pins
inside the gate scripts, and by the evidence docs' own digests.

This document is self-contained: a new session can act on it alone. At this
boundary that means section 3 item 1 — measure the VQ GEMM, then optimize it —
and nothing else without the owner's word. History is deliberately absent: git
log, the phase doc and the evidence docs are the historical record.
