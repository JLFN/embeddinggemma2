---
project: embeddinggemma2
plan_start_commit: 0ad31fe
last_updated_commit: 3e5e3b2
branch: dev
remote: 
handoff_written_at_context_usage: 34%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-20T13:40:00+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ in ds4-rs (rewritten 2026-10-20 at the end of the BATCH: the plan's last code row landed, so the port is code-complete and section 3 becomes the FINAL rule 19 QA pass)

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
before we are complete with the entire code". The port IS complete now (section 2:
this was the plan's last code row), so the instruction's other half binds and
section 3 is the single rule 19 pass it asks for: the deferral file is DELETED and
ONE QA-tester pass runs over the whole accumulated port. Until that pass writes a
fresh qa-evidence/qa-report.md reading `verdict: overall PASS`, tests/qa-gate.sh
keeps printing the deferral — which now means "not done yet", not "not due yet".
Unit 24's report on disk is the last QA evidence that exists.

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

2. Current state (2026-10-20, at the unit-27 close: THE PORT IS CODE-COMPLETE — every phase 1-8 row is landed and gated, and the plan's last code row was the batch)

- UNITS 19-27 LANDED; phases 1-8 complete. What the last three units added: the image path's end-to-end wire (unit 26: /health's image half, the OpenAI content-part item shape, the Rust sequence builder, six refusals by name) and the BATCH (unit 27: several items through the tower in ONE call). The audio tower, a multi-image item and the C CLI's tokenizer gap remain OUT of scope (section 7).
- UNIT 27 (row P2.3's last slice, ds4-rs 0b28cb40) IS DONE: `ds4_gpu_eg2vq_embed_batch` / `ds4_eg2vq_ref_embed_batch` / `ds4_engine_embed_gemma_vq_batch` / `ds4_bridge_eg2vq_embed_batch` / `Model::embed_gemma_vq_batch`, the packed row block with ITEM-RELATIVE attention key ranges and a per-item pool (the reference's own batched semantics: src/llama-batch.cpp:140-193 packs sequences contiguously with per-token seq_id/pos, src/llama-graph.cpp:482-517 masks between them), the row budget `DS4_EG2VQ_MAX_ROWS` 8192 held equal to the Rust `BATCH_ROWS_MAX` by the caps test, and the route handing its TEXT items to one call while each media item keeps its own (one device vision workspace per engine is serial).
- MEASURED by gate 27 (tools/eg2_embed_batch_gate27.sh, fast-path step 10): the batched vectors are BYTE-IDENTICAL to the per-item runs (max per-element delta 0.000e+00) over 6 items / 3881 rows and over 64 items / 1280 rows; one call 1283.3 ms against the per-item 1492.6 ms (1.18x the warm rate) on the mixed set, and 114.4 ms against 967.2 ms (1323 -> 11190 rows/s, 8.65x) on 64 small items; 8192 rows accepted in one call (912.13 MiB) while 8193 refused; five malformed span tables refused with the caller's buffer untouched; on the wire an array request equals the same items one per request BYTE FOR BYTE on both backends, and four items of exactly 2048 tokens are one accepted call while a fifth is a 400 quoting embed.batch_rows_max.
- THE ROW COST WAS CORRECTED, and a stale number found: the layout charges 116,748 B a row (27,648 f32 planes + ids/pos/len + the per-row projection plane and the per-item result plane), so 228.03 MiB at 2048 rows and 912.13 MiB at the 8192 budget. The tree had been quoting 108,548 B / 848.0 MiB since unit 20 -- one plane stale, because unit 25's image splice added the 512-float in_soft plane and nobody folded it in (the layout charged 110,592 B a row / 216.00 MiB while the docs and the caps test still said 212.01 MiB, gate 7's pre-splice measurement). Both are corrected, and gate 27 cross-checks the model against the workspaces a real call allocates.
- ONE REGRESSION, CAUGHT BY THIS UNIT'S OWN GATES: the first batch-core revision took the image span's start from the item-span table instead of deriving it from the ids, which refused the device splice outright; gate 25 caught it on the unit's first fast-path run and its numbers after the fix are exactly the unit-25 record (device vs the artifact's own CPU leg 0.00057164 / 0.004348 / 0.99999055).
- THE SINGLE-ITEM ARITHMETIC changed in one place, measured: the projection's accumulation is now the item's own rows in row order (the CPU reference's order) instead of an atomicAdd whose order the hardware chose -- before/after maxabs 1.000e-07, rel_rms 4.655e-07, device-vs-CPU agreement unchanged at the printed precision. Gate 7's bounds hold by a factor of a thousand.
- Machine-facing facts RE-RUN this session (2026-10-20): tests/run.sh fast path GREEN in 2 m 47 s (steps 1-10 PASS -- gates 22-27 -- step 11 the recorded deferral), `cargo test -q -p ds4-core` 664 passed / 0 failed over 41 suites, `cargo check --workspace --all-targets` clean, tests/test_eg2vq_attn and tests/test_eg2vq_ops PASS with their new batch cases, gate 25 PASS with the unit-25 numbers, `make ds4 ds4-server` clean, `sha256sum -c SHA256SUMS` all FOURTEEN entries OK, no server left running.
- Workspace: /data/embeddinggemma2, branch dev, HEAD = the commit that carries this handoff, tree clean at write time. Unpushed: outer, every commit since publish/dev d45394f (26 before unit 27, plus this unit's own gate, evidence and handoff commits) (`git rev-list --count publish/dev..HEAD` is the authority); ds4-rs ELEVEN over 29873445 (tip 0b28cb40). Both pushes are OFFERS (section 7). OUT/, .models/ and ds4-rs/ are gitignored: the artifacts are identified by the digests in SHA256SUMS.
- WHAT IS LEFT: (1) THE FINAL RULE 19 PASS, now due — the code is complete, so the deferral is satisfied and the pass is the plan's next step (section 3); (2) the two pushes, when the owner answers (offers only, rule 10 OSV re-run first); (3) the C CLI's tokenizer gap (owner's call); (4) the AUDIO tower (owner's call); (5) not planned but the next faithfulness gap: a multi-image item (the checkpoint's card allows several `<|image|>`s; this port supports one, `embed.image.per_item_max`, because the splice fills one contiguous run — widening it needs the CPU graph, the device ops and the engine's span check together).

3. Next unit plan — unit 28: THE FINAL RULE 19 QA PASS over the whole accumulated port

Goal: the owner's third directive's second half, executed once: delete the deferral,
run ONE QA-tester pass over every surface the port accumulated since the last QA
evidence, leave a fresh qa-evidence/qa-report.md reading `verdict: overall PASS`, and
have `bash tests/qa-gate.sh` exit 0. Nothing else is planned before the pushes.

WHY NOW: the port is code-complete (section 2). The owner's instruction was explicit
that QA must NOT run earlier and MUST run when the code is complete. There is no
further code row to defer for.

THE MECHANICS, in order:

 1. READ FIRST: tests/qa-gate.sh (what the gate checks: a fresh report, newer than
    publish/dev, reading `verdict: overall PASS`, covering every surface the gate's
    own patterns extract from BOTH repositories), qa-evidence/qa-deferral.txt (the
    instruction being discharged), and qa-evidence/qa-report.md (unit 24's, the last
    one that exists -- it is also the diff base for "what is new").
 2. DELETE qa-evidence/qa-deferral.txt in the same commit that lands the report (the
    file's own text says exactly this). Do not delete it and leave the pass undone:
    tests/qa-gate.sh would then fail, which is the correct reading, but it would also
    mean the deferral was used as an alibi.
 3. RUN THE QA-TESTER on the project's recorded model: deepseek-v4.1-CC-flash
    (tests/qa-gate.sh:9-10, the model this project recorded; never silently
    substitute another). The unit-26 handoff recorded the expected shape of this
    run: the whole accumulated port from unit 24 onward -- units 25, 26 and 27 --
    verified LIVE where the environment permits, with the gates as the live runs.
 4. WHAT THE PASS MUST COVER, at minimum (each is a surface this port added and no
    QA pass has looked at):
      - unit 25: ds4_eg2vq_ref_embed_image / ds4_gpu_eg2vq_embed_image, the engine's
        probe/encode_memory entries for gemma_vq, the image caps row;
      - unit 26: the six refusals, the parse-time item shape, decode_image_data_uri,
        the /health image contract, the two new bridges (eg2vq_embed_image,
        model_vision_encode_rows), the ds4.c --vision guard's gemma_vq exemption;
      - unit 27: ds4_engine_embed_gemma_vq_batch, ds4_gpu_eg2vq_embed_batch,
        ds4_eg2vq_ref_embed_batch, the five span-table refusals, the row budget and
        the new cost model, Model::embed_gemma_vq_batch's Rust-side checks, the
        route's two-pass grouping and its serve_embed_batch_test.rs;
      - F6 stands: the surface extractor matches single-line definitions, so the
        multi-line C entry points above MUST be read by hand (the unit-26 and unit-27
        evidence docs list them).
 5. RE-RUN, as part of the pass, whichever gates the findings question (the fast
    path is 2 m 47 s; gate 7's device harness is the oracle's own run with the CPU
    leg; gate 25/26/27 are the image and batch paths).
 6. THEN: commit the report + the deleted deferral + any corrections the pass
    produces (corrections land as their own commits, each with the gate that
    measures them), confirm `bash tests/qa-gate.sh` exits 0, and rotate this handoff
    again. The two pushes stay OFFERS: rule 10's OSV re-run and the owner's word.

WHAT UNIT 28 MUST NOT DO: re-open feature work (the port is complete; the audio
tower, a multi-image item and the C CLI's tokenizer gap are the owner's calls, not
QA's), weaken a gate's declared bound to make a finding go away, or commit the
report while a finding is open -- a finding is either fixed (with its gate) or
recorded in the report as an accepted risk with its reason.

4. Reference map — the exact repos and files (pinned; verified at this boundary)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (all the work) | /data/embeddinggemma2/ds4-rs @ 0b28cb40, branch feat/ds41-dsml-carriage, gitignored by the outer repo, own git | ds4.c, ds4.h, ds4_eg2vq.{h,inc,graph.inc,gpu.cuh}, ds4_eg2vq_vision.{h,inc}, ds4_glmvq*.{h,inc}, cuda/eg2vq_{attn,ops,vq,vis}.cuh, native/bridge/ds4_bridge.{c,h}, crates/ds4-core/src/**, crates/ds4-server/src/**, crates/ds4-sys/src/lib.rs, tests/parity/, tests/test_eg2vq_{attn,ops}.cu, the Makefile |
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
- RULE 19 IS NOW DUE, not deferred: the code is complete, so qa-evidence/qa-deferral.txt must be deleted and ONE pass run (section 3). tests/qa-gate.sh currently prints the deferral because that file still exists; leaving it there from here on is a rule 19 violation with the deferral as its alibi.
- F6 stands: the surface extractor matches single-line definitions, so multi-line C entry points never appear in its list. Unit 27's are ds4_engine_embed_gemma_vq_batch, ds4_gpu_eg2vq_embed_batch, ds4_eg2vq_ref_embed_batch, eg2vq_ops_proj768_pool_batch, ds4_bridge_eg2vq_embed_batch and Model::embed_gemma_vq_batch's Rust checks; unit 26's are listed in section 3 of its evidence doc.
- OPEN FLAKE, RECORDED NOT FIXED: `cargo test -p ds4-server --lib` fails `cache_identity::tests::bounded_sidecar_and_ple` intermittently (3 runs: 1 pass, 2 fail; the two identity arrays come out identical, so the test's own premise — that a rewrite changes the stat — does not hold on this filesystem). cache_identity.rs is not in units 25-27's diffs; the fast path does not run it. Fixing it touches the shared engine branch: a finding for the owner.
- CLOSED: the container, the quality floor, the caps, the llama-free boundary, phase 7's device path, the vision tower on both legs, the splice, the engine entries, the wire, and now THE BATCH (units 23-27).
- OPEN (owner, NEW): the two pushes. Outer: `git push publish dev`, range d45394f..HEAD (read `git rev-list --count publish/dev..HEAD` for the exact number; docs/tools/tests/gates 22-27 and their evidence — no artifact: out/, .models/ and ds4-rs/ are gitignored; the tracked binaries in bin/ are rebuilt by the gates, and the anchor binary is deliberately NOT tracked because it links the pin). Target publish = https://github.com/JLFN/embeddinggemma2 (rule 6 wants a fresh `gh repo view` immediately before the push). Nested: `git push publish feat/ds41-dsml-carriage:refs/heads/dev`, range 29873445..HEAD (ELEVEN commits as of 0b28cb40). Gates already run for both: rule 10 OSV 140 packages / 0 vulnerable (both roots, at an earlier boundary — re-run before an actual push), rule 19 DUE (section 3), rule 5/12 as recorded. Rollback if a push must be undone: `git push --force-with-lease publish d45394f:refs/heads/dev` (outer) — a force-push, so the owner's call; both local histories are kept either way.
- OPEN (owner): the C CLI's tokenizer gap (the family's gemma4 BPE is Rust-side; the C CLI refuses the artifact by name) and the AUDIO tower (out of scope; the owner asked for vision).
- STANDING (owner, 2026-10-19): AFK; automatic continuation after each handoff and compaction; pushes stay offers. Recorded in section 1 with the two later directives.
- Nothing is in flight in either repository. The nested checkout's working tree is clean at 0b28cb40; the outer tree carries the unit-27 commits and this handoff.
- The unit-20, -23 and -24 QA findings are closed (four corrections committed, one informational; unit 24's dispositions are docs/eg2vq-vision-tower-gate24.txt section 9). NO QA pass has run since unit 24, by the owner's instruction, and section 3 is the pass that discharges it.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git log --grep "Unit: .* complete" d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git status --short && git branch -vv                 # clean, on dev; every commit since publish/dev d45394f is unpushed (read the count, never this comment)
    tests/run.sh                                         # the fast path (2 m 47 s, gates 22-27 included), ending in the rule 19 gate
    sha256sum -c SHA256SUMS                              # FOURTEEN entries
    tools/eg2_loader_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/artifact-manifest.tsv
    bash tests/qa-gate.sh                                # prints the deferral until section 3 deletes the file; must be read, not trusted as evidence
    git -C ds4-rs log --oneline -3 && git -C ds4-rs status --short   # 0b28cb40, clean
    cuobjdump --list-elf ds4-rs/ds4_cuda.o | tail -2     # ds4_cuda.sm_89.cubin = the arch record is real
    cat ds4-rs/.ds4-cuda-config.mk                       # CUDA_ARCH := sm_89
    tail -8 out/phase5-gate.log                          # the agreement + retrieval tables
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'
    # the deep gates, when a number is questioned (each log is in out/):
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

Trust git over this document's headings: continue with the first unit whose work is
not in the diff, which at this boundary is section 3 (unit 28 = THE FINAL RULE 19 QA
PASS — the code is complete, so the deferral is discharged and the pass is due). Every
phase through 8 is landed and gated; the audio tower and the C CLI's tokenizer gap
are the owner's calls. The outer repo's HEAD carries every commit since publish/dev d45394f (the
range above d45394f, this handoff's own docs commits included); ds4-rs is at 0b28cb40 with its
published dev at 29873445. Anything else in either unpushed range must be read before
sections 2-6 are trusted. The artifacts, the retrieval source, every out/ log and the
CUDA objects live under gitignored paths (out/, .models/, ds4-rs/), so they never
appear in a diff: they are identified by the digests in SHA256SUMS, by the pins
inside the gate scripts, and by the evidence docs' own digests.

This document is self-contained: a new session can execute the remaining unit from
this file alone. History is deliberately absent — git log, the phase doc and the
evidence docs are the historical record.
