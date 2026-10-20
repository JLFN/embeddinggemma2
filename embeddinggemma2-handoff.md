---
project: embeddinggemma2
plan_start_commit: d42fc8fdf8cef8dc0f6806d249160ae5472abe77
last_updated_commit: 48c86626ca401d12734acbc79ec52d6d99c80d1e
branch: dev
remote: 
handoff_written_at_context_usage: 54%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-20T12:03:46+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ in ds4-rs (rewritten 2026-10-20 at the end of the IMAGE PATH: the wire landed, phase 8 is complete end to end, and section 3 returns to the plan's own last code row)

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
before we are complete with the entire code". So: keep implementing; do NOT run the
per-unit rule 19 QA-tester pass while the port is still being written. It is
recorded in qa-evidence/qa-deferral.txt, and tests/qa-gate.sh reads that file and
prints a DEFERRAL (no verdict, never a pass) instead of failing on stale evidence.
The session that finishes the code DELETES that file and runs ONE pass over the
whole accumulated port. Unit 24's report on disk is the last QA evidence that exists.

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

2. Current state (2026-10-20, at the unit-26 close: THE IMAGE PATH IS COMPLETE END TO END — preprocessing, tower on both legs, splice, engine entries, bridge, route, caps and refusals — and the plan's last in-scope code row is unit 21, batching)

- OWNER DIRECTIVES in force: vision IS wanted (2026-10-19, "we need to add vision also see how the other model variant handle it"), the pin is read/cited but never executed by a default path (2026-10-20), and QA is DEFERRED until the code is complete (2026-10-20, section 1). The audio tower and the C CLI's tokenizer gap are the owner's calls; unit 21 was deferred on 2026-10-19 as a PRIORITY redirect (vision first), not cancelled, and its spec is parked whole.
- WHAT THE ARTIFACT CARRIES (docs/tensor-inventory.tsv + out/artifact-kv.txt): of 1376 tensors, 208 are vision_tower.encoder.layers.*, 2 are vision_tower.patch_embedder.*, 744 are audio_tower.*; the text tower is 217 of the 461 vqblob payloads, the VISION TOWER IS DENSE BF16, and the two tie-in projections are in-file (embed_vision 512x768, embed_audio 512x1536). THE IMAGE PATH end to end: image -> preprocessing -> the 16-block tower -> 3x3 pool + sqrt(768) -> weightless RMS -> 768-to-512 projection -> 88 soft tokens filling the text's image span -> the 768-dim pooled embedding. Graph: /data/llama.cpp @ 31288c9c tools/mtmd/models/gemma4v.cpp (133 lines), variant GEMMA4V (conversion/gemma.py:1001, :1055). The WIRE's own contract is the checkpoint's model card (.models/EmbeddingGemma-2-914f7f8/README.md section 5): `<|image|>` marks an image's position, prefixes apply to text only, and the reference processor expands that one position into [<|image>] + one placeholder per soft row + [<image|>].
- THE ARTIFACT MUST NOT BE RE-ENCODED, and the owner's GPU-rebuild question is answered by RECIPE.md section 8 (read live): a CUDA build of the packer exists but at this tier the kernels cannot run (81920 bytes of codebook against vq_qc.h's 49152-byte shared-memory gate) and where they can run they are 1.25-3x SLOWER here (89.6 s vs 58.8 s over a 92-tensor subset, identical cosines). The committed file records `encoder=cpu (eg2_pack)` and its digest defines the release.
- THE OWNER'S QUALITY FLOOR is RECORDED and MET (docs/retrieval-cost-gate5.txt section 9): recall@10 >= 0.983333, MRR@10 >= 98% of baseline (0.932742), recall@1 >= 0.855, mean cosine >= 0.95 — margins 0.006667 / 0.003077 / 0.006611 / 0.006040, all PASS. The floor covers the 768 width only.
- UNITS 23, 24, 25 ARE DONE: the vision tower on the CPU leg (gate 23, 0.82%/0.53% relrms), the DEVICE tower (gate 24: device vs CPU 0.15841668 / 0.009006 / 0.99933243 worst, 0.229 s against the CPU leg's 4.336 s, ~19x; one defect found and fixed: the rope pairing), the SPLICE and the engine entry (gate 25, four legs; the frozen splices are the BASELINE model's own vectors — comparing the artifact against them measures the tier, not the splice).
- UNIT 26 (row P4.7, the WIRE) IS DONE: ds4-rs 4cb4f3d9. /health's `embed` object gained the image half (soft tokens 70..1120, pixels 161280..2580480, embd 512, per_item_max 1, serial), rendered by EmbedCaps::to_json and pinned textually against ds4_eg2vq.h / ds4_eg2vq_vision.h by crates/ds4-core/tests/eg2vq_serving_caps.rs; POST /v1/embeddings takes an input item that is a string or the chat route's OpenAI content-part shape (one image_url data URI, decoded by the SAME decoder — crate::parse::decode_image_data_uri); the route runs probe -> Rust sequence (ds4_core::embed_media_ids) -> caps -> rows -> splice; new C/bridge surfaces ds4_engine_embed_gemma_vq_image, ds4_bridge_eg2vq_embed_image, ds4_bridge_model_vision_encode_rows; six refusals by name (no placeholder, placeholder with no image, two placeholders, past the item ceiling, a second image, a non-data-URI url). MEASURED by gate 26 (tools/eg2_embed_media_gate26.sh, fast-path step 9): the wire's own sequence = the anchor's 92 positions, span [2,90); device vs CPU 0.00057160 / 0.004348 / 0.99999055 (pattern), 0.00025820 / 0.002304 / 0.99999735 (gradient), 0.00048280 / 0.004064 / 0.99999174 (pattern+prefix); device vs the FROZEN BASELINE splice 0.03686520 / 0.242455 / 0.97060790 = gate 25's leg-3 band; pattern vs gradient cosine 0.78589224, vs a text item 0.63687671; one image end to end 0.290780 s (device) / 5.965929 s (CPU). ONE C BEHAVIOUR CHANGE, measured and named: ds4.c's --vision guard no longer demands the CUDA backend for gemma_vq (its tower has a complete CPU leg), which is what makes the two legs comparable on one machine; every other family's guard is unchanged.
- RULE 19 IS DEFERRED (section 1, third directive): qa-evidence/qa-deferral.txt + tests/qa-gate.sh's deferral branch. F6 stands: the surface extractor matches single-line definitions, so multi-line C entry points (this unit's three) never appear in its list — the whole-port pass must read the diffs by hand.
- Machine-facing facts RE-RUN this session (2026-10-20): tests/run.sh fast path GREEN in 3 m 45 s (steps 1-9 PASS, step 10 the recorded deferral; gate 26 in it as step 9), `cargo test -q -p ds4-core` green (all suites: 442 + 13 + ... , 0 failed), `make ds4 ds4-server` clean, `sha256sum -c SHA256SUMS` all FOURTEEN entries OK. ONE PRE-EXISTING FLAKE FOUND AND NOT FIXED: `cargo test -p ds4-server --lib` fails cache_identity::tests::bounded_sidecar_and_ple intermittently (3 runs: 1 pass, 2 fail; the two identity arrays come out byte-identical; cache_identity.rs is NOT in this unit's diff). Recorded in section 7.
- Workspace: /data/embeddinggemma2, branch dev, HEAD = the commit that carries this handoff, tree clean at write time. Unpushed: outer TWENTY-SIX commits over publish/dev d45394f (this handoff's own docs commits included; `git rev-list --count publish/dev..HEAD` is the authority); ds4-rs TEN over 29873445 (tip 4cb4f3d9). Both pushes are OFFERS (section 7). OUT/, .models/ and ds4-rs/ are gitignored: the artifacts are identified by the digests in SHA256SUMS.
- WHAT IS LEFT, in order: (1) unit 21, batching several items into one tower call — the plan's own last code row, spec whole at docs/eg2vq-batch21-plan.txt (section 3); (2) the FINAL rule 19 pass over the whole port, which must delete qa-evidence/qa-deferral.txt first (the code is complete once unit 21 lands — then the owner's instruction is satisfied); (3) the C CLI's tokenizer gap — the owner's call (the Rust host owns encoding; the C CLI refuses the artifact by name); (4) the AUDIO tower — the owner's call; (5) the two pushes, when the owner answers. NOT planned, but the next faithfulness gap if the owner wants it: the model card allows SEVERAL image placeholders per item (its own example has two), and this port deliberately supports one (`embed.image.per_item_max`), because the splice fills one contiguous run — a multi-span unit would need the CPU graph, the device ops and the engine check widened together.

3. Next unit plan — unit 27: unit 21, BATCHING several items into one tower call (phase-doc row P2.3's last slice, deferred 2026-10-19 as a priority redirect)

Goal: several embedding items in ONE tower call, with the per-call row budget (`ds4_core::GEMMA_VQ_BATCH_ROWS_MAX`, 8192) the caps already advertise becoming the
thing the route and the CLI can actually use, instead of looping item by item.

WHY THIS IS THE NEXT UNIT: it is the plan's own remaining code row (the handoff's
"what is left" has listed it since unit 20), its deferral was a redirect ("do the
vision tower first"), and the wire that would consume it landed in unit 26. The one
thing that is NOT mine to decide is whether the owner wants it at all — if the next
session reads a newer instruction that drops it, the audio tower and the pushes are
the remaining owner-gated items.

STEP 0 — read first, in this order:

 1. THE PARKED SPEC, whole: docs/eg2vq-batch21-plan.txt. It was written at the
    unit-20 boundary and holds the reference reading already done (the batch shape,
    the KV/attention plan, the row budget's arithmetic, the packer's row layout).
    Where the spec and the code disagree AFTER unit 26, the code wins and the spec
    is corrected in the same commit.
 2. WHAT UNIT 26 ADDED AND THE BATCH MUST NOT BREAK: the per-call row budget is
    counted in serve.rs's run_embed loop (total_tokens vs caps.batch_rows_max) and
    the item ceiling is counted per item; the media path uses ONE device vision
    workspace per engine (media_serial) and the text tower's device plan is bound
    once per engine (ds4.c's eg2vq_device_ensure, which REBUILDS the workspace when
    a longer item arrives — a batch is a size class, so it must size once).
 3. THE DEVICE LAYOUT'S OWN COST TABLE: ds4_eg2vq_gpu.cuh's workspace fields and the
    108,548 bytes per row (4-byte id + 27,136 f32 planes), which
    crates/ds4-core/tests/eg2vq_serving_caps.rs already pins arithmetically.
 4. THE C ENTRY'S SHAPE: ds4_engine_embed_gemma_vq (one item, out768) and the CPU
    reference's packed rows (ds4_eg2vq_ref_embed).  A batch entry must be a NEW
    entry (ds4_engine_embed_gemma_vq_batch) rather than a changed one: the CLI, the
    route and the oracles all call the single-item entry today.

THE WORK, in the order it has to land:

 A. THE DEVICE BATCH. The tower's workspace is already a plane of rows; the batch
    entry must (i) take n items' ids and lengths, (ii) run them as ONE sequence
    plane with an attention mask that keeps items apart (the reference's batch), or
    as n sequential rows through the same bound plan if the mask is not what the
    reference does — READ tools/mtmd and src/llama-batch.cpp FIRST and mirror it
    (rule 0.2), (iii) write n x 768 outputs.
 B. THE CPU LEG, the same call, so the two legs stay comparable (gate 26's method).
 C. THE ROUTE + the CLI: /v1/embeddings already loops; a batch entry changes that
    loop into one call when the request's items fit the budget, with the same caps
    and refusals.  The CLI's --embed path likewise.
 D. A GATE (gate 27): the batch's vectors against the single-item ones, item by
    item, with the bound declared and printed first, plus the row budget's
    arithmetic at the boundary (a request that exactly fits and one that crosses).
 E. THE USUAL CLOSE: evidence doc, fast-path step, phase-doc row, and the deferral
    note if this unit is the last code row (then the QA pass follows).

WHAT UNIT 27 MUST NOT DO: change the single-item entry's behaviour (every existing
gate measures it), invent a second caps mechanism, batch across the media path
before the text path is measured (one vision workspace is serial by design), or
claim a throughput number without measuring it under the condition it claims.

4. Reference map — the exact repos and files (pinned; verified at this boundary)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (all the work) | /data/embeddinggemma2/ds4-rs @ 70278c19, branch feat/ds41-dsml-carriage, gitignored by the outer repo, own git | ds4.c, ds4.h, ds4_eg2vq.{h,inc,graph.inc,gpu.cuh}, ds4_eg2vq_vision.{h,inc}, ds4_glmvq*.{h,inc}, cuda/eg2vq_{attn,ops,vq,vis}.cuh, native/bridge/ds4_bridge.{c,h}, crates/ds4-core/src/**, crates/ds4-server/src/**, crates/ds4-sys/src/lib.rs, tests/parity/, the Makefile |
| Form and metadata authority | /data/llama.cpp @ 31288c9c (b11723, MIT, read-only, source only) | src/models/gemma-embedding2.cpp, src/llama-{arch,hparams,model,graph,vocab,batch}.cpp, ggml/include/gguf.h, conversion/{gemma,base}.py, gguf-py/gguf/**, examples/embedding/embedding.cpp, tools/mtmd/** (phase 8) |
| The VISION reference (phase 8) | /data/llama.cpp @ 31288c9c, the same pin | tools/mtmd/models/gemma4v.cpp (:5-117), tools/mtmd/clip.cpp (:337-450, :144-147, :254-256, :973-979, :1719-1736, :4419-4428, :1342-1352), clip-impl.h:112-157, mtmd.cpp:953-961, mtmd-image.cpp:57-96 / :122-156 / :204-390 / :7-14, clip-model.h:66-68 / :204-211, mtmd-helper.cpp:404, conversion/gemma.py:978-1063 |
| The WIRE's own contract | our own committed data | .models/EmbeddingGemma-2-914f7f8/README.md section 5 (the model card: `<|image|>` marks the position, prefixes apply to text only, the 70..1120 budget) and config.json's ids (image 258880, boi 255999, eoi 258882) |
| The VISION weights file, and the frozen image reference | our own committed data | out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf (sha256 9ffb86e3.., 211 tensors, written by tools/eg2_mmproj.c) and tests/vision/{eg2-vision-gradient-512x384.png,eg2-vision-pattern-512x384.png} with tests/vision/reference/*.{soft,text}.tsv (the pin's own numbers, produced once by tools/eg2_vision_anchor.sh) |
| The anchor build (phase 2) | /data/llama-anchor-build (CPU, plus an mtmd target built for the vision anchor) | bin/llama-embedding; the frozen text vectors it produced are phase 2's deliverable, and nothing but tools/eg2_vision_anchor.sh touches this tree |
| Writer and payload precedent | /data/YoungAi (a4e6d31) | src/common/ds4_gguf_write.h, gguf-tools/quantize/v41_to_gguf.c, vq_fmt.h, vq_qc_bytes.h |
| Verification method | /data/ds4-dfm-rs (the live ds4 project; never modified from here) | AGENTS.md, tests/parity/, Makefile (the false-green guard) |
| The anchor vectors | our own committed data | tests/anchor/anchor-items.tsv (113 items) and tests/expected/anchor-embeddinggemma2.jsonl (768/512/256/128 vectors, sha256 7561254c..); tools/eg2_anchor.sh / eg2_anchor_cmp.sh |
| The artifact, and the container behind it | our own committed recipe | out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf (sha256 5bcf2216.., 1376 tensors), out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf (0d84f1a9..), out/...-dqvd.bin (8902fb8a..), out/artifact-manifest.tsv, out/artifact-kv.txt |
| The family's contract, loader, tower | our own code | crates/ds4-core/src/gemma_vq.rs (+ the caps and the image constants), gemma_vq/layout.rs, tok/gemma_vq.rs; ds4_eg2vq_graph.inc + ds4_eg2vq_bind.inc; tools/eg2_{loader_check,phase4_gate,tok_check}.sh; docs/{loader-gate3,text-tower-gate4}.txt |
| The gate-5 harness and the frozen set | our own code and data | tools/eg2_retrieval_set.sh, tests/retrieval/scifact-test-{queries,pool,qrels}.tsv, tools/eg2_{phase5_gate,oracle_to_jsonl,retrieval_score,retrieval_score_selftest}.sh, docs/retrieval-cost-gate5.txt |
| The embedding entry, its caps and the wire (phases 6 and 8) | our own code | ds4.c's ds4_engine_embed_gemma_vq{,_image}, ds4_bridge_eg2vq_*, crates/ds4-core/src/{serving.rs,gemma_vq.rs}, crates/ds4-core/tests/eg2vq_serving_caps.rs, crates/ds4-core/examples/{gemma_vq_tok,gemma_vq_media_ids,gemma_vq_vocab}.rs, crates/ds4-server/src/{serve.rs,parse.rs,generate.rs,serve_embed_media_test.rs}, tools/eg2_{embed_route_gate,embed_media_gate26}.sh, docs/{eg2vq-serving-caps-gate20,eg2vq-embed-entry-gate6,eg2vq-embed-media-gate26}.txt |
| The device path (phase 7) | our own code | cuda/eg2vq_*.cuh, ds4_eg2vq_gpu.cuh, tests/parity/eg2vq_embd_gpu_oracle.c, docs/eg2vq-device-gate7.txt, out/cuda-objects-backup-sm121a |
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
- A SECOND image decoder, a second caps mechanism, or a C-side sequence builder: the
  wire builds ids in Rust (the vocabulary lives there), decodes images with the chat
  route's own decoder, and renders the contract through EmbedCaps::to_json.
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
  eg2vq-splice-gate25,eg2vq-embed-media-gate26}.txt; qa-evidence/;
  tests/{anchor,expected,retrieval,vision}/ + tests/{run.sh,qa-gate.sh};
  tools/eg2_*.{c,sh} + bin/; out/ (all artifacts and logs, gitignored).
- The caps (unit 20 + unit 26): crates/ds4-core/src/gemma_vq.rs holds ITEM_TOKENS_MAX
  2048 (the device attention key plane) and BATCH_ROWS_MAX 8192 (a PORT policy:
  108,548 B of workspace per row, 848.03 MiB at the cap), plus the image constants
  (IMAGE_TOKEN_ID 258880, the two markers, IMAGE_SOFT_TOKENS_MIN/MAX 70/1120,
  IMAGE_PIXELS_MIN/MAX 161280/2580480, IMAGE_EMBD 512, IMAGE_PER_ITEM_MAX 1),
  reached through `ds4_core::embed_caps(family)`; /health reports them under `embed`
  (with the image half under `embed.image`); the route refuses past either text
  ceiling by name as 400, and refuses the six image shapes section 2 lists.
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
  137 MiB weights, workspace sized by max_tokens: 212.01 MiB at 2048 rows); the
  oracle is `tests/parity/eg2vq_embd_c_oracle <gguf> <ids.tsv>`.
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
  single-instance lock, so two servers cannot run at once — gate 26 starts and stops
  one per backend.
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

- P-1 (no Python in the runtime, ever) and P-2 hold; all five P-1 checks return zero and `sha256sum -c SHA256SUMS` passes (re-verified at this boundary).
- RULE 19 IS DEFERRED by the owner's recorded instruction (section 1): tests/qa-gate.sh prints the deferral and holds no verdict while qa-evidence/qa-deferral.txt exists. The finishing session DELETES that file and runs ONE pass over the whole accumulated port (every unit since unit 24, multi-line C entry points included — F6). Leaving the file in place after the code is complete is a rule 19 violation with the deferral as its alibi.
- NEW FLAKE, RECORDED NOT FIXED: `cargo test -p ds4-server --lib` fails `cache_identity::tests::bounded_sidecar_and_ple` intermittently (3 runs: 1 pass, 2 fail; the two identity arrays come out identical, so the test's own premise — that a rewrite changes the stat — does not hold on this filesystem). cache_identity.rs is not in this unit's diff; the fast path does not run it (step 12 runs ds4-core only). Fixing it touches the shared engine branch, so it is a finding for the owner, not a sideways edit.
- CLOSED: the container, the quality floor, the caps, the llama-free boundary, phase 7's device path, the vision tower on both legs, the splice, the engine entries, and now THE WIRE (units 23-26).
- OPEN (owner, NEW): the two pushes. Outer: `git push publish dev`, range d45394f..HEAD (TWENTY-SIX commits; read `git rev-list --count publish/dev..HEAD` for the exact number; README/docs/tools/tests/gates 22-26 and their evidence — no artifact: out/, .models/ and ds4-rs/ are gitignored; the tracked binaries in bin/ are rebuilt by the gates, and the anchor binary is deliberately NOT tracked because it links the pin). Target publish = https://github.com/JLFN/embeddinggemma2 (rule 6 wants a fresh `gh repo view` immediately before the push). Nested: `git push publish feat/ds41-dsml-carriage:refs/heads/dev`, range 29873445..HEAD (TEN commits as of 4cb4f3d9). Gates already run for both: rule 10 OSV 140 packages / 0 vulnerable (both roots, at an earlier boundary — re-run before an actual push), rule 19 DEFERRED by the owner's instruction, rule 5/12 as recorded. Rollback if a push must be undone: `git push --force-with-lease publish d45394f:refs/heads/dev` (outer) — a force-push, so the owner's call; both local histories are kept either way.
- OPEN (owner): the C CLI's tokenizer gap (the family's gemma4 BPE is Rust-side; the C CLI refuses the artifact by name) and the AUDIO tower (out of scope; the owner asked for vision).
- STANDING (owner, 2026-10-19): AFK; automatic continuation after each handoff and compaction; pushes stay offers. Recorded in section 1 with the two later directives.
- Nothing is in flight in either repository. The nested checkout's working tree is clean at 4cb4f3d9; the outer tree carries only the unit-26 commits and this handoff.
- The unit-20, -23 and -24 QA findings are closed (four corrections committed, one informational; unit 24's dispositions are docs/eg2vq-vision-tower-gate24.txt section 9). NO QA pass has run since unit 24, by the owner's instruction, and the deferral file says so.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git log --grep "Unit: .* complete" d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git status --short && git branch -vv                 # clean, on dev, TWENTY-SIX unpushed commits (read the count, not this comment)
    tests/run.sh                                         # the fast path (3 m 45 s, gates 22-26 included), ending in the rule 19 gate
    sha256sum -c SHA256SUMS                              # FOURTEEN entries
    tools/eg2_loader_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/artifact-manifest.tsv
    bash tests/qa-gate.sh                                # prints the deferral; must be read, not trusted as evidence
    git -C ds4-rs log --oneline -3 && git -C ds4-rs status --short   # 4cb4f3d9, clean
    cuobjdump --list-elf ds4-rs/ds4_cuda.o | tail -2     # ds4_cuda.sm_89.cubin = the arch record is real
    cat ds4-rs/.ds4-cuda-config.mk                       # CUDA_ARCH := sm_89
    tail -8 out/phase5-gate.log                          # the agreement + retrieval tables
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'
    # the deep gates, when a number is questioned (each log is in out/):
    #   gate 4   (~20 min, the token dump is reused): tools/eg2_phase4_gate.sh <artifact> <baseline> tests/anchor/anchor-items.tsv .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
    #   gate 5   (minutes, the legs are reused):      tools/eg2_phase5_gate.sh
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
    # the sequence driver the wire's leg 0 uses (vocabulary only, no weights):
    #   (cd ds4-rs && cargo run -q -p ds4-core --release --example gemma_vq_media_ids -- <artifact> '<|image|>' 88 Raw)

Trust git over this document's headings: continue with the first unit whose work is
not in the diff, which at this boundary is section 3 (unit 27 = unit 21, batching —
the plan's own last code row). Every phase through 8 is landed and gated; the QA pass
is deferred by the owner's instruction until the code is complete; the audio tower
and the C CLI's tokenizer gap are the owner's calls. The outer repo's HEAD is at
publish/dev d45394f (TWENTY-SIX unpushed, this handoff's own docs commits included); ds4-rs is at 4cb4f3d9
with its published dev at 29873445. Anything else in either unpushed range must be
read before sections 2-6 are trusted. The artifacts, the retrieval source, every
out/ log and the CUDA objects live under gitignored paths (out/, .models/, ds4-rs/),
so they never appear in a diff: they are identified by the digests in SHA256SUMS, by
the pins inside the gate scripts, and by the evidence docs' own digests.

This document is self-contained: a new session can execute the remaining unit from
this file alone. History is deliberately absent — git log, the phase doc and the
evidence docs are the historical record.
