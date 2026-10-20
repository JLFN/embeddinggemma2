QA report - unit 28: the FINAL accumulated rule 19 pass (units 25, 26 and 27)
============================================================================

Unit and repositories at verification time

Outer /data/embeddinggemma2, branch dev, HEAD 84d0849 (read with
git rev-parse --short HEAD); its only remote is the publish remote, whose last
pushed dev is d45394f, and the outer tree carries 30 commits over it.  Nested
/data/embeddinggemma2/ds4-rs, branch feat/ds41-dsml-carriage, HEAD 0b28cb40
(unit 27) at verification time, with the three unit-28 fixes since committed on
top of it as e01eba8e; its publish/dev is 29873445, and the nested tree carries
12 commits over it.  This is the owner's third directive executed ONCE over the
WHOLE accumulated port since unit 24's report -- units 25, 26 and 27 -- and it
discharges qa-evidence/qa-deferral.txt: the parent session deletes that file
together with this report's commit.

State of the three unit-28 fixes: COMMITTED.  The nested fixes
(ds4_eg2vq.h's and ds4_eg2vq_gpu.cuh's cost figures, crates/ds4-core/src/lib.rs's
span rise check) are nested commit e01eba8e, on top of 0b28cb40.  The outer
gate-20 fix (tools/eg2_embed_route_gate.sh's caps check plus the LATER NOTE in
docs/eg2vq-serving-caps-gate20.txt) is landed in the same outer commit that
carries this report: a file cannot name the commit that contains it, and the
report is finalized at that commit.  Every check named below was run against
exactly the code those commits carry -- it was verified while the fixes were
working-tree edits, and the commits change none of that content.

QA model: deepseek-v4.1-CC-flash (tests/qa-gate.sh:9-10; the model this project
recorded, never substituted).  Date 2026-10-20.  Host: RTX 4070 SUPER, sm_89,
CUDA 13.3, 28 CPUs.

Method: the project's own gates, suites and oracle, run live on this machine,
plus a file:line read for every surface the gate's extractor cannot see
(finding F6).  No code was modified by this pass; the only file it writes is
this report.  Scratch logs: /tmp/qa28-routegate-after.log,
/tmp/qa28-fastpath-after.log, /tmp/qa28-coretests-after.log.  No Python, ONNX
or PyTorch anywhere in the verification (P-1/P-2 respected), and the pin
(/data/llama.cpp) was neither built nor executed.

SURFACES COVERED (the eleven the gate's own extractor names, each verbatim)

- decode_image_data_uri - live: gate 26 leg 4's "not a data URI" 400 and the
  serve test an_image_uri_that_is_not_a_data_uri_is_refused_by_name; read:
  crates/ds4-server/src/parse.rs:486-495 (the chat route's own decoder and its
  own refusal words).
- ds4_bridge_eg2vq_embed_batch - live: gate 27's wire legs (the array request
  byte-identical to the single requests on BOTH backends) and this pass's
  device-oracle run; read: native/bridge/ds4_bridge.c:989-998.
- ds4_bridge_eg2vq_embed_image - live: gate 26's four device and three CPU
  image requests; read: native/bridge/ds4_bridge.c:1003-1011.
- ds4_bridge_model_vision_encode_rows - live: every image request (the engine's
  encode half); read: native/bridge/ds4_bridge.c:1015-1050, including the
  caller-owned-buffer facts (the expect_rows refusal at :1035-1038 and the
  buffer-size refusal at :1042-1047).
- embed_caps - live: gate 26 leg 1's served /health body, which is exactly the
  contract the route enforces, and cargo test -q -p ds4-core --test
  eg2vq_serving_caps (4 passed, 0 failed, which holds the Rust numbers equal to
  ds4_eg2vq.h and ds4_eg2vq_vision.h as text); read:
  crates/ds4-core/src/serving.rs:243-261.
- embed_gemma_vq_batch - live: gate 27's wire legs on both backends and the
  four serve_embed_batch_test tests; read: crates/ds4-core/src/lib.rs:2474-2560,
  span loop included (and now the fixed loop at :2509-2543, see U28-3).
- embed_gemma_vq_image - live: gate 26's device and CPU media requests; read:
  crates/ds4-core/src/lib.rs:2672-2761 (item ceiling, soft-row width, the
  contiguous-run check).
- embed_item_media_ids - live: every media item of gate 26 builds its sequence
  through it; read: crates/ds4-core/src/lib.rs:2604-2612.
- embed_media_ids - live: gate 26 leg 0's 92-position sequence check (BOS 2,
  boi 255999, 88 x 258880, eoi 258882, EOS 1, span [2, 90)); read:
  crates/ds4-core/src/lib.rs:2777-2845.
- to_json - live: /health's embed object exactly as served (gate 26 leg 1's own
  line, image half included); read: crates/ds4-core/src/serving.rs:269-279 for
  the image half and :282-291 for the whole object, whose two callers are the
  /health body (crates/ds4-server/src/serve.rs:1646) and the plan diagnostics
  (serving.rs:1908).
- vision_encode_rows - live: the encode half of every image request on both
  legs; read: crates/ds4-core/src/lib.rs:2623-2661 (empty image, the expect_rows
  window, exactly expect_rows x 512 floats).

F6 HAND-READ SET (the multi-line C entries the extractor cannot match), each
read by hand and reached live by the gates named

- ds4_engine_embed_gemma_vq_batch - ds4.c:73827-73923; gate 27's wire legs and
  its CPU backend leg.
- ds4_engine_embed_gemma_vq_image - ds4.c:73945-74040 (the placeholder run
  checked by name before any arithmetic); gate 26's six refusals and its four
  device requests.
- ds4_gpu_eg2vq_embed_batch - ds4_eg2vq_gpu.cuh:657-660, into
  eg2vq_gpu_embed_core, which derives pos/len from the spans in one place and
  pools per item; the device oracle's --batch and --batch-bad legs.
- ds4_eg2vq_ref_embed_batch - ds4_eg2vq_graph.inc:418-434 (it loops the gated
  one-item core, deliberately); gate 27's CPU leg and the wire's CPU leg.
- eg2vq_ops_proj768_pool_batch - cuda/eg2vq_ops.cuh:405-414 with
  eg2vq_ops_pool_item_kernel at :364-398; every batched run of gate 27 and the
  ops unit test's packed case.
- the route's POST /v1/embeddings item shape (crates/ds4-server/src/serve.rs's
  embed_input_item, :942-1010: OpenAI content parts, one image_url, the URI
  through the chat route's decoder, every other part refused by name) and its
  two-pass grouping in run_embed (:1703-1909: all items prepared and
  caps-checked, then the TEXT items in ONE batch call and each media item in
  its own); gate 27's wire legs and the four serve_embed_batch_test tests.

LIVE EVIDENCE FROM THIS PASS, ON THE FIXED TREE

- tests/run.sh: exit 0, 2 m 46 s, ending "fast path complete".  Steps 1-10
  green: 14/14 recorded digests; the DQVD container's structure; the scorer
  self-test; loader gate 3; gate 22 EG2VQ_VISION_MMPROJ_PASS; gate 23
  EG2VQ_VISION_GATE23_PASS; gate 24 EG2VQ_VISION_GATE24_PASS; gate 25
  EG2VQ_SPLICE_GATE25_PASS; gate 26 EG2VQ_EMBED_MEDIA_GATE26_PASS; gate 27
  EMBED BATCH GATE 27: PASS.  Step 11 prints the deferral, which is correct and
  expected until the parent deletes qa-evidence/qa-deferral.txt.
- Gate 25 reproduced its evidence: the splice leg worst 0.00018850 / 0.001579 /
  0.99999875, device vs CPU 0.00057164 / 0.004348 / 0.99999055, refusals 4 of 4.
- Gate 26 reproduced its evidence: the served /health body (item_tokens_max
  2048, batch_rows_max 8192, the whole image half); device vs CPU
  0.00057160 / 0.004348 / 0.99999055 for the pattern fixture; six refusals each
  a 400; the frozen-baseline cross-check 0.03686520 / 0.242455 / 0.97060790.
- Gate 27 reproduced its evidence: "every element byte-identical to the
  per-item run, max per-element delta 0.000e+00" over 6 items / 3881 rows and
  64 items / 1280 rows; 8192 rows accepted in one call (one workspace of
  912.13 MiB) while 8193 is refused; five malformed span tables "refused,
  output untouched"; the wire byte-identical on both backends; four items of
  exactly 2048 tokens accepted (200) and the fifth a 400 quoting
  embed.batch_rows_max.
- The device oracle run earlier in this pass (--batch --batch-bad --no-cpu):
  exit 0, the same byte-identity line and the same five refusals.
- cargo test -q -p ds4-core with the fixes in place: 664 passed, 0 failed,
  20 ignored.  cargo test -p ds4-server --lib serve_embed: 12 passed, 0 failed
  (the 8 media and 4 batch route tests).  The two CUDA unit tests:
  EG2VQ_ATTN_PASS (packed two items) and EG2VQ_OPS_PASS (3 items packed).
- bash tools/eg2_embed_route_gate.sh <artifact>: exit 0 and
  EG2VQ_EMBED_ROUTE_PASS (U28-1's confirming check, below).

FINDINGS FROM THIS PASS - ALL THREE CLOSED

U28-1 CLOSED - the route gate's stale /health expectation (a checked-in gate
that exited 1 on HEAD).
  Fix: tools/eg2_embed_route_gate.sh:176 now greps the PREFIX of the served
  object, '"embed":{"item_tokens_max":2048,"batch_rows_max":8192,', with the
  reason in the comment above it (:169-175: the whole literal could not survive
  unit 26's growth of the contract, and gate 26 leg 1 owns the image half's
  numbers); docs/eg2vq-serving-caps-gate20.txt:198-220 records the drift and
  the fix as a dated LATER NOTE.
  Confirming check: bash tools/eg2_embed_route_gate.sh
  out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf -> exit 0,
  EG2VQ_EMBED_ROUTE_PASS, its 4b section printing the served embed object with
  unit 26's image half.

U28-2 CLOSED - the C sources' per-row workspace cost was one plane stale.
  Fix: ds4_eg2vq.h:192 and :195 now say 116,748 bytes a row and 912.13 MiB
  (pointing at the layout as the authority and at gate 27 as the measurement);
  ds4_eg2vq_gpu.cuh:60-66 now states 116,748 with the plane list (the tower's
  27,648 f32 = 110,592 bytes a row, in_soft included) and :519 says 116,748.
  Confirming check: the three sites re-read, and the layout recomputed from the
  constants in the same file (ds4_eg2vq_gpu.cuh:84-122) -> 8192 rows =
  956,432,400 bytes = 912.1250 MiB, with the interior per row 110,592 + 4 + 4
  + 4 + 3,072 + 3,072 = 116,748; gate 27's own output prints the same 116,748
  and the 8192-row call's 912.13 MiB.

U28-3 CLOSED - the Rust batch entry had no rise check, so a descending span
table underflowed (a debug-build panic) instead of the documented Err.
  Fix: crates/ds4-core/src/lib.rs:2510-2534 checks item_off[i + 1] ==
  item_off[i] (the empty item, :2515-2523) and item_off[i + 1] < item_off[i]
  (does not rise, :2525-2533) BEFORE the subtraction at :2535, each with its
  own Err message, so no build profile can underflow; the comment at
  :2510-2514 states why.
  Confirming check: the loop read; cargo check -q -p ds4-core exit 0; and
  cargo test -q -p ds4-core with the fix in place 664 passed, 0 failed.

RESIDUAL RISK ACCEPTED, AND WHAT WAS NOT TESTED

- No sanitizers, no valgrind and no concurrency stress were run; the gates'
  clean exits are not a leak or race proof.  The engine is single-owner and the
  media path is serial by design, and none of the evidence claims otherwise.
- The pin (/data/llama.cpp) was neither built nor executed; its citations are
  read as the pinned revision through the units' own docs.
- At the oracle level the 8193-row refusal is reached through the workspace
  create (8193 rounds to 8448 rows, past the 8192 clamp, refused by name)
  rather than through the device core's own per-call row check; the engine
  refuses 8193 rows by name before any allocation and gate 27 exercises the
  boundary at the wire, so the boundary itself is covered.  Recorded for
  precision, not as an open defect.
- The cache_identity flake (crates/ds4-server/src/cache_identity.rs:500-501)
  is UNCHANGED and is not this pass's to fix: it is recorded in
  embeddinggemma2-handoff.md section 7 as an open, owner-owned finding, it is
  outside units 25-27's diffs, and the fast path does not run it.  Re-confirmed
  live during the pass (the full ds4-server lib suite 357 passed / 1 failed;
  focused re-runs 1 fail / 2 pass).
- Out of scope by the owner's decisions: the audio tower, a multi-image item,
  the C CLI's tokenizer gap, and the two pushes.

post-write confirmation: bash tests/qa-gate.sh -- after this report was written
and with the owner's deferral bypassed (QA_DEFERRAL=/dev/null, exactly the
check that runs once the parent deletes the file) -- prints
QA GATE: overall PASS (deepseek-v4.1-CC-flash evidence fresh and covering) and
exits 0, covering every surface named above.

verdict: overall PASS
