QA FINDINGS - unit 28: the final rule 19 pass over units 25, 26 and 27
=====================================================================

Result: FINDINGS OPEN. This pass writes NO PASS report (the unit-28 brief: one
or more real defects means this file, not qa-evidence/qa-report.md). Three
findings, each reproducible on HEAD (U28-3 to the extent its own evidence
states), each in checked-in artifacts rather than in the arithmetic gates 25-27
measure: a project gate that now exits 1, a per-row
cost figure in the C sources that contradicts the corrected Rust figure and the
measured workspace, and a missing rise check in the Rust batch entry that a
debug build turns into a panic where the C core refuses. Everything this pass
could drive LIVE came back green and reproduced the evidence documents'
numbers; the clean list is section 2, and the surfaces and their hand-read
sites are section 1.

0. WHAT THIS PASS IS

- The owner's third directive (embeddinggemma2-handoff.md section 1, recorded
  verbatim in qa-evidence/qa-deferral.txt): rule 19 QA runs ONCE, when the port
  is code-complete. The port is code-complete (handoff section 2: unit 27 was
  the plan's last code row), so this is that pass, over the WHOLE accumulated
  port since unit 24's report.
- QA model: deepseek-v4.1-CC-flash (tests/qa-gate.sh:9-10, the model this
  project recorded; no substitution).
- Date 2026-10-20. Host: RTX 4070 SUPER, sm_89, CUDA 13.3, 28 CPUs.
  Workspace /data/embeddinggemma2, branch dev, outer HEAD 84d0849 (30 commits
  over publish/dev d45394f); nested /data/embeddinggemma2/ds4-rs, branch
  feat/ds41-dsml-carriage, HEAD 0b28cb40 (11 commits over its publish/dev
  29873445). Both trees clean at the start of this pass.
- No code was modified by this pass. The only file written in either tree is
  this one. qa-evidence/qa-deferral.txt is left in place for the parent, which
  owns its deletion. Scratch output went to /tmp (fast-path and route-gate
  logs, and the two rustc scratch binaries named in Finding U28-3).

1. THE SURFACES, AND HOW EACH WAS COVERED

1.1 The eleven surfaces the gate's own extractor names (printed live by the
    deferral branch, tests/qa-gate.sh:150-159):

    decode_image_data_uri, ds4_bridge_eg2vq_embed_batch,
    ds4_bridge_eg2vq_embed_image, ds4_bridge_model_vision_encode_rows,
    embed_caps, embed_gemma_vq_batch, embed_gemma_vq_image,
    embed_item_media_ids, embed_media_ids, to_json, vision_encode_rows.

    Hand-read sites and the live exercise for each:

    - decode_image_data_uri - crates/ds4-server/src/parse.rs:486-495 (the
      chat route's own decoder and its words: add_image_data_uri at :460-477,
      "image_url must be a base64 data URI" at :467-473). Live: gate 26 leg 4's
      "not a data URI" 400, and the unit test
      serve_embed_media_test::an_image_uri_that_is_not_a_data_uri_is_refused_by_name.
    - ds4_bridge_eg2vq_embed_batch - native/bridge/ds4_bridge.c:989-998 (the
      engine owns every refusal).
    - ds4_bridge_eg2vq_embed_image - native/bridge/ds4_bridge.c:1003-1011.
    - ds4_bridge_model_vision_encode_rows - native/bridge/ds4_bridge.c:1015-1050
      (a CALLER-owned buffer; expect_rows re-check at :1035-1038, buffer-size
      check at :1042-1047, memcpy at :1048).
    - embed_caps - crates/ds4-core/src/serving.rs:243-261. Live:
      cargo test -q -p ds4-core --test eg2vq_serving_caps -> 4 passed, 0 failed;
      gate 26 leg 1's served /health body is the documented row, image half
      included.
    - embed_gemma_vq_batch - crates/ds4-core/src/lib.rs:2474-2560 (its checks
      at :2477-2533, the FFI call at :2544-2555). Live: gate 27's wire legs on
      both backends, the four serve_embed_batch_test tests, and this pass's own
      device-oracle run.
    - embed_gemma_vq_image - crates/ds4-core/src/lib.rs:2672-2761 (item
      ceiling :2680-2689, soft-row width :2692-2709, the contiguous-run check
      :2711-2731, FFI :2745-2757).
    - embed_item_media_ids - crates/ds4-core/src/lib.rs:2604-2612 (the thin
      wrapper the route calls, over embed_media_ids).
    - embed_media_ids - crates/ds4-core/src/lib.rs:2777-2845 (the sequence
      rules; its own refusals listed at :2773-2776; marker ids read out of the
      vocabulary at :2848-2862, never written down).
    - to_json - crates/ds4-core/src/serving.rs:269-279 (EmbedImageCaps) and
      :282-291 (EmbedCaps). Its two callers are /health
      (crates/ds4-server/src/serve.rs:1646) and the plan diagnostics
      (serving.rs:1908). Live: the exact /health body gate 26 leg 1 prints.
    - vision_encode_rows - crates/ds4-core/src/lib.rs:2623-2661 (empty image,
      expect_rows window, exactly expect_rows x IMAGE_EMBD floats - the buffer
      C is allowed to write).

1.2 The multi-line C entries that finding F6 keeps out of the extractor, read
    here by hand and driven live by the gates named:

    - ds4_engine_embed_gemma_vq_image - ds4.c:73945-74040. The placeholder run
      is checked by name BEFORE any arithmetic (count and contiguity at
      :73989-74004), the soft block's ceiling at :73978-73984, then the
      text entry's own checks, then the device or CPU leg.
    - ds4_engine_embed_gemma_vq_batch - ds4.c:73827-73923. Span coverage
      :73857-73862, row budget against DS4_EG2VQ_MAX_ROWS :73863-73867,
      per-item non-empty and item-ceiling checks :73868-73882, id range
      :73885-73892, one workspace for the whole call (:73902, via
      eg2vq_device_ensure at :73656-73693).
    - ds4_gpu_eg2vq_embed_batch - ds4_eg2vq_gpu.cuh:657-660, the one-item
      entries being one-item calls of the same core (:663-681). The core
      eg2vq_gpu_embed_core (:462-651) holds the refusals (:466-501), derives
      pos/len from the spans in ONE place (:518-530) and pools per item
      (:641-645).
    - ds4_eg2vq_ref_embed_batch - ds4_eg2vq_graph.inc:418-434: loops the gated
      one-item core, deliberately (the CPU leg is the oracle).
    - eg2vq_ops_proj768_pool_batch - cuda/eg2vq_ops.cuh:405-414, with
      eg2vq_ops_pool_item_kernel at :364-398 (one block per item, the item's own
      rows in order, its own mean).
    - POST /v1/embeddings' new item shape - crates/ds4-server/src/serve.rs's
      embed_input_item (:942-1010: content parts read in order, text parts
      concatenate, one image_url allowed, the URI through decode_image_data_uri,
      every other part type refused by name) and run_embed (:1703-1909: every
      item prepared and caps-checked first, then the TEXT items in ONE batch
      call and each media item in its own).
    - Also read for unit 25's part of the hand list: ds4_engine_vision_probe
      (ds4.c:73475-73537) and ds4_engine_vision_encode_memory (:73546-73608),
      ds4_gpu_eg2vq_embed_image (ds4_eg2vq_gpu.cuh:673-680) and
      ds4_eg2vq_ref_embed_image (ds4_eg2vq_graph.inc:437-443).

2. WHAT WAS VERIFIED LIVE, AND ITS KEY LINES

- tests/run.sh (run twice; the second with the full output kept at
  /tmp/qa28-fastpath.log): exit 0, 2 m 47.5 s and 2 m 48 s, ending "fast path
  complete (deep path: tests/run.sh --deep)". Steps 1-10 green: 14/14 recorded
  digests OK; the DQVD container "structure OK: 1376 slots"; the scorer
  self-test PASSED (8 cases); loader gate 3; EG2VQ_VISION_MMPROJ_PASS;
  EG2VQ_VISION_GATE23_PASS; EG2VQ_VISION_GATE24_PASS;
  EG2VQ_SPLICE_GATE25_PASS; EG2VQ_EMBED_MEDIA_GATE26_PASS and
  EMBED BATCH GATE 27: PASS. Step 11 printed the recorded deferral, which is
  the expected state until the parent deletes qa-evidence/qa-deferral.txt.

- gate 25 (the splice) reproduced its evidence numbers: leg 1 worst
  maxabs 0.00018850 / relrms 0.001579 / cosine 0.99999875 against its declared
  0.005 / 0.010 / 0.999; leg 4 device-vs-CPU 0.00057164 / 0.004348 / 0.99999055
  against 0.005 / 0.005 / 0.9999; refusals 4 of 4; the engine probe/encode
  lines (source 512x384, padded 528x384, grid 11x8, 88 tokens; device soft rows
  vs the CPU leg 0.15841668 / 0.009006 / 0.99995945 and
  0.04474682 / 0.004954 / 0.99998773).

- gate 26 (the image wire): leg 1's /health line is exactly the documented
  contract, image half included -
    {"status":"ok",...,"embed":{"item_tokens_max":2048,"batch_rows_max":8192,
     "image":{"soft_tokens_min":70,"soft_tokens_max":1120,"pixels_min":161280,
     "pixels_max":2580480,"embd":512,"per_item_max":1,"serial":true}}}
  leg 2/3 device-vs-CPU: pattern 0.00057160 / 0.004348 / 0.99999055, gradient
  0.00025820 / 0.002304 / 0.99999735, pattern+prefix 0.00048280 / 0.004064 /
  0.99999174; six refusals, each a 400 quoting its own reason; leg 5 device
  wire vs the frozen baseline splice 0.03686520 / 0.242455 / 0.97060790 - gate
  25's leg-3 band, i.e. the wire's sequence is the reference's.

- gate 27 (the batch): leg 1 (6 items, 3881 rows) "equality: every element
  byte-identical to the per-item run, max per-element delta 0.000e+00"; leg 2
  (64 items, 1280 rows) the same; leg 3 "accepted: 8192 rows in one call"
  (one workspace of 912.13 MiB) and "refused: the device batch entry took none
  of this call" for 8193; five malformed span tables "refused, output
  untouched"; the wire's array-vs-singles byte-identical on BOTH backends; four
  items of exactly 2048 tokens are one accepted call (200) while the same four
  plus a fifth is a 400 quoting embed.batch_rows_max.

- This pass's own oracle run:
    ./ds4-rs/tests/parity/eg2vq_embd_gpu_oracle <artifact> out/eg2vq-batch27-ids.tsv --batch --batch-bad --no-cpu
  exit 0; the same byte-identity line over 6 items x 768 values; the five
  refusals with the caller's buffer untouched; the layout's own workspaces
  (2048-row context 228.0 MiB, packed 3373 rows 399.05 MiB).

- The two CUDA unit tests (make tests/test_eg2vq_attn tests/test_eg2vq_ops):
  EG2VQ_ATTN_PASS with "two items packed (40 + 1200 rows) rel 1.04e-06 mincos
  1.000000000" and "5 query heads (not the tower's 4) refused, output
  untouched"; EG2VQ_OPS_PASS with "3 items packed (1/349/40 rows) worst
  rel_rms 3.99e-07".

- cd ds4-rs && cargo test -q -p ds4-core: 664 passed, 0 failed, 20 ignored
  (summed from the run's own "test result" lines; the handoff's count is
  reproduced).

- cd ds4-rs && cargo test -p ds4-server --lib serve_embed: 12 passed, 0 failed
  (the 8 serve_embed_media_test and 4 serve_embed_batch_test tests).

- bash tests/qa-gate.sh: the deferral branch prints (tests/qa-gate.sh:148-158),
  exit 0 - the owner's recorded instruction, not a verdict.
  QA_DEFERRAL=/dev/null bash tests/qa-gate.sh: the extractor prints exactly the
  eleven surfaces above; against unit 24's report the exists/fresh/verdict
  checks PASS and the ten coverage checks FAIL - the correct reading of "no
  report covers units 25-27 yet", and the reason a PASS report cannot be
  written while section 3 has open findings.

- bash tools/eg2_embed_route_gate.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf:
  exit 1, EG2VQ_EMBED_ROUTE_FAIL - Finding U28-1.

3. FINDINGS

U28-1 (medium; a checked-in project gate now exits 1 on HEAD; introduced by
unit 26 and unnoticed because the gate is not in the fast path)

  WHERE: tools/eg2_embed_route_gate.sh:170, section 4b:
      if ! grep -q '"embed":{"item_tokens_max":2048,"batch_rows_max":8192}' "$WORK/health.json"; then
  The literal predates the image half: it was added in outer commit e407734
  (unit 20) and has never been touched since (git log --oneline -L 164,172 --
  tools/eg2_embed_route_gate.sh lists e407734 as the only commit that wrote
  those lines; the file's --follow history also lists its docs sibling
  2f52c12). Unit 26's nested commit 4cb4f3d9 then changed
  EmbedCaps::to_json (crates/ds4-core/src/serving.rs:282-291) to append
  "image":..., so the served body now reads

    "embed":{"item_tokens_max":2048,"batch_rows_max":8192,"image":{...}}

  and the literal-with-closing-brace no longer occurs anywhere in the body.

  EVIDENCE (commands and their lines):
  - bash tools/eg2_embed_route_gate.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf
    exits 1. Its own output:
      == 4b. the embedding caps /health advertises (phase 6 row P2.3) ==
      embed: {"item_tokens_max":2048,"batch_rows_max":8192,"image":{"soft_tokens_min":70,...}}
      FAIL: /health does not advertise the family's embedding caps
      ...
      EG2VQ_EMBED_ROUTE_FAIL
    The line the gate prints under its FAIL is itself the proof that the
    numbers ARE advertised: the check's expectation, not the served contract,
    is what is stale.
  - Every other section of the gate is green (section 5: route vector vs CLI
    cos 1.000000000000; section 4c's two refusals name the caps; section 3's
    256-dim vector is unit length). The failure is section 4b alone.
  - The gate's intent is still satisfied by the live server: gate 26 leg 1
    (fast-path step 9) asserts the same two numbers plus the image half and
    passes; the serve test health_reports_the_image_half_when_the_family_has_one
    passes.
  - docs/eg2vq-serving-caps-gate20.txt:95 records EG2VQ_EMBED_ROUTE_PASS from
    unit 20; that artifact no longer reproduces on HEAD, and no document
    records the supersession.

  WHAT IT SHOULD BE: the check updated to the current contract, e.g. the grep
  pattern widened to `"embed":{"item_tokens_max":2048,"batch_rows_max":8192,"image":`
  or the body parsed as JSON, after which the gate must again print
  EG2VQ_EMBED_ROUTE_PASS and exit 0. A gate cannot be left red on HEAD: either
  it is fixed or it is recorded as superseded and replaced, and neither has
  happened.

U28-2 (low; the per-row workspace cost in the C sources is one plane stale -
the same defect class unit 27 corrected in two other places, written by the
same commit that corrected them)

  WHERE (all three written by unit 27, nested commit 0b28cb40, per
  git log -L 190,196:ds4_eg2vq.h and git log -L 60,65:ds4_eg2vq_gpu.cuh):
  - ds4-rs/ds4_eg2vq.h:192-195 - "The workspace is linear in rows at 114,700
    bytes each ... so the cap is 896.2 MiB of device workspace."
  - ds4-rs/ds4_eg2vq_gpu.cuh:60-65 - "...against the 108,548 bytes a row
    already costs; ... which is what makes the batch's row 114,700 bytes
    rather than 108,548".
  - ds4-rs/ds4_eg2vq_gpu.cuh:517 - "8 bytes a row against the 114,700 the
    row's own planes cost".
  The number is short by exactly the image splice's in_soft plane: 512 f32 =
  2,048 B a row, 16 MiB at 8,192 rows.

  EVIDENCE:
  - The same unit's own corrections say 116,748 B a row and 912.13 MiB:
    crates/ds4-core/src/gemma_vq.rs:102-109 ("exactly linear in rows at 116,748
    bytes each ... 912.13 MiB of workspace"), and
    crates/ds4-core/tests/eg2vq_serving_caps.rs:96-139 (the plane list and
    PER_ROW 116,748 at :112-116; the assertion is against 912.13 MiB at
    :126-130; the comment at
    :103-109 states the 108,548 figure was stale).
  - The layout itself, computed from the constants in
    ds4_eg2vq_gpu.cuh:84-122:
      awk 'BEGIN{t=8192; spans=int(((t+1)*4+15)/16)*16;
      n=t*4+t*768*4+4*(t*512*4)+t*4*256*4+t*4*512*4+2*(t*2*256*4)+t*4*256*4+
      t*4*512*4+2*(t*2048*4)+t*512*4+2*(t*512*4)+t*512*24*4+t*512*4+t*4+t*4+
      spans+t*768*4; printf "%.4f MiB, %.1f B/row\n", n/1048576.0, n/t}'
    prints 912.1250 MiB, 116752.0 B/row. 8192 x 114,700 B is 896.05 MiB.
  - The gate's own cost model uses the corrected number
    (tools/eg2_embed_batch_gate27.sh:93-100) and its live output prints both
    "the row cost the layout charges: 116,748 B/row" and "912.13 MiB at the
    8192-row budget", and the 8192-row call's actual workspace line
    "call: 4 items, 8192 rows, one workspace of 912.13 MiB".
  - The handoff records the correction (embeddinggemma2-handoff.md section 2,
    the row-cost bullet); the C side was missed.

  IMPACT: nothing computes with the number (the layout computes its own), but
  a reader sizing for the cap - the one reason the comment exists - is told
  896.2 MiB where the code and the measurement say 912.13 MiB.

  WHAT IT SHOULD BE: 116,748 B a row and 912.13 MiB in ds4_eg2vq.h:192-195,
  and the ds4_eg2vq_gpu.cuh comment's plane list/arithmetic reworked to name
  the in_soft plane (its own base of 108,548 B is the stale figure the
  unit-27 evidence explicitly repudiates).

U28-3 (low; the Rust batch entry has no rise check, so a debug build panics
where both C layers refuse)

  WHERE: crates/ds4-core/src/lib.rs:2509-2512 -
      for i in 0..n_items {
          let len = (item_off[i + 1] - item_off[i]) as usize;
  - in a table that is not monotonic (e.g. {0, 5, 3, 8}) the subtraction
  underflows. The docstring at lib.rs:2469-2474 promises the opposite
  behaviour: "The refusals mirror the C entry's ... a span table that
  does not rise from 0 to n_ids (an empty item, a hole, a disagreement)".
  The C layers both refuse: ds4_eg2vq_gpu.cuh:475 checks
  "if (item_off[i] >= item_off[i + 1u]) { return 0; }" explicitly, and
  ds4.c:73870 relies on C's defined unsigned wrap (a huge len is caught by the
  item ceiling).

  EVIDENCE:
  - The code and its promise: lib.rs:2509-2512 and :2469-2474 above; the C
    twin's explicit rise check at ds4_eg2vq_gpu.cuh:473-476.
  - Rust's own semantics, demonstrated with rustc in this pass (scratch
    binaries under /tmp/qa28-underflow*):
      printf 'fn main() { let a = std::hint::black_box(3u32); let b =
      std::hint::black_box(5u32); let c = a - b; println!("{c}"); }' |
      rustc - -o /tmp/qa28-underflow
      /tmp/qa28-underflow
    -> panicked at <anon>:1:93: attempt to subtract with overflow, exit 101.
    The same source with -O prints 4294967294 and exits 0. What this
    demonstrates is the language behaviour the entry depends on; the
    function-level call was not run because it needs an open Model and a
    scratch crate, which is outside this pass's write scope.
  - Consequence, read from the code: in a debug build (cargo test's profile)
    Model::embed_gemma_vq_batch returns a panic for a descending table instead
    of the documented Err; in the shipped release build the wrap turns the
    same input into the "input[i] is 4294967294 tokens, past the 2048 ..."
    refusal. The wire builds monotonic spans, so no served request can reach
    it, and no test exercises it today.

  WHAT IT SHOULD BE: an explicit rise check in the Rust loop (reject
  item_off[i + 1] <= item_off[i] by name, or subtract with checked_sub),
  matching the device core, so the documented refusal holds in every build
  profile.

4. CONFIRMED PRE-EXISTING, ALREADY RECORDED (not a new finding)

- cache_identity::tests::bounded_sidecar_and_ple (crates/ds4-server/src/
  cache_identity.rs:500-501) fails intermittently: after fs::write changes
  part.bin's bytes, assert_ne!(old, changed) sometimes does not hold, so the
  test's premise - that the rewrite changes the snapshot - fails on this
  filesystem. Live this pass: the full suite
  (cargo test -q -p ds4-server --lib) ran 357 passed / 1 failed
  (cache_identity::tests::bounded_sidecar_and_ple - FAILED), and three focused
  re-runs were 1 fail / 2 pass. The handoff section 7 already records this as
  an open flake ("a finding for the owner"), cache_identity.rs is not in units
  25-27's diffs, and the fast path does not run it. Not counted as a new
  finding; re-confirmed live so the record is current.
- The 3 warnings the ds4-server lib-test build prints include a dead-code
  warning on ImagePlan.window (crates/ds4-core/src/step37/media.rs:32) - a
  Step 3.7 family file untouched by these units.

5. WHAT WAS NOT TESTED, AND RESIDUAL RISK

- Leaks, sanitizers and races: none run (no valgrind/ASan in this pass); the
  gates' own runs exit cleanly, which is not a leak proof.
- Concurrency: the engine is single-owner and the media path is serial by
  design; no concurrent-request stress was run, and none is claimed by the
  evidence docs.
- The exact 8193-row refusal at the oracle is reached through the workspace
  create (8193 rounds to 8448 rows, past the 8192 clamp, printed by name)
  rather than by the core's own row check; the engine refuses 8193 rows by
  name before any allocation either way, so the boundary holds - noted for
  precision, not as a defect.
- The pin (/data/llama.cpp) was not built or run; its citations were read as
  recorded by the units' own docs.
- Out of scope by the owner's decisions: the audio tower, a multi-image item,
  the C CLI's tokenizer gap, and the two pushes.

6. WHAT CLOSING THIS PASS REQUIRES

- Fix U28-1 and U28-2 (both are one-line-class corrections to
  tools/eg2_embed_route_gate.sh and the two C comments; U28-3 is a three-line
  Rust addition), re-run the affected checks, and only then write
  qa-evidence/qa-report.md fresh, naming every surface in section 1 and reading
  "verdict: overall PASS" as its last non-empty line - at which point
  qa-evidence/qa-deferral.txt can be deleted and
  QA_DEFERRAL=/dev/null bash tests/qa-gate.sh (and tests/run.sh step 11) can go
  green.
- This file is the evidence for that work list; nothing in it weakens a
  gate's declared bound, and nothing in it re-opens feature work.

Method note: shell, awk, grep, sed, make, cargo, nvcc/CUDA tests; the only
non-runtime scratch artifacts are the two rustc demo binaries named in U28-3
and the logs under /tmp/qa28-*. No Python, ONNX or PyTorch was used anywhere in
this verification (P-1/P-2 respected), and no pin binary was executed.
