QA report - unit 20: phase 6 row P2.3, slice 1, the embedding-shaped serving caps

Unit: 20. Outer repo /data/embeddinggemma2, branch feature/p23-serving-cap-batching,
HEAD 92c4014 (last pushed publish/dev = d45394f). Nested engine checkout
/data/embeddinggemma2/ds4-rs, branch feat/ds41-dsml-carriage, HEAD 68837a53 (last
pushed publish/dev = 29873445). Both trees were clean at the start and end of this
pass (git status --short empty in both); the only file this pass wrote in either
tree is this report (plus the route gate's own out/phase6-route-gate scratch dir,
which the gate deletes and recreates on every run by design; scratch work otherwise
under /tmp/qa20).

QA model: deepseek-v4.1-CC-flash (the model tests/qa-gate.sh records for rule 19).
Date: 2026-10-19. Host: RTX 4070 SUPER (sm_89), CUDA backend exercised live.
Method: shell, curl, jq, awk, sha256sum, cmp/diff, rustc, cargo, make. No Python,
no ONNX, no PyTorch anywhere in this verification (P-1/P-2 respected).

SURFACES COVERED

- embed_caps - ds4_core::embed_caps(ModelFamily) -> Option(EmbedCaps), re-exported at
  crates/ds4-core/src/lib.rs:121 and defined at crates/ds4-core/src/serving.rs:206.
  This is the exact surface tests/qa-gate.sh extracts from the nested diff. Verified
  live (its values are what /health reports and what both refusals enforce) and
  falsified cross-file (CLAIM 1).
- The /health field embed, built in serve.rs's GET /health branch (serve.rs:1227-1246).
- POST /v1/embeddings: both cap refusals and the accepted path in run_embed
  (crates/ds4-server/src/serve.rs:1596-1632).
- The CLI refusal, ds4 -m ARTIFACT --embed TEXT, through Model::embed_gemma_vq's
  Rust pre-check (crates/ds4-core/src/lib.rs:2397-2420).
- The new test crates/ds4-core/tests/eg2vq_serving_caps.rs (all three tests), and
  the family native-limit arm in crates/ds4-core/src/serving_host.rs:1105-1200.
- Documents checked: docs/eg2vq-serving-caps-gate20.txt (the unit's evidence),
  docs/retrieval-cost-gate5.txt sections 2, 5, 5b, 5c, 7, 9, RECIPE.md section 5b,
  docs/ds4-gemma-port-phases.md row P2.3 (line 192), tools/eg2_embed_route_gate.sh,
  tests/qa-gate.sh, tests/run.sh.

CLAIM 1 - the caps exist and are the family's own numbers: PASS
(one wrong number in a comment, FINDING F2)

- ds4_core::embed_caps(ModelFamily::GemmaVq) returns
  Some(EmbedCaps { item_tokens_max: 2048, batch_rows_max: 8192 }): read at
  crates/ds4-core/src/serving.rs:206-213; the constants are
  crates/ds4-core/src/gemma_vq.rs:94 (ITEM_TOKENS_MAX: u32 = 2048) and :108
  (BATCH_ROWS_MAX: u32 = 8192). The match arm is GemmaVq => Some(...), _ => None,
  so every decode family is None by construction; the test pins three decode
  families and the plan-JSON test pins the null for a decode family.
- The one number in three files: ds4-rs/ds4_eg2vq.h:140 (the macro
  DS4_EG2VQ_MAX_TOKENS = 2048u); ds4-rs/cuda/eg2vq_attn.cuh:49
  (EG2VQ_ATTN_KEYS_MAX = 2048); the compile-time equality assert at
  ds4-rs/ds4_eg2vq_gpu.cuh:45-46. The new test reads BOTH C sources at compile
  time (include_str! at crates/ds4-core/tests/eg2vq_serving_caps.rs:60-61) because
  the crate carries no C build (no build.rs in crates/ds4-core or crates/ds4-sys).
- I falsified the test live, on COPIES only, never in place: copied the two
  headers plus the test source verbatim into /tmp/qa20/divtree and recompiled the
  test against the tree's own built crate with
  rustc --edition 2021 --test .../eg2vq_serving_caps.rs --extern
  ds4_core=target/debug/deps/libds4_core-d910397024fc9869.rlib
  -L dependency=target/debug/deps (run from ds4-rs so the tree's rust-toolchain
  override, 1.98.0, applies). Results:
    - unmodified copies: 3 passed.
    - CUDA enum 2048 -> 4096: FAILED with "the item ceiling diverged: Rust
      ITEM_TOKENS_MAX=2048, C DS4_EG2VQ_MAX_TOKENS=2048, CUDA
      EG2VQ_ATTN_KEYS_MAX=4096".
    - C macro 2048u -> 1024u: FAILED with the same message, C=1024.
    - macro renamed to DS4_EG2VQ_MAX_TOKENS_V2: FAILED with "ds4_eg2vq.h:
      DS4_EG2VQ_MAX_TOKENS has no numeric value".
  So a divergence in either C file is a red test, and a rename fails loudly, as the
  unit's section 2 claims.
- Workspace arithmetic. The layout at ds4_eg2vq_gpu.cuh:72-99 is linear in rows;
  summing its coefficients from the constants at ds4_eg2vq.h:53-62 gives 27,136
  f32 planes per row (h/x/f/a 4 x 512, q_swa 1,024, q_full 2,048, kv 2 x 512,
  attn_swa 1,024, attn_full 2,048, fg 2,048, fu 2,048, fd 512, pg 512, pp 512,
  ple 12,288) plus the 4-byte id = 108,548 B per row. 2048 rows = 2048 x 108,548 B
  = 222,306,304 B; 8192 rows = 889,225,216 B = 848.03125 MiB, so the test's
  848.0 +/- 1.0 MiB assert and the comment's 848.0 MiB hold.
- Measured cross-check: the runtime itself prints the workspace at the top of the
  range. My CLI run at exactly 2048 tokens (below) logged "ds4: gemma_vq device
  path calls=1 tokens=2048 payload=55.94 MiB weights=137.04 MiB workspace=212.01
  MiB", matching docs/eg2vq-device-gate7.txt:407-411 ("At max_tokens 2048 it is
  212.01 MiB") to the printed precision.
- FINDING F2 is the only fault here: the test comment and the nested commit
  message state 212.0086 MiB where the arithmetic is 212.0078 MiB (or 212.0107
  including the constant out plane). Behavior and gates are unaffected.

CLAIM 2 - enforced live, by name, at the boundaries: PASS, with the measured
boundary counts below

Live setup: ds4-rs/ds4-server -m out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-
gemma_vq.gguf --host 127.0.0.1 --port 18200 (the route gate's server had already
exited; no other model process ran at any point; the server and CLI take a
machine-global model lock and were never run concurrently). Inputs are repeated
words ("word " k times); this construction tokenizes to exactly k + 10 tokens,
measured through usage.prompt_tokens on accepted requests and through the refusal
text on refused ones (k=100 -> 110, k=1400 -> 1410, k=2037 -> 2047, k=2038 ->
2048, k=3000 -> 3010).

- /health (curl, live): HTTP 200 with the exact object
  "embed":{"item_tokens_max":2048,"batch_rows_max":8192} (plus "family":"gemma_vq",
  "embedding":true, "backend":"cuda").
- Item ceiling, exact pair: k=2038 words (2048 tokens) -> HTTP 200 with
  usage.prompt_tokens 2048; k=2039 words (2049 tokens) -> HTTP 400
  "input[0] is 2049 tokens, past this model's 2048-token item ceiling (GET /health
  reports embed.item_tokens_max)" in the OpenAI invalid_request_error envelope.
  Accepted at 2048, refused at 2049.
- Row budget, exact pair: five 1400-word items (5 x 1410 = 7050 tokens) -> HTTP 200
  with 5 vectors; five 1400-word items plus one 1132-word item (7050 + 1142 =
  8192 tokens) -> HTTP 200 with 6 vectors; the same request with a 1133-word last
  item (8193 tokens) -> HTTP 400 "the 6 inputs add up to 8193 tokens, past the
  8192-token budget of one embedding call (GET /health reports
  embed.batch_rows_max); split the request". Accepted at 8192, refused at 8193.
- Which cap fires first, measured on three orderings: (a) a 3010-token item first
  in an eight-item request that would also exceed the budget -> the ITEM-ceiling
  message for input[0]; (b) five 1400-word items then a 3010-token item, budget
  still fine when the long item is reached -> the ITEM-ceiling message for
  input[5]; (c) eight 1400-word items then a 3010-token item, the budget already
  exceeded at the sixth item -> the ROW-budget message. So the boundary reached
  first in input order fires, and within one item the per-item check precedes the
  cumulative check (serve.rs:1605-1630 reads in exactly that order). The unit's
  documents describe each cap firing on its own ground and batch_rows_max as a
  running total of the request's item tokens (commit 68837a53, note 1); they state
  no precedence order, so nothing here contradicts them. See also FINDING F4 for
  the message's wording when the trip is early.
- Inside both caps: three 1400-word items at 128 dims -> HTTP 200 with 3 vectors
  (the route gate's own case, reproduced below); my 8192-token accepted request
  returned 6 vectors and the 7050-token one 5 - one vector per input throughout.
- CLI: ds4 -m ARTIFACT --embed with a 3010-token item exits 1 with "ds4-rs: ds4
  bridge error 1: the item is 3010 tokens, past the 2048 the embedding tower's
  attention kernel holds (crate::gemma_vq::ITEM_TOKENS_MAX)" - the Rust pre-check,
  and it fired in 0.56 s, before any tower work. At exactly 2048 tokens (k=2038,
  --dims 128) the CLI exits 0 and prints the vector, with its own log confirming
  the device run at tokens=2048.

CLAIM 3 - the build and the suites: PASS

All run in /data/embeddinggemma2/ds4-rs (warm tree), exactly as reported:
- cargo test -p ds4-core --lib serving -> "test result: ok. 212 passed; 0 failed;
  0 ignored; 0 measured; 237 filtered out".
- cargo test -p ds4-core --test eg2vq_serving_caps -> "test result: ok. 3 passed;
  0 failed" (item_ceiling_is_one_number_in_rust_c_and_cuda,
  batch_row_budget_costs_what_the_layout_charges,
  only_the_embedding_family_states_embedding_caps).
- cargo test -p ds4-core --lib -- the_embedding_family_reports_item_and_batch_caps
  gemma_vq_native_arm -> "2 passed" (the serving plan-JSON test and
  serving_host::tests::gemma_vq_native_arm_is_the_tower_ceiling_not_a_prefill_chunk).
- cargo check -p ds4-server --all-targets -> exit 0, "Finished"; two warnings
  (ds4-core dead-code on ImagePlan.window; an unused import in the ds4-server test
  "generate"), no errors.
- make ds4 ds4-server -> "'ds4' is up to date." and "'ds4-server' is up to date.",
  exit 0; re-hashing gave the same binaries (ds4-server
  3fbac283c15e0fcd9ca86c885bb807fc1d6f0d7f596e75b0d1d6237e09341b79, ds4
  1132dbe753e32f12414dfdf6150d58f95d2c5ab20d27a08499ab592530a7c941), so the
  artifacts the live runs used are the tree's own.

CLAIM 4 - the route gate reproduction: PASS (digests verified; line-for-line
rerun, one digit of device noise)

- Digests re-read in this pass: artifact
  out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf sha256
  5bcf22161dda46bccbd5123b26d519efccad981cbfe11cce7227415b3373b33f (the evidence
  doc quotes the 5bcf22161dda46bccbd5123b... prefix); ds4-rs/ds4-server sha256
  3fbac283c15e0fcd9ca86c885bb807fc1d6f0d7f596e75b0d1d6237e09341b79 (doc prefix
  3fbac283c15e0fcd9ca86c88...); out/eg2vq-route-gate-unit20.log sha256
  5bc8874c90500cbf0c5e282952b351b0493f9a9607d1a6a4e22174addd0e1be1, the full digest
  the doc quotes. All match.
- Re-ran tools/eg2_embed_route_gate.sh out/...gguf, output to
  /tmp/qa20/route-gate-rerun.log: exit 0, EG2VQ_EMBED_ROUTE_PASS; the gate picked
  port 18108 (free) and the whole run took 5.7 s. All of sections 1, 2, 3, 4, 4b,
  4c and 5 printed, including the exact /health object, the 3010-token refusal,
  the eight-item row-budget refusal reading "the 8 inputs add up to 8460 tokens...",
  the three-vector accepted case, and dim=768 cos=1.000000000000.
- diff of the recorded log against the rerun: identical except two lines. (1) the
  recorded "L2 norm: 1.000000041" versus the rerun "L2 norm: 1.000000000" - the
  norm the gate computes from the printed %.7f values of the 256-dim vector.
  (2) the recorded log's final "route gate exit=0" is not printed by the script;
  the unit appended the exit status when capturing. Every other line, including
  both refusal bodies, is byte-identical.
- The norm difference is device run-to-run arithmetic noise, measured directly:
  three consecutive identical POSTs of the gate's step-3 request against my own
  server returned byte-different JSON (first difference at offset 818) and norms
  1.000000011 / 1.000000022 / 1.000000018. All are within the gate's declared
  bound (abs(norm - 1) <= 1e-5), and no unit document claims bit-determinism
  for the
  device path, so this is an observation (FINDING F5), not a contradiction. The
  recorded 1.000000041 is the same phenomenon.

CLAIM 5 - the declared quality floor is reported honestly: PASS
(all four verdicts and every percentage recomputed; two wrong cross-references,
FINDING F3)

Re-derived with awk from the document's own tables - the mean cosine from section
5b (0.956039571) and the recall/MRR table at section 5c lines 281-288 - NOT from
section 9's arithmetic:

- recall@10 must not fall below the baseline's own 0.983333: artifact 0.990000,
  margin 0.006667 -> PASS (change +0.678%, the doc's +0.68%).
- MRR@10 must stay within 2% of baseline: floor 0.98 x 0.951778 = 0.93274244;
  artifact 0.935819, margin 0.00307656 (printed 0.003077) -> PASS (change
  -1.676757%, the doc's -1.68%).
- recall@1 must stay at or above 0.855: artifact 0.861611, margin 0.006611 ->
  PASS (change -3.002101%, the doc's -3.00%).
- mean vector cosine must stay at or above 0.95: artifact 0.956039571, margin
  0.006039571 (printed 0.006040) -> PASS (shortfall from 1.0 -4.396043%, the
  doc's -4.40%).

So all four verdicts and all four margins printed in section 9 (and RECIPE.md
section 5b's "margins of 0.006667, 0.003077, 0.006611 and 0.006040") reproduce
exactly.

- RECIPE.md section 5b's percentages, each recomputed from the same tables:
  95.60% agreement (0.956039571); MRR kept 98.32% (98.323243); recall@1 97.00%
  (96.997899); recall@5 99.59% (99.591559); recall@10 100.68% (100.678000);
  top-1 changed 7.67% (23/300 = 7.666667); MRR retained at 512/256/128 =
  98.64% / 97.81% / 96.76% (98.640117 / 97.808306 / 96.763788); the 128-dim
  losses cited there, 4.06% of recall@5 and 2.63% of recall@10 (4.061940 /
  2.628020). All correct.
- RECIPE.md section 5b's table is digit-for-digit equal to the document's 5c
  table (diff over the eight rows, clean).
- FINDING F3: section 9's two pointers to "section 7" name the wrong section.

CLAIM 6 - the rule 19 mount behaves: PASS

- Red run captured BEFORE this report was written, in place (the gate only reads):
  bash tests/qa-gate.sh printed the surface list { embed_caps }; PASS report
  exists; FAIL fresh (report mtime 1792439313 < the newer base epoch 1792439494 =
  outer publish/dev d45394f; the nested base 29873445 is older); PASS operative
  verdict (the unit-15 report's last line); FAIL covers embed_caps; exit 1. The
  same red appears as tests/run.sh step 5: the fast path (steps 1-4: the recorded
  digests, the DQVD structure check, the scorer self-test, loader gate 3 with 1376
  tensors) passed in 3.9 s and step 5 failed red, exit 1.
- Last-line contract, in a sandbox clone at /tmp/qa20/gate/outer (QA_NESTED_REPO
  pointed at the real nested checkout, read-only): a report covering embed_caps
  whose last non-empty line is "trailing note after the verdict" -> exactly one
  FAIL, "operative verdict is PASS (report's last non-empty line: trailing note
  after the verdict)", exit 1; the same report ending with "verdict: overall PASS"
  -> all checks PASS, exit 0.
- Both repositories are scanned, mechanism and current reach: with
  QA_NESTED_BASE=4892fb4b (unit 18's commit) the list grows to 10 surfaces
  (ds4_bridge_eg2vq_embed, ds4_bridge_eg2vq_has_embed, embed_caps,
  embed_gemma_vq, embed_item_ids, embeds_text, mrl_truncate, st_prompt,
  tokenizer_special_flags, /v1/embeddings); with QA_NESTED_BASE=68837a53 (the
  nested HEAD) the list is empty and the gate prints ALL PASS "no new ...
  surfaces relative to publish/dev and ds4-rs/68837a53" - so embed_caps comes
  from the nested scan. With QA_BASE=ed2fa0b~1 (an older outer base whose window
  spans many tools/ commits) the list did not change: no added line in the outer
  repo's tools/ or tests/ matches the C-definition pattern, because the outer C
  tools are not named ds4_... and define functions with inline bodies (checked:
  tools/eg2_gguf.c's definitions are die, xmalloc, jw, jstr_end, ...; and
  git log -S 'int ds4_' -- tools/ finds nothing). A synthetic added line
  "int ds4_qa_probe(" in the sandbox's tools/ does appear in the list, so the
  outer extraction path works; for this unit's window the effective surface was
  the nested one. None of this weakens the gate's purpose (the nested scan is
  what catches this project's engine work), it is recorded so a later session
  reads the gate's coverage honestly.
- Post-write green, in place, on this report: bash tests/qa-gate.sh prints four
  PASS lines (exists; fresh; operative verdict; covers embed_caps) and "QA GATE:
  overall PASS", exit 0; and bash tests/run.sh (fast path) completes all five
  steps in one run and exits 0 with "fast path complete". This is the state the
  unit's commit-for-delivery and push will see.

FINDINGS (all low severity or informational; none changes a measured verdict)

F1 (low; wrong number in the unit's evidence doc and both commit messages).
docs/eg2vq-serving-caps-gate20.txt:96-97 says "1400 words tokenize to about 1057
tokens (8460/8), so the eight-item request was refused by the ROW budget and not
by the per-item ceiling". The conclusion is right, the number is wrong: a
1400-word item is 1410 tokens, measured three ways (single-item request
usage.prompt_tokens 1410; five items usage 7050 = 5 x 1410; six items refused at
total 8460 = 6 x 1410). The eight-item request holds 8 x 1410 = 11280 tokens and
trips the budget at its sixth item, which is why the server's message says 8460;
1057.5 is no item's token count. The same misreading is in the nested commit
message 68837a53 ("eight 1400-word items (8460 tokens)"), and the same
1400-words-equal-1400-tokens assumption is in tools/eg2_embed_route_gate.sh:200
("three such items is 4200 rows"; measured 3 x 1410 = 4230 tokens). The refusals
themselves are correct and were re-measured at the boundary.

F2 (low; one wrong constant, comment and commit message only).
crates/ds4-core/tests/eg2vq_serving_caps.rs:78 says
"// (2048 x 108,548 B = 212.0086 MiB)."; 2048 x 108,548 = 222,306,304 B =
212.0078125 MiB exactly, so the correct figures are 212.0078 MiB to four
decimals (or 212.0107 MiB if the constant 768-float out plane at
ds4_eg2vq_gpu.cuh:75-77 is included; the runtime prints 212.01 MiB either way).
The nested commit message 68837a53 repeats it ("the per-row charge 108,548 =
212.0086 MiB at 2048 rows"). No asserted value depends on the typo: the test
asserts 848.0 +/- 1.0 for 8192 rows, which is 848.03125 MiB and correct.

F3 (low; wrong cross-references in the new section 9).
docs/retrieval-cost-gate5.txt:534: "The percentages below are the arithmetic of
the table at section 7 applied to the same two legs"; :588: "the margin table in
section 7 (mean top1-top2 margin 0.085722 at 768)". Section 7 of that document is
"The -O2 aside, and where the timing came from" (lines 361-439) and contains no
recall/MRR table and no margin table; both sentences mean the tables at section 5c
(retrieval table lines 279-288, top-1-change/margin table lines 291-296), with the
mean cosine from section 5b. Every number those sentences lead to was re-derived
and is correct, so only the pointer is wrong.

F4 (low; the row-budget refusal can misstate the request's own total).
run_embed's budget refusal prints the request's declared input count together
with the RUNNING total of the items processed so far (serve.rs:1620-1630). When
the trip happens before the last input, the sentence is false about its own
subject: measured with input of eight 1400-word items and one 3000-word item the
400 body reads "the 9 inputs add up to 8460 tokens", while those nine inputs
tokenize to 8 x 1410 + 3010 = 14290; the unit's own eight-item case shows the
same effect ("the 8 inputs add up to 8460 tokens" versus the true 11280). The
commit message's note 1 documents the running-total semantics of the cap itself,
so the enforcement is as intended; the message text is what mislabels. The
exact-boundary case is accurate in the same message ("the 6 inputs add up to 8193
tokens"), and so is the six-item case (8460 = 6 x 1410).

F5 (informational; the device path is not run-to-run bit-identical).
Three identical 256-dim POSTs on the same running server returned byte-different
JSON (first difference at offset 818) and norms 1.000000011 / 1.000000022 /
1.000000018; this explains the only differing line in the route-gate reproduction
(recorded 1.000000041 versus rerun 1.000000000). The gate's declared tolerance
(within 1e-5 of 1.0) covers it, and no unit document claims device-path
determinism; recorded so a future comparison of two runs does not expect
byte-identical vectors, and treats the ninth-decimal norm as noise.

WHAT I DID NOT TEST

- The CPU backend over HTTP: the evidence doc's section 5 says the same Rust
  refusal is reached before the FFI call on the CPU backend but was not driven
  over HTTP; I reproduced the unit's live run on the CUDA backend only, as it did.
- A full 8192-row device workspace: the batch budget's memory figure is arithmetic
  (verified by summing the layout coefficients and by the live 2048-token
  workspace print), not a run at 8192 rows; no code path sizes a workspace past
  one item today, and the unit says so.
- /health's embed for a decode family (null): verified through the unit tests and
  the match arm, not by serving a decode model (no decode artifact was loaded).
- The batching half of row P2.3: explicitly not in this unit (its section 5 and the
  phases row say so), hence untested.
- tools/eg2_embed_cli_gate.sh and the older phase-4/5 gates: not re-run; the
  route gate's route-vs-CLI comparison (cos 1.000000000000) and the suites above
  cover this unit, and the older gates' evidence is unchanged by it.
- The device noise (F5) was characterized with three samples, not a distribution.

verdict: overall PASS
