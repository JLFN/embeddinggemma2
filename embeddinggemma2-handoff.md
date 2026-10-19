---
project: embeddinggemma2
plan_start_commit: d6da36680643c48c144536caf658e360b6d30d82
last_updated_commit: 2f52c1280adfdd361aeccee11dacdea647d46e99
branch: dev
remote: 
handoff_written_at_context_usage: 18%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-19T22:35:34+02:00
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

2. Current state (2026-10-19, re-verified at the 2f52c12 boundary: unit 20 landed and merged, dev is clean, and nothing is in flight)

- Workspace: /data/embeddinggemma2, branch dev, HEAD 2f52c12 (fast-forwarded from feature/p23-serving-cap-batching, which was then deleted per rule 23), tree CLEAN. The outer repo has no origin, only publish = https://github.com/JLFN/embeddinggemma2, whose dev is at d45394f — so FIVE commits are unpushed: 68a3a28 (hook frontmatter refresh), e407734 (the rule 19 QA mount + tests/run.sh + the route gate's cap sections), c08163f (unit evidence + the declared quality floor), 92c4014 (the gate scans the nested repo; definitions not mentions), 2f52c12 (the QA report + four prose corrections). The push is an OFFER, recorded in section 7.
- Nested ds4-rs: CLEAN at 70278c19 on feat/ds41-dsml-carriage, TWENTY commits ahead of fork/feat/ds41-dsml-carriage; its publish = https://github.com/JLFN/embeddinggemma2-ds4-rs stands at 29873445 (refs/heads/dev), so TWO ds4-rs commits are unpushed: 68837a53 (the caps) and 70278c19 (the QA corrections). Push with `git push publish feat/ds41-dsml-carriage:refs/heads/dev`; plain `git push publish dev` fails (the local dev tracks fork/dev and is behind).
- Unit 20 (phase 6, row P2.3, slice 1) is DONE and merged: the family now has an EMBEDDING-SHAPED cap entry instead of a decode-shaped one. `ds4_core::embed_caps(ModelFamily::GemmaVq) = Some(EmbedCaps { item_tokens_max: 2048, batch_rows_max: 8192 })`, `None` for every decode family; reported as `controls.embed` in the plan JSON and as `embed` by GET /health; refused by name as 400 in `run_embed` and before the FFI call in `Model::embed_gemma_vq`; the item ceiling is held equal across Rust, ds4_eg2vq.h and cuda/eg2vq_attn.cuh by the new crates/ds4-core/tests/eg2vq_serving_caps.rs; `family_native_limit(GemmaVq)` is that ceiling and DS4_GEMMAVQ_PREFILL_CHUNK cannot lift it. Live: tools/eg2_embed_route_gate.sh sections 4b/4c, log out/eg2vq-route-gate-unit20b.log, exit 0, EG2VQ_EMBED_ROUTE_PASS. Evidence: docs/eg2vq-serving-caps-gate20.txt.
- The owner's declared quality floor for this tier is RECORDED and MET (docs/retrieval-cost-gate5.txt section 9; RECIPE.md 5b): recall@10 >= 0.983333, MRR@10 >= 98% of baseline (0.932742), recall@1 >= 0.855, mean cosine >= 0.95 — margins 0.006667 / 0.003077 / 0.006611 / 0.006040, all PASS. In percentages at 768: vectors keep 95.60% agreement, MRR@10 98.32%, recall@1 97.00%, recall@5 99.59%, recall@10 100.68% (better), and 7.67% of queries change top-1. The floor covers the 768 width only; 512/256/128 have none.
- Rule 19 is now MECHANICAL: tests/qa-gate.sh (a copy of the canonical template with three recorded deviations: QA_BASE = publish/dev because there is no origin; the three surface patterns retargeted at C/Rust/shell; it scans BOTH repositories, because ds4-rs is gitignored in the outer tree and the first version therefore passed a unit with no surfaces visible) and tests/run.sh (the fast path: digests, container, scorer self-test, loader gate 3, QA gate; `--deep` adds gate 1b, cargo test -p ds4-core and gate 4). Verified: red on the committed unit, green after the QA report; `bash tests/qa-gate.sh` exit 0 as of this boundary.
- The unit-20 QA pass (deepseek-v4.1-CC-flash) returned `verdict: overall PASS`, qa-evidence/qa-report.md, and found five things: four wrong/misleading sentences (a 1400-word item is 1410 tokens, not ~1057; 212.0078 MiB not 212.0086; two "section 7" pointers that meant 5c/5b; and the row-budget message's "the 8 inputs add up to 8460") and one informational run-to-run difference in the device path's last printed digit. All four corrections are committed (ds4-rs 70278c19, outer 2f52c12) and recorded in docs/eg2vq-serving-caps-gate20.txt section 6.
- What is left of the plan, in order: (1) THIS row's slice 2 — batching several items into one tower call (section 3); (2) the C CLI's tokenizer gap (ds4-c refuses the artifact by name; the family's gemma4 BPE is Rust-side, so porting it duplicates the encoder) — owner's call; (3) phase 8, the vision and audio towers — owner's call; (4) the two pushes, when the owner answers.
- Nothing measured to date claims: an MTEB-comparable score; a serving-shaped latency or concurrency number for a batch (the tower still runs one item at a time); anything about the vision and audio towers; or any quality verdict past the four declared floors.
- Machine-facing facts re-verified this session: artifact sha256 5bcf2216.., baseline 0d84f1a9.., container 8902fb8a.., ids out/phase5-ids.tsv, the -O3 oracle a9f74d0b.., ds4-rs 70278c19, `sha256sum -c SHA256SUMS` all five OK, loader gate 3 1376 tensors + 8 refusals, scorer self-test 8 cases, `CUDA_ARCH := sm_89` with the sm_89 cubin present, and rule 10 OSV 140 packages / 0 vulnerable in BOTH repos (run at this boundary).

3. Next unit plan — unit 21: batching several items into one tower call (row P2.3, slice 2)

Goal, in the row's own words (docs/ds4-gemma-port-phases.md:192): "batching of
several items into one tower call". Today an array input embeds its items one at a
time over the single bound plan (`run_embed` loops; `ds4_engine_embed_gemma_vq` and
`ds4_gpu_eg2vq_embed` each take ONE item), so an N-item request pays N host round
trips, N kernel-wave launches and N syncs. After this unit one call carries all N
items, the caps already advertised (item_tokens_max 2048, batch_rows_max 8192) bound
it, and the equality claim is that a batched item's vector EQUALS its single-item
vector.

THE DESIGN, and where it is read from (rule 0.2 — read these before writing):

- The reference semantics are llama.cpp's own batched path at the pin
  (/data/llama.cpp @ 31288c9c): src/llama-batch.cpp's `llama_ubatch` packs the
  tokens of several sequences contiguously and carries per-token `seq_id[]` and
  `pos[]`, and src/llama-graph.cpp builds the attention mask so tokens of different
  sequences cannot attend to each other (the same mechanism the anchor's
  `llama-embedding` run used when it embedded a batch). Read both files first and
  mirror the packing: the batch is ONE row block of `sum(item tokens)` rows, not a
  padded grid (padding wastes the budget this project just declared).
- The device tower's mask TODAY is the symmetric +-512 window over [0, n_tok)
  (cuda/eg2vq_attn.cuh:105-114, `half_swa`). For a packed batch the window and the
  item's own bounds combine into ONE contiguous key range, so no mask array is
  needed: for query row t with item start s and item length L, the keys are
  [s + max(0, pos - half), s + min(L - 1, pos + half)], where pos = t - s. That is
  the only attention change, and it makes the batch exactly equal to the per-item
  run (same keys, same order, same reduction), which is what the gate asserts.
- Three per-row facts must travel with the ids, and nothing else:
  `pos[t]` (position within its item; RoPE reads it instead of the row index) and
  `item_len[t]` (the row's item length; the attention bound), plus
  `item_off[n_items + 1]` (the per-item row ranges; the pool needs them). Store
  `pos` and `item_len` as int32 rows in the workspace (`ds4_eg2vq_gpu.cuh:72-99`'s
  layout, three new fields, ~8 B per row against the 108,548 B the row already
  costs), and pass `item_off` in the same arena.
- Pool: today one block per token writes the mean over ALL rows
  (eg2vq_ops.cuh's proj768_pool). It becomes one block per ITEM, averaging its own
  `item_off[i]..item_off[i+1]` rows and writing out768 + i * 768.
- RoPE: eg2vq_ops.cuh's rope kernel indexes the cos/sin tables by row. It takes a
  `const int32_t *pos` and indexes by pos[t] (the tables are already sized by
  max_tokens, and pos < item_len <= 2048 holds).

FILES, in the order the change has to land (each one read before it is written):

 1. cuda/eg2vq_attn.cuh — the key range from the item's bounds (above). Update its
    header comment: the "window over the whole item" text becomes item-relative.
 2. cuda/eg2vq_ops.cuh — rope takes pos; proj768_pool becomes per-item (grid
    n_items) and reads item_off.
 3. ds4_eg2vq.h — the API: a batch entry
    `int ds4_gpu_eg2vq_embed_batch(ctx, const int32_t *ids, const int32_t *pos,
     const int32_t *item_len, const uint32_t *item_off, uint32_t n_items,
     uint32_t n_tokens, float *out)`; `ds4_gpu_eg2vq_workspace`/`create` gain the
    row-budget side (they are already sized by max_tokens = rows).
 4. ds4_eg2vq_gpu.cuh — the layout fields, the entry, and the refusal set (ids
    range, n_items == 0, rows past max_tokens, an item longer than the plane).
 5. ds4_eg2vq_graph.inc — `ds4_eg2vq_ref_embed_batch(...)`; the CPU reference may
    simply loop `ds4_eg2vq_ref_embed` (it is the oracle, not the speed path) but it
    MUST exist so the engine has one entry on both backends, and its comment must
    say which of the two it is.
 6. ds4.c + ds4.h — `ds4_engine_embed_gemma_vq_batch`, keeping the single-item
    entry as the one-item call; the workspace is ensured once for the whole batch
    (rows = n_tokens, rounded up in 256 steps as today).
 7. crates/ds4-sys/src/lib.rs, crates/ds4-core/src/lib.rs — the bridge and
    `Model::embed_gemma_vq_batch(&[(pos, ids)])`-shaped API (choose the Rust shape
    after reading how the other multi-item bridged calls in this tree carry their
    arguments; do not invent one).
 8. crates/ds4-server/src/serve.rs `run_embed` — tokenize all inputs (it already
    loops), then ONE tower call, then the per-item MRL truncation and the same JSON
    the route emits today; the response must stay byte-comparable with the
    single-item path (the QA pass will diff them).
 9. The caps stay as they are: batch_rows_max is already the packed row count, so
    no number moves.

THE GATE (write it as tools/eg2_embed_batch_gate.sh, evidence docs/eg2vq-batch-gate21.txt):

 a. EQUALITY, the whole point: over the frozen SciFact set (tests/retrieval/), the
    same items through the batch entry and through the per-item entry must agree
    bit-for-bit or to a declared floor — measure the per-element delta and the
    cosine on both the device and the CPU reference, and declare the floor BEFORE
    the run. Same ids file, same weights, one binary.
 b. The batch is not a different answer: the route's array response for the same
    inputs, byte-compared with N single-input requests (the QA pass will repeat
    this; make it a scripted check, not a manual one).
 c. The caps bind: a batch whose rows would pass 8192 is refused by name (already
    true at the route), and a batch exactly AT 8192 rows is accepted.
 d. Speed, measured, with the claim's limits stated: items/s and tokens/s for
    one call of N items against N calls of one item, on the same set, with N and
    the set named in the evidence. Report the launch/sync count implied. Do NOT
    claim a serving latency or a concurrency number: the plan never had one.
 e. Regression: `cargo test -q -p ds4-core`, `cargo check --workspace --all-targets`,
    the phase-7 device gate (docs/eg2vq-device-gate7.txt's harness) re-run, and
    tests/run.sh (which includes the rule 19 gate: run the QA pass BEFORE the push
    offer, on the model the project recorded, deepseek-v4.1-CC-flash).
 f. Then: phase-doc row P2.3 to DONE (it is the row's last slice), this handoff's
    section 2/3 rotated again, the project memory entry refreshed, and the push
    offer recorded — NOT executed.

TWO THINGS TO GET RIGHT, or the unit will be wrong rather than slow:

- The batch must not change any item's arithmetic. A packed layout keeps every
  item's keys contiguous and its positions item-relative, so the per-row inputs are
  identical to the single-item run; anything that makes a row's result depend on
  the batch's composition (a shared row max in the softmax, a mask that lets one
  item's pads leak into another's window, positions counted from the batch start)
  is a bug the gate's equality check must catch. The first version of that check
  should therefore run at least one batch whose items have very different lengths.
- The device path is NOT run-to-run bit-identical at the last printed digit
  (measured at this boundary: the 256-dim L2 norm read 1.000000017 in one run and
  1.000000000 in another, with byte-differing JSON). Declare the equality floor
  accordingly — a per-element tolerance and a cosine floor, like gate 4's — and do
  not claim bit-identity on the device unless the measurement supports it.

4. Reference map — the exact repos and files (pinned; verified at this boundary)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (all the work) | /data/embeddinggemma2/ds4-rs @ 70278c19, branch feat/ds41-dsml-carriage, gitignored by the outer repo, own git | ds4.c, ds4.h, ds4_eg2vq.{h,inc,graph.inc,gpu.cuh}, ds4_glmvq*.{h,inc}, cuda/eg2vq_{attn,ops,vq}.cuh, cuda/*.cuh, crates/ds4-core/src/**, crates/ds4-server/src/**, crates/ds4-sys/src/lib.rs, tests/parity/, the Makefile |
| Form and metadata authority | /data/llama.cpp @ 31288c9c (b11723, MIT, read-only, source only) | src/models/gemma-embedding2.cpp, src/llama-{arch,hparams,model,graph,vocab,batch}.cpp, ggml/include/gguf.h, conversion/{gemma,base}.py, gguf-py/gguf/**, examples/embedding/embedding.cpp, tools/mtmd/** (phase 8) |
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
- OPEN (owner): whether the vision/audio towers (phase 8) are wanted; whether KV checkpoints mean anything for an embeddings family.
- OPEN (environment): the canonical builder cannot build this tree until it can pass `--features native`; fixing it changes a builder shared by every project on this machine, so it is PROPOSED, not done.
- OPEN (owner, NEW): the two pushes. Outer: `git push publish dev`, range d45394f..2f52c12 (five commits, docs/tests/tools only — no artifact: out/, .models/ and ds4-rs/ are gitignored), target publish = https://github.com/JLFN/embeddinggemma2 (PUBLIC with dev as default, verified when it was created; rule 6 wants a fresh `gh repo view` immediately before the push). Nested: `git push publish feat/ds41-dsml-carriage:refs/heads/dev`, range 29873445..70278c19. Gates already run for both: rule 10 OSV 140 packages / 0 vulnerable (both roots, this boundary), rule 19 QA fresh and PASS for unit 20, rule 5/12 as recorded. Rollback if a push must be undone: `git push --force-with-lease publish d45394f:refs/heads/dev` (outer) — a force-push, so the owner's call; both local histories are kept either way.
- STANDING (owner, 2026-10-19): the owner is AFK and wants automatic continuation after each handoff and compaction; pushes stay offers. Recorded in section 1.
- The unit-20 QA findings are all closed (four corrections committed, one recorded as informational).
- Nothing in flight. Both repositories are clean.

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

Trust git over this document's headings: continue with the first unit whose work is
not in the diff, which at this boundary is section 3 (unit 21), the last slice of
phase-doc row P2.3. Every phase through 7 is landed; the outer repo's HEAD is
2f52c12 and its published dev is d45394f; ds4-rs is at 70278c19 with its published
dev at 29873445. Anything else in either unpushed range must be read before sections
2-6 are trusted. The artifacts, the retrieval source, every out/ log and the CUDA
objects live under gitignored paths (out/, .models/, ds4-rs/), so they never appear
in a diff: they are identified by the digests in SHA256SUMS, by the pins inside the
gate scripts, and by the evidence docs' own digests.

This document is self-contained: a new session can execute the remaining unit from
this file alone. History is deliberately absent — git log, the phase doc and the
evidence docs are the historical record.
