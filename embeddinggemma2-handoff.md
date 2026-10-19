---
project: embeddinggemma2
plan_start_commit: d6da36680643c48c144536caf658e360b6d30d82
last_updated_commit: cada4bdc3d372b0dcbba855ffe098903f81e6e2b
branch: dev
remote: 
handoff_written_at_context_usage: 17%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-19T07:56:53+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ in ds4-rs (rewritten 2026-10-19: the plan exact, the dropped routes named)

Note on the frontmatter: it MUST be the first thing in the file, with
bare values (no inline comments after the value) — the handoff-hooks
parser reads it verbatim, and a commented or misplaced frontmatter
silently breaks the SessionStart probe readout. The scaffold writes
this exact shape; hand-written files must match it. <COMMIT> means the
full commit hash (plan_start_commit = the commit the current plan
began from); <PERCENT> = the context usage read from the session's
signals.json (target 25-35, hard cap 40).

0. Rules for writing THIS document (read before editing, not just before reading)

- Only write or replace this file at a clean, verified, committed boundary. Never checkpoint mid-work.
- Start drafting once context usage crosses ~25%. Finalize and commit before the auto-compact threshold — do not wait for it to begin writing. The writing is a summarization task and degrades like any other task under context pressure. Treat 25-35 as the drafting target and the threshold as the hard cap. The threshold is the harness's own cut-over: [session] auto_compact_threshold_percent in ~/.opengrok/config.toml (this machine: 60), which makes it compact this session in place — no new session is opened.
- Every claim in "Current state" must be re-verified by running the relevant command in this session, not copied or paraphrased from the previous handoff. If you cannot re-run it, mark it unverified rather than asserting it.
- Hard caps: "Current state" max 15 lines. "Guardrails and open items" max 15 lines. "Next unit plan" has no cap — it is the actual work spec. If a section would exceed its cap, cut detail; do not compress into denser prose (denser prose is where drift hides).
- Do NOT: replay prior units' logs (git log is the record), re-explain global rules already in the loaded guardrails, restate anything the staleness check in section 8 already covers, or record acceptance evidence you did not personally observe this session.
- This file is the plan of record. docs/ds4-gemma-port-phases.md holds the row-by-row work list behind it; where the two disagree, THIS file wins and the phase doc is corrected in the same commit.

1. Session protocol (read first)

This project is a sequence of units; each unit ends with a verified, locally
committed step of the plan, and a push is offered, never taken (no remote is
configured on the outer repo).

1. Read context usage from the session's signals.json (contextTokensUsed /
   contextWindowTokens) under ~/.opengrok/sessions/<percent-encoded-cwd>/
   <session-id>/, newest session dir. If signals.json is absent, fall back to
   chat_history.jsonl bytes / 4 and say in the session report that the number
   is a fallback, not the gauge.
2. If usage is below the threshold (~/.opengrok/config.toml
   [session].auto_compact_threshold_percent, this machine 60): continue
   immediately — write the next unit's plan into section 3, execute it,
   verify, commit, re-check usage, repeat.
3. At the threshold the harness compacts this session in place; nothing is
   launched. This handoff must be written and committed BEFORE the threshold,
   because a compaction carries a summary of the conversation, not unwritten
   plan prose.
4. Preferred cadence: once the finished unit's handoff, the project memory
   entry and the commits are in place, call the compact_session tool as the
   last action of that turn.

2. Current state (2026-10-19, verified in this session)

- Workspace: /data/embeddinggemma2, branch dev, HEAD cada4bd, tree clean but for this handoff. Nested ds4-rs untouched at eb67d493 on feat/ds41-dsml-carriage, 0 modified, 8,582 files, 2.5 GB.
- Artifact: out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin, 421,758,160 B; `bin/eg2_pack_cpu check` says kind=0 slots=1376, recipe `towers=vq4x4096 sharedcb=0 cbfp16 token=q8_0 frontend=vq other=keep nocal src=914f7f89142e`, structure OK. SHA256SUMS carries one line, 8902fb8a… (the container is the DQVD one being retired in phase 1 of section 3).
- Tools built: bin/eg2_vq, bin/eg2_pack, bin/eg2_pack_cpu, bin/eg2_gpuprobe.
- Checkpoint: .models/EmbeddingGemma-2-914f7f8, 1.5 GB, pinned revision 914f7f8.
- References verified today: /data/llama.cpp at 31288c9c (2026-10-18), src/models/gemma-embedding2.cpp is 232 lines; /data/YoungAi carries the native GGUF writer precedent (section 3, phase 1) and vq_fmt.h.
- Guardrails: all five P-1 checks return zero, and the ldd check is now non-vacuous — ds4-rs/ds4-server is a built 76 MB CUDA-linked ELF (libcudart.so.13, libcublas.so.13) with 0 python/torch entries.
- Behavior correction recorded: llama.cpp declares this model's sliding window as LLAMA_SWA_TYPE_SYMMETRIC (src/models/gemma-embedding2.cpp:4), masked at ±n_swa/2 (src/llama-hparams.h:524-533) — not ds4's trailing cut. Phase 4 and 6 must implement the symmetric mask.

3. Next unit plan — run EmbeddingGemma-2 in ds4-rs, phase by phase (2026-10-19)

Status: not started. Goal: the ds4 runtime loads our GGUF artifact (a GGUF carrying a
`gemma_vq` payload, per the owner's container decision) and emits the 768-dim
embeddings this model exists for, on the GPU, with the measurements that show the
port matches the pinned original. Runtime is Rust + C + CUDA; no Python anywhere,
no ONNX, no PyTorch (section 5 lists every route removed).

One rule for the whole plan: every mechanism below is READ from the cited file
before it is written, per project rule 0.2. Where the plan says "mirror", it means
open both files side by side and copy the shape, not paraphrase it.

Commit convention: each unit's final commit carries the trailer
"Unit: <N> complete" as its last body line, so unit boundaries are recoverable
with `git log --grep`.

PHASE 0 — the `gemma_vq` payload contract (phase-doc row X.8). No code is written in this phase.

- Read (all paths relative to /data/embeddinggemma2/ds4-rs unless another root is given):
  - crates/ds4-core/src/glm_vq.rs:276 (`glm_vq.codebooks`), :353 and :366 (`glm_vq.logical_shapes`, the mechanism that carries rank), :435 (`glm_vq.format_version`), :466 (`glm_vq.native_policy`).
  - crates/ds4-core/src/glm_vq/layout.rs and crates/ds4-core/src/glm_vq/release.rs (how a VQ release is described and validated).
  - crates/ds4-core/src/ling31_vq.rs:23-29 (ARCH / VARIANT / SOURCE_MODEL / RECIPE / RECIPE_V2) and crates/ds4-core/src/ling31_vq/release.rs:139.
  - crates/ds4-core/src/identify.rs:183-186 (the metadata gate that makes a family self-identifying in GGUF).
  - ds4_glmvq_bind.inc:1-3 (the stated host/device split) and ds4_glmvq.h (the C-side types).
  - The payload carrier itself: /data/YoungAi/vq_fmt.h (DQVL blob v2.2 and the DQVQ matrix payload), our vendored copy ds41_vq_fmt.h:1-30 and its dim-generic decoder :72-104, and /data/YoungAi/gguf-tools/quantize/v41_to_gguf.c (493 lines: the only shipping example of a ds4 VQ family being written into a GGUF, and the mapping of VQ bytes to GGUF tensor types).
  - The writer helper that existing tool uses: /data/YoungAi/src/common/ds4_gguf_write.h (127 lines, header-only).
- Land: docs/ds4-gemma-payload-contract.md — the metadata keys, their GGUF types, the tensor-name convention, the payload byte layout, the recipe string, where rank lives, and the exact host/device split.
- Gate: every key, type, offset and magic in that document cites a file:line from a shipping family or the writer precedent. Two determinations must be settled here, not assumed:
  1. Whether our kind-0 payload bytes are already exactly a DQVQ payload (`ds41_vq_fmt.h:7-12` layout: magic, u16 dim, u16 nc, u32 rows, u32 cols, codebook, row gains, index stream) — settle it by hashing the first payload of a slot in the DQVD container against a freshly packed one.
  2. If they are identical, phase 1's artifact step is a re-wrap (fast, byte-preserving) instead of a re-encode (~14.5 min shard wall time, out/encode-run.txt) — and the plan states which of the two was measured and why.

PHASE 1 — the writer, and the baseline GGUF (phase-doc rows X.5, X.6, plus the encoder change)

- Read: the two GGUF sources side by side — /data/llama.cpp/ggml/include/gguf.h with ggml/src/gguf.cpp (the format authority: header, KV value types, tensor-info layout, alignment), and ds4-rs's own reader ds4.c:1607 (DS4_GGUF_MAGIC) with :2348 (the parse entry) plus crates/ds4-core/src/gguf.rs (the Rust reader: open at :208, getters at :295-466, split handling at :509). Reuse the writer helper /data/YoungAi/src/common/ds4_gguf_write.h (ds4gw_begin / ds4gw_kv_* / ds4gw_tensor / ds4gw_header / ds4gw_write / ds4gw_end) exactly as v41_to_gguf.c uses it; write our own only if the helper cannot express a key this model needs, and say so in the commit.
- Metadata and tensor set to write: /data/llama.cpp/src/models/gemma-embedding2.cpp:3-13 (the hparams keys the loader resolves) and :34-71 (the 413 text-tower tensors); arch string at src/llama-arch.cpp:64; tokenizer keys as llama.cpp reads them (src/llama-arch.cpp LLM_KV_TOKENIZER_* block, and the vocab loader src/llama-vocab.cpp), sourced from .models/EmbeddingGemma-2-914f7f8/tokenizer.json and tokenizer_config.json.
- Our own inputs to the writer: docs/tensor-inventory.tsv (1377 lines, the per-tensor truth) and docs/encode-plan.tsv (1377 lines); the DQVD writer being replaced is tools/eg2_pack.c:51-88, :221-225, :666-724, and its checker :980-1077 defines what the new writer must still guarantee.
- Land: the vendored writer header in our tree, tools/eg2_gguf.c + bin/eg2_gguf, the changed emit path in tools/eg2_pack.c, and RECIPE.md section 3 rewritten for the GGUF container. The GGUF must carry the DQVD-only information the old container lost: rank and shape per tensor (the container's slot table records rows/cols only), via the declared layout, mirroring `glm_vq.logical_shapes`.
- Gate 1a (baseline): metadata plus one tensor first, round-tripped through OUR OWN reader (ds4.c's parser path, or crates/ds4-core/src/gguf.rs) back to the same name, dtype, dims and byte-identical payload; then the whole checkpoint, 413 text-tower tensors, key set equal to gemma-embedding2.cpp:3-13.
- Gate 1b (artifact): all 1376 slots present in one GGUF; a ds4-side dump of every tensor (name, dtype, dims, kind, geometry, payload length, payload hash) equals what `bin/eg2_pack_cpu check` plus the per-slot re-encode comparison already produce; SHA256SUMS regenerated for the new file.
- Cost, measured: ~14.5 minutes of shard wall time plus the merge for a full re-encode (out/encode-run.txt); a re-wrap, if phase 0 proves the payloads identical, is minutes. Say which one ran.

PHASE 2 — the external anchor, once the baseline GGUF exists (phase-doc row X.5; owner-approved 2026-10-19)

- Act: build /data/llama.cpp at 31288c9c in a scratch directory (never in that tree), run examples/embedding/embedding.cpp over the frozen sentence set with the baseline GGUF from phase 1a, and commit the resulting vectors as data (tests/expected/anchor-embeddinggemma2.jsonl) with the revision, the exact command line, the prompt strings used (from the checkpoint's config_sentence_transformers.json) and digests.
- Read before trusting any of it: examples/embedding/embedding.cpp:70 (`common_embd_normalize`) — the normalization is the application's, not the graph's; the graph output is the raw pooled vector.
- Gate: at least 100 sentences, all four MRL widths (768 / 512 / 256 / 128, each re-normalized after truncation per the checkpoint README:166-185), file committed with its provenance, and a re-run reproducing the same digests.
- Why here and not later: this freezes the denominator before any port code is written, and it is a second opinion beside our own chain, never it — the port's C reference path stays the oracle for Rust and CUDA, exactly as /data/ds4-dfm-rs/AGENTS.md states for its retained `-c` oracles.

PHASE 3 — the loader: identity, payload reader, layouts, registration (phase-doc rows P0.4-P0.10; P0.1-P0.3 are retired with DQVD)

- Read: the two shipping families end to end, as the pattern — crates/ds4-core/src/identify.rs:169-207 (routing) and :183-186 (the family gate), crates/ds4-core/src/shape.rs (ModelFamily values through Glm53Vq = 14 at :35 and Ling31Vq = 15, Ling31FlashVq = 18 at :104, name arms :49 and :71), crates/ds4-core/src/layout.rs (LayoutSpec :119, family layouts() :3791), crates/ds4-core/src/bind.rs:993, crates/ds4-core/src/lib.rs (the Glm53Vq arms :1248, :1274, :1528), crates/ds4-core/src/tensors.rs (TensorInfo / ShardPlan / TensorInventory and the error variants), C side ds4.c:382-383 (family enum), :657-660 (shape catalog), :10240-10250 (bind dispatch), with ds4_glmvq_bind.inc / ds4_glmvq.h as the `.inc` shape to mirror.
- Land: crates/ds4-core/src/gemma_vq.rs (payload reader + validator + TensorInfo fill, the contract from phase 0), its error variant in tensors.rs, the gemma arms in identify.rs / shape.rs / layout.rs / bind.rs / lib.rs, the C identity + new ds4_eg2vq_bind.inc with ds4_eg2vq.h, and the family arms in crates/ds4-cli/src/lib.rs (:184 parse_args, family arms :451, :491).
- Host decode for the CPU path is mostly free: `ds4vq_dequant_f32` (ds41_vq_fmt.h:72-104) is dim- and width-generic and already handles dim 4; check its magic whitelist against what our encoder writes before relying on it.
- Gate: the ds4-side dump of all 1376 tensors equals the encoder's verify output line for line, and loading the artifact twice is deterministic (same dump, same hashes).

PHASE 4 — the text tower on the CPU reference path (phase-doc rows P1.1-P1.20)

- Authority, every line from /data/llama.cpp/src/models/gemma-embedding2.cpp unless named: embedding lookup scaled by sqrt(512) :86; no-cache, non-causal inputs :90-92 (causal_attn = false at :7); per-layer input block :97 with :215-232; attn_norm :109; q/k/v and heads :113-123; weighted RMS on Q and K, weightless on V :125-127; per-layer RoPE base (:132-137, base selection src/llama-model.cpp:2748-2750, the SWA split rule src/llama-hparams.cpp:8-20) ; attention scale 1.0 :139-141; attn_post_norm + residual :149-152; gated tanh-GELU FFN :156-170; the per-layer embedding block :172-193; layer_scalar :195-196; the control-vector hook is a no-op without a cvec file :199 (implement nothing, record the decision); output_norm + the 768 projection :206-211; mean pool src/llama-graph.cpp:4049-4053; L2 normalize in the caller, examples/embedding/embedding.cpp:70.
- The one mechanism to get exactly right, and the reason: the sliding layers use a SYMMETRIC window — declared at gemma-embedding2.cpp:4 (LLAMA_SWA_TYPE_SYMMETRIC) and masked at |p1 - p0| > n_swa/2 (src/llama-hparams.h:524-533), so with n_swa = 512 a token sees ±256, a 513-wide span. With causal_attn = false every query sees every row, so a trailing cut (what ds4's kernel does at ds4_cuda.cu:42214) is the wrong mask for sequences longer than 256 tokens.
- Tokenizer: .models/EmbeddingGemma-2-914f7f8/tokenizer.json + tokenizer_config.json + tokenizer.model (262,144 entries, byte fallback), ported in the pattern of crates/ds4-core/src/tok/ling31.rs.
- Land: the family's C graph (ds4_eg2vq_graph.inc) plus its CPU reference entry, and the tokenizer module.
- Gate: the frozen sentence set (phase 2) through three things — the C reference path, the Rust host, and the anchor vectors. Report mean and minimum cosine and max element difference for C-vs-Rust and C-vs-anchor, at 768 and at 512/256/128 with re-normalization. Declared before the run: C-vs-anchor mean cosine ≥ 0.9999 (if it is not, the port is wrong, not the anchor), and Rust-vs-C within a stated element tolerance.

PHASE 5 — what VQ costs (phase-doc row X.7)

- Act: the same sentence set and a frozen labelled retrieval set through our own engine, once with the baseline GGUF and once with the VQ artifact; report the cosine agreement distribution (mean, min, quantiles) and recall@k / MRR at 768 and at each MRL width.
- Gate: the numbers are committed with the artifact hashes they were measured on. This is the owner's "not less smart" answer, and it needs no third party.

PHASE 6 — the entry point (phase-doc rows P2.1-P2.4)

- Land: a CLI verb (crates/ds4-cli/src/lib.rs:184 parse_args, family arms :451, :491), then /v1/embeddings in crates/ds4-server/src/serve.rs beside the existing arms (:1169, :1183, :1212 — measured today: no embeddings route and no pooling anywhere in crates/), plus the serving capability and native-limit arms (crates/ds4-core/src/serving.rs:715, serving_host.rs:1111), the task-prompt selection from config_sentence_transformers.json, and MRL truncation on request.
- Gate: the route returns the same vector as the CLI byte for byte for the same input and prompt.

PHASE 7 — the device path (phase-doc rows P3.1-P3.3)

- Read: cuda/ds41_vq_row.cuh:55 (the kernel refuses any dim but 8) and crates/ds4-core/src/vq.rs:42 (VQ_DIM = 8) against our dim-4 tier; the attention kernel ds4_cuda.cu:42201 with the row cap :42208 (`rows > 256u`), the `float score[256]` at :42211 and the trailing-window line :42214, its public API ds4_gpu.h:3645 (no window) and :3658 (window), and the VQ-in-kernel execution pattern ds4_glmvq_gpu.cuh.
- Land: a dim-4 VQ device kernel, a production bidirectional attention kernel with the symmetric ±n_swa/2 mask, and the tower end to end on the GPU with payloads consumed in-kernel (never expanded to dense).
- Gate: GPU vectors equal the CPU path within a stated tolerance, plus a latency and a VRAM number printed by the run.

PHASE 8 — vision and audio towers (phase-doc rows P4.1-P4.4), only if the owner wants them

- Their graphs live in /data/llama.cpp/tools/mtmd/models/gemma4v.cpp and gemma4a.cpp (wired at tools/mtmd/clip.cpp:973-979 and :1115-1121), the tie-in projections are already verbatim BF16 in our container, and the projector variant for this checkpoint is not stated by the conversion (conversion/gemma.py:978, :1057) and must be read out before porting. Gate: the phase 4 comparison, per tower.

Order of work, and the first command of the next session:

0 → 1 (baseline GGUF) → 2 (anchor freeze) → 1 (artifact GGUF) → 3 → 4 → 5 → 6 → 7 → 8 if wanted.
First command: `./bin/eg2_pack_cpu check out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin` (prints the 1376-slot verdict the phase 3 gate compares against), then open the phase-0 reading list.

4. Reference map — the exact repos and files (pinned; verified in this session)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (the work lands here) | /data/embeddinggemma2/ds4-rs @ eb67d493, branch feat/ds41-dsml-carriage, gitignored, own git | every crates/ds4-core/src path named in section 3, ds4.c, ds4_glmvq*.inc/.h, cuda/*.cuh, tests/parity/, the Makefile |
| Form and metadata authority | /data/llama.cpp @ 31288c9c (b11723, 2026-10-18), MIT, read-only, source only (no build/ present) | src/models/gemma-embedding2.cpp (232 lines), src/llama-arch.cpp, src/llama-hparams.*, src/llama-model.cpp, src/llama-graph.cpp, ggml/include/gguf.h, ggml/src/gguf.cpp, examples/embedding/embedding.cpp, tools/mtmd/** (phase 8) |
| Writer and payload precedent | /data/YoungAi (reference toolchain, revision a4e6d31) | src/common/ds4_gguf_write.h, gguf-tools/quantize/v41_to_gguf.c, vq_fmt.h |
| Verification method | /data/ds4-dfm-rs (the live ds4 project; not modified from here) | AGENTS.md (retained `-c` oracles), tests/parity/, tests/ds4_proof.py, Makefile (the false-green guard) |
| The model | /data/embeddinggemma2/.models/EmbeddingGemma-2-914f7f8 (1.5 GB, 15 files, gitignored) | config.json, per_layer_config, config_sentence_transformers.json, 1_Pooling/config.json, modules.json, tokenizer.{json,model}, tokenizer_config.json, README.md |
| Counter-example only | /data/llama-prism @ b10743 | not needed; read only to avoid copying its 1/sqrt(head_dim) attention scale |

5. Routes removed — do not resurrect any of them

- ONNX and onnxruntime, in any role (runtime, export, instrument). Owner, 2026-10-19: "remove this onnx i dont want it and no pytorch".
- PyTorch, transformers, sentence-transformers, and any Python ML stack: forbidden as a compute path by AGENTS.md P-1, and unusable here anyway — the installed transformers has no embedding_gemma module, and llama.cpp's converter imports torch at convert_hf_to_gguf.py:12. The isolated-venv item is dead.
- The DQVD container as the target format: retired by the owner's container decision. The 421 MB DQVD file remains only as the input to phase 1 (the re-wrap or re-encode source) and as the byte-verified record of the quantization itself.
- The `eg2_pack reconstruct` fallback (rebuild a safetensors directory from the DQVD container): DQVD-only and out of date the moment the GGUF exists; the GGUF plus its declared layout is the loadable form now.
- candle, burn, tch, and any Rust tensor framework: AGENTS.md P-1 item 5; the arithmetic is our own C/CUDA kernels.
- /data/llama-prism: a counter-example, never a source. Metal and macOS: not a target here (Linux, NVIDIA only).
- Not in scope unless the owner says otherwise: key-value checkpoints (phase-doc X.3), the distributed/agent surfaces of ds4, and the vision/audio towers (phase 8).

6. Project facts a fresh session cannot re-derive

- Workspace layout: RECIPE.md (the artifact recipe: sections 4 commands, 5 quality, 6 what the port must implement, 7 identity); RECIPE-deepseek41-vq.md (the ds4 VQ recipe we mirror); docs/ds4-gemma-port-phases.md (row list); docs/ds4-gemma-port-reference.md (the port map and the four-leg verification); docs/tensor-inventory.tsv and docs/encode-plan.tsv (the per-tensor truth); docs/artifact-verify-summary.txt; docs/ladder-embeddinggemma2.txt (the measured tier ladder); tools/{eg2_vq.c,eg2_pack.c,eg2_gpuprobe.cu} and bin/ (their binaries); out/ (all artifacts and logs, gitignored); AGENTS.md (project guardrails).
- The artifact's quantizer: nocal VQ (codebooks solved on weights alone), dim 4, nc 4096, text-tower mean cosine 0.9859; the ladder's measured comparison is dim 4/nc 4096 at cos 0.9869 versus dim 8/nc 4096 at 0.9218 on the same representative tensor (docs/ladder-embeddinggemma2.txt).
- Model shape to load: 24 layers, hidden 512, 4 heads, 2 KV heads, head_dim 256; layers 5/11/17/23 full attention with head_dim 512 and 1 KV head; intermediate 2048; rms eps 1e-6; vocab 262,144; embedding dim 768; sliding window 512; mean pool then L2 normalize.
- Machine: RTX 4070 SUPER (12,282 MiB, sm_89), CUDA 13.3, 28 CPUs; Rust binds through the canonical builder (/data/build/linux/build.sh -p /data/embeddinggemma2/ds4-rs, rule 5); the C/CUDA side builds through the tree's Makefile, and the carried-over objects and the 76 MB ds4-server binary make the first build incremental.
- ds4-rs remotes are shared with the live ds4 project (origin Baekpica/ds4-dfm-rs, fork JLFN/ds4-dfm-rs): a push from here is shared state and needs confirmation. /data/ds4-dfm-rs itself is not to be modified from this workspace.
- Guardrails: global ~/.opengrok/AGENTS.md (core) + the mount at .opengrok/rules/project-work-rules.md (symlink; do not edit for project rules) + this repo's AGENTS.md. .gitignore excludes .models/, ds4-rs/, graphify-out/, graphify-rs-out/, out/ and *.o.

7. Guardrails and open items

- P-1 (no Python in the runtime, ever) and P-2 (references are citations or one-shot instruments, never dependencies; no ONNX, no PyTorch) are in this repo's AGENTS.md with their gate commands. All five checks return zero today, including ldd against the real ds4-server binary.
- Owner decisions in force: the container is GGUF + `gemma_vq` payload (2026-10-19); the external anchor run is approved once the baseline GGUF exists; quality is paramount — VQ is for compactness, never a licence to lose accuracy.
- OPEN (owner): whether the dim-4 VQ device kernel (phase 7) is in scope now or the device path waits for a dim-8 sibling artifact. OPEN (owner): whether the vision/audio towers (phase 8) are wanted. OPEN: whether KV checkpoints mean anything for an embeddings family.
- OPEN before any push: rule 12 graphify, rule 19 QA evidence, rule 10 OSV scan for ds4-rs (it carries a Cargo.lock); the outer repo has no remote, so publishing is an offer.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git diff --stat d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git log --grep "Unit: .* complete" d6da36680643c48c144536caf658e360b6d30d82..HEAD
    ./bin/eg2_pack_cpu check out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin
    sha256sum -c SHA256SUMS
    git -C ds4-rs log --oneline -1 && git -C ds4-rs status --short
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'

Trust git over this document's headings: continue with the first phase whose work is
not in the diff. Staleness check at the time of writing: after cada4bd only plan
documents are expected — this handoff and any phase-doc correction committed with
it. Anything else in the diff must be read before sections 2-6 are trusted.

This document is COMPLETE and self-contained: a new session can execute the project
end to end from this file alone. It carries the plan ahead, the exact reference
files, the facts git cannot provide, and the guardrails. History is deliberately
absent — git log and the project memory entry are the historical record.
