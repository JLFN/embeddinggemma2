# EmbeddingGemma-2 in ds4-rs — the phase doc

What has to be ported, in what order, and where each piece comes from. This is the
document the handoff points at; `docs/ds4-gemma-port-reference.md` holds the
method, the authority trees and the forward-pass citations, and this file turns
that into a work list with one row per item, grouped by phase.

Status: written 2026-10-19. Every row is OPEN unless its status cell says
otherwise. A row closes when its cited reference has been read and its own gate
passes; nothing closes on "looks right".

How to use it:

- Execute phase by phase, P0 first. Within a phase the rows are independent
  unless a row names another row.
- A row is a citation, not a summary. Where a row gives `file:line`, read that
  line before writing code against it; where it gives a line range, that range is
  the authority for the row's whole subject (project rule P-2, global rule 0.2).
- Every row's port lands in the ds4-rs family pattern: Rust host (identify,
  validate, tokenize, layout) → C bind `.inc` (name → tensor, flat record table)
  → C graph `.inc` (the forward pass) → CUDA kernels. The GLM VQ family is the
  closest analog in shape and the Ling-3.1 VQ release the closest analog in
  container handling.
- A deprecation trigger, not a rule: if a row's reference line no longer says
  what the row claims, the reference moved. Re-read it, then fix the row.

Status legend: `OPEN` (not started), `WIP` (in progress), `DONE` (gate passed,
evidence committed), `BLOCKED` (names what it waits on).

## 1. The three inputs, pinned

| Role | What | Revision / identity | Verified |
|---|---|---|---|
| Reference (the mathematics and the metadata) | `/data/llama.cpp`, upstream `ggml-org/llama.cpp`, MIT | `b11723`, commit `31288c9c66e59377f7788d920678a565bb89dd37`, 2026-10-18 | `git log -1` this session; `src/models/gemma-embedding2.cpp` is 232 lines |
| Port base (where the work lands) | `/data/embeddinggemma2/ds4-rs`, working copy of the ds4 runtime | `eb67d493` on `feat/ds41-dsml-carriage`, no `target/`, 9,126 files | `git log -1`, `git branch --show-current` this session |
| Artifact (what the runtime loads) | `out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin` | 421,758,160 B, one DQVD v1 container, kind 0, 1376 slots | `bin/eg2_pack_cpu check` this session: `structure OK: 1376 slots, names, hashes, alignment, payload ranges` |
| Writer and payload precedent | `/data/YoungAi` (the reference toolchain, revision `a4e6d31`) | `src/common/ds4_gguf_write.h` (127 lines, the writer helper), `gguf-tools/quantize/v41_to_gguf.c` (493 lines, the only shipping example of a ds4 VQ family being written into a GGUF), `vq_fmt.h` (the DQVL/DQVQ payload contract) | listed this session; our vendored copy of the payload contract is `ds41_vq_fmt.h`, whose decoder `:72-104` is dim- and width-generic |

Cross-check tree, not authority: `/data/llama-prism` at `b10743` (`adfffbe`),
which carries only the older 300m `gemma-embedding` and differs at its attention
scale (`/data/llama-prism/src/models/gemma-embedding.cpp`). Never port from it.

## 2. What the artifact carries, measured

Slot counts and payload bytes aggregated from `docs/encode-plan.tsv` (1377 lines,
one header) this session:

| Tower | kind | slots | payload bytes |
|---|---|---|---|
| `language_model` | 0 vq (dim 4, nc 4096) | 217 | 58,658,409 |
| `language_model` | 1 q8_0 (the token table) | 1 | 142,606,336 |
| `language_model` | 2 verbatim (BF16) | 195 | 940,080 |
| `vision_tower` | 0 vq (dim 4, nc 4096) | 112 | 60,614,512 |
| `vision_tower` | 2 verbatim | 98 | 32,739,328 |
| `audio_tower` | 0 vq (dim 4, nc 4096) | 132 | 118,016,196 |
| `audio_tower` | 2 verbatim | 619 | 5,670,400 |
| `embed_vision` / `embed_audio` | 2 verbatim | 1 + 1 | 786,432 + 1,572,864 |
| total | | 1376 | 421,758,160 |

Text-tower detail (the P1 scope): 413 slots = 24 layers × 17 + 5 globals. Of
those, 217 are VQ (`4x4096`): the 24 layers' `q/k/v/o_proj`, `mlp.gate/up/down_proj`,
`ple_block.per_layer_input_gate`, `ple_block.per_layer_projection` (9 per layer =
216) plus `ple.per_layer_model_projection`. The 195 verbatim are the 24 × 8
norm/scalar tensors (`input_layernorm`, `pre_feedforward_layernorm`,
`post_feedforward_layernorm`, `post_attention_layernorm`,
`ple_block.post_per_layer_input_norm`, `self_attn.q_norm`, `self_attn.k_norm`,
`layer_scalar`) plus `language_model.norm.weight`,
`ple.per_layer_projection_norm.weight` and — note — `embedding_projection.weight`
(768 × 512 BF16, kept whole). The 1 q8_0 is `language_model.embed_tokens.weight`
(262144 × 512).

Two facts worth carrying into the design: the largest single verbatim payload in
the whole container is `vision_tower.patch_embedder.position_embedding_table`
(31,457,280 B), and the text tower's VQ tier is dim 4 while the only device VQ
kernel in ds4-rs today is dim 8 (row P3.1).

One trap measured while writing this: a DQVD slot records only `rows` and `cols`,
so a tensor of rank 3 loses its rank in the container. That vision position table
is `2x10240x768` in the checkpoint inventory (`docs/tensor-inventory.tsv`) while
its slot records `rows=2, cols=10240` — and the container's own verbatim rule
accepts that, because it only requires `payload_bytes / dtype_size >= rows * cols`
(`tools/eg2_pack.c:1059-1068`). For a VQ slot the geometry is fully pinned (the
byte account has to close), but for a verbatim slot it is a floor, not a shape.
The port must therefore take tensor shapes from a declared per-family layout — the
mechanism ds4-rs already uses, `layouts()` building `LayoutSpec` with an explicit
`ndim` (`crates/ds4-core/src/glm_vq/layout.rs`), fed from the reference's own
`create_tensor` calls (`src/models/gemma-embedding2.cpp:34-71`) and our
`docs/tensor-inventory.tsv` — never from the slot geometry.

Container layout and the validation contract, both read from the encoder that
wrote it (`tools/eg2_pack.c`): header 64 B at `:51-88` (documented) and `:221-225`
(the constants `DQVD_MAGIC`/`DQVD_VERSION`/`DQVD_HDR_BYTES`/`DQVD_SLOT_BYTES`/
`DQVD_ALIGN`); plan offsets `:666-690`; the checker `:980-1077`, whose rules are
the ones row P0.2 must reproduce.

## 3. P0 — container, identity, registration

Container decision applied (owner, 2026-10-19, row X.6): the artifact is a GGUF
carrying a `gemma_vq` payload, so the rows below change shape. P0.1 to P0.3
described a container of the project's own — DQVD, with its own magic, header and
checker rules — and are RETIRED as a target: that work is not ported, the DQVD
file survives only as the input to the re-wrap or re-encode done in X.5's unit, and
its reader stays in `tools/eg2_pack.c` as the tool that made the file we have. What
the payload must carry is derived first, in X.8, from the two VQ families that
already do this. One consequence is a saving: identity no longer needs a new
container signature, because the GGUF metadata carries it (the route
`glm_vq.format_version` takes at `crates/ds4-core/src/identify.rs:183-186`).

Gate for the phase: a ds4-rs dump of all 1376 slots equals the encoder's own
verify output (`bin/eg2_pack_cpu check` plus the per-slot re-encode comparison in
`docs/artifact-verify-summary.txt`, 1376 slots across 14 shards).

| id | Item to port | Reference | Lands in | Status |
|---|---|---|---|---|
| P0.1-P0.3 | RETIRED (X.6): the DQVD container of the project's own is not ported — no reader, no checker port, no container error surface. The reading stays here as the record of what produced the artifact we have, and the file is the input to X.5's re-wrap or re-encode. If the writer ever has to reproduce a DQVD byte, the authority is the header at `tools/eg2_pack.c:51-88` and `:221-225`, the plan writer `:666-724`, and the checker `:980-1077` (whose rules the new writer must still guarantee in GGUF terms: name/geometry/payload-length agreement per tensor) | `tools/eg2_pack.c:51-88`, `:221-225`, `:666-724`, `:980-1077`, `:302` (`fnv1a64`), payload accounts `tools/eg2_pack.c:18-19` → `gguf-tools/quantize/vq_qc_bytes.h:15`, `:20` | nowhere — retired | RETIRED (X.6) |
| P0.4 | Family ids: `ModelFamily` value and `Variant` value, the oracle-name parser and the `as_str` arm | last values in use are `Glm53Vq = 14` / `Ling31Vq = 15` (`crates/ds4-core/src/shape.rs:35`) and `Ling31FlashVq = 18` (`:104`); name arms at `:49` and `:71` | `crates/ds4-core/src/shape.rs` | DONE 2026-10-19 (unit 13): `ModelFamily::GemmaVq = 16`, `Variant::GemmaVqFlash = 19`, the oracle-name arms, `SHAPE_GEMMA_VQ` and the arch route. Verified three ways: `dump_oracle` prints the Rust table, `tests/parity/shape_c_oracle.c` prints its own copy, and `test-catalog-parity` compares them (`docs/loader-gate3.txt` section 4) |
| P0.5 | File identity, reduced by X.6: no new container signature — the GGUF holds the family's own metadata key and the existing GGUF route carries it, exactly as `glm_vq.format_version` does. The row is now "assert our key, then hand to the family reader", not a new magic | routing `crates/ds4-core/src/identify.rs:169-207`; the `glm_vq.format_version` gate at `:183-186`; `Identified` `:54` | `crates/ds4-core/src/identify.rs` | DONE 2026-10-19 (unit 13): the file asserts `general.architecture` `gemma_vq` and `gemma_vq.format_version` 1, in `identify.rs` beside GLM's gate; the reader reports `IDENTIFY EmbeddingGemma-2 VQ family=16 variant=19` |
| P0.6 | Layout + bind dispatch: the family's `layouts()` and its bind names | `crates/ds4-core/src/layout.rs:3791`, `crates/ds4-core/src/bind.rs:993`; `LayoutSpec` `crates/ds4-core/src/layout.rs:119` | `crates/ds4-core/src/layout.rs`, `bind.rs`, plus `crates/ds4-core/src/gemma_vq/layout.rs` if it mirrors the Ling-3.1 split | DONE 2026-10-19 (unit 13): `gemma_vq/layout.rs` builds all 1376 specs, the arms are at `layout.rs:3791` and `bind.rs:993`, and the reader reports `BIND ok slots=1376` / `LAYOUT ok specs=1376` |
| P0.7 | Inventory: DQVD slots become `TensorInfo` rows (name, dim, typ, rel/abs offset, elements, bytes) over one file-backed shard plan; the shape (`ndim`, `dim[]`) comes from the declared family layout, not from the slot's `rows`/`cols` (see the trap in section 2) | `crates/ds4-core/src/tensors.rs` (`TensorInfo`, `ShardPlan`, `TensorInventory` at `:153-180`); `ling31_vq/release.rs:139` fills the same types from `release.json` | `crates/ds4-core/src/gemma_vq.rs` + the family's `layouts()` | DONE 2026-10-19 (unit 13): `validate_part` runs from `tensors.rs::parse_one` and fills `TensorInfo`; the shape comes from the family layout, and each VQ payload's geometry is checked against the same byte account the layout computes from the tower architecture, so neither side can drift |
| P0.8 | Host wiring: the family's arms in the model/host open path | the `Glm53Vq` arms at `crates/ds4-core/src/lib.rs:1248`, `:1274`, `:1528`, `:2405`, `:2982`; payload layout `crates/ds4-core/src/payload.rs:78`, `:99` (only if checkpoints are in scope — see X.3) | `crates/ds4-core/src/lib.rs`, `payload.rs` | DONE 2026-10-19 (unit 13), reduced: the arms are `validate_file` → `gemma_vq::validate_metadata`, the serving profile, the native chunk bound and the tokenizer refusal, so the model/host open path stops at the tokenizer by name until phase 4. `payload.rs` is deliberately untouched: X.3 is still open, and an embeddings route has no decode loop to checkpoint |
| P0.9 | C identity: family enum value, shape catalog entry, and the `weights_bind` arm | `ds4.c:382-383` (enum), `ds4.c:657-660` (shape catalog shape), `ds4.c:10240-10250` (bind dispatch: `glmvq_bind` / `ling31_bind`) | `ds4.c` | DONE 2026-10-19 (unit 13): `DS4_MODEL_FAMILY_GEMMA_VQ` 16, `DS4_VARIANT_GEMMA_VQ` 19, `DS4_SHAPE_GEMMA_VQ`, the shape switch arm and the `weights_bind` arm. `ds4.o` builds in 38.9 s and `ds4_cpu.o` in 10.7 s; `test-catalog-parity` compares the shape three ways |
| P0.10 | C bind table: the flat record table the Rust reader hands to C, one `.inc` per family | `ds4_glmvq_bind.inc` (201 lines) as the pattern; header `ds4_glmvq.h` (179) | new `ds4_eg2vq_bind.inc`, `ds4_eg2vq.h` | DONE 2026-10-19 (unit 13), reduced and renamed: the files are `ds4-rs/ds4_eg2vq_bind.inc` and `ds4-rs/ds4_eg2vq.h` (the handoff's names, matching `tools/eg2_*.c`), not `ds4_gemma_vq_*`. The header declares the record table the host will fill; the bind stops by name, because a record table has no consumer before the graph (phase 4) and an unhandled family falls through to the DeepSeek binding |
| P0.11 | Host decode of a VQ payload for the CPU reference path: the dim-generic decoder already exists and takes the payload tier's dim/nc as arguments, so the text tower's dim-4 tier needs no new host decoder — only a check that its magic whitelist accepts what our encoder writes (`DS4VQ_MAT_MAGIC` v2 payloads; a `DQV3` payload is refused by design) | `/data/embeddinggemma2/ds4-rs/ds41_vq_fmt.h:73-105` (`ds4vq_dequant_f32`, dim- and width-generic, index width computed from `nc`), vendored from `/data/YoungAi/vq_fmt.h`; the Rust-side decoder is v3/dim-8 specific (`crates/ds4-core/src/vq.rs:42`, `VQ_DIM` = 8) | C side: reuse `ds41_vq_fmt.h`; Rust side: only if a Rust decode path is needed | DONE 2026-10-19 (unit 13) for the magic check: the loader asserts `DQVQ` on every payload and the reference decoder's whitelist accepts exactly that. The decode itself is phase 4's; the dim-4 host decode is the same function with our tier's arguments |

Footprint for sizing, measured this session: 35 files in ds4-rs mention the VQ
family identity (154,585 lines in total across them), and the VQ-specific core is
`ds4_glmvq_bind.inc` 201 + `ds4_glmvq_graph.inc` 391 + `ds4_glmvq_mtp.inc` 98 +
`ds4_glmvq.h` 179 + `ds4_glmvq_gpu.cuh` 2063 + `ds41_vq_fmt.h` 106 +
`cuda/ds41_vq_row.cuh` 267 + `cuda/ds41_vq_decode.cuh` 390 lines. An earlier
figure of "7,577 lines across 24 files" came from a wider file set that was not
re-enumerated here; treat it as unverified and use the measurement above.

## 4. P1 — the text tower on the CPU reference path

DONE 2026-10-19 (unit 14). Landed: `ds4-rs/crates/ds4-core/src/tok/gemma_vq.rs` (the
`gemma4` pre-tokenizer, the merge queue and the special-token partition) plus its
driver `crates/ds4-core/examples/gemma_vq_tok.rs`; the tower as the C reference
forward `ds4-rs/ds4_eg2vq_graph.inc` with its bind `ds4_eg2vq_bind.inc` (every
type-42 payload decoded once into a 520 MiB f32 cache) and its oracle
`ds4-rs/tests/parity/eg2vq_embd_c_oracle.c`; the Rust host's own leg as the
integration test `crates/ds4-core/tests/eg2vq_ref.rs`. Gate script
`tools/eg2_phase4_gate.sh` (its tokenizer leg is `tools/eg2_tok_check.sh`), evidence
`docs/text-tower-gate4.txt`. Rows P1.1-P1.20 below are all landed.

Two facts the run settled that the rows did not predict: the anchor's own model is
the BASELINE GGUF (llama.cpp refuses the artifact at its type-42 tensors), so the
declared C-vs-anchor cosine is a baseline-weights measurement and the artifact's
distance from it is the quantization, not the port; and the C oracle therefore
binds both weight sources by name.

Every line below is `src/models/gemma-embedding2.cpp` at `b11723` unless another
file is named. Gate: on a frozen set of at least 100 sentences, our 768-dim
vectors agree with the C reference path's (X.1 leg 2; the optional external anchor
is a second, independent number, not the denominator) — mean and minimum cosine,
maximum element difference — and the same three numbers under MRL truncation to
512, 256 and 128 with re-normalization (the checkpoint's own README documents the
truncation and the required re-normalization: `.models/EmbeddingGemma-2-914f7f8/README.md:91`,
`:166-185`).

| id | Item to port | Reference | Status |
|---|---|---|---|
| P1.1 | Embedding lookup scaled by `sqrt(n_embd)` = `sqrt(512)` (not the usual 1.0) | `:86` | DONE 2026-10-19 (unit 14) |
| P1.2 | Position input and no-cache attention inputs; the whole tower is non-causal | `:90`, `:92` (metadata `causal_attn = false` at `:7`) | DONE 2026-10-19 (unit 14) |
| P1.3 | Per-layer input: project with `per_layer_model_proj`, scale by `1/sqrt(n_embd)`, reshape to `[n_embd_per_layer, n_layer, n_tokens]`, RMS with `per_layer_proj_norm`, permute to `[n_embd_per_layer, n_tokens, n_layer]`; there is no per-layer token embedding | `:97`, `:215-232` (comment `:216-217`) | DONE 2026-10-19 (unit 14) |
| P1.4 | Per layer: `attn_norm` RMS with weight | `:109` | DONE 2026-10-19 (unit 14) |
| P1.5 | q/k/v projections and the head reshape | `:113-116`, `:121-123` | DONE 2026-10-19 (unit 14) |
| P1.6 | Weighted RMS on Q and K (`q_norm`, `k_norm` of width `n_embd_head`), weightless RMS on V | `:125-127` (no V-norm weight exists in our inventory, which matches) | DONE 2026-10-19 (unit 14) |
| P1.7 | RoPE per layer with that layer's own base (SWA layers 1e4, the four full layers 1e6) and `n_rot(il)` | `:104`, `:132-137`; base selection `src/llama-model.cpp:2748-2750`; which layers are SWA `src/llama-hparams.cpp:8-20` (`il % 6 < 5`) | DONE 2026-10-19 (unit 14) |
| P1.8 | Attention with scale 1.0 — `q_norm` makes the `1/sqrt(head_dim)` scaling unnecessary; sliding window 512 on the twenty sliding layers, full attention on 5/11/17/23. The window is SYMMETRIC: masked at `\|p1 - p0\| > n_swa/2` = 512, so a token sees a 1025-wide span — not a trailing cut | `:139-141` (scale), `:8` (scale value and reason), window key `:11`, SWA type `:4` (`LLAMA_SWA_TYPE_SYMMETRIC`), mask rule `src/llama-hparams.h:524-533` | DONE 2026-10-19 (unit 14). CORRECTION of this row: it said `n_swa/2` = 256 and a 513-wide span, which is wrong — `n_swa` is the file's 1024 (the converter's doubling of HF's one-sided 512), so the half-window is 512 and the span 1025 positions |
| P1.9 | `attn_post_norm` RMS with weight, then residual onto the layer input | `:149-152` | DONE 2026-10-19 (unit 14) |
| P1.10 | `ffn_norm` + gated FFN: `gelu(gate_proj(x)) * up_proj(x)`, then `down_proj`; the GELU is the tanh form | `:156-165` (`LLM_FFN_GELU` + `LLM_FFN_PAR` → `src/llama-graph.cpp:1995-2000` → `ggml_geglu_split` `ggml/src/ggml.c:3061`); formula `ggml/src/ggml-cpu/ops.cpp:3316` + `ggml_vec_geglu_f32` `ggml/src/ggml-cpu/vec.h:1462` (`gelu(x) * g`); tanh constants `ggml/src/ggml-cpu/vec.h:963-969` | DONE 2026-10-19 (unit 14) |
| P1.11 | `ffn_post_norm` RMS with weight, then residual onto the attention output | `:167-170` | DONE 2026-10-19 (unit 14) |
| P1.12 | Per-layer embedding block: `gelu(per_layer_inp_gate(x))` (a plain GELU, no up-projection and no `1/sqrt(2)` factor) × this layer's slice of the per-layer input, then `per_layer_proj`, `per_layer_post_norm`, residual | `:172-193` | DONE 2026-10-19 (unit 14) |
| P1.13 | `layer_scalar` multiply, one learned scalar per layer | `:195-196` | DONE 2026-10-19 (unit 14) |
| P1.14 | Control-vector hook: a no-op without a cvec file — implement nothing, record the decision | `:199` | DONE 2026-10-19 (unit 14) |
| P1.15 | Output: `output_norm` RMS, then the embedding projection to 768 (projecting per token ≡ projecting after pooling) | `:206-211` | DONE 2026-10-19 (unit 14) |
| P1.16 | Mean pooling over tokens | `src/llama-graph.cpp:4049-4053`; `1_Pooling/config.json` (`pooling_mode: mean`, `include_prompt: true`) | DONE 2026-10-19 (unit 14) |
| P1.17 | L2 normalization — the graph does not do it, the application does | `examples/embedding/embedding.cpp:70` (`common_embd_normalize`); `2_Normalize/` and `modules.json` in the checkpoint | DONE 2026-10-19 (unit 14) |
| P1.18 | Task prompts as data, taken from the checkpoint (20 named prompts) and defaulted per task | `.models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json` (read fully this session) | DONE 2026-10-19 (unit 14) |
| P1.19 | The BPE tokenizer (262,144 entries, byte fallback, merges, normalizer, Split pre-tokenizer, `▁`→space decoder) | ties to `tokenizer.json` / `tokenizer_config.json` / `tokenizer.model` in the checkpoint; ds4-rs pattern `crates/ds4-core/src/tok/ling31.rs` | DONE 2026-10-19 (unit 14) |
| P1.20 | The layer split itself, as a self-check on load: 24 layers, hidden 512, 4 heads, 2 KV heads, head_dim 256; layers 5/11/17/23 with head_dim 512 and 1 KV head; intermediate 2048; rms eps 1e-6 | `.models/EmbeddingGemma-2-914f7f8/config.json` (`layer_types`, `per_layer_config`), agreeing with P1.7's SWA rule | DONE 2026-10-19 (unit 14) |

## 5. P2 — the entry point

Gate: the route returns the same vector as the CLI, byte for byte, for the same
input and prompt.

| id | Item to port | Reference | Status |
|---|---|---|---|
| P2.1 | A CLI verb that runs the tower and prints the vector (text in, 768 floats out) | the flag-based CLI in `crates/ds4-cli/src/lib.rs` (`parse_args` `:184`); family arms today at `:451`, `:491` | DONE 2026-10-19 (unit 19, ds4-rs fa9d96a0): `ds4 -m <artifact> --embed TEXT [--task NAME] [--dims N]` (`--embed-file FILE`, `--embed-stdin`, `--prompt-map FILE`), JSON on stdout with the family, task, prefix, token count, the special-token flags, dims and the vector at 7 decimals. The tower itself is reached through the C engine's new `ds4_engine_embed_gemma_vq` (ds4.c, beside `ds4_engine_vision_encode_memory`), which binds the device plan and its workspace ONCE per engine (`eg2vq_device_ensure`, workspace grown in 256-token steps) and falls back to the same CPU reference forward (`ds4_eg2vq_ref_embed`) on a non-graph backend; the bridge slot is `ds4_bridge_eg2vq_embed`/`_has_embed` (the `ds4_bridge_v41_has_vision` pair's shape). Gate: `tools/eg2_embed_cli_gate.sh` takes RAW TEXT from the frozen set, runs the verb, and compares against the device parity raw produced by the harness -- a separate driver -- at cos 1.000000000 / rel_rms <= 5.568e-07 on six queries and cos 1.000000000 / rel_rms <= 5.745e-07 on six documents of 803-1414 tokens, with the token counts identical to the harness's own ids on all twelve (`out/phase6-embed-cli-gate.log`, `out/phase6-embed-cli-gate-docs.log`). Evidence `docs/eg2vq-embed-entry-gate6.txt` sections 2 and 5 |
| P2.2 | The embeddings route: an endpoint that is not there today | the router's existing GET arms `crates/ds4-server/src/serve.rs:1169` (`/v1/models`), `:1183` (`/v1/models/<id>`), `:1212` (`/v1/stats`); measured this session: no `/v1/embeddings` and no pooling anywhere in `crates/` | DONE 2026-10-19 (unit 19, ds4-rs fa9d96a0): `POST /v1/embeddings` (OpenAI-compatible: `input` string or array of strings, `model`, `task` defaulting to Retrieval-query, `dimensions` in {768,512,256,128}, `encoding_format` float) and `GET /health` (`{"status":"ok","model":..,"backend":..,"family":"gemma_vq","embedding":true}`), both in the Rust server's `prepare_client` beside the existing arms. An embedding job rides the port's own admission/lease plumbing to the owner thread that holds the model (`PreparedJob::embed` + `ParsedRequest::embedding_carrier`; the alternative, a `ReqKind::Embedding` variant, rippled through eight match sites and was rejected). Gate (`tools/eg2_embed_route_gate.sh`, `out/phase6-route-gate.log`): /health answers as above; /v1/embeddings returns 768 floats; `dimensions: 256` returns 256 floats with L2 norm 1.000000045 against a declared 1e-5; a bad task name is a 400 listing the checkpoint's own names; and the route's vector equals the CLI's for the same text and task at cos 1.000000000000. Checked once more against the device raw directly (outside the CLI): cos 1.000000000. Evidence `docs/eg2vq-embed-entry-gate6.txt` sections 3-5 |
| P2.3 | Serving capability and native-limit arms for the new family | `crates/ds4-core/src/serving.rs:715` (serving caps), `crates/ds4-core/src/serving_host.rs:1111` (native max), and the family-conditional sites `:654`, `:1179`, `:1647-1712` | PARTIAL 2026-10-19 (units 19 and 20; the batching half is what remains). Unit 19: the embedding route needs no per-request capability arm (it shares the inference admission gate and the same lease path: `preparse_shed(&g.admit, true, true, ..)`), and the engine's own limit is stated and enforced in C -- `DS4_EG2VQ_MAX_TOKENS` (2048, the attention kernel's key plane, static-asserted equal to `EG2VQ_ATTN_KEYS_MAX` in `ds4_eg2vq_gpu.cuh`) with an item longer than that REFUSED by name in `ds4_engine_embed_gemma_vq`. Unit 20 (ds4-rs 68837a53): the per-family cap entry now EXISTS and is embedding-shaped rather than decode-shaped -- `EmbedCaps { item_tokens_max: 2048, batch_rows_max: 8192 }` via `ds4_core::embed_caps(ModelFamily::GemmaVq)`, reported as `controls.embed` in the plan JSON and as `embed` by `GET /health`, refused by name (400 in the OpenAI envelope) in `run_embed` and before the FFI call in `Model::embed_gemma_vq`, with the item ceiling held equal across Rust, `ds4_eg2vq.h` and `cuda/eg2vq_attn.cuh` by the new `crates/ds4-core/tests/eg2vq_serving_caps.rs` (the tree's first check that Rust sees the C number at all); `DS4_GEMMAVQ_PREFILL_CHUNK` can no longer advertise a chunk past the tower's ceiling, and `family_native_limit(GemmaVq)` is that ceiling. Evidence: `docs/eg2vq-serving-caps-gate20.txt`, live via `tools/eg2_embed_route_gate.sh` sections 4b/4c (out/eg2vq-route-gate-unit20.log). STILL NOT DONE: the batching of several items into one tower call -- today an array input embeds its items in sequence over the one bound plan, which is the next slice of this row |
| P2.4 | Prompt selection and the request shape (task prompt applied to the input, mean pool, L2 normalize; MRL truncation on request with re-normalization) | P1.16-P1.18 rows; the checkpoint's README for the truncation contract | DONE 2026-10-19 (unit 19, ds4-rs fa9d96a0): the prompt table is the checkpoint's `config_sentence_transformers.json` "prompts" object verbatim (20 names) and lives in `ds4-core` as `ST_PROMPTS` / `st_prompt` so the CLI and the route cannot drift; the checkpoint's `default_prompt_name` is null, so the default task is `Retrieval-query` ("task: search result | query: ") and `Raw` is the no-prefix case the anchor script names; an unknown name is refused with the list of names it does carry. The prefix is applied BEFORE tokenizing, as the phase-4 driver does, and BOS/EOS come from the artifact's own `tokenizer.ggml.add_bos/eos_token` keys (both true: `out/artifact-kv.txt`), read through `ds4_core::tokenizer_special_flags`. Mean pool and L2 are the tower's own (inside the C forward); MRL truncation to 512/256/128 with renormalization is `ds4_core::mrl_truncate`, reachable from the CLI (`--dims`) and the request (`dimensions`), and any other width is refused rather than approximated (measured: `dimensions: 256` -> L2 norm 1.000000045). Evidence `docs/eg2vq-embed-entry-gate6.txt` sections 3 and 5 |

## 6. P3 — the device path

Gate: GPU vectors equal CPU vectors within a stated tolerance, plus a latency and
a VRAM number, both printed by the run.

| id | Item to port | Reference | Status |
|---|---|---|---|
| P3.1 | A dim-4 VQ device kernel. The existing device open rejects every dim but 8, and our whole text-tower VQ tier is dim 4 | `cuda/ds41_vq_row.cuh:55` (`d16 != 8u` → open fails); Rust mirror `crates/ds4-core/src/vq.rs` (`VQ_DIM = 8`); the tier measured in section 2 (217 slots, 4x4096) | DONE 2026-10-19 (unit 17): `ds4-rs/cuda/eg2vq_vq.cuh` takes a dim-4/nc-4096/12-bit payload and produces the tower's matrix product with the codebook in shared memory and the index stream decoded in-kernel (`out[t][r] = gain[r] * sum_i codebook[idx[r][i/4]][i%4] * x[t][i]`); a 64-token x 64-row tile on 16 x 16 threads, 40 KB of static shared (no dynamic opt-in), f32 activations and f32 accumulation because the CPU reference path is f32. Gate `tests/test_eg2vq_vq.cu` (make test-eg2vq-vq, exit 0, `out/phase7-p31-gate.log`): against the payload's own semantics (its own decoder, checked against the values that were packed) and against an FP64 dot of that decode, at all six tower geometries, rel RMS 4.01e-07 to 8.22e-07 and min per-token cosine 1.000000000 against declared bounds of 1e-5 and 1-1e-6; a payload bound to the wrong geometry decodes nothing (output untouched, measured); single-kernel 3.14 TFLOP/s at 2048x512x184 and 3.13 at 512x2048x1414. Two defects of its own were caught by that gate and are recorded: a slab stride taken as the bytes of 256 codewords instead of 256 columns, and an index-unpack row index that was block-local, so every row block above the first re-read row 0's stream (the second is invisible to a probe whose rows share one stream — it was isolated with ones-x and a Kronecker probe). Evidence `docs/eg2vq-device-gate7.txt` sections 1 and 2 |
| P3.2 | Bidirectional attention with a sliding window. Correction to an earlier claim in this project's handoff: the kernel does implement both a window and non-causal attention — the window arrives as a parameter and is applied as `first = window && visible > window ? visible - window : 0`, and `causal = false` makes every query see all rows. Two things make it unusable: it is a scalar one-thread-per-(query, head) kernel with `rows > 256` refused and a `float score[256]` stack buffer, and its window is TRAILING, which is the wrong mask for this model — the reference declares `LLAMA_SWA_TYPE_SYMMETRIC` (`src/models/gemma-embedding2.cpp:4`), masked at `pos_diff < -n_swa/2 || pos_diff > n_swa/2` with `n_swa` = 512, i.e. a token sees ±256 (a 513-wide span) (`src/llama-hparams.h:524-533`). The production kernel must implement the symmetric mask; a trailing cut diverges from the reference for any sequence longer than 256 tokens | kernel `ds4_cuda.cu:42201`, window/causal at `:42205` and `:42217-42219`, row cap `:42208`, score buffer `:42211`; public API with a window `ds4_gpu.h:3658` (`ds4_gpu_motif3_expanded_attention_window_tensor`), without `:3645`; a caller already passes a window `ds4.c:48259`; symmetric rule `/data/llama.cpp/src/llama-hparams.h:524-533`, SWA type `/data/llama.cpp/src/models/gemma-embedding2.cpp:4` | DONE 2026-10-19 (unit 17): `ds4-rs/cuda/eg2vq_attn.cuh`, new kernel code — one block per query token, one warp per query head (so a block's four heads share the KV rows in L2), the key range split across the lanes, pass 1 scores plus the warp maximum, pass 2 `expf(score - max)` plus the sum, pass 3 the weighted sum with the output dims spread over the lanes; scale exactly 1.0f, no causal mask, a zero denominator yields zero; 32 KB static shared bounds an item at 2048 tokens, and a longer one is refused by the launcher rather than truncated. Gate `tests/test_eg2vq_attn.cu` (make test-eg2vq-attn, exit 0, `out/phase7-p32-gate.log`) against a transcription of `eg2vq_attn_worker`'s arithmetic at all seven tower shapes (hd 256 / 2 KV heads / half 512 at 26, 184, 513, 514, 1414 tokens; hd 512 / 1 KV head / no mask at 349, 1414): rel RMS 5.77e-07 to 1.78e-06, min per-(query, head) cosine 1.000000000; the mask's edge measured with one value row at position 1100 and q = 0 (every in-window key weighted 1/nkey exactly, so an underflowed weight cannot mask a wrong cut): a query at distance exactly 512 sees it, at 513 and 1100 it does not, and the device output for a blind query is exactly zero. Evidence `docs/eg2vq-device-gate7.txt` section 3 |
| P3.3 | The tower on the GPU end to end, with the VQ payloads consumed in-kernel (never expanded to dense), plus the VRAM and latency measurement | the GLM VQ device pattern: `ds4_glmvq_gpu.cuh` (2063 lines, the GVQ1 execution adapter) over the `cuda/ds41_vq_*.cuh` kernel family (row-level decode `ds41_vq_row.cuh`, `ds41_vq_decode.cuh`, prefill variants) DONE 2026-10-19 (units 17-18): step 1 landed — `ds4-rs/cuda/eg2vq_ops.cuh` (commit 467f5431) with the tower's remaining primitives (q8_0 token-row gather at sqrt(n_embd), row and per-head RMS with the sum accumulated in double, NeoX RoPE over the whole head from a caller-built table whose angles use the graph's own f32 accumulation, gelu-mul, the 768 projection with output_norm, mean pool and L2), gated by `tests/test_eg2vq_ops.cu` (`make test-eg2vq-ops`, exit 0, `out/phase7-p33-ops-gate.log`): RMS rows and heads at f32 rounding (5.6e-08 to 6.0e-08), RoPE and the embedding gather bit-identical, proj768+pool+L2 at 4.9e-07 with a unit-length result, and gelu-mul at 2.39e-06 with its bound set to 1e-05 for that measured reason (--use_fast_math compiles tanhf differently from libm). STEPS 2-4 DONE 2026-10-19 (unit 18, ds4-rs 4892fb4b): the adapter `ds4-rs/ds4_eg2vq_gpu.cuh` (included once by `ds4_cuda.cu`, the `ds4_glmvq_gpu.cuh` pattern) binds the text tower's 217 payloads AND the 196 aux slots into two device arenas once per process — records from the CPU binder's own `abs_offset`/`dim[0]` via `eg2vq_device_tables()`, with the contract in `ds4_eg2vq.h` (record order, aux slot table, and a static assertion that the sliding layers' KV width 2x256 equals the full layers' 1x512, which is what lets ONE 512-wide KV plane serve both), the payload's own header re-checked against its record, rows%64 and cols%256 re-checked against the GEMM tile, the q8_0 token table (136.02 MiB) and the bf16 768 projection read in-kernel while everything else is widened to f32 on the host with the IEEE reading; a 17-plane workspace sized by max_tokens; and the RoPE table pair built with the graph's own f32 angle accumulation plus double trig. `ds4_gpu_eg2vq_embed` runs the tower in `ds4_eg2vq_graph.inc:169-370`'s exact op order — no new arithmetic anywhere, the adapter's contribution is the order, the buffers and the binding. Gate `tests/parity/eg2vq_embd_gpu_oracle.c` (compiled WITHOUT `-DDS4_NO_GPU` and linked against the CUDA objects; `make tests/parity/eg2vq_embd_gpu_oracle`), declared bounds `cos >= 0.999999`, `rel_rms <= 1e-03` stated before the run: run A, the frozen set's 300-item query partition, 8,059 tokens — min cos(device,cpu) 0.999999950, max rel_rms 3.149e-04, and the CPU leg reproduces the committed oracle raw at cos 1.000000000 / rel_rms 0.000e+00 over 255 comparable items; device 5.39 s = 0.018 s/item = 1494.8 tokens/s (steady state 1499.0), 414.0 MiB by cudaMemGetInfo delta = payload 55.94 + weights 137.04 + workspace 212.01 MiB (+ a 6.29 MiB RoPE pair), against a CPU leg whose own dense f32 decode is 520.0 MiB and which the device leg never builds. Run B, the twelve longest documents (803-1414 tokens): min cos 0.999999876, max rel_rms 4.978e-04, device 3.68 s = 2733.1 tokens/s, the 1414-token item 564.94 ms = 2503 tokens/s. THE SPEEDUP IS STATED AGAINST THE FAIR LEG, not the gate's own: the gate prints 65.1x and 125.2x against its `-O0` reference entry, and the same CPU source compiled with the tree's own CFLAGS (`-O3 -ffast-math -march=native`, `out/eg2vq_embd_c_oracle_o3`) takes 53.53 s (run A) and 66.54 s (run B) while agreeing with the `-O0` leg at cos 1.000000000 / rel_rms 9.5e-07, so the honest ratios are 9.9x on short queries and 18.1x on retrieval-length documents (150.6 and 151.1 tokens/s of CPU against 1494.8 and 2733.1 of device). Both runs are cross-checked by an independent instrument (`tools/eg2_parity_check.awk`, needs `-F'\t'`, recomputes from the raws: run A exact, run B to the last printed digit). Two corrections recorded in unit 18: the earlier "437 MB" was the whole artifact FILE — the text tower's payloads are 58,658,409 B / 55.9 MiB over 217 of the artifact's 461 (the rest are the audio and vision towers), and the cross-check instrument's first version split fields on whitespace, which left it comparing one value per item while printing a plausible result. What this row does NOT claim, both measured and both phase 6's job: the tower has NO entry point (its only callers in the tree are the two parity harnesses; `ds4.c` binds the family at `:10287` but nothing in the runtime calls the device tower), and the decode path is a CRASH rather than an error for this family — `./ds4 -m <artifact> -p hello -n 1` initializes CUDA, maps 0.41 GiB and then dies with SIGSEGV (exit 139, core dumped), identically with `--cpu`, because family 16 has a shape case (`ds4.c:2881`) and a bind arm but no forward/session arm, so the generic prefill reaches `embed_token_f16` and dereferences `w->token_embd`, which gemma_vq never populates (its token table is bound under `w->eg2vq`, `ds4_eg2vq_bind.inc:233-239`). Evidence `docs/eg2vq-device-gate7.txt` sections 4-6. |

## 7. P4 — vision and audio towers (owner decision pending)

The container already holds both towers (section 2); nothing in this phase is
needed for the text embeddings to work. Gate: the same comparison as P1 (X.1
leg 2), per tower, on the tower's own inputs.

| id | Item to port | Reference | Status |
|---|---|---|---|
| P4.1 | Vision tower forward: 16 layers, hidden 768, head_dim 64, 12 heads, patch 16, axial RoPE theta 100 | tower graphs live in the mtmd half of the reference: `tools/mtmd/models/gemma4v.cpp` (`PROJECTOR_TYPE_GEMMA4V` wired at `tools/mtmd/clip.cpp:973-979`, `:480`); shape from our `config.json` | OPEN |
| P4.2 | Audio tower forward: 12 layers, hidden 1024 | `tools/mtmd/models/gemma4a.cpp` (`PROJECTOR_TYPE_GEMMA4A`, `tools/mtmd/clip.cpp:1115-1121`); unified variants `gemma4uv.cpp` / `gemma4ua.cpp` | OPEN |
| P4.3 | The tie-in projections `embed_vision.embedding_projection.weight` (768 × 512) and `embed_audio.embedding_projection.weight` (1536 × 512), both verbatim BF16 in our container | section 2; conversions `conversion/gemma.py:1057` (`EmbeddingGemma2VisionAudioModel`, "same towers as Gemma4, but the tensor names have no `model.` prefix") and its parent `:978` (`Gemma4VisionAudioModel`) | OPEN |
| P4.4 | Which projector variant our model is: the reference has four (V/UV/A/UA) and the conversion class does not state which one this checkpoint maps to | `tools/mtmd/clip.cpp:973-979`, `:1115-1121`; check the conversion's own projector-type assignment before porting | OPEN |

## 8. X — cross-cutting rows

| id | Item | Reference | Status |
|---|---|---|---|
| X.1 | Verification, four legs, all native — no ONNX and no PyTorch anywhere (owner, 2026-10-19). Leg 1: read and cite the authority, llama.cpp `gemma-embedding2` (`src/models/gemma-embedding2.cpp:3-13` hparams keys, `:34-71` the 413 tensors). Leg 2: the port's own C reference path is the oracle and the Rust host is the candidate, with the false-green guard (distinct builds or the comparison is void). Leg 3: a frozen sentence set carrying the checkpoint's own task prompts, run once through the C path, its digests committed and reproduced by every later run. Leg 4: the CUDA path against the C path within a stated tolerance, latency and VRAM recorded beside the equality | `docs/ds4-gemma-port-reference.md` section 6; `/data/ds4-dfm-rs/AGENTS.md` (retained `-c` behavior oracles), `/data/ds4-dfm-rs/tests/parity/`, `/data/ds4-dfm-rs/tests/ds4_proof.py` (scenario + `--check-expected`), `/data/ds4-dfm-rs/Makefile` (the false-green guard) | OPEN |
| X.2 | Guardrails in force for every row: no Python in the runtime and no ONNX or PyTorch anywhere (P-1, five zero-hit greps, the fifth added this session) and references are citations or one-shot instruments, never dependencies (P-2, item 5) | `/data/embeddinggemma2/AGENTS.md` | DONE (rules written, commit `fd63233` plus this session's P-2 item 5; all five greps return zero) |
| X.3 | Decide whether key-value checkpoints matter for an embeddings model at all before wiring `payload.rs` (row P0.8). An embeddings route has no decode loop, so a checkpoint format may be pure cost | `crates/ds4-core/src/payload.rs` (the GLM/Ling VQ arms) | OPEN |
| X.4 | Before any push from ds4-rs: rule 12 graphify graph, rule 19 QA evidence, rule 10 OSV scan (ds4-rs carries a `Cargo.lock`); ds4-rs's remotes are shared with the live ds4 project, so a push is shared state and needs confirmation | `~/.opengrok/project-work-rules.md` rules 10/12/19/23 | OPEN |
| X.5 | The baseline model in GGUF, written natively: the pinned checkpoint re-expressed as GGUF with the metadata and tensor names the loader resolves, so the reference's embedding example and our own C reference path can load the same file. Owner-approved route (2026-10-19): the external anchor run happens once this file exists, as a second opinion beside our own chain. llama.cpp's own converter is not usable: `convert_hf_to_gguf.py:12` imports torch, which rule P-1 forbids. First slice: read the format authority, then emit metadata plus one tensor and prove a round trip. There is no need to invent a writer: the reference toolchain already ships one, `/data/YoungAi/src/common/ds4_gguf_write.h` (127 lines, header-only: `ds4gw_begin`, `ds4gw_kv_*`, `ds4gw_tensor`, `ds4gw_header`, `ds4gw_write`, `ds4gw_end`), and `gguf-tools/quantize/v41_to_gguf.c` (493 lines) is the only shipping example of a ds4 VQ family written into a GGUF — mirror both. The GGUF must also carry the tokenizer metadata llama.cpp resolves (`LLM_KV_TOKENIZER_*`, `src/llama-arch.cpp`, `src/llama-vocab.cpp`), or the anchor cannot load the file | `src/models/gemma-embedding2.cpp:3-71` (the key and tensor set), `src/llama-arch.cpp:64` (arch string), format authority `ggml/include/gguf.h` + `ggml/src/gguf.cpp`, our own reader `ds4.c:1607`, `:2348` and `crates/ds4-core/src/gguf.rs:208`; writer precedent `/data/YoungAi/src/common/ds4_gguf_write.h` + `/data/YoungAi/gguf-tools/quantize/v41_to_gguf.c`; the tokenizer arrays come from our `tokenizer.json` (P1.19) | DONE 2026-10-19 (X.5a): `tools/eg2_gguf.c` writes `out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf` (413 tensors, 557,949,600 B, sha256 `0d84f1a9…`, re-emitted in phase 2 with a tokenizer correction) and gate 1a passed — the writer's manifest and the ds4-rs reader's own dump agree on all 413 tensors, 219 are byte-identical to the checkpoint and 194 are its exact BF16→F32 widening, and llama.cpp at the pin loads the file and prints a 768-dim embedding. Four contract corrections came out of it: 1-D/`*_norm.weight` tensors must be F32 (else the CPU binary op aborts), `attention.sliding_window` is written doubled (1024), `attention.sliding_window_pattern` is a bool array, and the converter's key set also includes `causal`, both `rope.dimension_count*` and `pooling_type`. Evidence: `docs/baseline-gguf-gate1a.txt`. A fifth correction came in the anchor slice (see below): three tokenizer keys the reference's own `SpecialVocab` writes (`mask_token_id` 4, `add_eos_token` true, `add_sep_token` false) were missing; found by running that reference code (`tools/ref_vocab_probe.py`), not by reading it. ANCHOR SLICE DONE 2026-10-19: the frozen set `tests/anchor/anchor-items.tsv` (113 items, 91 with the checkpoint's own task prompt, three of them 1,175-1,398 tokens so the file's 512-token mask half-window is actually exercised) and the reference's vectors `tests/expected/anchor-embeddinggemma2.jsonl` (sha256 `7561254c…`, bit-reproducible on a from-scratch re-run), one item per model invocation so no vector depends on the set. Both open decisions settled by measurement against their alternatives: dropping the eos token moves the vectors by mean cosine 0.96, dropping the prefix by 0.87. Evidence: `docs/anchor-gate2.txt`. ARTIFACT SLICE DONE 2026-10-19 (X.5b): `tools/eg2_gguf.c`'s `artifact` mode re-wrapped the DQVD container into `out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf` (437,508,416 B, sha256 `5bcf2216…`) — arch `gemma_vq`, `format_version` 1, the recipe verbatim, 461 type-42 `vqblob` payloads at nd 1 (`ne` = payload bytes), one type-8 `q8_0` token table (`ne` 512x262144), 914 type-30 `bf16` tensors at the checkpoint's rank, and the same hparams/tokenizer block the baseline carries, under the `gemma_vq.` prefix. Gate 1b passed (`tools/eg2_artifact_check.sh`): 1376 tensors present, all 1376 byte-identical to the container's payloads, all 914 verbatim tensors equal to the checkpoint's own bytes, all 461 VQ tensors beginning with the `DQVQ` magic, the metadata read back value by value through the port's own reader (42 keys) and differing from the baseline's only in `gemma_vq.format_version`/`.recipe` (no `general.file_type`), the file bit-reproducible, and the writer's baseline path re-emitted byte-for-byte identical. A sixth contract correction came out of it: the file's rank is never 0 — the 480 checkpoint scalars ship as rank 1, `ne` {1}, because both readers reject nd 0 (`ds4.c:3353`, `tensors.rs:266`, `ggml.c:1786`). Evidence: `docs/artifact-gguf-gate1b.txt` |
| X.6 | Container: DECIDED by the owner, 2026-10-19 — the artifact moves onto GGUF with a `gemma_vq` payload, as `glm_vq` and `ling31_vq` do; DQVD is dropped. GGUF is ds4's native container (`ds4.c:1607`, `crates/ds4-core/src/identify.rs:199-208`) and the ds4 VQ recipe writes GGUF (`RECIPE-deepseek41-vq.md`, `v41_to_gguf ... .gguf`), so DQVD was the deviation. Cost accepted: a re-encode, measured at about 14.5 minutes of wall time for the shard phase (14 shards in parallel, 1017-1155 s each) plus the merge. Because the baseline model (X.5) is also a GGUF, one writer serves both | `crates/ds4-core/src/identify.rs:199-208`; `RECIPE-deepseek41-vq.md`; `out/encode-run.txt` (`19:47:47Z` -> `20:02:16Z`) | DONE (owner decision; builds on rows X.5 and X.8) |
| X.8 | Derive the `gemma_vq` payload contract before writing any reader or encoder code: the metadata namespaces and custom tensor types a ds4 VQ family puts in its GGUF, and the exact split of work between host and device. Read, not invented: `glm_vq.format_version` / `.codebooks` / `.fp8_scales` / `.logical_shapes` / `.residuals` / `.native_policy` (`crates/ds4-core/src/glm_vq.rs:276-466`), the arch and recipe keys (`crates/ds4-core/src/ling31_vq.rs:23-29`), the custom types bound in C (`DS4_TENSOR_GLM_VQ`, `DS4_TENSOR_GLM_BOOK` in `ds4_glmvq_bind.inc`), and the division of responsibility stated there: "Rust validates the complete GVQ1 release before this pointer adapter" (`ds4_glmvq_bind.inc:1-3`). The recipe string our encoder writes goes in this namespace, and `logical_shapes` is the mechanism that carries rank for tensors a payload cannot express (the section 2 trap). Two determinations belong to this row, settled by measurement rather than assumed: (a) whether our kind-0 payload bytes are already exactly a DQVQ payload — the layout at `ds41_vq_fmt.h:7-12` (`/data/YoungAi/vq_fmt.h`), magic `DS4VQ_MAT_MAGIC`, then u16 dim, u16 nc, u32 rows, u32 cols, codebook nc*dim f16, row gains rows f16, index stream — checked by hashing a slot's payload in the DQVD container against a freshly packed one; and (b) if they are identical, whether phase X.5's artifact step becomes a re-wrap (fast, byte-preserving) instead of a re-encode (about 14.5 min of shard wall time) | `crates/ds4-core/src/glm_vq.rs`, `crates/ds4-core/src/ling31_vq.rs`, `ds4_glmvq_bind.inc:1-3`, `ds4_glmvq.h`, `crates/ds4-core/src/glm_vq/layout.rs`, `/data/YoungAi/vq_fmt.h`, `/data/YoungAi/gguf-tools/quantize/v41_to_gguf.c`, `ds41_vq_fmt.h:7-12` | DONE 2026-10-19: contract in `docs/ds4-gemma-payload-contract.md`. Both determinations measured: (a) kind-0 payloads ARE DQVQ payloads byte for byte (slot payload and a fresh pack of the same tensor both sha256 `d9c02751…`, 1376/1376 slots bit-identical to a fresh encode); (b) the artifact step is a re-wrap, not a re-encode. `logical_shapes` is not needed here: all 461 VQ tensors are rank 2 and all 15 rank>2 tensors are verbatim |
| X.7 | What VQ costs, measured on our own engine: the same sentences through the port's path with the baseline weights (X.5) and with the VQ artifact, compared as a cosine agreement distribution (mean, min, quantiles) and as retrieval quality on a frozen labelled set (recall@k, MRR), at 768 and at each MRL truncation. This is the "not less smart" number, and it needs no third party | our own engine on both container paths; the MRL contract in the checkpoint README `:155-185` | DONE 2026-10-19 (unit 15): the harness is `tools/eg2_phase5_gate.sh` (with its scorer, its set builder and their self-tests) and the labels are BEIR SciFact's test split taken whole (300 queries / 283 documents / 339 labelled pairs, frozen in `tests/retrieval/`, digests pinned in the gate). Both legs are the same CPU reference tower over one ids file, only the weight source differing, run with the -O3 build of the oracle (measured equivalent to the tree's -O0 entry before use: min cos 1.000000000, max rel_rms 9.0e-07 over 24 items — section 7a of the evidence). The codec's intrinsic cost on these 583 items: mean cosine 0.956039571 / min 0.898186175 at 768, 0.972493258 at 128. The task-level cost at 768: MRR@10 0.951778 to 0.935819 (-0.015959) and recall@1 0.888278 to 0.861611, with recall@10 0.983333 to 0.990000 (the artifact finds the labelled evidence inside ten slightly more often) and 23 of 300 queries changing their top-1 document against a baseline mean top1-top2 margin of 0.085722; at 128 the two costs compound (-0.029414 MRR, recall@5 0.954667 to 0.915889). No threshold is declared: the numbers and the hashes they were measured on are the row's deliverable. Evidence: `docs/retrieval-cost-gate5.txt`, log `out/phase5-gate.log` |
| X.9 | Device-path precondition (owner direction 2026-10-19: "we need to get cuda support it should much faster"): the ds4-rs working copy's CUDA object set was built for sm_121a only, one architecture this card does not have (RTX 4070 SUPER, compute capability 8.9), so P3 had no runnable CUDA underneath it — a link against that set would have produced a host that cannot execute a kernel, and would have said so only at run time. Arch-guard the five sm_90-only instruction sites (four `cp.async.bulk.prefetch.L2.global`, one `mbarrier.try_wait.parity`), build the object set for sm_89, prove the kernels execute on this card, and link the Rust host against it. No speed claim belongs to this row: the tokens/s comparison is P3.3 | the tree's own split, read before the change: `ds4-rs/cuda/glm_vq_mma.cuh:83-90` and `cuda/ds41_vq_prefill_mma.cuh:291-297`; the fork's link recipe `Makefile:354-358`; the device tests it already ships (`Makefile:2162-2188`) | DONE 2026-10-19: five sites guarded in three headers (42 inserted lines, 0 deleted, so a checkout restores the sm_90-only form exactly); ptxas' refusal and its absence measured per arch (`out/cuda-sm90-probe.log`); the tree's own sm_121a build still yields an `sm_121a` cubin (`out/cuda-sm89-objectset.log`); the sm_89 object set builds with the Makefile's own flags (`out/cuda-sm89-build.log`, exit 0); the three device tests that compile the edited headers directly print `GLMVQ_FLASH_PASS`, `GLMVQ_Q4P_PASS`, `GLMVQ_VERIFY_PASS` on this card (`out/cuda-sm89-device-tests.log`); and `make ds4 CUDA_ARCH=sm_89` links, 49,388,088 B with ten `sm_89` cubins embedded, `./ds4 --help` exit 0 (`out/cuda-sm89-link.log`). X.9b, same day: the arch is now selected by the tree itself rather than by a flag on every invocation — `make cuda CUDA_ARCH=sm_89` records `CUDA_ARCH := sm_89` in the gitignored `.ds4-cuda-config.mk` (replacing `native`, which resolves to the same card here but does not name it), so a rebuild with no arch anywhere on the command line compiles `-arch=sm_89` and yields `ds4_cuda.1.sm_89.cubin`; and the card is named in the tree's tracked files (`README.md:91` and the Makefile help block) so a reader finds it without this project's docs. The alternative was measured and is not available: one object covering both this card and the Spark fails at ptxas, because the v0.5 mxf4 block (`ds4_cuda.cu:21024-21358`) carries no `__CUDA_ARCH__` guard (`.kind::mxf4` not supported on `sm_89`, exit 255; `out/cuda-multiarch-probe.log`). Building the whole target list from the record then surfaced a regression of our own: this working copy could not build its server at all, because `hidden_values` (`crates/ds4-server/src/worker_run.rs:16`) never received the `GemmaVq` arm its CLI twin got in `467af3a4` (E0004, `make cuda` exit 2 while the CUDA objects were already green); the arm is added exactly as the twin has it, a workspace-wide sweep finds no other stale arm (three `cargo check` configurations, zero errors), and the full list now builds, exit 0, with `ds4`, `ds4-server`, `ds4-bench` and `ds4-agent` each carrying ten `sm_89` cubins and running their `--help`. Rollback: `out/cuda-objects-backup-sm121a/` for the objects, and one `make cuda-spark` / `make cuda-generic` for the arch record. Evidence: `docs/cuda-sm89-port.txt` |

## 9. Order of work and the first gate

The container decision is in (X.6), so the order is now fixed:

Sequence: X.8 (DONE 2026-10-19: the `gemma_vq` payload contract is
`docs/ds4-gemma-payload-contract.md`, and it settled both questions — the payloads
are DQVQ byte for byte, so X.5b is a re-wrap) → X.5a (DONE 2026-10-19: the baseline
GGUF exists, gate 1a passed against the port's own reader and against llama.cpp at
the pin, which loads it and prints a 768-dim embedding) → the external anchor run
(owner-approved; DONE 2026-10-19: llama.cpp at the pin over the frozen 113-item set,
its vectors committed as `tests/expected/anchor-embeddinggemma2.jsonl`, bit-reproducible;
the same slice corrected three tokenizer keys in the baseline and settled the eos
and prompt decisions by measurement — `docs/anchor-gate2.txt`) → X.5b (DONE
2026-10-19: the artifact GGUF is a re-wrap of the container, which X.8 measured;
`out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf`, sha256 `5bcf2216…`,
gate 1b passed — `docs/artifact-gguf-gate1b.txt`) → P0 (DONE 2026-10-19: the
loader, identity and registration; `IDENTIFY … family=16`, `META ok`, `BIND ok
slots=1376`, `LAYOUT ok specs=1376`, gate 3 passed with 8 tampered copies refused —
`docs/loader-gate3.txt`) → P1 (DONE 2026-10-19, unit 14: the whole text tower on
the CPU reference path plus the tokenizer, gated against the frozen anchor —
`docs/text-tower-gate4.txt`) →
P1 gate → X.7 (DONE 2026-10-19, unit 15: the codec's cost measured on BEIR SciFact's
test split, both legs the same CPU reference tower over one ids file — at 768 the
vectors agree at mean cosine 0.956039571 and the rankings move by MRR@10 0.951778 to
0.935819 with recall@10 0.983333 to 0.990000; `docs/retrieval-cost-gate5.txt`) → P2
(DONE 2026-10-19, unit 19: the family refuses generation by name instead of crashing,
`ds4_engine_embed_gemma_vq` carries the device plan once per engine, and the `--embed`
verb and `POST /v1/embeddings` route serve it; note `gemma-embedding2`'s own
per-layer mask rule — the file's `attention.sliding_window` is 1024, so the
symmetric mask spans ±512 — `docs/eg2vq-embed-entry-gate6.txt`) → X.9 (DONE 2026-10-19, unit 16: the CUDA object set now
builds for this card's sm_89, the tree's device tests pass on it, the arch is recorded
so later builds select this card by themselves, and the whole target list builds — after
the server's missing `GemmaVq` arm was found and fixed —
`docs/cuda-sm89-port.txt`) → P3 (P3.1 DONE 2026-10-19, unit 17: the dim-4 VQ device
GEMM, `ds4-rs/cuda/eg2vq_vq.cuh`, rel RMS 4.0e-07 to 8.2e-07 against FP64 at all six
tower geometries and 3.1 TFLOP/s single-kernel; P3.2 DONE, unit 17: the symmetric-mask
attention kernel, `ds4-rs/cuda/eg2vq_attn.cuh`, rel RMS 5.8e-07 to 1.8e-06 with the mask
edge measured at 512/513 keys; P3.3 DONE, unit 18: the tower end to end on the device
at min cos(device, cpu) 0.999999950 over 300 items, 1494.8 to 2955.4 tokens/s, 414.0
MiB and 9.9x/18.1x against an -O3 CPU leg —
`docs/eg2vq-device-gate7.txt`) → P4 (owner's
call).

X.8 comes first because both the writer and the reader are written against the
contract it extracts, and because that is the difference between a design taken
from the reference and one invented: the two VQ families already shipping in ds4
carry their payload in a GGUF namespace, and project rule 0.2 says to mirror that
rather than approximate it. The anchor sits directly after the baseline GGUF
because that is the first moment it can run, and because freezing the denominator
before any port code exists is what keeps the later comparison a measurement
rather than a chasing of two moving numbers. P1's gate depends on P0 for something
to run, and on the frozen anchor vectors for its target.

First gate to run, concretely: `./bin/eg2_pack_cpu check
out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin` (the encoder's verdict,
re-run this session) against the new ds4-rs dump of the same 1376 slots — equal
line for line, including names, hashes, kinds, dims, `nc`, row/col geometry and
payload lengths.

## 10. Open owner decisions carried by this doc

1. DECIDED (owner, 2026-10-19): the container becomes a GGUF with a `gemma_vq`
   payload, as `glm_vq` and `ling31_vq` do (X.6), and the external anchor run is
   wanted (X.5), to be made once the baseline GGUF exists. The re-encode the
   decision costs is measured at about 14.5 minutes of shard wall time.
2. DECIDED (owner, 2026-10-19, "we need to get cuda support, it should be much
   faster"): the device path was taken in scope immediately, and the dim-4 kernel
   landed (P3.1-P3.3, units 17-18). A dim-8 sibling artifact remains an option on
   quality grounds alone — dim 4 / K=4096 is the higher-quality tier and dim 8 buys
   size, not quality (RECIPE.md section 5).
3. Whether P4 (vision and audio towers) is wanted at all.
4. Whether key-value checkpoints are in scope for an embeddings family (X.3).
5. NEW (2026-10-19, after X.7): what the codec's task-level cost is worth. The
   measurement says the VQ container keeps recall@10 (0.983333 to 0.990000 at 768)
   while loosening the very top of the ranking (MRR@10 0.951778 to 0.935819, recall@1
   0.888278 to 0.861611), and that at 128 dims truncation and quantization compound
   (-0.029414 MRR). No threshold was declared, so the decision whether that trade is
   acceptable for the intended use is the owner's, and it is now made on labelled
   data (`docs/retrieval-cost-gate5.txt`).
