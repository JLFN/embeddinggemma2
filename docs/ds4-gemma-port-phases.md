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
(2 × 10240, 31,457,280 B), and the text tower's VQ tier is dim 4 while the only
device VQ kernel in ds4-rs today is dim 8 (row P3.1).

Container layout and the validation contract, both read from the encoder that
wrote it (`tools/eg2_pack.c`): header 64 B at `:51-88` (documented) and `:221-225`
(the constants `DQVD_MAGIC`/`DQVD_VERSION`/`DQVD_HDR_BYTES`/`DQVD_SLOT_BYTES`/
`DQVD_ALIGN`); plan offsets `:666-690`; the checker `:980-1077`, whose rules are
the ones row P0.2 must reproduce.

## 3. P0 — container, identity, registration

Gate for the phase: a ds4-rs dump of all 1376 slots equals the encoder's own
verify output (`bin/eg2_pack_cpu check` plus the per-slot re-encode comparison in
`docs/artifact-verify-summary.txt`, 1376 slots across 14 shards).

| id | Item to port | Reference | Lands in | Status |
|---|---|---|---|---|
| P0.1 | DQVD v1 reader: 64 B header, recipe string, NUL-terminated name blob, 48 B slot table, 16 B-aligned payload region; fields in the order `tools/eg2_pack.c:51-88` documents | `tools/eg2_pack.c:51-88`, `:221-225`, plan writer `:666-724` | new `crates/ds4-core/src/eg2vq.rs`; shape of the reader follows `crates/ds4-core/src/ling31_vq/release.rs` (single-file variant) | OPEN |
| P0.2 | Validation, exactly the reference's rules: magic `:986`, version `:987`, container kind `:990`; file size == `payload_off + payload_bytes` `:1007`; per slot name range `:1030`, `strlen` = `name_len` and `fnv1a64` = `name_hash` `:1032`, payload offsets monotone `:1033`, inside the region `:1034`, 16-aligned `:1035`, container kind ≤ 3 `:1037`, VQ metadata usable and `payload_len == vq_payload_bytes(rows, cols, dim, nc)` `:1040-1046`, q4_K/q8_0 `cols % block == 0` and `row_size × rows == payload_len` `:1047-1057`, verbatim dtype known and elements ≥ rows × cols `:1059-1068`, names strictly ascending `:1070`; region not padded past its end `:1072` | `tools/eg2_pack.c:980-1077`, `:302` (`fnv1a64`), payload accounts `tools/eg2_pack.c:18-19` → `gguf-tools/quantize/vq_qc_bytes.h:15` (`vq_idx_bytes`), `:20` (`vq_payload_bytes`) | same file; block/row size via the existing host helpers used at `tools/eg2_pack.c:1049-1051` | OPEN |
| P0.3 | Container error surface: a per-family variant on the host error type plus its `token()` arm, following the two VQ families already there | template `crates/ds4-core/src/tensors.rs` (`TensorError`, its `GlmVq`/`Ling31Vq` variants and tokens), `fail()` helper convention `crates/ds4-core/src/glm_vq.rs:82`, `crates/ds4-core/src/ling31_vq.rs:168` | `crates/ds4-core/src/tensors.rs` + `eg2vq.rs` | OPEN |
| P0.4 | Family ids: `ModelFamily` value and `Variant` value, the oracle-name parser and the `as_str` arm | last values in use are `Glm53Vq = 14` / `Ling31Vq = 15` (`crates/ds4-core/src/shape.rs:35`) and `Ling31FlashVq = 18` (`:104`); name arms at `:49` and `:71` | `crates/ds4-core/src/shape.rs` | OPEN |
| P0.5 | File identity: a file (not a directory) carrying the DQVD magic routes to the new family; today `identify_gguf` hands every directory to a release reader and every file to the GGUF parser | routing decision `crates/ds4-core/src/identify.rs:200-205`; `Identified` `:54`; family-specific gate at `:183` | `crates/ds4-core/src/identify.rs` | OPEN |
| P0.6 | Layout + bind dispatch: the family's `layouts()` and its bind names | `crates/ds4-core/src/layout.rs:3791`, `crates/ds4-core/src/bind.rs:993`; `LayoutSpec` `crates/ds4-core/src/layout.rs:119` | `crates/ds4-core/src/layout.rs`, `bind.rs`, plus `crates/ds4-core/src/eg2vq/layout.rs` if it mirrors the Ling-3.1 split | OPEN |
| P0.7 | Inventory: DQVD slots become `TensorInfo` rows (name, dim, typ, rel/abs offset, elements, bytes) over one file-backed shard plan | `crates/ds4-core/src/tensors.rs` (`TensorInfo`, `ShardPlan`, `TensorInventory` at `:153-180`); `ling31_vq/release.rs:139` fills the same types from `release.json` | `crates/ds4-core/src/eg2vq.rs` | OPEN |
| P0.8 | Host wiring: the family's arms in the model/host open path | the `Glm53Vq` arms at `crates/ds4-core/src/lib.rs:1248`, `:1274`, `:1528`, `:2405`, `:2982`; payload layout `crates/ds4-core/src/payload.rs:78`, `:99` (only if checkpoints are in scope — see X.3) | `crates/ds4-core/src/lib.rs`, `payload.rs` | OPEN |
| P0.9 | C identity: family enum value, shape catalog entry, and the `weights_bind` arm | `ds4.c:382-383` (enum), `ds4.c:657-660` (shape catalog shape), `ds4.c:10240-10250` (bind dispatch: `glmvq_bind` / `ling31_bind`) | `ds4.c` | OPEN |
| P0.10 | C bind table: the flat record table the Rust reader hands to C, one `.inc` per family | `ds4_glmvq_bind.inc` (201 lines) as the pattern; header `ds4_glmvq.h` (179) | new `ds4_eg2vq_bind.inc`, `ds4_eg2vq.h` | OPEN |
| P0.11 | Host decode of a VQ payload for the CPU reference path: the dim-generic decoder already exists and takes the payload tier's dim/nc as arguments | `ds41_vq_fmt.h:72-104` (`ds4vq_dequant_f32`, dim- and width-generic, index width follows `nc`); the Rust-side decoder is v3/dim-8 specific (`crates/ds4-core/src/vq.rs`, `VQ_DIM` = 8) | C side: reuse `ds41_vq_fmt.h`; Rust side: only if a Rust decode path is needed | OPEN |

Footprint for sizing, measured this session: 35 files in ds4-rs mention the VQ
family identity (154,585 lines in total across them), and the VQ-specific core is
`ds4_glmvq_bind.inc` 201 + `ds4_glmvq_graph.inc` 391 + `ds4_glmvq_mtp.inc` 98 +
`ds4_glmvq.h` 179 + `ds4_glmvq_gpu.cuh` 2063 + `ds41_vq_fmt.h` 106 +
`cuda/ds41_vq_row.cuh` 267 + `cuda/ds41_vq_decode.cuh` 390 lines. An earlier
figure of "7,577 lines across 24 files" came from a wider file set that was not
re-enumerated here; treat it as unverified and use the measurement above.

## 4. P1 — the text tower on the CPU reference path

Every line below is `src/models/gemma-embedding2.cpp` at `b11723` unless another
file is named. Gate: on a frozen set of at least 100 sentences, our 768-dim
vectors agree with the reference instrument's — mean and minimum cosine, maximum
element difference — and the same three numbers under MRL truncation to 512, 256
and 128 with re-normalization (the checkpoint's own README documents the
truncation and the required re-normalization: `.models/EmbeddingGemma-2-914f7f8/README.md:91`,
`:166-185`).

| id | Item to port | Reference | Status |
|---|---|---|---|
| P1.1 | Embedding lookup scaled by `sqrt(n_embd)` = `sqrt(512)` (not the usual 1.0) | `:86` | OPEN |
| P1.2 | Position input and no-cache attention inputs; the whole tower is non-causal | `:90`, `:92` (metadata `causal_attn = false` at `:7`) | OPEN |
| P1.3 | Per-layer input: project with `per_layer_model_proj`, scale by `1/sqrt(n_embd)`, reshape to `[n_embd_per_layer, n_layer, n_tokens]`, RMS with `per_layer_proj_norm`, permute to `[n_embd_per_layer, n_tokens, n_layer]`; there is no per-layer token embedding | `:97`, `:215-232` (comment `:216-217`) | OPEN |
| P1.4 | Per layer: `attn_norm` RMS with weight | `:109` | OPEN |
| P1.5 | q/k/v projections and the head reshape | `:113-116`, `:121-123` | OPEN |
| P1.6 | Weighted RMS on Q and K (`q_norm`, `k_norm` of width `n_embd_head`), weightless RMS on V | `:125-127` (no V-norm weight exists in our inventory, which matches) | OPEN |
| P1.7 | RoPE per layer with that layer's own base (SWA layers 1e4, the four full layers 1e6) and `n_rot(il)` | `:104`, `:132-137`; base selection `src/llama-model.cpp:2748-2750`; which layers are SWA `src/llama-hparams.cpp:8-20` (`il % 6 < 5`) | OPEN |
| P1.8 | Attention with scale 1.0 — `q_norm` makes the `1/sqrt(head_dim)` scaling unnecessary; sliding window 512 on the twenty sliding layers, full attention on 5/11/17/23 | `:139-141` (scale), `:8` (scale value and reason), window key `:11` | OPEN |
| P1.9 | `attn_post_norm` RMS with weight, then residual onto the layer input | `:149-152` | OPEN |
| P1.10 | `ffn_norm` + gated FFN: `gelu(gate_proj(x)) * up_proj(x)`, then `down_proj`; the GELU is the tanh form | `:156-165` (`LLM_FFN_GELU` + `LLM_FFN_PAR` → `src/llama-graph.cpp:1995-2000` → `ggml_geglu_split` `ggml/src/ggml.c:3061`); formula `ggml/src/ggml-cpu/ops.cpp:3316` + `ggml_vec_geglu_f32` `ggml/src/ggml-cpu/vec.h:1462` (`gelu(x) * g`); tanh constants `ggml/src/ggml-cpu/vec.h:963-969` | OPEN |
| P1.11 | `ffn_post_norm` RMS with weight, then residual onto the attention output | `:167-170` | OPEN |
| P1.12 | Per-layer embedding block: `gelu(per_layer_inp_gate(x))` (a plain GELU, no up-projection and no `1/sqrt(2)` factor) × this layer's slice of the per-layer input, then `per_layer_proj`, `per_layer_post_norm`, residual | `:172-193` | OPEN |
| P1.13 | `layer_scalar` multiply, one learned scalar per layer | `:195-196` | OPEN |
| P1.14 | Control-vector hook: a no-op without a cvec file — implement nothing, record the decision | `:199` | OPEN |
| P1.15 | Output: `output_norm` RMS, then the embedding projection to 768 (projecting per token ≡ projecting after pooling) | `:206-211` | OPEN |
| P1.16 | Mean pooling over tokens | `src/llama-graph.cpp:4049-4053`; `1_Pooling/config.json` (`pooling_mode: mean`, `include_prompt: true`) | OPEN |
| P1.17 | L2 normalization — the graph does not do it, the application does | `examples/embedding/embedding.cpp:70` (`common_embd_normalize`); `2_Normalize/` and `modules.json` in the checkpoint | OPEN |
| P1.18 | Task prompts as data, taken from the checkpoint (20 named prompts) and defaulted per task | `.models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json` (read fully this session) | OPEN |
| P1.19 | The BPE tokenizer (262,144 entries, byte fallback, merges, normalizer, Split pre-tokenizer, `▁`→space decoder) | ties to `tokenizer.json` / `tokenizer_config.json` / `tokenizer.model` in the checkpoint; ds4-rs pattern `crates/ds4-core/src/tok/ling31.rs` | OPEN |
| P1.20 | The layer split itself, as a self-check on load: 24 layers, hidden 512, 4 heads, 2 KV heads, head_dim 256; layers 5/11/17/23 with head_dim 512 and 1 KV head; intermediate 2048; rms eps 1e-6 | `.models/EmbeddingGemma-2-914f7f8/config.json` (`layer_types`, `per_layer_config`), agreeing with P1.7's SWA rule | OPEN |

## 5. P2 — the entry point

Gate: the route returns the same vector as the CLI, byte for byte, for the same
input and prompt.

| id | Item to port | Reference | Status |
|---|---|---|---|
| P2.1 | A CLI verb that runs the tower and prints the vector (text in, 768 floats out) | the flag-based CLI in `crates/ds4-cli/src/lib.rs` (`parse_args` `:184`); family arms today at `:451`, `:491` | OPEN |
| P2.2 | The embeddings route: an endpoint that is not there today | the router's existing GET arms `crates/ds4-server/src/serve.rs:1169` (`/v1/models`), `:1183` (`/v1/models/<id>`), `:1212` (`/v1/stats`); measured this session: no `/v1/embeddings` and no pooling anywhere in `crates/` | OPEN |
| P2.3 | Serving capability and native-limit arms for the new family | `crates/ds4-core/src/serving.rs:715` (serving caps), `crates/ds4-core/src/serving_host.rs:1111` (native max), and the family-conditional sites `:654`, `:1179`, `:1647-1712` | OPEN |
| P2.4 | Prompt selection and the request shape (task prompt applied to the input, mean pool, L2 normalize; MRL truncation on request with re-normalization) | P1.16-P1.18 rows; the checkpoint's README for the truncation contract | OPEN |

## 6. P3 — the device path

Gate: GPU vectors equal CPU vectors within a stated tolerance, plus a latency and
a VRAM number, both printed by the run.

| id | Item to port | Reference | Status |
|---|---|---|---|
| P3.1 | A dim-4 VQ device kernel. The existing device open rejects every dim but 8, and our whole text-tower VQ tier is dim 4 | `cuda/ds41_vq_row.cuh:55` (`d16 != 8u` → open fails); Rust mirror `crates/ds4-core/src/vq.rs` (`VQ_DIM = 8`); the tier measured in section 2 (217 slots, 4x4096) | OPEN |
| P3.2 | Bidirectional attention with a sliding window. Correction to an earlier claim in this project's handoff: the kernel does implement both a window and non-causal attention — the window arrives as a parameter and is applied as `first = window && visible > window ? visible - window : 0`, and `causal = false` makes every query see all rows. What makes it unusable for traffic is that it is a scalar one-thread-per-(query, head) kernel with `rows > 256` refused and a `float score[256]` stack buffer | kernel `ds4_cuda.cu:42201`, window/causal at `:42205` and `:42217-42219`, row cap `:42208`, score buffer `:42211`; public API with a window `ds4_gpu.h:3658` (`ds4_gpu_motif3_expanded_attention_window_tensor`), without `:3645`; a caller already passes a window `ds4.c:48259` | OPEN |
| P3.3 | The tower on the GPU end to end, with the VQ payloads consumed in-kernel (never expanded to dense), plus the VRAM and latency measurement | the GLM VQ device pattern: `ds4_glmvq_gpu.cuh` (2063 lines, the GVQ1 execution adapter) over the `cuda/ds41_vq_*.cuh` kernel family (row-level decode `ds41_vq_row.cuh`, `ds41_vq_decode.cuh`, prefill variants) | OPEN |

## 7. P4 — vision and audio towers (owner decision pending)

The container already holds both towers (section 2); nothing in this phase is
needed for the text embeddings to work. Gate: the same instrument comparison as
P1, per tower, on the tower's own inputs.

| id | Item to port | Reference | Status |
|---|---|---|---|
| P4.1 | Vision tower forward: 16 layers, hidden 768, head_dim 64, 12 heads, patch 16, axial RoPE theta 100 | tower graphs live in the mtmd half of the reference: `tools/mtmd/models/gemma4v.cpp` (`PROJECTOR_TYPE_GEMMA4V` wired at `tools/mtmd/clip.cpp:973-979`, `:480`); shape from our `config.json` | OPEN |
| P4.2 | Audio tower forward: 12 layers, hidden 1024 | `tools/mtmd/models/gemma4a.cpp` (`PROJECTOR_TYPE_GEMMA4A`, `tools/mtmd/clip.cpp:1115-1121`); unified variants `gemma4uv.cpp` / `gemma4ua.cpp` | OPEN |
| P4.3 | The tie-in projections `embed_vision.embedding_projection.weight` (768 × 512) and `embed_audio.embedding_projection.weight` (1536 × 512), both verbatim BF16 in our container | section 2; conversions `conversion/gemma.py:1057` (`EmbeddingGemma2VisionAudioModel`, "same towers as Gemma4, but the tensor names have no `model.` prefix") and its parent `:978` (`Gemma4VisionAudioModel`) | OPEN |
| P4.4 | Which projector variant our model is: the reference has four (V/UV/A/UA) and the conversion class does not state which one this checkpoint maps to | `tools/mtmd/clip.cpp:973-979`, `:1115-1121`; check the conversion's own projector-type assignment before porting | OPEN |

## 8. X — cross-cutting rows

| id | Item | Reference | Status |
|---|---|---|---|
| X.1 | The measuring instrument: an ONNX execution path in Rust (`ort` crates, already in the local cargo cache), producing frozen reference vectors. Four checks before any number from it is a denominator: pick the artifact and record its revision, dump its graph I/O, check the conversion against the pinned reference, then freeze | `docs/ds4-gemma-port-reference.md` section 6 (the four checks listed there) | OPEN |
| X.2 | Guardrails in force for every row: no Python in the runtime (P-1, with its four zero-hit greps) and references are citations or one-shot instruments, never dependencies (P-2) | `/data/embeddinggemma2/AGENTS.md` | DONE (rule written, commit `fd63233`; gate greps return zero) |
| X.3 | Decide whether key-value checkpoints matter for an embeddings model at all before wiring `payload.rs` (row P0.8). An embeddings route has no decode loop, so a checkpoint format may be pure cost | `crates/ds4-core/src/payload.rs` (the GLM/Ling VQ arms) | OPEN |
| X.4 | Before any push from ds4-rs: rule 12 graphify graph, rule 19 QA evidence, rule 10 OSV scan (ds4-rs carries a `Cargo.lock`); ds4-rs's remotes are shared with the live ds4 project, so a push is shared state and needs confirmation | `~/.opengrok/project-work-rules.md` rules 10/12/19/23 | OPEN |

## 9. Order of work and the first gate

X.1 (the instrument) and P0 (the container) can run in parallel and both precede
P1's gate: without X.1 the quality gate has no denominator, and without P0 there
is nothing to run.

Sequence: P0.1-P0.3 → P0.4-P0.10 → P0 gate → X.1's four checks → P1 → P1 gate →
P2 → P3 → P4 (owner's call).

First gate to run, concretely: `./bin/eg2_pack_cpu check
out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin` (the encoder's verdict,
re-run this session) against the new ds4-rs dump of the same 1376 slots — equal
line for line, including names, hashes, kinds, dims, `nc`, row/col geometry and
payload lengths.

## 10. Open owner decisions carried by this doc

1. Which ONNX artifact becomes the instrument (X.1 item 1) — and whether the
   export includes the per-layer input, the embedding projection, pooling and
   normalization or only the encoder body.
2. Whether the dim-4 device kernel (P3.1) is in scope now, or the device path
   waits for a dim-8 sibling artifact. Dim 4 / K=4096 is the higher-quality tier.
3. Whether P4 (vision and audio towers) is wanted at all.
4. Whether key-value checkpoints are in scope for an embeddings family (X.3).
