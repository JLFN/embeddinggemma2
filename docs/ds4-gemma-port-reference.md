# EmbeddingGemma-2 → ds4-rs: the porting reference and the name contract

Unit 5 evidence, 2026-10-19. This file is the port map: what is authority, what is
the measuring instrument, and exactly where every piece of the forward pass comes
from. It exists so the port can be written and audited without re-deriving any of
it, and so no step has to be invented (global rule 0.2, project rule P-2).

## 1. Method

Three roles, never mixed:

- Authority for the mathematics: the pinned upstream implementation, read
  first-hand and cited by file and line. For this model that is llama.cpp's
  `gemma-embedding2` (section 2) — a full reference implementation of exactly
  this architecture, in C++, which also means the port's arithmetic can be
  compared line for line rather than translated from another language's
  conventions.
- Port shape: the ds4-rs family pattern (Rust host, C bind `.inc`, C graph
  `.inc`, CUDA kernels), per the GLM VQ family as the closest analog.
- Verification: the ds4-dfm-rs method itself, with no foreign instrument. The
  candidate (Rust host, CUDA kernels) is checked against a native oracle — the
  port's own C reference path, in the ds4 idiom of the retained `-c` behavior
  oracles and their parity harnesses — with frozen numbers committed as golden
  snapshots, and the pinned llama.cpp run once (C++, in GGUF) as an optional
  external anchor (section 6). No ONNX, no PyTorch, and no Python in the runtime
  at any point (owner decision 2026-10-19; project rule P-1).

## 2. Pinned references

| Tree | Identity | Revision | Role |
|---|---|---|---|
| `/data/llama.cpp` | upstream `ggml-org/llama.cpp` (MIT) | `b11723`, commit `31288c9c66e59377f7788d920678a565bb89dd37`, 2026-10-18 | the reference implementation for this model |
| `/data/llama-prism` | `PrismML-Eng/llama.cpp` fork (MIT) | `b10743`, commit `adfffbe41b2cabcd51fff326ab045662265062bb`, 2026-09-25 | cross-check only: it predates `gemma-embedding2` and carries the older `gemma-embedding` (`src/models/gemma-embedding.cpp`, the 300m text-only model) |

Both trees are read-only inputs. Neither becomes a runtime dependency: llama.cpp
is not built, linked, or invoked by the port, and its kernels are not used — the
port reimplements the graph in the ds4 pattern and cites what it copied.

Files that matter in `/data/llama.cpp`:

| Path | What it holds |
|---|---|
| `src/models/gemma-embedding2.cpp` | the whole text-tower forward and its loader |
| `src/llama-arch.h:69`, `src/llama-arch.cpp:64` | the arch id `LLM_ARCH_GEMMA_EMBEDDING2` / name `"gemma-embedding2"` |
| `src/llama-hparams.cpp:8-20` | `set_swa_pattern`: which layers are sliding-window |
| `src/llama-model.cpp:3905-3912` | `load_swa_pattern` (metadata or default period) |
| `src/llama-model.cpp:2748-2750` | `get_rope_freq_base`: SWA layers use the SWA base, the rest the global base |
| `src/llama-graph.cpp:4049-4053` | mean pooling over tokens |
| `src/llama-graph.cpp:1995-2000` | `LLM_FFN_GELU` + `LLM_FFN_PAR` → `ggml_geglu_split` |
| `ggml/src/ggml-cpu/vec.h:963`, `ggml/src/ggml-cuda/unary.cuh:103` | `GELU_COEF_A = 0.044715` (ggml's GELU is the tanh form) |
| `examples/embedding/embedding.cpp:70` | `common_embd_normalize` — normalization is the caller's step, not the graph's |
| `conversion/gemma.py:813-816`, `:1055-1057` | the two conversion classes: text-only, and the vision+audio composite |
| `gguf-py/gguf/tensor_mapping.py` | HF ↔ GGUF name Rosetta stone (the `# embeddinggemma` comments mark our names) |

## 3. Shape contract (measured from our own checkpoint)

From `.models/EmbeddingGemma-2-914f7f8/config.json` (text_config):

- 24 layers, hidden 512, 4 query heads, 2 KV heads, head_dim 256, intermediate
  2048, vocab 262144, `max_position_embeddings` 262144, rms eps 1e-6.
- `per_layer_config` = layers 5, 11, 17, 23 with head_dim 512 and 1 KV head;
  `layer_types` marks exactly those four as `full_attention` and the other twenty
  as `sliding_attention` with `sliding_window` 512. This is the same split as
  llama.cpp's `set_swa_pattern(6, dense_first=false)`, where `is_swa(il)` is
  `il % 6 < 5` (`src/llama-hparams.cpp:8-20`): layers 5, 11, 17, 23 are the
  non-sliding ones. Two independent sources agree.
- `rope_parameters`: `full_attention` theta 1e6, `sliding_attention` theta 1e4 —
  the mapping to llama.cpp's `rope_freq_base_train` / `rope_freq_base_train_swa`
  is `get_rope_freq_base` (`src/llama-model.cpp:2748`).
- `hidden_activation` gelu_pytorch_tanh, which is the tanh approximation — the
  same function as ggml's `GELU_COEF_A` form (`ggml/src/ggml-cpu/vec.h:963`).
- `hidden_size_per_layer_input` 512, `embedding_dim` 768.
- Vision tower: 16 layers, hidden 768, head_dim 64, 12 heads, patch 16,
  `default_output_length` 280, rope type axial with theta 100.
- Audio tower: 12 layers, hidden 1024 (its tensors are 744 of the 1376 slots).

Name contract, verified against `docs/tensor-inventory.tsv` (our artifact stores
Hugging Face names, so the C bind maps these directly):

Per text layer, 17 tensors:

| HF name | shape | role (llama.cpp) |
|---|---|---|
| `input_layernorm.weight` | 512 | attn_norm |
| `self_attn.q_proj.weight` | 512x1024 | wq |
| `self_attn.k_proj.weight` | 512x512 | wk |
| `self_attn.v_proj.weight` | 512x512 | wv |
| `self_attn.o_proj.weight` | 1024x512 | wo |
| `self_attn.q_norm.weight` | 256 | attn_q_norm |
| `self_attn.k_norm.weight` | 256 | attn_k_norm |
| `post_attention_layernorm.weight` | 512 | attn_post_norm |
| `pre_feedforward_layernorm.weight` | 512 | ffn_norm |
| `mlp.gate_proj.weight` | 2048x512 | ffn_gate |
| `mlp.up_proj.weight` | 2048x512 | ffn_up |
| `mlp.down_proj.weight` | 512x2048 | ffn_down |
| `post_feedforward_layernorm.weight` | 512 | ffn_post_norm |
| `ple_block.per_layer_input_gate.weight` | 512x512 | per_layer_inp_gate |
| `ple_block.per_layer_projection.weight` | 512x512 | per_layer_proj |
| `ple_block.post_per_layer_input_norm.weight` | 512 | per_layer_post_norm |
| `layer_scalar` | 1 | out_scale |

There is no V-norm weight in the inventory, which matches the reference applying a
weightless RMS to V (`src/models/gemma-embedding2.cpp:127`). Global tensors:
`language_model.embed_tokens.weight` (262144x512),
`language_model.norm.weight` (512), `language_model.embedding_projection.weight`
(768x512), `language_model.ple.per_layer_model_projection.weight` (12288x512),
`language_model.ple.per_layer_projection_norm.weight` (512), plus
`embed_vision.embedding_projection.weight` (768x512) and
`embed_audio.embedding_projection.weight` (1536x512) for the tower side.

## 4. The forward pass, cited

Every line below is `src/models/gemma-embedding2.cpp` unless stated otherwise.

Metadata (`:1-19`): `swa_type = SYMMETRIC`, pattern default 6, `causal_attn =
false` ("embeddings do not use causal attention"), `f_attention_scale = 1.0` with
the stated reason "same as Gemma4, q_norm makes the scaling unnecessary",
`rope_freq_base_swa`, `n_swa`, `f_norm_rms_eps`, `n_embd_per_layer` (`:12`).
Note this differs from the older 300m implementation
(`/data/llama-prism/src/models/gemma-embedding.cpp:28`), which scales by
`1/sqrt(head_dim)` — the two models are not the same graph, and using the older
file for this model would silently change every attention score.

Tensors (`:21-72`): token embedding; the per-layer projection
`{n_embd, n_embd_per_layer * n_layer}` and its norm (`:36-37`); output norm and
the embedding projection `{n_embd, n_embd_out}` (`:39-41`, the comment calls it
"projects the final hidden state to the embedding dimension"); per layer, the 17
tensors above with per-layer head geometry read from `hparams.n_head(i)` /
`n_embd_head_k(i)` / `n_embd_k_gqa(i)` (`:43-48`).

Per-layer input (`:94-97`, `:215-232`): project the (already scaled) input with
`per_layer_model_proj`, multiply by `1/sqrt(n_embd)` (`:223`), reshape to
`[n_embd_per_layer, n_layer, n_tokens]` (`:224`), RMS norm with
`per_layer_proj_norm` (`:226`), then permute to `[n_embd_per_layer, n_tokens,
n_layer]` (`:229-230`). The comment at `:216-217` states this model has no
per-layer token embedding; the per-layer input comes only from this projection.

Input scaling (`:86`): `build_inp_embd(model.tok_embd, sqrtf(n_embd))` — the
token embedding is multiplied by `sqrt(512)`.

Per layer (`:99-201`):

1. `attn_norm` RMS with weight, on `inpL` (`:109`).
2. q, k, v projections (`:113-115`), reshape to heads (`:121-123`).
3. RMS with weight on Q and K (`:125-126`); weightless RMS on V (`:127`).
4. RoPE on Q and K (`:133-136`) with the per-layer base (`:104`),
   `n_rot_l = hparams.n_rot(il)` (`:107`).
5. Attention with `hparams.f_attention_scale` = 1.0 (`:139-141`), no KV cache
   (`build_attn_inp_no_cache`, `:92`), non-causal, sliding window on the twenty
   sliding layers.
6. `attn_post_norm` RMS with weight, then residual add with `inpL` (`:149-152`).
7. `ffn_norm` RMS with weight (`:155`), then the FFN (`:159-165`) with
   `LLM_FFN_GELU` and `LLM_FFN_PAR`, which is `ggml_geglu_split`
   (`src/llama-graph.cpp:1995-2000`): `gelu(gate_proj(x)) * up_proj(x)`, then
   `down_proj`. The GELU is the tanh form (`GELU_COEF_A`, `ggml/src/ggml-cpu/vec.h:963`).
8. `ffn_post_norm` RMS with weight, then residual add with the attention output
   (`:168-170`).
9. Per-layer embedding block (`:172-191`): `gelu(per_layer_inp_gate(x))` (`:178`)
   multiplied elementwise by this layer's slice of the per-layer input (`:180-184`)
   — a plain `ggml_gelu` with no `1/sqrt(2)` factor and no up-projection, unlike
   the FFN's geglu; then `per_layer_proj` (`:188`), `per_layer_post_norm` (`:189`),
   and a residual add onto the pre-block value (`:191`).
10. `layer_scalar` multiply (`:194-195`) — one learned scalar per layer.
11. `build_cvec` (`:199`), llama.cpp's control-vector hook: a no-op unless a cvec
    file was supplied, so the port has no equivalent to implement.

End (`:207-211`): `output_norm` RMS, then the embedding projection to 768
(`:211`). The comment at `:210` records that projecting per token is equivalent to
projecting after mean pooling, which is why the graph emits per-token 768-dim
vectors and the pooling happens afterwards.

Pooling and normalization (outside this file): mean pooling is
`ggml_mul_mat(transpose(inp), inp_mean)` over the token axis
(`src/llama-graph.cpp:4049-4053`), and L2 normalization is the application's step
(`examples/embedding/embedding.cpp:70`, `common_embd_normalize`). Our route must
therefore do mean pool then L2 normalize itself; `1_Pooling/config.json` says
`pooling_mode: mean` with `include_prompt: true`, and `modules.json` has exactly
Transformer → Pooling → Normalize, so there is no Dense module to apply
(llama.cpp's optional `dense_2`/`dense_3` stay absent, which its loader already
treats as `TENSOR_NOT_REQUIRED`).

Prompts are data, not graph: our checkpoint carries them in
`config_sentence_transformers.json` (`STS` and `SentenceSimilarity` →
`"task: sentence similarity | query: "`, `Retrieval`/`query` → `"task: search
result | query: "`, `document` → `"title: none | text: "`, and so on). llama.cpp
contains no such table, so the port's route must take the prompt explicitly and
default it from this file.

## 5. What the reference does not cover

- The vision and audio towers. Upstream converts the composite as
  `EmbeddingGemma2VisionAudioModel` (`conversion/gemma.py:1055-1057`), a subclass
  of the gemma4 vision+audio model, so the tower graphs live in
  `src/models/gemma4.cpp` and the sibling tower code, not in
  `gemma-embedding2.cpp`. Those are the P4 phase, and the `embed_vision` /
  `embed_audio` projections in our inventory are the tie-in tensors.
- Tokenization. The reference reads pretokenized ids; the port owns the BPE
  tokenizer (our `tokenizer.json`: 262144 entries, `byte_fallback: true`, merges
  present, a normalizer, a Split pre-tokenizer and a Replace `▁`→space decoder),
  implemented in the ds4 pattern (`crates/ds4-core/src/tok/ling31.rs` is the
  template: assert the whole pipeline shape, then build tables).

## 6. Verification: the ds4 method, and no foreign instrument

Owner decision, 2026-10-19: this port carries no ONNX and no PyTorch, in the
runtime or in its verification. The method is the one `/data/ds4-dfm-rs` already
uses, read out of that tree:

- The oracle is native code of the project's own. "C executables with `-c`
  suffixes are retained behavior oracles, not the production hosts"
  (`/data/ds4-dfm-rs/AGENTS.md`). The Rust host is the candidate, and each
  surface has a parity harness against the C oracle
  (`tests/parity/<surface>_c_oracle.c`, with Makefile targets
  `test-route-parity`, `test-catalog-parity`, `test-tokenizer-parity`,
  `test-kv-parity`, `test-server-parity`, `test-agent-parity`).
- A false green is a named failure, not an accident. `proof-rust-cuda-opp-c`
  refuses to run when candidate and oracle are the same binary or the same hash:
  "Same inode or same hash is a false-green (Rust-vs-Rust) and must die"
  (`/data/ds4-dfm-rs/Makefile`).
- Frozen numbers are committed snapshots, not a library call:
  `tests/proof/expected/<scenario>.json`, checked by `tests/ds4_proof.py
  --scenario <name> --check-expected <path>`; the report carries the same fields
  across scenarios so the same commit and artifact produce a byte-identical plan
  (`tests/ds4_proof.py:2007-2019`).
- Evidence is a committed JSON that names the artifact and its revision
  (`docs/evidence/`), and a release ledger records what was qualified
  (`docs/releases/`).

For this port that is four legs, all native.

Leg 1, the authority (read and cited). llama.cpp's `gemma-embedding2` at the
pinned revision (section 2), read first-hand and cited in the port's comments and
in the evidence file (global rule 0.2). This is where the arithmetic comes from,
including the GGUF key and tensor set the loader resolves:
`src/models/gemma-embedding2.cpp:3-13` — `attention.sliding_window`,
`attention.layer_norm_rms_epsilon`, `rope_freq_base_swa`,
`embedding_length_per_layer`, the SWA pattern through `load_swa_pattern(ml, 6)`,
non-causal attention and `f_attention_scale = 1.0f` — and `:34-71`, the 413
tensors: 5 globals plus 17 per layer. Arch string `src/llama-arch.cpp:64`,
loader wiring `src/llama-model.cpp:161`.

Leg 2, the oracle (run, native). The port's own C reference path is the oracle
and the Rust host is the candidate, exactly as `ds4-c` is to `ds4`. Same
container, same weights, same input: the 768-dim pooled vector, the per-tensor
values and their digests must agree. The false-green guard applies in our terms
too — distinct builds, or the comparison is void.

Leg 3, the golden snapshot (committed, reproducible). A frozen sentence set with
the checkpoint's own task prompts, run once through the C reference path, its
digests committed; every later run must reproduce them. MRL truncation is part of
the snapshot rather than an afterthought: 768/512/256/128, each re-normalized
after the slice, per the checkpoint README lines 166-185.

Leg 4, device parity (run). The CUDA path must match the C reference path within
a stated tolerance, and the VQ device path must reproduce the host decode of the
same payload. Latency and VRAM are recorded beside the equality, because a
correct kernel slower than the CPU path is not the deliverable.

External anchor, optional and by owner decision (phase doc X.5). llama.cpp
itself, built at the pinned revision and run once over the frozen set, is the
only independent implementation of this architecture on hand. It is C++ and needs
no Python, but it needs the model in GGUF, and its own converter is not usable:
`convert_hf_to_gguf.py:12` imports torch, which rule P-1 forbids. The anchor
route is therefore a native GGUF writer of our own, plus one run of the
reference's embedding example (`examples/embedding/embedding.cpp`, normalization
at `:70`), its output committed as data with the GGUF's sha256 and the build's
revision. The anchor is a second opinion, never the denominator: the denominator
is our own C reference path, checked against a cited implementation.

The anchor never becomes an execution dependency (owner directive 2026-10-20,
"We will not use llama it must be ported all"). Measured that day, both trees
touched: `ds4-rs` contains no llama call and links no llama library — every
`llama` occurrence in the engine is a citation in a comment or an identifier
name — so the runtime boundary was already clean, and what remained were two
default-path runs of the pin: gate 22's vision harness (the mmproj run and the
frozen-dump reproduction) and the tokenizer dump's regeneration branch. Both are
now opt-in and the default path is native end to end: gate 22 reads the mmproj
back with the Rust host's own GGUF reader (`gguf_inventory`), and the vision
anchor moved to `tools/eg2_vision_anchor.sh` with its source renamed
`tests/parity/eg2vq_vision_anchor.cpp`. The frozen numbers in
`tests/vision/reference/`, `tests/expected/` and `out/tok-anchor-verbose.log`
stay: they are data, produced once, and reproducing them natively is exactly
what gates 4 and 23 measure.

The container, decided (owner, 2026-10-19, phase doc X.6): the artifact becomes a
GGUF carrying a `gemma_vq` payload, as `glm_vq` and `ling31_vq` do — not a
container of our own, and DQVD is dropped. GGUF is ds4's native container, not a
foreign one. The C host parses it itself (`ds4.c:1607` carries `DS4_GGUF_MAGIC`,
the parser starts at `:2348`) and the Rust host opens it
(`crates/ds4-core/src/identify.rs:199-208`); the VQ families extend it with their
own metadata namespaces and custom tensor types rather than replacing it
(`glm_vq.format_version` and the `glm_vq.*` namespaces in
`crates/ds4-core/src/glm_vq.rs:276-466`, custom types bound in
`ds4_glmvq_bind.inc`). The ds4 VQ recipe our own `RECIPE-deepseek41-vq.md`
follows writes a `.gguf` too (`v41_to_gguf ... .gguf`), so DQVD was the deviation.
The same writer serves the baseline model (X.5), which is why the two are one work
item rather than two.

## 7. Consequences for the port

- The artifact is a GGUF (section 6, decided 2026-10-19), so the host's container
  work is a family payload rather than a new container. Rust validates the payload
  and binds a flat record table for C — the split `ds4_glmvq_bind.inc` states about
  itself in its opening comment: "Rust validates the complete GVQ1 release before
  this pointer adapter" (`ds4_glmvq_bind.inc:1-3`). Identity needs no new
  signature: the family's own metadata key carries it through the existing GGUF
  route (`crates/ds4-core/src/identify.rs:183-186` is the `glm_vq.format_version`
  gate). The contract to mirror is extracted first, in phase doc row X.8.
- The family id lands in `ModelFamily`/`Variant`
  (`crates/ds4-core/src/shape.rs:20-35`, `:84-104`), the five Rust dispatch
  tables, the C family enum (`ds4.c:383`), the catalog (`ds4.c:659`) and the
  `weights_bind` switch (`ds4.c:10190-10250`).
- The host decoder for our payload tier already exists and is dim-generic
  (`ds41_vq_fmt.h:72-104`), so a CPU reference forward is possible with today's
  code. The device VQ path is dim-8 only (`cuda/ds41_vq_row.cuh:55`,
  `crates/ds4-core/src/vq.rs:40-44`), so the GPU phase needs a dim-4 kernel.
- Bidirectional attention exists in the device API (`ds4_gpu.h:3645-3656`,
  kernel at `ds4_cuda.cu:42201-42230`, full attention when `causal=0`), but it is
  a scalar reference kernel capped at 256 rows per launch (`ds4_cuda.cu:42208`,
  score buffer `float score[256]` at `:42211`, guarded again at `:42358`), so it
  can prove correctness, not serve traffic. A windowed entry point also exists and
  is in use (`ds4_gpu_motif3_expanded_attention_window_tensor`, `ds4_gpu.h:3658`,
  called at `ds4.c:48259`), but its window is trailing: `first = window &&
  visible > window ? visible - window : 0` (`ds4_cuda.cu:42214`), and with
  `causal=0` `visible` is the whole row set, so every query would attend only to
  the last `window` tokens rather than to a window around itself. Twenty of our
  twenty-four layers need the symmetric form, so P3 is a new kernel, not a
  parameter (phase doc P3.2). Correction of 2026-10-19: an earlier draft of this
  bullet claimed no sliding window existed in that kernel, which the signature at
  `ds4_gpu.h:3658` and the use at `:42214` disprove.
- There is no embeddings route and no mean pooling anywhere in the tree
  (`crates/ds4-server/src/serve.rs:1169-1240` dispatches only models, metrics,
  stats and batch), so both are new work for the port.
