# ds4-gemma-payload-contract.md — the `gemma_vq` contract (phase 0, row X.8; extended by phase 1a, row X.5a, phase 2, row X.5, and phase 1b, row X.5b)

Derived 2026-10-19 in `/data/embeddinggemma2`, reading only: no reader, writer or
encoder code was written for this document. Phase 1a then wrote the baseline GGUF
against it and corrected four of its rows by measurement; phase 1b wrote the
artifact GGUF and corrected a sixth (section 4: the file's rank is never 0).
Section 7 states each correction and the evidence. It is subordinate to `embeddinggemma2-handoff.md` —
where the two disagree, the handoff wins and this file is corrected in the same
commit.

Every key, type, offset and magic below carries a `file:line` citation that was
opened and read in this session, per project rule 0.2. Sections 2 and 3 settle
the two determinations row X.8 owns, by measurement.

## 0. What the three families are, and what "family" means here

A ds4 VQ family is not a new file format. It is a set of conventions laid over
GGUF: an architecture string, one metadata key that makes the file
self-identifying, optional private tensor types, a recipe string, and a stated
split of work between the Rust host (validation) and the C/CUDA side (kernels).
Two families ship today and are the pattern for the third.

| part | `glm_vq` (GLM-5.3-Flash) | `ling31_vq` (Ling-3.1-flash) | `gemma_vq` takes |
| --- | --- | --- | --- |
| architecture string | `glm5_next` (`crates/ds4-core/src/glm_vq.rs:425`, `:436`) | `ling31_vq` (`crates/ds4-core/src/ling31_vq.rs:23`) | `gemma_vq` |
| self-identifying gate | `glm_vq.format_version` in 1..=2 (`crates/ds4-core/src/glm_vq.rs:22`; gate `crates/ds4-core/src/identify.rs:183-186`) | directory release `format: ling31-vq8` (`crates/ds4-core/src/ling31_vq/release.rs:139-146`), `FORMAT_VERSION = 1` (`ling31_vq.rs:33`) | `gemma_vq.format_version` = 1 (u32), checked in `identify.rs` |
| alignment | 128 (`glm_vq.rs:18`), enforced `glm_vq.rs:437` | 128 (`ling31_vq.rs:30-31`) | 32, the writer's default (`/data/YoungAi/src/common/ds4_gguf_write.h:37`) and the precedent's choice (`/data/YoungAi/gguf-tools/quantize/v41_to_gguf.c:361`) |
| private tensor types | FP8 `0x47514638`, CODEBOOK `0x47564342`, VQ `0x47565138`, RESIDUAL `0x47565238` (`glm_vq.rs:24-27`) | FP8/CODEBOOK/VQ, the same three (`ling31_vq.rs:34-36`) | none new: the payload tensor is ds4's existing type 42 |
| family metadata keys | `glm_vq.codebooks` (`glm_vq.rs:276`), `.fp8_scales` (`:324`), `.logical_shapes` (`:353`, `:366`), `.residuals` (`:381`), `.format_version` (`:435`), `.native_policy` (`:466`) | `ARCH`/`VARIANT`/`SOURCE_MODEL`/`RECIPE`/`RECIPE_V2` (`ling31_vq.rs:23-29`) | `general.architecture`, `general.alignment`, `gemma_vq.format_version`, `gemma_vq.recipe` |
| recipe | `deepseek4.recipe` string (`v41_to_gguf.c:363`) | `RECIPE` / `RECIPE_V2` (`ling31_vq.rs:26`, `:29`) | `gemma_vq.recipe` (str) |
| rank for tensors a payload cannot express | `glm_vq.logical_shapes`, a JSON object of name → shape (`glm_vq.rs:353-366`) | not needed | not needed (section 5, measured) |
| host/device split | "Rust validates the complete GVQ1 release before this pointer adapter" (`ds4_glmvq_bind.inc:1-3`) | same shape, one `.inc` and one header per family | same rule (section 6) |

The C side has a table of private GGUF type numbers, and only two of them are VQ:
42 `vqblob` (opaque bytes, 1 byte per element) and 43 `fp4x32`
(`/data/YoungAi/src/core/core_gguf.c:94-97`; the enum that names them,
`/data/YoungAi/src/common/ds4_quantfmt.h:29` and `:34`). `gemma_vq` uses 42 and
nothing else custom.

## 1. The payload: a DQVQ matrix payload, exactly as the reference writes it

Authority: `/data/YoungAi/vq_fmt.h`, the file the quantizer and the engine share;
our tree already carries a vendored copy at `ds4-rs/ds41_vq_fmt.h`, with the
layout documented at `:4-12` and the decoded value rule at the head of the file
(`vq_fmt.h:7`: `value = codebook[idx][d] * g_r[row]`).

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | magic `DQVQ` = `DS4VQ_MAT_MAGIC` `0x51565144` (`ds41_vq_fmt.h:32`) |
| 4 | 2 | `dim` — codebook vector length (u16 LE) |
| 6 | 2 | `nc` — codebook entry count (u16 LE) |
| 8 | 4 | `rows` (u32 LE) |
| 12 | 4 | `cols` (u32 LE) |
| 16 | `nc*dim*2` | codebook, `nc` vectors of `dim` f16 |
| 16 + `nc*dim*2` | `rows*2` | `g_r`, one f16 row gain |
| then | `idx_bytes` | index stream, `ceil(log2 nc)` bits per index |

Byte account, the single copy both the writer and the size allocator include
(`/data/YoungAi/gguf-tools/quantize/vq_qc_bytes.h:15` and `:20`):

    vq_payload_bytes(rows, cols, dim, nc) =
        16 + nc*dim*2 + rows*2 + vq_idx_bytes(rows, cols, dim, nc)
    vq_idx_bytes = nidx = rows*cols/dim, one byte each when the width is exactly
                   8 bits, else (nidx*bits + 7)/8 + 1 (bitstream LE plus one
                   guard byte against an over-read at the tail)

The decoder is dim- and width-generic — the index width is computed from `nc`,
not hardcoded (`ds41_vq_fmt.h:73-105`, `ds4vq_dequant_f32`). Two properties
matter for us:

- It refuses any payload whose magic is not `DQVQ` (`:75`, returns -1). The v3
  form carries a different magic (`DQV3` = `0x33565144`, `vq_fmt.h:18`) precisely
  so an older binary fails to recognize it instead of mis-decoding it
  (`vq_fmt.h:16-18`). Our payloads are v2 form, so the whitelist accepts them.
- For our tier (`dim` 4, `nc` 4096) the index width is 12 bits, computed by the
  same `while ((1 << nbit) < nc) nbit++` loop the writer uses (`ds41_vq_fmt.h:87`
  against `vq_qc.h:250`). The stale "dim4 -> 9bit" note in the file header
  describes the old hardcoded path, not this code.

  MEASURED 2026-10-19 (phase 3): 12 bits is exactly the codebook's address
  space, so at this tier **no out-of-range index can be encoded** — the mask is
  `0xFFF` and `nc` is 4096, so the index range and the codebook are the same
  size. A loader therefore cannot refuse an out-of-range index here, and the
  reachable tamper is a header whose geometry stops reproducing the declared
  length, which the byte account refuses. A future tier whose `nc` is not a power
  of two has slack above `nc - 1` and would need a scan; that is an open item,
  not a check that was skipped.

## 2. Determination 1 — our kind-0 payloads ARE DQVQ payloads, byte for byte

Settled by measurement, both at the source and on disk.

Source: the packer never writes a payload of its own. `emit_payload` allocates
`vq_payload_bytes(...)` (`tools/eg2_pack.c:513`) and fills it with the reference's
own `vq_pack(Cb, idx, gr, s->rows, s->cols, dim, nc, out)` (`tools/eg2_pack.c:514`,
writer at `/data/YoungAi/gguf-tools/quantize/vq_qc.h:241-267`). So the container's
kind-0 bytes are the reference payload by construction, and the DQVD checker's
independent rule for them is exactly the DQVQ byte account
(`tools/eg2_pack.c:1044`, `const size_t want = vq_payload_bytes(...)`).

On disk, measured this session on
`out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin`:

    container out/...-dqvd.bin: kind=0 slots=1376
    recipe: towers=vq4x4096  sharedcb=0 cbfp16 token=q8_0 frontend=vq other=keep nocal encoder=cpu (eg2_pack) src=914f7f89142e
    layout: recipe=64 names=192(+73377) slots=73584 payloads=139632+421618528
    structure OK: 1376 slots, names, hashes, alignment, payload ranges

    slot kinds: 0 = 461 slots, 2 = 914, 3 = 1
    every VQ slot: dim 4, nc 4096

The first VQ slot (the DQVD slot table entry at file offset 73584+2*48, name
`audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight`, payload at
139664, length 1613841) begins, in hex:

    44 51 56 51 | 04 00 | 00 10 | 00 10 00 00 | 00 04 00 00
    'DQVQ'      | dim=4  | nc=4096| rows=4096    | cols=1024

and the length closes the account exactly:
`16 + 4096*4*2 + 4096*2 + ((1048576*12+7)/8 + 1) = 1613841`.

The same tensor packed fresh by the same binary, in a container of its own
(`./bin/eg2_pack_cpu encode .models/EmbeddingGemma-2-914f7f8 /tmp/eg2-phase0-slot2.bin
/tmp/eg2-phase0-slot2.log --only audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight
--limit 1`, 31.6 s), produced a payload with the identical digest:

    artifact slot payload : d9c02751e1757834af34a8235ea00a1a9af7db92ee7266541d62746068ed62bd
    fresh pack of the same: d9c02751e1757834af34a8235ea00a1a9af7db92ee7266541d62746068ed62bd

Corroborating, from the artifact's own verification run: all fourteen shards
report "verify OK (all payloads bit-identical to a fresh encode)", 1376 slots
covered across the shards (`docs/artifact-verify-summary.txt`).

Verdict: identical. No payload has to be re-encoded for byte-identity reasons,
and no encoder change is needed to reach the contract — only a container change.

## 3. Determination 2 — the artifact step is a re-wrap, not a re-encode

Because section 2 is settled, phase 1's artifact step reads the DQVD payload
region and writes each payload unchanged as a GGUF tensor. It needs neither the
checkpoint nor the kmeans nor the GPU. The measured alternative it replaces is
about 14.5 minutes of shard wall time plus the merge (`out/encode-run.txt`), and
those bytes would be identical anyway (section 2).

What the re-wrap must still guarantee — the rules the DQVD checker enforced
(`tools/eg2_pack.c:980-1077`) restated in GGUF terms:

1. every one of the 1376 tensors present, once, under a unique name;
2. for every VQ tensor, `ne[0] == vq_payload_bytes(rows, cols, dim, nc)` for the
   declared geometry and tier, and the payload's own header fields equal that
   declared geometry — so a mistyped `dim`/`nc` cannot slip through as garbage
   (the DQVD rule at `tools/eg2_pack.c:1044-1045`);
3. the token table's bytes equal the q8_0 account for its declared shape
   (`tools/eg2_pack.c:1047-1058`);
4. no payload overlaps another, every tensor start respects the file's alignment,
   and no payload runs past the region (`tools/eg2_pack.c:1033-1035`).

A re-encode stays available as the fallback and was not needed; if gate 1b
(section 7) ever fails on a payload, the fallback is a re-encode, not a
hand-repaired byte.

## 4. The artifact GGUF (family `gemma_vq`)

Metadata, all values from the DQVD artifact's own header or the checkpoint:

| key | GGUF type | value |
| --- | --- | --- |
| `general.architecture` | str | `gemma_vq` |
| `general.name` | str | `EmbeddingGemma-2 VQ` (mirrors `v41_to_gguf.c:360`) |
| `general.alignment` | u32 | 32 (`ds4_gguf_write.h:37`, `v41_to_gguf.c:361`) |
| `gemma_vq.format_version` | u32 | 1 — the family gate, in the shape of `glm_vq.format_version` |
| `gemma_vq.recipe` | str | the DQVD recipe string verbatim (114 bytes, measured at offset 64 of the artifact) |
| hparams | | `gemma_vq.*` with the same values as the baseline's `gemma-embedding2.*` (section 7), because the writer is parameterized by the arch prefix |
| `general.file_type` | — | **not written**: the one shipping VQ-to-GGUF tool writes it only for the arch's own file types and omits it here (`v41_to_gguf.c:357-363`); the baseline carries 32 (MOSTLY_BF16) because it is an unquantized BF16 file |
| the tokenizer block | | the same `tokenizer.ggml.*` keys as the baseline, including the three `SpecialVocab`-derived ones: the runtime tokenizes the artifact from this block, and phase 4's token port is checked against it |

Tensors, one per checkpoint tensor (1376), named with the checkpoint's own names
— the convention `glm_vq` uses for its matrices (`ds4_glmvq_bind.inc:4-9`, which
builds `model.language_model.layers.%u.%s` from the source name) and which makes
gate 1b a name-by-name diff against `bin/eg2_pack_cpu check`:

| class | count | GGUF type | dims | why |
| --- | --- | --- | --- | --- |
| VQ payload | 461 | 42 `vqblob` | `nd = 1`, `ne = [payload_bytes]` | mirrors the precedent: `plan_vqblob` registers `ne[1] = {total}` with `DS4_GGT_VQBLOB` (`v41_to_gguf.c:137-142`), `gen_vqblob` fills the bytes (`:285-313`), and type 42's geometry is 1 element = 1 byte (`core_gguf.c:95`) |
| token table | 1 | 8 `q8_0` | `nd = 2`, `ne = [512, 262144]` | the checkpoint row is 262144 x 512; 16 blocks of 32 per row at 34 B = 544 B, so 142,606,336 B, which is the account the encoder already produced |
| kept verbatim | 914 | 30 `bf16` | `nd` 1..4 from the checkpoint (a scalar ships rank 1 — see below), `ne` inner-dim-first | all 914 DQVD verbatim slots record src_dtype BF16 |

CORRECTION (phase 1b, measured 2026-10-19): this table previously said the 914
verbatim tensors ship "nd 0..4 from the checkpoint". They cannot. 480 of them are
rank-0 scalars in the checkpoint (shape `[]`, for example
`audio_tower.layers.0.feed_forward1.ffw_layer_1.input_max`, 2 bytes), and **both**
readers of this file reject a rank-0 tensor: the engine dies "tensor has an
unsupported number of dimensions" (`ds4.c:3353`, mirrored at
`crates/ds4-core/src/tensors.rs:266`, `TensorError::BadDims`) and ggml asserts
`n_dims >= 1` (`ggml.c:1786`). A scalar therefore ships as the one-element vector it
is — rank 1, `ne = {1}` — which is the shape the baseline gives its 24
`layer_output_scale` tensors. The written artifact's rank census is the check:
1354 rank 1, 7 rank 2, 13 rank 3, 2 rank 4, and no rank 0. Evidence:
`docs/artifact-gguf-gate1b.txt` section 6.

A VQ tensor is **not** written 2-D. Type 42 is one byte per element, so a
`[cols, rows]` declaration would make any generic reader compute `cols*rows`
bytes, which is not the payload length (`8,388,608` against `1,613,841` for the
example above). `glm_vq` can declare 2-D because each of its VQ tensors begins
with a canonical 128-byte header that carries the byte size at offset 56
(`glm_vq.rs:121-172`); a DQVQ payload has no size field — its length is the byte
account — so the byte-exact declaration is `ne = [payload_bytes]`, and the
geometry comes from the payload's own 16-byte header cross-checked against the
declared layout.

## 5. Where shape and rank live — and the keys we do not write

The layout (the family's `layouts()`, mirroring `crates/ds4-core/src/glm_vq/layout.rs`
and `LayoutSpec` at `crates/ds4-core/src/layout.rs:119`) is the shape authority,
never the container geometry. That is phase-doc row P0.7, and GGUF makes the
`logical_shapes` mechanism unnecessary here, measured:

WRITTEN 2026-10-19 (phase 3): `crates/ds4-core/src/gemma_vq/layout.rs` is that
table — all 1376 tensors, their rank and their dimensions as the FILE declares
them, with "as the file declares them" doing real work for the 461 VQ tensors:
their single dimension is the payload byte length, computed here from the tower's
architecture through the reference's own account (`vq_payload_bytes(rows, cols,
4, 4096)`, `vq_qc_bytes.h:15`/`:20`). The loader computes the same number from
the payload's own header, so the two are compared on every load and a mistyped
geometry on either side is a load-time failure. The table matched the artifact
tensor for tensor on its first run (`docs/loader-gate3.txt` section 1); the
family's rank census in the file is 1354 rank 1 / 7 rank 2 / 13 rank 3 / 2 rank 4,
so the table states a rank-1 VQ payload and the file's true rank for everything
else.

- the checkpoint carries 1376 tensors; all 461 VQ tensors are rank 2
  (`docs/encode-plan.tsv`, `kind = vq`), and all 15 tensors of rank > 2 (13 rank
  3, 2 rank 4 — the audio `lconv1d` kernels, the two subsample convolutions and
  the vision position table) are verbatim BF16, because the classifier sends
  every `nd != 2` tensor to `C_KEEP` (`tools/eg2_pack.c:331-334`);
- GGUF carries a tensor's rank natively, up to four dimensions: the tensor info
  is name, `n_dims`, `ne[n_dims]`, type, offset (the format's own field list in
  the comment at the head of `/data/llama.cpp/ggml/include/gguf.h`, and the
  writer's implementation of it at `ds4_gguf_write.h:73-105`).

So the 15 rank>2 tensors keep their true rank in the tensor table, and no VQ
tensor needs a shape override. Rank 0 does not occur in the file (section 4's
correction): the 480 checkpoint scalars ship as rank 1, `ne = {1}`. Keys the two
shipping families have that we deliberately do not carry, with the reason:

| key | reason |
| --- | --- |
| `gemma_vq.codebooks` | each payload carries its own codebook; the recipe says `sharedcb=0` |
| `gemma_vq.fp8_scales` | no FP8 tensor exists in this model (`glm_vq.rs:324` is GLM's) |
| `gemma_vq.residuals` | no residual stage; that is a container-v2 GLM feature (`glm_vq.rs:372-381`) |
| `gemma_vq.logical_shapes` | not needed, measured above |
| `gemma_vq.native_policy` | its only use in GLM is selecting the Q4_K non-expert role set (`glm_vq.rs:466-473`); this artifact has no Q4_K |

## 6. Host/device split

The shipping families state the rule in the first line of their bind include:
"Rust validates the complete GVQ1 release before this pointer adapter"
(`ds4_glmvq_bind.inc:1-3`). `gemma_vq` keeps it:

- Rust: opens the GGUF, checks `general.architecture` and
  `gemma_vq.format_version`, walks the tensor table, and for every VQ tensor
  validates the payload header (magic, `dim`/`nc`/`rows`/`cols` against the
  declared layout and tier), the byte account, the index-stream bound (the last
  index must fit in `idx_bytes`), the row gains (finite, nonzero) and the
  codebook (finite) — the checks the DQVD checker makes now
  (`tools/eg2_pack.c:1044-1045`) plus the finiteness checks GLM's header parser
  makes (`glm_vq.rs:168` for the row gains, `:260-270` for the book). It fills
  `TensorInfo`/`TensorInventory` (`crates/ds4-core/src/tensors.rs`).- The Rust-to-C hand-off is a flat record table, the shape of
  `ds4_gvq_record` (`ds4-rs/ds4_glmvq.h:31-34`: offset, bytes, type, rows, cols,
  scale, book, res) with `res` and `scale` unused by this family. C never parses
  the container; `ds4-rs/ds4_eg2vq.h` declares the family's version of that record
  (offset, bytes, type, rows, cols) and its tier constants, and nothing consumes
  one yet.
- The device consumes the payload in-kernel, never expanded to dense: the
  artifact's payload region is 421,618,528 B while its dense equivalent would be
  a multiple of that (one 4096x1024 matrix alone decodes to 8,388,608 elements).

Two device-side facts are already known and are phase 7's scope, not a
contract question: the existing row kernel refuses any `dim` but 8
(`ds4-rs/cuda/ds41_vq_row.cuh:55`, against `crates/ds4-core/src/vq.rs:42`
`VQ_DIM = 8`) and the existing attention kernel caps rows at 256 with a trailing
window (`ds4_cuda.cu:42208`, `:42214`). Owner decision, still open: whether the
dim-4 device kernel is in scope now.

## 7. The baseline GGUF (phase 1a / row X.5a) — the same writer, the other arch

STATUS 2026-10-19: WRITTEN AND VERIFIED. `tools/eg2_gguf.c` emits it, the port's own
reader reads it back tensor by tensor, and `/data/llama.cpp` at 31288c9c loads it and
computes a 768-dimensional embedding. Evidence: `docs/baseline-gguf-gate1a.txt`. The
tables below are corrected against what the writer had to do; the four corrections
(sliding_window doubled, the pattern as a bool array, 1-D tensors in F32, and the
converter's extra keys) are each cited and were each measured, the first as a crash
and the third as a crash, before they were fixed.

The baseline is a plain GGUF that `/data/llama.cpp` can load, because phase 2's
anchor run feeds it to `examples/embedding/embedding.cpp`. Arch string
`gemma-embedding2` (`/data/llama.cpp/src/llama-arch.cpp:64`), alignment 32. The
authority for the key set is the converter that would write this file,
`/data/llama.cpp/conversion/gemma.py:676-733` and `:815-832`, read against the loader
that consumes it.

Key set, every name read from the loader that consumes it:

| key | type | value | read at |
| --- | --- | --- | --- |
| `general.architecture` | str | `gemma-embedding2` | `llama-model.cpp:385` (`ml.get_arch()`) |
| `general.name` | str | `EmbeddingGemma-2` | `llama-model.cpp:1376` |
| `general.file_type` | u32 | 32 (MOSTLY_BF16) | `gguf-py/gguf/constants.py:6274` |
| `general.alignment` | u32 | 32 | `gguf.h:29-30`, `:46` |
| `gemma-embedding2.block_count` | u32 | 24 | `llama-model.cpp:1394` |
| `.context_length` | u32 | 262144 | `llama-model.cpp:1382` |
| `.embedding_length` | u32 | 512 | `llama-model.cpp:1383` |
| `.embedding_length_out` | u32 | 768 | `llama-model.cpp:1384` (`n_embd_out_impl`, `llama-hparams.cpp:136-137`) |
| `.embedding_length_per_layer_input` | u32 | 512 | `src/models/gemma-embedding2.cpp:13` |
| `.feed_forward_length` | u32 | 2048 | `llama-model.cpp:1462` |
| `.attention.head_count` | u32 | 4 | `llama-model.cpp:1463` |
| `.attention.head_count_kv` | array u32 (24) | 2, and 1 at layers 5/11/17/23 | `llama-model.cpp:1471`; the converter builds it from `per_layer_config` (`conversion/gemma.py:714-726`) |
| `.attention.key_length` / `.value_length` | u32 | 512 | `llama-model.cpp:1514`, `:1517` |
| `.attention.key_length_swa` / `.value_length_swa` | u32 | 256 | `llama-model.cpp:1539-1540` |
| `.attention.layer_norm_rms_epsilon` | f32 | 1e-6 | `gemma-embedding2.cpp:12` |
| `.attention.sliding_window` | u32 | **1024** = 2 x the checkpoint's 512 | `gemma-embedding2.cpp:11`; the doubling is the converter's (`conversion/gemma.py:824-825`) so the loader's symmetric mask reproduces HF's one-sided window |
| `.attention.sliding_window_pattern` | **array bool (24)** | per-layer, from `layer_types`: true for the sliding layers | `llama-model.cpp:3902-3912` (the array is preferred over the scalar); `conversion/gemma.py:681-682` |
| `.attention.causal` | bool | false | `llama-model.cpp:1387`; `conversion/gemma.py:828` |
| `.rope.freq_base` / `.rope.freq_base_swa` | f32 | 1e6 / 1e4 | `llama-model.cpp:1482`, `gemma-embedding2.cpp:10` |
| `.rope.dimension_count` / `_swa` | u32 | 512 / 256 | `llama-model.cpp:1528`, `:1543`; `conversion/gemma.py:729-733` |
| `.pooling_type` | u32 | 1 (MEAN) | `llama-model.cpp:1386`; the converter writes it from `1_Pooling/config.json` (`conversion/gemma.py:829`) |
| `.vocab_size` | u32 | 262144 | `llama-arch.cpp:202` |
| `tokenizer.ggml.model` | str | `gemma4` | `llama-vocab.cpp:2236` (BPE; a missing `tokenizer.ggml.merges` is a hard throw at `:2242`) |
| `tokenizer.ggml.tokens` | array str (262144) | the checkpoint's `model.vocab` keys by id | `llama-vocab.cpp:2595-2602` (required) |
| `tokenizer.ggml.scores` | array f32 | -1000.0 throughout | `llama-vocab.cpp:2609`; `gguf-py/gguf/vocab.py:659-662` |
| `tokenizer.ggml.token_type` | array i32 | BYTE for `<0xNN>`, CONTROL for a special id, else NORMAL | `llama-vocab.cpp:2629`; `gguf-py/gguf/vocab.py:651-657` |
| `tokenizer.ggml.merges` | array str (514906) | the checkpoint's `model.merges`, pairs joined with one space (parts containing a space escaped as U+0101 — inert for this vocabulary) | `llama-vocab.cpp:2239-2269`; `gguf-py/gguf/vocab.py:197-220` |
| `tokenizer.ggml.bos_token_id` / `.eos_token_id` / `.unknown_token_id` / `.padding_token_id` / `.mask_token_id` | u32 | 2 / 1 / 3 / 0 / 4 | `llama-vocab.cpp:2734-2738`; resolved by `SpecialVocab` from `tokenizer_config.json` by name against the `added_tokens` entries (`gguf-py/gguf/vocab.py:358-368`) |
| `tokenizer.ggml.add_bos_token` | bool | true | `conversion/gemma.py:675`, and `SpecialVocab` agrees (`gguf-py/gguf/vocab.py:274-286`) |
| `tokenizer.ggml.add_eos_token` | bool | true | `SpecialVocab` derives it from `tokenizer.json`'s `post_processor` — `TemplateProcessing<single>` is `[<bos>, A, <eos>]` here, so `add_special_token['eos'] = true` (`gguf-py/gguf/vocab.py:191-330`); the loader appends EOS on that flag (`llama-vocab.cpp:3625`). Measured 2026-10-19 by running that reference code: `tools/ref_vocab_probe.py` |
| `tokenizer.ggml.add_sep_token` | bool | false | same source; `add_special_token['sep'] = false` for this template (llama.cpp's default is also false, so the key is fidelity rather than behaviour) |
| `tokenizer.ggml.add_space_prefix` | bool | false | `conversion/gemma.py:674` |
| `tokenizer.chat_template` | str | the checkpoint's `chat_template.jinja` | `gguf-py/gguf/vocab.py:107-110` |
| `tokenizer.ggml.pre` | — | **not written**: the loader pins `"gemma4"` for this model type itself (`llama-vocab.cpp:2275`) | — |

Values are the checkpoint's own, from `.models/EmbeddingGemma-2-914f7f8/config.json`
(`text_config`: hidden 512, layers 24, heads 4, kv heads 2, head_dim 256,
`hidden_size_per_layer_input` 512, intermediate 2048, `embedding_dim` 768, rms
1e-6, `sliding_window` 512, `layer_types` with full attention at 5/11/17/23,
`per_layer_config` giving those four `head_dim` 512 and `num_key_value_heads` 1,
`rope_parameters` 1e6 full / 1e4 sliding, `max_position_embeddings` 262144, bos 2,
eos 1, pad 0).

Tensor dtypes: **219 tensors are BF16 and 194 are F32**. The rule is the converter's
own (`conversion/base.py:1244-1246`): "most of the codebase that takes in 1D tensors
or norms only handles F32 tensors", so anything with `n_dims <= 1` or a name ending
`_norm.weight` is written F32 even in a BF16 conversion, which for this model is the
24 x {attn_norm, attn_q_norm, attn_k_norm, post_attention_norm, ffn_norm,
post_ffw_norm, post_norm, layer_output_scale} plus `output_norm` and
`per_layer_proj_norm`. The widening is exact (the BF16 pattern is the F32 word's high
half). Without it the file loads and then dies in the graph — measured:
`binary_op: unsupported types: dst: f32, src0: f32, src1: bf16` — so the rule is not
cosmetic.

Norm weights ship **verbatim**: `Gemma3Model` adds +1.0 to every `norm.weight`
(`conversion/gemma.py:131-132`) but `Gemma4Model` overrides `norm_shift` to 0.0
(`conversion/gemma.py:637-639`) and `EmbeddingGemma2Model` inherits that
(`:815-816`). The port must not add a shift either.

Tensor names, the checkpoint name → the GGUF name
(`/data/llama.cpp/src/llama-arch.cpp` name table):

| checkpoint | GGUF | cite |
| --- | --- | --- |
| `language_model.embed_tokens.weight` | `token_embd.weight` | `:480` |
| `language_model.norm.weight` | `output_norm.weight` | `:481` |
| `language_model.embedding_projection.weight` | `output.weight` | `:483` |
| `language_model.ple.per_layer_model_projection.weight` | `per_layer_model_proj.weight` | `:632` |
| `language_model.ple.per_layer_projection_norm.weight` | `per_layer_proj_norm.weight` | `:633` |
| `layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` | `:486` |
| `layers.N.self_attn.q_proj.weight` | `blk.N.attn_q.weight` | `:487` |
| `layers.N.self_attn.k_proj.weight` | `blk.N.attn_k.weight` | `:488` |
| `layers.N.self_attn.v_proj.weight` | `blk.N.attn_v.weight` | `:489` |
| `layers.N.self_attn.o_proj.weight` | `blk.N.attn_output.weight` | `:490` |
| `layers.N.self_attn.q_norm.weight` | `blk.N.attn_q_norm.weight` | `:505` |
| `layers.N.self_attn.k_norm.weight` | `blk.N.attn_k_norm.weight` | `:506` |
| `layers.N.post_attention_layernorm.weight` | `blk.N.post_attention_norm.weight` | `:504` |
| `layers.N.pre_feedforward_layernorm.weight` | `blk.N.ffn_norm.weight` | `:493` |
| `layers.N.post_feedforward_layernorm.weight` | `blk.N.post_ffw_norm.weight` | `:510` |
| `layers.N.mlp.gate_proj.weight` | `blk.N.ffn_gate.weight` | `:494` |
| `layers.N.mlp.up_proj.weight` | `blk.N.ffn_up.weight` | `:496` |
| `layers.N.mlp.down_proj.weight` | `blk.N.ffn_down.weight` | `:495` |
| `layers.N.ple_block.per_layer_input_gate.weight` | `blk.N.inp_gate.weight` | `:636` |
| `layers.N.ple_block.per_layer_projection.weight` | `blk.N.proj.weight` | `:637` |
| `layers.N.ple_block.post_per_layer_input_norm.weight` | `blk.N.post_norm.weight` | `:638` |
| `layers.N.layer_scalar` | `blk.N.layer_output_scale.weight` | `:530` |

The tensor set is `src/models/gemma-embedding2.cpp:34-72`: 5 top-level tensors
plus 24 layers of 17 = 413, which is exactly the count of `language_model.*`
tensors in `docs/encode-plan.tsv` (217 VQ, 195 verbatim, 1 table). Dims are GGUF
order, inner dimension first: a checkpoint row `[rows, cols]` becomes
`ne = [cols, rows]`; the writer checks every declared dim against the checkpoint's
own shape and refuses on any disagreement, so the file cannot drift from the
loader's declaration.

The baseline writes **every** tensor from the checkpoint: the token table is BF16
here, not `q8_0`, and no tower is quantized. That is deliberate and it is a named
deviation from this document's earlier text: the baseline is the unquantized
original, so phase 2's anchor is the true model and phase 5's delta against this
file is the whole cost of the artifact recipe, token table included. The `q8_0`
token table belongs to the artifact (section 4), where it is part of the recipe
string and the byte account.

## 8. Reference map for this contract (pinned)

| role | path | revision |
| --- | --- | --- |
| port base | `/data/embeddinggemma2/ds4-rs` | `eb67d493` |
| form authority (read-only) | `/data/llama.cpp` | `31288c9c` |
| writer and payload precedent | `/data/YoungAi` | `a4e6d31` for the quantizer sources cited by the packer, `3946dbc` for the vendored `vq_fmt.h` |
| our artifact and its evidence | `out/`, `docs/artifact-verify-summary.txt`, `docs/encode-plan.tsv`, `docs/tensor-inventory.tsv` | artifact sha256 `8902fb8a86ef7fdbd1dd079f08a0d1c4210237c1e9681d5545c983ccf0ce43fe` |
| the artifact GGUF, written in phase 1b (the runnable form) | `out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf`, `out/artifact-manifest.tsv`, `out/artifact-inventory.txt`, `out/artifact-kv.txt`, `tools/eg2_artifact_check.sh` | sha256 `5bcf22161dda46bccbd5123b26d519efccad981cbfe11cce7227415b3373b33f`; evidence `docs/artifact-gguf-gate1b.txt` |
| the baseline and its evidence (phase 1a, re-emitted in phase 2) | `out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf`, `docs/baseline-gguf-gate1a.txt`, `docs/anchor-gate2.txt` | baseline sha256 `0d84f1a93dcfe825d7391d0bc57bf2966be823757d0f1b349179733caebec201` |
| the anchor: the frozen set and the reference's vectors (phase 2) | `tests/anchor/anchor-items.tsv`, `tests/expected/anchor-embeddinggemma2.jsonl` | anchor sha256 `7561254c85591521716ec801336e3cbc2bf47e7f00c945c4f97cb643d0d4961d` |
| the anchor build (phase 2) | `/data/llama-anchor-build` (built from the pin above, never inside its tree) | llama-embedding, CPU backend |

## 9. Open items this document does not decide

1. Whether the dim-4 device kernel (row P3.1) is in scope now — owner.
2. Whether the vision and audio towers (row P4) are ported at all — owner. The
   artifact carries their payloads either way, since the DQVD container does.
3. Whether KV checkpoints mean anything for an embeddings family (row X.3).
4. Phase 2's prompt construction: RESOLVED 2026-10-19 by measurement. The file
   sets `tokenizer.ggml.add_eos_token` true and `add_sep_token` false, because the
   reference's own `SpecialVocab` derives exactly that from this checkpoint's
   tokenizer (section 7's key table, and `docs/anchor-gate2.txt` section 1). The
   checkpoint's task prompts are prepended to the text and pooled with it
   (`1_Pooling/config.json` `include_prompt` true; model card `README.md:128-151`),
   which is what the frozen anchor carries. Both choices were measured against
   their alternatives over the same 113 items: dropping EOS moves the vectors by
   mean cosine 0.96, dropping the prefix by 0.87.

Gate for phase 1: the writer may be written only against sections 4 to 7, and no
key or type may enter it that is not in those tables. Gate 1a is one tensor
round-tripped through our own reader and then loaded by the reference; gate 1b is
all 1376 tensors dumped and compared with `bin/eg2_pack_cpu check` plus the
per-payload hashes.

Gate 1a: PASSED 2026-10-19, and re-passed the same day on the re-emitted baseline
(three tokenizer keys corrected, see `docs/anchor-gate2.txt` section 1). One tensor
round-tripped first
(`output_norm.weight`, abs 15765504, byte-identical to the checkpoint), then all
413: the writer's manifest and the ds4-rs reader's own dump agree on every tensor's
name, type, rank, dims, byte count and offset, 219 tensors are byte-for-byte the
checkpoint's and 194 are its exact BF16->F32 widening; and llama.cpp at the pin
loads the file and prints a 768-dimensional embedding for a prompt.

Gate 2: PASSED 2026-10-19. 113 frozen items, one model invocation each, all four
MRL widths stored, the file bit-reproducible on a from-scratch re-run, and both
open decisions settled by measurement against their alternatives (EOS, prefix).
`docs/anchor-gate2.txt` holds the numbers.

Gate 1b: PASSED 2026-10-19. `tools/eg2_gguf.c`'s `artifact` mode wrote
`out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf` (437,508,416 B,
sha256 `5bcf2216…`) from the container, and `tools/eg2_artifact_check.sh` reports
1376 tensors where the writer's manifest and the port's own reader agree on
name/type/rank/dims/bytes/offset, 1376 copied byte-for-byte from the container, 914
verbatim tensors equal to the checkpoint's own bytes, and 461 VQ tensors beginning
with the `DQVQ` magic. The metadata was read back through the same reader with
values and differed from the baseline's in exactly the two intended keys
(section 4's table); the file is bit-reproducible; and the writer's baseline path
was re-emitted byte-for-byte identical, so the shared metadata functions changed no
existing byte. Evidence: `docs/artifact-gguf-gate1b.txt`.

Gate 3: PASSED 2026-10-19. The family loader exists
(`crates/ds4-core/src/gemma_vq.rs` with its layout module) and the ds4 host reads
the artifact four ways: `IDENTIFY EmbeddingGemma-2 VQ family=16 variant=19`,
`META ok` (the artifact's hparams against `SHAPE_GEMMA_VQ`), `BIND ok slots=1376`
and `LAYOUT ok specs=1376`. Gate 1b still passes on the same file with the family
validator in the path, the reader's tensor dump and the writer's manifest agree for
all 1376, two loads are identical, and eight tampered copies are refused by name
(a wrong payload magic, a wrong `dim`, a wrong `rows`, a zero row gain, a
nonfinite codebook word, a nonzero padding byte, a truncated file, and a wrong
metadata value). The C side gained the family enum value 16, the variant 19,
`DS4_SHAPE_GEMMA_VQ` and a bind that stops by name; the shape exists in three
copies and `test-catalog-parity` compares them. Evidence:
`docs/loader-gate3.txt`; gate: `tools/eg2_loader_check.sh`.

Gate 4 and onward are open: there is no graph, no tokenizer and no forward pass
yet. `Vocab::load` refuses this family by name and `eg2vq_bind` stops by name, so
nothing can silently run the artifact through another family's mathematics.
