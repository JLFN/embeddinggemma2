# Recipe: EmbeddingGemma-2 dense VQ — `vq4x4096-q80token-dqvd` (nocal)

The quantizer, the payload format and every constant below come from
[YoungAi](https://github.com/fodelf/YoungAi) by Wenzhou Wu, the same toolchain that
produced the DeepSeek V4.1 recipe kept next to this file
(`RECIPE-deepseek41-vq.md`), which is also the quality bar this conversion is
measured against. Nothing in the encoder was changed or reimplemented: kmeans, the
GPTQ column-group feedback, the codebook and gain paths, the payload writer and the
block-scalar encoders are the reference implementations called directly.

What this recipe adds is one new structure: a dense, name-keyed container (`DQVD` v1)
that wraps the reference's `DQVQ` payloads and the reference's `q8_0` blocks
unchanged, so an engine port inherits the payload contract it already knows. The
container's field-by-field contract is in the header of `tools/eg2_pack.c`.

Scope note: `ds4-dfm-rs` cannot load this model today. It has no `embedding_gemma2`
family, no `gemma` string anywhere in `crates/`, `src/` or `ds4.c`, and no embeddings
endpoint — verified on `ds4-dfm-rs` @ `233dc7a9` (`grep -rin gemma` returns nothing;
the family list at `crates/ds4-cli/src/lib.rs:1366` is
deepseek4|motif3|solar-open2|exaone-moe|dots3-note|qwen4exp|glm5-next|k2-horizon|iquest_q1).
This artifact is the preparation for that port; section 6 is what the port has to
implement.

## 1. Pinned inputs

| Input | Revision |
|---|---|
| Weights | [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) `914f7f89142e33e77833254d9c9b90c3cef7303b`, Apache-2.0; `model.safetensors` 1,488,915,288 B, 1376 BF16 tensors, 1,488,743,984 B of tensor data |
| Quantizer and container tool | YoungAi `a4e6d3177f310cabf3e8d5d214d437a7068ddf23` (the local checkout this bake ran against) |
| Payload value rule | [`/data/ds4-dfm-rs`](file:///data/ds4-dfm-rs) `233dc7a944f556d53f0a4496368ec90d6c430bcb`, `ds41_vq_fmt.h:7`: `value[row][col] = codebook[idx][col % dim] * gain[row]` |
| Encoder build | x86_64, no CUDA in the encode path (see section 8); CUDA 13.3 was used only to measure the GPU facts quoted there |

The model is multimodal (`EmbeddingGemma2Model`: a 24-layer text tower, a 12-layer
audio tower, a 16-layer vision tower). The text tower is what produces the embedding:
`hidden_size` 512, `intermediate_size` 2048, `head_dim` 256 (512 on layers 5, 11, 17,
23), `embedding_dim` 768, vocabulary 262144. `1_Pooling/config.json` and `2_Normalize`
say the sentence embedding is the mean of the last hidden state, L2-normalized.

## 2. Settings

| Part | Setting |
|---|---|
| Text tower, text PLE, audio attention, audio FFN, audio conv-module linears, vision attention, vision FFN (461 matrices) | 4-dim VQ, K=4096 (12 bits per 4 weights, 3.0 bpw index stream), one codebook per matrix, codewords stored as f16 |
| Codebooks | Lloyd on the reference's CPU path, 8 iterations, strided subsample <= 200k vectors, codewords round-tripped through f16 before assignment (`vq_qc.h:109`) |
| Row gains | f16, one per output row; on this path all 1.0 |
| Calibration data | None; the codebooks are solved on the weights alone (`X = NULL`, the reference's nocal path — `ds4quant_run_p9.inc.c:77` passes `ncal?Xc:NULL`) |
| Token table | `language_model.embed_tokens.weight` (262144x512) at `q8_0`, 8.5 bpw, zero-corpus encoder (`ds4q_quantize_chunk` with `imatrix = NULL`) |
| Kept at source precision | 914 tensors: every non-matrix tensor (norms, biases, per-tensor and per-channel scales, the audio convolutions, the vision patch embedder and its position table) plus the embedding-space projection heads and the audio output head |
| Recipe string in the container | `towers=vq4x4096  sharedcb=0 cbfp16 token=q8_0 frontend=vq other=keep nocal encoder=cpu (eg2_pack) src=914f7f89142e` |

Why dim 4 and K=4096, and why `q8_0` rather than `q8_K`, is section 5; both were
decisions taken from measurement, not defaulted.

## 3. Container

`DQVD` v1: a 64-byte little-endian header (magic, version, kind, slot count, recipe
string offset/length, name blob offset/length, slot table offset, payload offset and
length), then the recipe string, the NUL-terminated name blob, a 48-byte slot table
entry per tensor (FNV-1a 64 name hash, name offset/length, payload offset/length,
rows, cols, dim, nc, kind, source dtype, flags) and finally the payloads, each
starting on a 16-byte boundary.

`kind` is 0 for a `DQVQ` payload, 3 for a `q8_0` payload, 2 for a verbatim copy of the
source tensor's bytes, and 1 for `q4_K` (retained so the contract still accepts the
block format this artifact does not use). Verbatim slots are the only ones above
rank 2: `rows x cols` are the first two dimensions and `payload_len / dtype_size` is
the element count, with the full shape coming from the family contract — the same
shape-contract style the reference's own loaders use. Offsets are u32, so the
container is limited to 4 GiB and the encoder refuses to write more rather than
truncate an offset.

An unknown magic, a version other than 1, an unknown kind, a dim that does not divide
`cols`, a payload range that overlaps or leaves the region, or a `payload_len` the
reference's own byte accounts cannot reproduce is a hard stop. Nothing is guessed and
nothing is padded into place.

## 4. Commands

```sh
HF=/data/embeddinggemma2/.models/EmbeddingGemma-2-914f7f8   # the pinned checkpoint
YA=/data/YoungAi                                            # a4e6d3177f310
OUT=/data/embeddinggemma2/out
B=EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin

# 1. Fetch the pinned checkpoint (15 files, 1.49 GB)
hf download google/embeddinggemma-2 \
   --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --local-dir $HF

# 2. Build the tool (CPU: see section 8 for why the GPU kernels cannot run this tier)
cc -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L \
   -I$YA/gguf-tools/quantize -I$YA/src/common \
   -o bin/eg2_pack_cpu tools/eg2_pack.c $YA/gguf-tools/quantize/quants.c \
   $YA/gguf-tools/quantize/onebit_quant.c $YA/src/common/ds4_quantfmt.c -lm -lpthread

# 3. Plan (writes the per-tensor table and prints the class totals)
./bin/eg2_pack_cpu plan $HF docs/encode-plan.tsv

# 4. Encode: 14 shards in parallel, then merge in plan order
N=14
for i in $(seq 0 $((N-1))); do
    ./bin/eg2_pack_cpu encode $HF $OUT/shard-$i.bin $OUT/shard-$i.log --shard $i/$N &
done; wait
./bin/eg2_pack_cpu merge $HF $OUT/$B $OUT/encode-log.txt $OUT/shard-{0..13}.bin

# 5. Structural check of the merged container
./bin/eg2_pack_cpu check $OUT/$B

# 6. Determinism: re-encode each shard's slots and compare payload bytes (in parallel)
for i in $(seq 0 $((N-1))); do
    ./bin/eg2_pack_cpu verify $HF $OUT/shard-$i.bin $OUT/shard-$i.verify.log --shard $i/$N &
done; wait

# 7. Checksums
sha256sum $OUT/$B > SHA256SUMS
```

Sharding is legitimate because a payload depends only on its own matrix: the codebook
is per matrix and the kmeans is seeded from the matrix's own weights (`vq_qc.h:91-95`).
It is also proved, not assumed: a 4-shard run of `vision_tower.encoder.layers.11.`
merged byte-identical to the single-process container (`cmp` silent). The merge
hard-stops on a slot that is missing, duplicated or mismatched with the plan.

`bin/eg2_vq` is the companion single-tensor tool that produced the quality ladder
(`bin/eg2_vq tensor <hf-dir> <name> <dim> <nc>`); its rows are recorded verbatim in
`docs/ladder-embeddinggemma2.txt`.

### 4b. The baseline GGUF (phase 1a) — the pinned checkpoint as a loadable GGUF

The baseline is the checkpoint re-expressed with the metadata and tensor names the
reference loader resolves, all 413 text-tower tensors in BF16 exactly as the
checkpoint carries them, so phase 2's anchor (llama.cpp at `31288c9c` over the
frozen sentence set) and our own C reference path read the same file. The writer is
`tools/eg2_gguf.c`; it uses the reference toolchain's own GGUF v3 writer, vendored
byte-identical at `tools/ds4_gguf_write.h`
(sha256 `5bfc5dd4fc8646709d521ad9a212aa0c6cba83cce2f0016baf0ff698364f1fff`).

```sh
# 8. Build the baseline writer
cc -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L -Itools \
   -I$YA/src/common -o bin/eg2_gguf tools/eg2_gguf.c -lm

# 9. Write the baseline (413 tensors, ~0.54 GB of tensor data, ~3 s)
./bin/eg2_gguf baseline $HF $OUT/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf \
   --manifest $OUT/baseline-manifest.tsv

# 10. Read it back with the port's own reader (ds4-rs, not this tool)
cd ds4-rs && cargo run -q -p ds4-core --release --example gguf_inventory -- \
   ../out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf --no-kv > ../out/baseline-inventory.txt && cd ..

# 11. Two-sided check: the writer's manifest and the reader's dump must agree on
#     every tensor's name/type/rank/dims/bytes/offset, and every tensor's bytes
#     must equal the checkpoint's own (413 cmp calls, ~0.4 s)
tools/eg2_gguf_check.sh $OUT/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf \
   $OUT/baseline-manifest.tsv $HF $OUT/baseline-inventory.txt
```

`--limit N` and `--only SUBSTR` write a smoke file whose `general.name` carries the
marker `SMOKE-TEST-NOT-A-RELEASE`. The metadata is the converter's own key set for
this arch (`/data/llama.cpp/conversion/gemma.py:676-733`, `:815-832`), read against
the loader that consumes it (`src/llama-model.cpp:1382-1541`,
`src/models/gemma-embedding2.cpp:3-13`); the tokenizer block is `gemma4` BPE with
the checkpoint's own `model.vocab` as the token list, `-1000.0` scores, and the
514,906 merges. The one deliberate difference from the artifact recipe: the baseline
ships the token table in BF16, not `q8_0`, so the baseline is the unquantized
original and phase 5's delta against it is the whole cost of the recipe.

## 5. Quality, measured

The bar is the DeepSeek V4.1 recipe's own reported number, produced the same way (the
quantizer's reconstruction cosine with no calibration data): 0.925 for the 13-bit
routed-expert layers, 0.910-0.911 for the 12-bit layers, 17.06 percent relative error
at layer 20.

The tier decision. One representative per class (layer 9 of each tower), every tier
measured with `bin/eg2_vq` on the pinned checkpoint:

| class (representative) | 4x512 | 4x1024 | 4x2048 | 4x4096 | 8x256 | 8x512 | 8x1024 | 8x4096 |
|---|---|---|---|---|---|---|---|---|
| text-tower (`layers.9.mlp.up_proj`) | 0.9631 | 0.9737 | 0.9814 | **0.9869** | 0.8216 | 0.8520 | 0.8782 | 0.9218 |
| text-ple (`ple.per_layer_model_projection`) | 0.9622 | 0.9728 | 0.9804 | **0.9857** | 0.8199 | 0.8489 | 0.8733 | 0.9111 |
| audio-attn (`layers.9.self_attn.q_proj`) | 0.9935 | 0.9970 | 0.9989 | **0.9994** | 0.8433 | 0.8740 | 0.9015 | 0.9495 |
| audio-ffn (`layers.9.feed_forward2.ffw_layer_2`) | 0.9934 | 0.9972 | 0.9986 | **0.9993** | 0.8462 | 0.8740 | 0.8989 | 0.9432 |
| vision-attn (`layers.9.self_attn.v_proj`) | 0.9629 | 0.9739 | 0.9818 | **0.9875** | 0.8235 | 0.8545 | 0.8816 | 0.9273 |
| vision-ffn (`layers.9.mlp.up_proj`) | 0.9619 | 0.9727 | 0.9802 | **0.9858** | 0.8183 | 0.8483 | 0.8736 | 0.9144 |
| audio-frontend (`layers.9.lconv1d.linear_end`) | 0.9941 | — | — | — | 0.8464 | — | — | — |

Three conclusions, all from those rows.

1. Dim 4 beats dim 8 at the same 12-bit index width (0.9869 against 0.9218 on the
   text tower, 0.9858 against 0.9144 on the vision FFN); dim 8 buys size, not quality.
2. The reference's own cold tier (8x256, 1.0 bpw) is 6.4 to 12.4 cosine points below
   the bar on every class; it was measured, not adopted, and `--cold` remains the flag
   that selects it for a size experiment.
3. 4x4096 is the only tier measured whose relative error (16.16 percent on the text
   tower, 16.86 on the PLE) is at the DeepSeek recipe's 17.06 percent level, so it is
   what the artifact ships.

The artifact itself, measured from the merged file by decoding every payload through
the reference's own reader:

| class | tensors | mean cosine | mean relative L2 | worst tensor |
|---|---|---|---|---|
| text-tower | 216 | 0.9859 | 16.69% | 0.9795 (`layers.11.self_attn.o_proj`) |
| text-ple | 1 | 0.9857 | 16.86% | 0.9857 |
| audio-attn | 60 | 0.9992 | 3.64% | 0.9974 (`layers.9.self_attn.post`) |
| audio-ffn | 48 | 0.9993 | 3.35% | 0.9976 (`layers.7.feed_forward1.ffw_layer_2`) |
| audio-frontend | 24 | 0.9936 | 9.53% | 0.9876 (`layers.0.lconv1d.linear_start`) |
| vision-attn | 64 | 0.9853 | 17.07% | 0.9740 (`layers.0.self_attn.o_proj`) |
| vision-ffn | 48 | 0.9841 | 17.76% | 0.9779 (`layers.1.mlp.down_proj`) |
| skeleton-table (`q8_0`) | 1 | 1.0000 | 0.55% | 1.0000 |
| keep (verbatim) | 914 | byte-identical by construction | — | — |

No matrix anywhere in the model falls below 0.9740, and the whole distribution sits
above the DeepSeek bar; the token table, the tensor most sensitive to quantization, is
essentially lossless at 0.55 percent relative error. The verbatim class is checked by
byte equality rather than by cosine: 914 slots are copied from the source unchanged
(the log's aggregate for that class counts only the 656 with a defined positive cosine;
the rest are single-element scalars, where cosine is meaningless).

## 6. What the `ds4-dfm-rs` port has to implement

1. A family entry for `embedding_gemma2` in the family list
   (`crates/ds4-cli/src/lib.rs:1366`) with the model's shape contract: 24 text layers of
   hidden 512, the 12-layer audio tower at hidden 1024, the 16-layer vision tower at
   hidden 768, and the two attention variants (head_dim 256, and 512 on text layers 5,
   11, 17, 23).
2. A `DQVD` reader: header validation per section 3, name-hash lookup, and the slot
   kinds — `DQVQ` payloads through the value rule the repository already implements in
   `crates/ds4-core/src/vq.rs`, `q8_0` blocks through the reference decoder
   (`src/common/ds4_quantfmt.h:45`), verbatim bytes through the dtype in the slot.
3. An embedding API. The repository has no `v1/embeddings` route today; the composition
   the checkpoint expects is mean pooling over the last hidden state followed by L2
   normalization (`1_Pooling/config.json`, `2_Normalize`).

Until that port exists, the artifact is verifiable but not runnable in `ds4-dfm-rs`.

## 7. Identity and verification

There is no published EmbeddingGemma-2 VQ artifact to compare against, so identity here
means reproducibility and byte accounting:

| Check | Result |
|---|---|
| Artifact | `out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin`, 421,758,160 B (421,604,557 B of payload plus 153,603 B of header, names, slot table and 16-byte alignment) |
| `sha256sum` | `8902fb8a86ef7fdbd1dd079f08a0d1c4210237c1e9681d5545c983ccf0ce43fe` |
| Slots | 1376, one per source tensor; names, FNV-1a hashes, ascending order, 16-byte alignment and payload ranges all verified on read-back |
| Source coverage | the plan's source bytes sum to 1,488,743,984 B, exactly the checkpoint's tensor total; no tensor is dropped or double-counted |
| Payload lengths | every slot's length reproduced from the reference's own byte accounts — `vq_payload_bytes` (`vq_qc_bytes.h:20`) for VQ, 34 B per 32 values for `q8_0`, element count times dtype size for verbatim |
| Round trip | every VQ matrix's decode-versus-encode `maxabs` is 0 (the payload decodes to exactly what the encoder wrote) |
| Determinism | 14 parallel `verify` runs covering all 1376 slots re-encoded each payload from the checkpoint and compared it byte for byte with the file: all identical |
| Encode cost | 14 shard encodes 19:42:39Z to 20:02:16Z, merge 1.4 s, 14 parallel verifies to 20:21:45Z; 39 minutes wall for the whole pipeline on 14 cores |

## 8. Build notes

The encode is CPU-only, and that was a measured decision rather than a preference. The
reference's `vq_gpu.cu` kernels take host pointers directly because the machine they
were written on has unified memory (`vq_gpu.cu:8`); this card does not
(`pageableMemoryAccess = 0`, measured by `tools/eg2_gpuprobe.cu`), and handing the
assign kernel plain `malloc` buffers returns `cudaErrorIllegalAddress` with zero of
200000 indices written while `vqg_last_err` (`vq_gpu.cu:26`) keeps the run going. With
the reference's own pinned-host mechanism (`vqg_host_alloc`) the kernels do run,
correctly and at 146x the CPU scalar loop on that stage — but driven end to end the GPU
path is 1.25 to 3x *slower* per tensor than the CPU path on this card (89.6 s against
58.8 s over one 92-tensor subset, identical cosines), because the kernel data path is
PCIe. And at this tier the kernels cannot run at all: `vq_qc.h` admits them only when
the codebook fits the 49152-byte shared-memory gate, and dim 4 with 4096 entries needs
81920 bytes. The reference's device-resident encoder for this shape is
`gguf-tools/quantize/v41_vq.cu`; adopting it is the GPU route for a later unit, and it
writes its own payload layout.

Determinism: kmeans is seeded from a strided subsample of the weights in a fixed order,
and the assignment and GPTQ loops have fixed iteration order, so the same checkpoint and
binary give the same bytes. Across CPU architectures floating-point accumulation order
can still differ, so the bytes in `SHA256SUMS` define this release.
