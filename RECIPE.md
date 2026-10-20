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

To use the finished artifact — the one-shot CLI, the HTTP route, batching, images,
and turning a PDF into a searchable index — see `docs/USAGE.md` (every command there
was run and every number measured on 2026-10-20).

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

The shipping artifact is the `gemma_vq` GGUF of section 4d. The container described
here is the one that produced it: the encoder's own output, whose payloads the GGUF
step copies unchanged (phase 0 measured them byte-identical to a fresh encode, and
it is the byte-verified record of the quantization itself). It is not the form the
runtime loads.

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

Three tokenizer keys come from the reference's `SpecialVocab` rather than from the
converter, and were corrected into the writer in phase 2 after running that
reference code (`tools/ref_vocab_probe.py`): `mask_token_id` (4), `add_eos_token`
(true), `add_sep_token` (false). The `add_*` flags are derived from the
checkpoint's `tokenizer.json` `post_processor` — `TemplateProcessing<single>` is
`[<bos>, A, <eos>]` here, so the checkpoint's own tokenizer appends `<eos>` to
every sequence, and without the flag llama.cpp tokenizes without it. That is not
cosmetic: the same 113 frozen items differ by mean cosine 0.96 with the token
present or absent (`docs/anchor-gate2.txt` section 4).

### 4c. The frozen anchor (phase 2) — the reference's own vectors over a frozen set

The anchor is llama.cpp at the pin over a frozen input set, with the baseline GGUF.
It is a second opinion beside this project's own C reference path, never it.

```sh
# 12. The frozen input set is committed: 113 items, TSV, id/prompt_name/text.
#     The model input is prompts[prompt_name] + text, where the prefix strings are
#     the checkpoint's own task prompts (config_sentence_transformers.json).
#     tests/anchor/anchor-items.tsv is the authored source; # lines are comments.

# 13. Produce the anchor (one item per model invocation, ~116 s for 113 items)
tools/eg2_anchor.sh

# 14. Check it against another anchor file (per MRL width: cosine and element
#     deltas, plus the worst items)
tools/eg2_anchor_cmp.sh tests/expected/anchor-embeddinggemma2.jsonl out/legB-noeos.jsonl

# 15. Re-run and confirm the digest is reproducible (it is, bit for bit)
tools/eg2_anchor.sh --out /tmp/rerun.jsonl --tag rerun && \
    sha256sum tests/expected/anchor-embeddinggemma2.jsonl /tmp/rerun.jsonl
```

The file carries its own provenance as its first line (reference revision, the
model's sha256, the items' sha256, the exact command) and then one line per item:
the input string, the reference's token count for it, its sha256, and four vectors
(768 and the three Matryoshka widths, each L2-renormalized after truncation). The
three long items (1,175 / 1,221 / 1,398 tokens) exist to exceed the file's
512-token mask half-window, so phase 4's symmetric mask is actually exercised.

### 4d. The artifact GGUF (phase 1b) — the container re-wrapped, `gemma_vq`

The runnable artifact is a GGUF of the `gemma_vq` family, not the DQVD container:
the container's 1376 slots are copied into one GGUF under the family's own arch,
metadata and tensor types (`docs/ds4-gemma-payload-contract.md` section 4), which
is the form the ds4 port loads. Phase 0 measured the container's payloads
byte-identical to a fresh encode, so this step re-wraps bytes — it re-encodes
nothing and needs neither the kmeans nor the GPU. Shapes and ranks come from the
checkpoint, cross-checked against the container's slot table and against each
payload's own header; every payload's byte account is the reference's
(`vq_qc_bytes.h:20`).

```sh
# 16. Confirm phase 0's measurement still holds on the current inputs
./bin/eg2_pack_cpu check $OUT/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin
sha256sum -c SHA256SUMS

# 17. Re-wrap the container as the artifact GGUF (~1 s; 461 vqblob + 1 q8_0 + 914 bf16)
./bin/eg2_gguf artifact $OUT/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin \
   $HF $OUT/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
   --manifest $OUT/artifact-manifest.tsv

# 18. Read it back through the port's own reader, twice: the tensor table, and the
#     metadata with values (--kv-values resolves every key through the reader)
cd ds4-rs && cargo run -q -p ds4-core --release --example gguf_inventory -- \
   ../out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf --no-kv \
   > ../out/artifact-inventory.txt && \
   cargo run -q -p ds4-core --release --example gguf_inventory -- \
   ../out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf --kv-values \
   > ../out/artifact-kv.txt && cd ..

# 19. Gate 1b: writer vs reader field by field, every tensor's bytes against the
#     container, the verbatim ones against the checkpoint, and every VQ tensor's
#     payload magic (~1.7 s)
tools/eg2_artifact_check.sh $OUT/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
   $OUT/artifact-manifest.tsv $OUT/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin \
   $HF $OUT/artifact-inventory.txt

# 20. Gate 3: the family loader (~1.5 s). The reader identifies the family, checks
#     the artifact's hparams against the family shape, resolves all 1376 bind names,
#     checks the family layout against the file, loads twice, and refuses eight
#     tampered copies by name. The `--family` leg is the four verdict lines.
tools/eg2_loader_check.sh $OUT/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
   $OUT/artifact-manifest.tsv

# 21. The C half of the family identity: the object compiles and the three copies of
#     the shape (ds4.c, crates/ds4-core/src/shape.rs, tests/parity/shape_c_oracle.c)
#     are compared by the existing parity test
cd ds4-rs && make ds4.o tests/parity/shape_c_oracle && \
   cargo test -q -p ds4-core --test shape && cd ..

# 22. The text tower, three ways (gate 4): the host's tokenizer against the
#     reference's own ids, the C reference forward over the frozen 113-item set on
#     both weight sources, and the anchor. The declared floor (768 mean cosine
#     >= 0.9999) is measured on the BASELINE weights, because the anchor was
#     produced on the baseline: the artifact's own distance from it is the
#     quantization of section 5, not the port. Writes out/phase4-{ids,baseline-*
#     ,artifact-*}.{tsv,txt,log,jsonl}
cd ds4-rs && make tests/parity/eg2vq_embd_c_oracle && cd ..
tools/eg2_phase4_gate.sh \
   $OUT/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
   $OUT/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf \
   tests/anchor/anchor-items.tsv .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
```

Two deliberate properties of the file. First, its hparams carry the `gemma_vq.`
prefix and it writes no `general.file_type` — the VQ family's own convention
(`v41_to_gguf.c:357-363`) — while the hparams and the tokenizer values come from
the same two functions the baseline uses, so the two files cannot drift on a
geometry value or a vocabulary entry. Second, it is a ds4 file and not a
llama.cpp file: ds4's private type 42 is `vqblob` and llama.cpp's 42 is `q2_0`,
so the reference loader refuses the artifact at its first VQ tensor while it
loads the baseline (bf16/f32) happily — which is exactly why the anchor leg runs
on the baseline.

### 4e. What the codec costs, measured on a task (phase 5)

RECIPE.md section 5 reports the codec's intrinsic cost (weight reconstruction
cosines). Phase 5 reports what that cost does to retrieval, with the same engine
over both weight sources and labels this project did not write: BEIR's SciFact test
split, taken whole. The set is frozen in `tests/retrieval/` and its builder refuses
any `scifact.zip` whose digest is not the pinned one; the gate pins the three frozen
files' digests in turn, so a regenerated set cannot silently stand behind an old
number.

```sh
# 23. The frozen labelled retrieval set — BEIR SciFact's test split, taken whole:
#     300 labelled queries, 283 pooled documents, 339 labelled pairs, formatted with
#     the checkpoint README's own retrieval-query prompt and titled-document rule
mkdir -p $OUT/beir && cd $OUT/beir && \
   curl -sSLO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip && \
   unzip -oq scifact.zip && cd ../.. && \
   sha256sum $OUT/beir/scifact.zip
tools/eg2_retrieval_set.sh $OUT/beir/scifact tests/retrieval

# 24. Gate 5: the codec's cost on that task. One ids file drives both legs (built
#     from the artifact's own vocabulary, which is the same in both files), each leg
#     runs the same CPU reference tower — baseline weights, then the artifact's VQ
#     payloads — and the scorer reports the agreement distribution and recall@k /
#     MRR@10 at 768 and at each MRL width, with the invariants that would make the
#     numbers wrong. The oracle is the tree's own entry, -O0 by default (about 80
#     minutes per leg on this set); EG2_PHASE5_ORACLE selects another build of the
#     SAME source instead, which cannot bias the artifact-versus-baseline delta
#     because both legs use it. The numbers in section 5 came from the -O3 build,
#     measured equivalent before use (min cos 1.000000000, max rel_rms 9.0e-07 over
#     24 items, ~12 minutes per leg instead of ~80; docs/retrieval-cost-gate5.txt
#     section 7a). The legs run concurrently (EG2_PHASE5_SERIAL=1 runs them one after
#     the other); a re-run reuses a leg whose raw file, ids and oracle binary still
#     match. Writes out/phase5-{ids.tsv,items.tsv,baseline-*,artifact-*}.{txt,log,jsonl,meta}
cd ds4-rs && make tests/parity/eg2vq_embd_c_oracle && cd ..
cc -O3 -ffast-math -march=native -Wall -Wextra -std=c99 -D_GNU_SOURCE \
   -fno-finite-math-only -DDS4_NO_GPU -ffunction-sections -fdata-sections \
   -Wno-unused-function -I ds4-rs -o $OUT/eg2vq_embd_c_oracle_o3 \
   ds4-rs/tests/parity/eg2vq_embd_c_oracle.c -Wl,--gc-sections -lm -pthread
EG2_PHASE5_ORACLE=$OUT/eg2vq_embd_c_oracle_o3 tools/eg2_phase5_gate.sh \
   $OUT/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
   $OUT/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf

# 25. The scorer's own check: two fixtures whose answers are worked out by hand,
#     including the tie break's numeric rule (ids 9 and 10) and a deliberate rotation
tools/eg2_retrieval_score_selftest.sh

# 26. The same agreement distribution over the phase-2 sentence set, in the scorer's
#     agreement-only mode (no labels there, so no retrieval metric)
tools/eg2_retrieval_score.sh $OUT/phase4-baseline.jsonl $OUT/phase4-artifact.jsonl -
```

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

### 5b. Quality on a retrieval task (gate 5)

The rows above are weight-level: they say the codec reconstructs the weights closely.
The owner's requirement was stated at the model level — "we should not make it less
smart, just use the VQ to make it compact" — so gate 5 measures the same CPU
reference tower over two weight sources — the baseline GGUF and the VQ artifact —
over one shared ids file, on labels this project did not write: BEIR SciFact's test
split, taken whole (300 labelled queries, 283 pooled documents, 339 labelled pairs).
Ranking is by cosine over the pool, ties broken by document id:

| width | leg | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| 768 | baseline | 0.888278 | 0.979333 | 0.983333 | 0.951778 |
| 768 | VQ artifact | 0.861611 | 0.975333 | 0.990000 | 0.935819 |
| 512 | baseline | 0.886611 | 0.979333 | 0.986667 | 0.949861 |
| 512 | VQ artifact | 0.864944 | 0.975333 | 0.986667 | 0.936944 |
| 256 | baseline | 0.872611 | 0.968667 | 0.980000 | 0.935167 |
| 256 | VQ artifact | 0.833278 | 0.965333 | 0.980000 | 0.914671 |
| 128 | baseline | 0.829278 | 0.954667 | 0.976667 | 0.908902 |
| 128 | VQ artifact | 0.805944 | 0.915889 | 0.951000 | 0.879488 |

The vectors behind those rankings agree at mean cosine 0.956039571 over the 583 items
at 768 (min 0.898186175, one item below 0.90 and 124 below 0.95) and 0.972493258 at
128. So the task-level cost at 768 is 1.6 MRR points and 2.7 recall@1 points, with
recall@10 marginally better for the artifact (0.990000 against 0.983333): the
labelled evidence is still retrieved within the top ten at the same rate or better,
while the very top of the ranking is a few percent looser. At 128 dims the two costs
compound (MRR -0.029414, recall@5 0.954667 to 0.915889).

In the percentages a decision is actually made in, at 768: the vectors keep 95.60% of
their agreement with the original; MRR@10 keeps 98.32% of the original's ranking
quality (1.68% worse); recall@1 keeps 97.00% (3.00% worse); recall@5 keeps 99.59%;
recall@10 keeps 100.68% (0.68% better, no loss); and 7.67% of queries (23 of 300)
changed their single best answer. The other widths, as MRR retained: 512 dims 98.64%,
256 dims 97.81%, 128 dims 96.76% (that width also loses 4.06% of recall@5 and 2.63% of
recall@10, where MRL truncation and the codec compound).

The owner declared the floor for this tier (dim 4 / nc 4096) on 2026-10-19, and it is
tier-tolerant by his own word: recall@10 at or above the baseline's own 0.983333;
MRR@10 within 2% of the baseline (floor 0.932742); recall@1 at or above 0.855; and a
mean vector cosine at or above 0.95. All four are met, with margins of 0.006667,
0.003077, 0.006611 and 0.006040 respectively; the floors, the verdict and the three
things they do not cover (the 512/256/128 widths, other data, and the top-1 change
rate) are recorded in section 9 of docs/retrieval-cost-gate5.txt. Full evidence:
docs/retrieval-cost-gate5.txt.

## 6. What the `ds4-dfm-rs` port has to implement

1. A family entry for `embedding_gemma2` in the family list
   (`crates/ds4-cli/src/lib.rs:1366`) with the model's shape contract: 24 text layers of
   hidden 512, the 12-layer audio tower at hidden 1024, the 16-layer vision tower at
   hidden 768, and the two attention variants (head_dim 256, and 512 on text layers 5,
   11, 17, 23).
2. A `gemma_vq` loader over the artifact GGUF, not a DQVD reader: the family gate
   (`gemma_vq.format_version`), the per-tensor payload validation the contract assigns
   to Rust (`docs/ds4-gemma-payload-contract.md` section 6: magic, dim/nc/rows/cols
   against the declared layout, the byte account, the index-stream bound, the row gains
   and the codebook), and the three tensor classes — 461 `vqblob` payloads through the
   value rule the repository already implements in `crates/ds4-core/src/vq.rs`, one
   `q8_0` token table through the reference decoder (`src/common/ds4_quantfmt.h:45`),
   914 `bf16` tensors at their true rank.

   DONE 2026-10-19: `crates/ds4-core/src/gemma_vq.rs` with its layout module, the
   `ModelFamily::GemmaVq` family, the C identity and the family arms. The reader
   identifies the file, checks its hparams, resolves all 1376 bind names and
   validates every payload (`docs/loader-gate3.txt`).
   DONE 2026-10-19 (unit 14): the tokenizer (`crates/ds4-core/src/tok/gemma_vq.rs`:
   the `gemma4` pre-tokenizer, the BPE queue and the special-token partition, with
   its ids equal to llama.cpp's own over all 113 items) and the text tower itself as
   the C reference forward (`ds4-rs/ds4_eg2vq_graph.inc` with its bind, the
   `ds4_eg2vq_matrix` dense decode of every type-42 payload, and the oracle
   `ds4-rs/tests/parity/eg2vq_embd_c_oracle`). Evidence: `docs/text-tower-gate4.txt`.
   The device path is phase 7.
3. An embedding API. The repository had no `v1/embeddings` route when this recipe was
   written; the composition the checkpoint expects is mean pooling over the last hidden
   state followed by L2 normalization (`1_Pooling/config.json`, `2_Normalize`). The C
   reference path does exactly that composition (`ds4_eg2vq_ref_embed`).
   DONE 2026-10-19 (unit 19, phase 6): the generation path now refuses this family by
   name instead of segfaulting (one function, three C sites plus the Rust CLI), the C
   engine entry `ds4_engine_embed_gemma_vq` binds the device plan once per engine and
   drives the device tower on a graph backend, the Rust CLI grew `--embed TEXT
   [--task NAME] [--dims N]` with JSON output (`--embed-file`, `--embed-stdin`,
   `--prompt-map`), and the server serves `GET /health` and `POST /v1/embeddings`
   (OpenAI shape: input string or array, model, task, dimensions 768/512/256/128,
   `encoding_format` float). Both gates are committed: the verb against the phase-7
   device raw (cos 1.000000000 on 12 items, rel_rms <= 5.745e-07) and the route against
   the verb (cos 1.000000000000; 256 dims with L2 norm 1.000000045). Evidence:
   `docs/eg2vq-embed-entry-gate6.txt`. What is still not done there: one P2.3 item
   (an embedding-shaped serving cap, batching several items into one tower call) and
   the C CLI's tokenizer gap (its legacy vocab loader wants DeepSeek's special tokens
   while this family's gemma4 BPE is Rust-side, so `ds4-c` refuses the artifact by
   name rather than decoding it).

As of 2026-10-19 the artifact is loadable, validated and runnable in `ds4-rs` on the
CPU reference path and on the GPU: `tests/parity/eg2vq_embd_c_oracle` prints a 768-dim
vector per item for the frozen set, the same graph reproduced the anchor's baseline
vectors at 768 mean cosine 0.99999681, the device tower matches the CPU tower at min
cos 0.999999950 over 300 items (`docs/eg2vq-device-gate7.txt`), and the codec's
task-level cost is measured and committed (section 5b). What is still missing is the
two tower encoders (phase 8) and the P2.3 remainder above.

## 7. Identity and verification

There is no published EmbeddingGemma-2 VQ artifact to compare against, so identity here
means reproducibility and byte accounting:

| Check | Result |
|---|---|
| Artifact (runnable form) | `out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf`, 437,508,416 B: 421,604,557 B of tensor data plus 15,881,824 B of header, metadata, tensor table and 32-byte alignment |
| `sha256sum` | `5bcf22161dda46bccbd5123b26d519efccad981cbfe11cce7227415b3373b33f` |
| Tensors | 1376, one per source tensor, under the checkpoint's own names: 461 type 42 `vqblob` (nd 1, ne = payload bytes), 1 type 8 `q8_0` (512x262144), 914 type 30 `bf16` at the checkpoint's rank (a rank-0 scalar ships as rank 1, ne {1}: both readers refuse nd 0) |
| Metadata | 42 keys: the family gate and recipe, the hparams under `gemma_vq.`, the gemma4 tokenizer with the checkpoint's own ids; read back value by value through the port's own reader, and 38 of them compared against the baseline's after mapping the arch prefix — 0 differing |
| Payload identity | all 1376 GGUF tensors byte-identical to the container's payloads, all 914 verbatim tensors byte-identical to the checkpoint's own bytes, all 461 VQ payloads starting with the `DQVQ` magic |
| Determinism | a second run produced the identical digest (0.94 s wall); the writer's baseline path is byte-transparent, re-emitted as the same sha256 as the committed baseline |
| Container (the input) | `out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin`, 421,758,160 B (421,604,557 B of payload plus 153,603 B of header, names, slot table and 16-byte alignment) |
| `sha256sum` | `8902fb8a86ef7fdbd1dd079f08a0d1c4210237c1e9681d5545c983ccf0ce43fe` |
| Slots | 1376, one per source tensor; names, FNV-1a hashes, ascending order, 16-byte alignment and payload ranges all verified on read-back |
| Source coverage | the plan's source bytes sum to 1,488,743,984 B, exactly the checkpoint's tensor total; no tensor is dropped or double-counted |
| Payload lengths | every slot's length reproduced from the reference's own byte accounts — `vq_payload_bytes` (`vq_qc_bytes.h:20`) for VQ, 34 B per 32 values for `q8_0`, element count times dtype size for verbatim |
| Round trip | every VQ matrix's decode-versus-encode `maxabs` is 0 (the payload decodes to exactly what the encoder wrote) |
| Tokenizer (gate 4, 2026-10-19) | the Rust host's ids equal llama.cpp's own over the frozen set, element for element: 113 items, 7088 tokens, 0 count and 0 id mismatches (`tools/eg2_tok_check.sh`) |
| Text tower (gate 4, 2026-10-19) | the C reference forward of the same graph on the BASELINE weights reproduces the anchor at 768 mean cosine 0.99999681 / min 0.99998727 (declared floor 0.9999), 6.3e-4 worst element; the same graph on this artifact is reported separately at mean 0.96007 / min 0.82945 — that gap is the codec, which this section's ladder measures as a 0.9859 weight cosine (`docs/text-tower-gate4.txt`) |
| Determinism | 14 parallel `verify` runs covering all 1376 slots re-encoded each payload from the checkpoint and compared it byte for byte with the file: all identical |
| Encode cost | 14 shard encodes 19:42:39Z to 20:02:16Z, merge 1.4 s, 14 parallel verifies to 20:21:45Z; 39 minutes wall for the whole pipeline on 14 cores |
| Re-wrap cost | 0.94 s wall for the whole artifact GGUF (a copy of the container's bytes, no re-encode) |
| Family identity | `IDENTIFY EmbeddingGemma-2 VQ family=16 variant=19` — the arch route, the `gemma_vq.format_version` gate, and the family's own metadata contract (`META ok`: 16 integer hparams, 3 floats, the causal flag, and both per-layer arrays) |
| Family layout | `LAYOUT ok specs=1376`: the family's table (rank and dims as the FILE declares them, a VQ payload's single dimension being the byte length the reference's own account computes from the tower architecture) equals the artifact tensor for tensor |
| Bind set | `BIND ok slots=1376`: every name the family's bind list declares is present |
| Tamper refusal | 8 tampered copies refused by name — wrong payload magic, wrong `dim`, wrong `rows`, zero row gain, nonfinite codebook word, nonzero padding byte, truncated file, wrong metadata value — while two loads of the untouched file are identical (`tools/eg2_loader_check.sh`) |
| C identity | `DS4_MODEL_FAMILY_GEMMA_VQ` 16 / `DS4_VARIANT_GEMMA_VQ` 19 / `DS4_SHAPE_GEMMA_VQ`; `ds4.o` builds in 38.9 s, `ds4_cpu.o` in 10.7 s, and `test-catalog-parity` compares the shape's three copies |
| Retrieval (gate 5, 2026-10-19) | the codec's cost on a task, on labels this project did not write: BEIR SciFact's test split taken whole (300 queries / 283 documents / 339 pairs), the same CPU reference tower over both weight sources and one shared ids file. At 768 the vectors agree at mean cosine 0.956039571 / min 0.898186175 over 583 items, and the rankings move by MRR@10 0.951778 to 0.935819 with recall@10 0.983333 to 0.990000 (`docs/retrieval-cost-gate5.txt`) |
| Device tower (phase 7, 2026-10-19) | the same tower on the GPU with the payloads consumed in-kernel: min cos(device, cpu) 0.999999950 over the 300-item query partition, max rel_rms 3.149e-04 against a declared 1e-03, 1494.8 to 2955.4 tokens/s and 414.0 MiB, against an -O3 CPU leg at 150.6 tokens/s (9.9x on short queries, 18.1x on retrieval-length documents; `docs/eg2vq-device-gate7.txt`) |
| Entry point (phase 6, 2026-10-19) | `--embed` against the phase-7 device raw: cos 1.000000000 on 12 items, rel_rms <= 5.745e-07, token counts identical; `POST /v1/embeddings` against the verb: cos 1.000000000000; `ds4 -m <artifact> -p hello -n 1` refuses by name, exit 1, no core dump, on both backends (`docs/eg2vq-embed-entry-gate6.txt`) |

## 8. Build notes

Runtime builds on this card: the port's `ds4-rs` working copy selects the card's
architecture through the tree's own mechanism rather than a hand-edited file — `cd
ds4-rs && make cuda CUDA_ARCH=sm_89`, which builds the whole target list and records
`CUDA_ARCH := sm_89` in the gitignored `.ds4-cuda-config.mk` the Makefile includes on
every invocation, so later builds need no flag. The record earns its place twice over:
nvcc 13.3's bare default here is sm_75 SASS (measured), and one object cannot cover the
Spark and this card at once — with the Spark's `-DDS4_CUDA_HAVE_MXF4=1`, adding a second
sm_89 gencode is refused outright by ptxas (`.kind::mxf4` not supported on `.target
'sm_89'`). Switching cards is `make cuda-spark` for a GB10 and `make cuda-generic` for
the host-detected default. The device path for the tower itself is not this note: it is
P3 (phase 7), and the tree's existing kernels running on this card is the precondition
row. Evidence: `docs/cuda-sm89-port.txt` sections 3e to 3g.

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
