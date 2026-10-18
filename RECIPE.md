# Recipe: YoungAi `vq8sh14-q4k-mtpnative` on DeepSeek-V4.1-Flash-1017

The recipe, the tools and every flag below come from
[YoungAi](https://github.com/fodelf/YoungAi) by Wenzhou Wu. They reproduce the
base published at
[wenzhouwu/YoungAi-DeepSeek-V4.1-Flash](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash),
whose build is recorded in YoungAi
[`docs/113.md`](https://github.com/fodelf/YoungAi/blob/7fb26cc332d740416792204a29c7d13d2986caf5/docs/113.md)
(steps B and G'). Nothing in the algorithm, bit allocation or file format
was changed. The only differences are the input checkpoint and the Engram
and vision weights that come with it.

## Pinned inputs

| Input | Revision |
|---|---|
| Weights | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Quantizer, main converter, constant generators | YoungAi [`7fb26cc332d740416792204a29c7d13d2986caf5`](https://github.com/fodelf/YoungAi/tree/7fb26cc332d740416792204a29c7d13d2986caf5) |
| Vision sidecar converter (`v41_to_gguf vision`) | YoungAi [`e4ea07889144d85acd33b2490ec4f44032644fb0`](https://github.com/fodelf/YoungAi/tree/e4ea07889144d85acd33b2490ec4f44032644fb0) |

`7fb26cc` is the commit the ds4-dfm-rs DeepSeek V4.1 port is pinned to. Its
`v41_to_gguf` writes the base without vision tensors, as in the V4.1
reference artifact. The `vision` subcommand was added later, in YoungAi
`320c25d` ("把视觉塔单独转成侧车 GGUF"), so the main file can stay
byte-compatible with that layout. The vision code has not changed between
`320c25d` and `e4ea078`. The quantizer sources are the same at both commits.

## Settings

| Part | Setting |
|---|---|
| Routed experts, layers 0–13 | 8-dim VQ, K=8192 (13-bit, 12-bit stream + 1-bit plane), one shared codebook per layer |
| Routed experts, layers 14–39 | 8-dim VQ, K=4096 (12-bit), one shared codebook per layer |
| Codebooks | Lloyd on the GPU, 8 iterations, sampling stride 8, codewords stored as E4M3 |
| Row gains | One F16 RMS gain per output row |
| Calibration data | None; the codebooks are solved on the weights alone |
| Skeleton | q4_K for every FP8 projection outside the routed experts and for the BF16 embedding and output head |
| MTP towers (3 × 128 experts) | Experts: 8-dim VQ, K=4096, one shared codebook per tower. Other tower weights: copied from the original checkpoint |
| Everything else | Shipped precision; small BF16 tensors are widened losslessly to F32 |
| Engram tables | Not in the GGUF. The base records the two 1017 shards' file names and byte offsets |
| Vision | Separate sidecar GGUF: original BF16 matrices, small tensors widened losslessly to F32 |

## Commands

```sh
HF=/models/DeepSeek-V4.1-Flash-1017             # the pinned 1017 checkpoint
YA=YoungAi-7fb26cc                              # checkout of 7fb26cc
OUT=DeepSeek-V4.1-Flash-1017-vq8sh14-q4k        # quantized safetensors directory

make -C $YA/gguf-tools v41_quantize v41_to_gguf

# 1. Quantize (YoungAi docs/113.md step B)
mkdir -p $OUT
for f in config.json tokenizer.json tokenizer_config.json encoding inference README.md; do ln -sfn $HF/$f $OUT/$f; done
# The converter stores realpath() of each Engram shard in the header. A 77-character path, the
# length the V4.1 reference recorded, keeps the header and file size identical to it.
EG=/work/hf/DeepSeek-V4.1-Flash-1017-1bae0e1088     # hard links, no copy
mkdir -p $EG
for s in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
    ln $HF/$s $EG/$s && ln -sfn $EG/$s $OUT/$s
done
$YA/gguf-tools/quantize/v41_quantize $HF $OUT \
    --skel q4k --mtp-vq --vq-nc 4096 --vq-nc-layers 0:14=8192 --vq-shared-cb --vq-cb-fp8 \
    --stats-out $OUT/stats.txt

# 2. Constants from the 1017 tokenizer and inference code
python3 $YA/gguf-tools/scripts/v41_tokenizer_consts.py $HF tokenizer_consts.bin
python3 $YA/gguf-tools/scripts/v41_engram_consts.py   $HF engram_consts.bin

# 3. Base GGUF with the MTP towers (YoungAi docs/113.md step G')
$YA/gguf-tools/quantize/v41_to_gguf $OUT engram_consts.bin tokenizer_consts.bin \
    DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf $HF

# 4. Vision sidecar (YoungAi e4ea078)
YoungAi-e4ea078/gguf-tools/quantize/v41_to_gguf vision $OUT $HF DeepSeek-V4.1-Flash.vision.gguf

# 5. Publish as 40 byte-range parts with the reference's sizes
#    (parts 1-24: 2,838,915,986 B, parts 25-40: 2,838,915,985 B)
B=DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
for i in $(seq 1 40); do
    if [ $i -le 24 ]; then n=2838915986; off=$(( (i-1)*n )); else n=2838915985; off=$(( 24*2838915986 + (i-25)*n )); fi
    tail -c +$((off+1)) $B | head -c $n > $B.part$(printf %02d $i)-of-40
done
```

The V4.1 reference puts the remainder of 113,556,639,424 / 40 at the front,
one extra byte in each of parts 1–24, and YoungAi's `install.sh` resumes
merges with that layout (`PART_BASE=2838915985`, `PART_EXTRA=24`). GNU
coreutils 8.32 `split -n 40` puts the whole remainder in the last part
instead, so this release was cut with explicit byte ranges.

## Identity with the V4.1 reference

Both bases carry the same recipe string (`deepseek4.recipe`):

```
routed=vq8x8192/4096 sharedcb cbfp8 skeleton=q4_K mtp=vq8x4096 engram=sidecar nocal (v41_quantize)
```

Checked on this artifact against the V4.1 reference
(`wenzhouwu/YoungAi-DeepSeek-V4.1-Flash`, base part 1 header and vision
sidecar):

| Check | V4.1 reference | This 1017 artifact |
|---|---|---|
| Quantized directory written by `v41_quantize` | 114.759 GB in 41 shards (YoungAi `docs/113.md` §0.5) | 114.759 GB in 41 shards |
| Base GGUF | 1,000 tensors, 73 metadata keys, 113,556,639,424 bytes | same; tensor names, shapes, types and offsets identical |
| Metadata values | — | 71 of 73 identical; the two Engram `table_path` strings differ (same length) |
| Engram offsets | 664 / 672 | 664 / 672, recomputed from the 1017 safetensors headers |
| Vision sidecar | 306 tensors, 17 keys, 971,066,016 bytes | same layout and keys; 532,623,100 data bytes differ (1017 retrained the tower) |
| ds4-dfm-rs `ds41_inspect` | `bind: slots=1000 bound=1000 required-missing=0` | same, and both 1017 Engram shards open |

Routed-expert weight reconstruction reported by the quantizer: layers 0–13
(13-bit) cosine 0.925, layers 14–39 (12-bit) cosine 0.910–0.911; YoungAi
recorded a 0.17062 relative error for layer 20 of V4.1 Flash, and this
conversion reports 17.06% for layer 20 of 1017.

## Build notes

The conversion ran on x86_64 with one NVIDIA H100 80 GB (sm_90), CUDA 13.0
and clang 14. YoungAi targets DGX Spark (aarch64, GB10). Two
build-environment changes were needed and no source file was edited:

- clang instead of gcc 11, because `vq_fmt.h` uses the ARM `__fp16` type,
  which clang accepts on x86_64 with the same round-to-nearest-even
  conversion;
- the CUDA cuSOLVER, cuSPARSE and nvJitLink development packages.

YoungAi describes its codebook training as deterministic for a given device.
Across GPU architectures, floating-point accumulation order can still differ,
so the bytes in `SHA256SUMS` define this release.
