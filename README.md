# embeddinggemma2 — EmbeddingGemma-2 in 4-bit VQ, ported to a Rust + C + CUDA runtime

![github](https://img.shields.io/badge/github-JLFN_embeddinggemma2-blue)
![engine](https://img.shields.io/badge/engine-Rust_+_C_+_CUDA-orange)
![weights](https://img.shields.io/badge/weights-4--bit_VQ_417_MiB-green)
![release](https://img.shields.io/badge/Hugging_Face-embeddinggemma2--vq4x4096-yellow)

This repository is the recipe, the tooling and the frozen evidence behind a 4-bit
vector-quantized build of Google's
[EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2) and its port into
the `ds4` inference engine. The result is one GGUF that turns text — and optionally a
single image — into a **768-dimensional, L2-normalized** vector, while the full-precision
tower it came from stays available as the accuracy reference.

The work splits in two. This repository **produces and proves the artifact**: the
quantizer wrapper, the container format, the writer, the gates, and the measured
numbers. The **runtime lives in the sibling repository**
[JLFN/embeddinggemma2-ds4-rs](https://github.com/JLFN/embeddinggemma2-ds4-rs), a Rust
host over the `ds4` C inference core with a CUDA backend; artifacts here are what that
engine loads.

Nothing about the quantizer was reimplemented. The kmeans, GPTQ column-group feedback,
codebooks, gain paths, payload writer and block-scalar encoders are the reference
implementation, called directly; what this project adds is a dense, name-keyed container
(`DQVD` v1) that wraps the reference payloads so an engine port inherits a contract it
already knows. `RECIPE.md` is the full account, with every pinned input.

## Get the artifact

The two runnable files are published with a model card, the license text and the digests at
**[huggingface.co/Jlfn/embeddinggemma2-vq4x4096-q80token](https://huggingface.co/Jlfn/embeddinggemma2-vq4x4096-q80token)**.

| File | Size | sha256 | What it is |
|---|---|---|---|
| `EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf` | 417.2 MiB | `5bcf2216…` | The embedding model: 1376 tensors, family `gemma_vq` |
| `EmbeddingGemma-2-914f7f8-vision-mmproj.gguf` | 350.1 MiB | `9ffb86e3…` | The vision encoder, for image input (211 tensors) |
| `EmbeddingGemma-2-914f7f8-baseline-bf16.gguf` | 532.1 MiB | `0d84f1a9…` | Full-precision reference tower, for accuracy comparison |
| `EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin` | 402.2 MiB | `8902fb8a…` | The `DQVD` container the GGUF is wrapped from |

`SHA256SUMS` holds all fourteen recorded digests, including the frozen test inputs and
the anchor files. The GGUFs are build outputs and are not committed here (`out/` is
ignored); take them from the release.

## Quick start

Build the engine from its own repository, then point it at the artifact:

```sh
git clone https://github.com/JLFN/embeddinggemma2-ds4-rs.git
cd embeddinggemma2-ds4-rs
make cuda CUDA_ARCH=sm_89      # Ada (RTX 4070 SUPER class); see the repo's README for other targets

# one text through the CLI
./ds4 -m <artifact>.gguf --embed "Vad gäller för uppsägningstiden?" --task Retrieval-query
```

Output is JSON on stdout — `family`, `task`, the resolved prefix, `n_tokens`, `dims`,
`secs`, and the `embedding` array.

Serving, with the vision encoder attached:

```sh
./ds4-server --host 127.0.0.1 --port 8098 \
  -m <artifact>.gguf --vision <mmproj>.gguf --backend cuda \
  --model-id embeddinggemma-2-vq4x4096

curl -s http://127.0.0.1:8098/health | jq .
curl -s http://127.0.0.1:8098/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["Vad gäller för uppsägningstiden?"],"task":"Retrieval-query"}' | jq '.data[0].embedding | length'
```

The route is OpenAI-compatible: `input` is a string or an array, `task` selects a prompt
prefix from the checkpoint's own map, `dimensions` truncates to a Matryoshka width, and an
item may carry one image as a `data:` URI. **`docs/USAGE.md` is the step-by-step guide**,
with the exact commands, the task-name table, batching, images and a PDF-to-searchable-index
workflow; every command and number in it was run and measured.

Task prefixes are not cosmetic. The prefix is prepended before tokenizing, so it changes
the vector: embed queries with `Retrieval-query` (`task: search result | query: `) and
documents with `Retrieval-document` (`title: none | text: `), and compare across the two.

## What it does

- 768-dimensional L2-normalized output, Matryoshka-truncatable to 512, 256 or 128 (renormalized), on a 24-layer text tower with a 262144-entry vocabulary.
- Batched embedding over the HTTP route, up to 8192 tokens per request and 2048 tokens per item.
- One image per item through a 16-layer vision tower: 70 to 1120 soft tokens, 161,280 to 2,580,480 pixels.
- No language-model head. This family embeds; a generation request against it is refused by name.
- Deterministic: the same input and task give the same vector.

## Measured performance

Same-machine numbers (RTX 4070 SUPER 12 GB, CUDA 13.3), including the full pipeline that
matters in practice:

| Measurement | Result |
|---|---|
| One item through the CLI | 19 tokens in 0.035 s (1.0 s wall including the 0.41 GiB model load) |
| Indexing a 143-page PDF through the route | 80,747 tokens in 20.8 s — **3,887 tokens/s** |
| Retrieval over that index | 4 of 4 Swedish queries returned the correct section at rank 1 |

## Quality

Measured against the frozen `llama.cpp` anchor at the pinned revision and against full
precision, on the committed evidence in `docs/`:

| Comparison | Result |
|---|---|
| Port's full-precision leg vs the frozen anchor, 768-d | mean cosine 0.999996809, min 0.999987274 (floor 0.9999) |
| The 4-bit artifact vs full precision, 768-d | mean cosine 0.960067, min 0.829454 |
| Vision tower vs the frozen reference (CPU) | min row cosine 0.99978457 (floor 0.998) |
| Vision tower, device vs CPU | min row cosine 0.99933243, relative RMS 0.009006 |

On the frozen BEIR SciFact test split the 4-bit codec costs a small amount of ranking
quality at full width and slightly *improves* recall at 10:

| Width | Leg | recall@1 | recall@10 | MRR@10 |
|---|---|---|---|---|
| 768 | baseline | 0.888278 | 0.983333 | 0.951778 |
| 768 | 4-bit VQ | 0.861611 | **0.990000** | 0.935819 |
| 512 | 4-bit VQ | 0.864944 | 0.986667 | 0.936944 |
| 256 | 4-bit VQ | 0.833278 | 0.980000 | 0.914671 |
| 128 | 4-bit VQ | 0.805944 | 0.951000 | 0.879488 |

The honest caveat: on same-domain text the spread between a right and a wrong answer is
narrow, so ordering is reliable while the margin is around 0.05, not 0.3.

## Known limitations

- **No audio path.** The container carries the audio tower's tensors and the loader's layout knows their names, but the port has no audio inference entry point.
- **One image per item** (`per_item_max` 1); an item carrying two is refused by name.
- **No PDF reader.** A PDF is converted first — `pdftotext` for a text layer, or `pdftoppm` plus the image route for a scan. There is no OCR.
- **Truncated widths cost more**: 256 costs about 2.0 MRR points and 128 about 2.9 against full precision.
- The C CLI's tokenizer gap is a recorded open item; the Rust CLI and the server route are unaffected.

## Repository layout

| Path | What it is |
|---|---|
| `RECIPE.md` | The recipe: pinned inputs, container format, quantization decisions |
| `docs/USAGE.md` | The practical guide: CLI, route, batching, images, PDF workflow |
| `docs/*.txt` | The evidence walls, one per gate, quoted from the runs |
| `docs/ds4-gemma-port-reference.md` | The port's reference notes and phase map |
| `tools/` | The quantizer wrapper, GGUF writer, probes, and every gate script |
| `tests/` | `run.sh` (the test entry point), frozen anchor and retrieval sets, vision fixtures |
| `qa-evidence/` | The rule 19 QA report and findings |
| `SHA256SUMS` | Fourteen recorded digests |
| `embeddinggemma2-handoff.md` | The working plan of record |

## Verification

```sh
bash tests/run.sh            # fast path: digests, container, loader gate, gates 22-27, QA gate
bash tests/run.sh --deep     # adds the text tower against the anchor and the cargo test suite
bash tests/qa-gate.sh        # the rule 19 gate on the surface in flight
```

The fast path is seconds to a minute; the deep path runs the tower against the frozen
anchor and takes about 25 minutes. Every gate prints its own bound before it runs, so a
pass states what it proved.

## Documentation

`RECIPE.md` explains what was built and why. `docs/USAGE.md` explains how to use it.
The model card at the release carries the short version. The engine's own documentation
— build targets, model family support, HTTP compatibility — is in the
[engine repository](https://github.com/JLFN/embeddinggemma2-ds4-rs).

## License

The model weights derive from `google/embeddinggemma-2` at revision `914f7f8`, which is
Apache-2.0; the published release carries that license text. The `ds4` engine is MIT.
The reference quantizer tool is YoungAi, pinned in `RECIPE.md`. This repository itself
does not carry a `LICENSE` file yet, so its own scripts and documents are currently
unlicensed by default — that is the owner's choice to make, and until it is made no
license is granted for them.
