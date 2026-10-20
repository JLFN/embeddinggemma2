# Using the EmbeddingGemma-2 port — step by step

This is the practical guide: how to load the artifact, get an embedding, talk to
the HTTP route, batch, send an image, and turn a PDF into a searchable index.
Every command and every number below was run on this machine on 2026-10-20 and
is reproduced from the run's own output, not from memory.

The reference for what was built and why is `RECIPE.md`; the phase-by-phase
evidence lives in `docs/`. This file is only about using it.

## 0. What you are running, and what you get

| Item | Path |
|---|---|
| Model (4-bit VQ artifact) | `out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf` (417 MiB, sha256 `5bcf2216…`) |
| Vision encoder | `out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf` (211 tensors) |
| One-shot CLI | `ds4-rs/ds4` |
| Server | `ds4-rs/ds4-server` |

The model takes text (and/or one image) and returns one **768-dimensional
L2-normalized** vector per item. Because every vector has unit length, the
cosine between two of them *is* their dot product, and the numbers below are
directly comparable to each other.

There is no language-model head in this family: `ds4 -m <artifact> -p hello -n 1`
refuses by name (exit 1) instead of trying to generate. Embedding is the only
thing this artifact does.

## 1. One-shot text embedding (CLI)

```sh
cd ds4-rs
./ds4 -m ../out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
      --embed "The cat sat on the warm windowsill."
```

Output is JSON on stdout:

```json
{
  "family": "gemma_vq",
  "task": "Retrieval-query",
  "prompt": "task: search result | query: ",
  "n_tokens": 17,
  "add_bos": true,
  "add_eos": true,
  "dims": 768,
  "secs": 0.036292,
  "embedding": [0.0158068, 0.0440282, ...]
}
```

Measured on four sentences in a row: 17, 17, 16 and 18 tokens, 0.0317–0.0370 s
each, every vector 768 wide with L2 norm 1.000000.

Other sources for the same verb — exactly one is allowed per call:

```sh
./ds4 -m <artifact> --embed-file item.txt
./ds4 -m <artifact> --embed-stdin < item.txt
```

## 2. Task prefixes matter — pick one deliberately

The checkpoint ships a prompt map, and this port carries it verbatim
(`crates/ds4-core/src/lib.rs`, `ST_PROMPTS`). The prefix is prepended to your
text **before** tokenizing, so it changes the vector. `--task` selects it;
the default is `Retrieval-query`.

| Task name | Prefix |
|---|---|
| `Retrieval-query`, `SearchQuery`, `Retrieval`, `Reranking`, `BitextMining` | `task: search result \| query: ` |
| `Retrieval-document`, `document`, `Document` | `title: none \| text: ` |
| `STS`, `SentenceSimilarity`, `PairClassification`, `Summarization` | `task: sentence similarity \| query: ` |
| `Classification`, `Clustering`, `MultilabelClassification` | `task: classification \| query: ` |
| `FactChecking` | `task: fact checking \| query: ` |
| `QuestionAnswering` | `task: question answering \| query: ` |
| `CodeRetrieval`, `InstructionRetrieval` | `task: code retrieval \| query: ` |
| `Raw` | (nothing) |

Measured effect (same sentence, two tasks): `Retrieval-query` 15 tokens versus
`Raw`, and different vectors. The rule that follows from it: **embed your query
with a query-style task and your documents with `Retrieval-document`, then
compare across the two.** An unknown name is refused by name and the valid names
are printed — there is no silent fallback, because the checkpoint's own
`default_prompt_name` is null.

## 3. Narrower vectors (Matryoshka)

```sh
./ds4 -m <artifact> --embed "..." --dims 256
```

`--dims` accepts 512, 256 or 128 and **renormalizes** after truncation, so the
result is still unit length and still directly comparable. In the committed
BEIR measurement (`docs/retrieval-cost-gate5.txt`) truncation cost almost
nothing in agreement (0.9560 at 768 → 0.9725 at 128) and is the cheapest way to
shrink an index.

## 4. The HTTP route

Start the server (one model per machine: the CLI and the server take the same
lock, so run them one after the other, not at once):

```sh
cd ds4-rs
./ds4-server -m ../out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
             --vision ../out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf \
             --model-id eg2vq --host 127.0.0.1 --port 8791 --backend cuda
```

It answers `GET /health` with the contract you can program against — measured:

```json
{"status":"ok","model":"eg2vq","backend":"cuda","family":"gemma_vq","embedding":true,
 "embed":{"item_tokens_max":2048,"batch_rows_max":8192,
          "image":{"soft_tokens_min":70,"soft_tokens_max":1120,
                   "pixels_min":161280,"pixels_max":2580480,
                   "embd":512,"per_item_max":1,"serial":true}}}
```

Read those numbers as the request budget: one item may be at most 2048 tokens;
one request may total 8192 rows; one image per item.

Embed a query:

```sh
curl -s -X POST http://127.0.0.1:8791/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"Vad gäller för övertidsersättning?","task":"SearchQuery","model":"eg2vq"}'
```

Measured: one vector, 768 wide, `"usage":{"prompt_tokens":18,"total_tokens":18}`.

The response is OpenAI-shaped: `{"data":[{"index":0,"embedding":[...],...}],...}`.

## 5. Many items in one call

The route takes an array and drives **one tower call** for the whole batch, so
batching is where the throughput is. Send the query in one call and the
documents in another (one `task` per request):

```sh
jq -n --rawfile a doc1.txt --rawfile b doc2.txt \
   '{input:[$a,$b],task:"Retrieval-document",model:"eg2vq"}' \
 | curl -s -X POST http://127.0.0.1:8791/v1/embeddings \
        -H 'Content-Type: application/json' -d @-
```

Measured on the Riksavtalet PDF (143 pages, one page per item, 8 pages per
request, 18 requests): **76,504 tokens in 19.0 s → about 4,029 tokens/s** on the
RTX 4070 SUPER. For comparison, the committed per-item device measurement is
1494.8 tokens/s (`docs/eg2vq-device-gate7.txt`) — the batch is what buys the
rest. Keep an eye on `batch_rows_max`: a request whose items together exceed
8192 rows is refused with a 400 that names the limit.

## 6. Images

An item may carry exactly one image, and the text must mark where it goes with
the placeholder `<|image|>` (the family's own contract; markers `<|image>` and
`<image|>` delimit it internally, token id 258880). The item is the OpenAI
content-part shape:

```sh
# uri.txt holds:  data:image/png;base64,<base64>     (no newline)
printf 'data:image/png;base64,' > uri.txt
base64 -w0 page.png >> uri.txt
jq -n --rawfile u uri.txt \
   '{input:[{content:[{type:"text",text:"<|image|>"},
                      {type:"image_url",image_url:$u}]}],
     task:"Raw",model:"eg2vq"}' > req.json
curl -s -X POST http://127.0.0.1:8791/v1/embeddings \
     -H 'Content-Type: application/json' -d @req.json
```

Two practical notes, both learned the hard way: build the JSON **from a file**
(a 268 KB data URI blows past the shell's argument-list limit — `jq --arg` fails
with `Argument list too long`), and stay inside the pixel window from `/health`.

Measured: page 19 of the Riksavtalet PDF rendered with `pdftoppm -r 150` to
874×1241 px (1,084,634 px, inside 161,280…2,580,480), posted as above → HTTP 200
in 2.83 s, 472 prompt tokens, a 768-wide vector. The same page's *text* embedded
separately agrees with the *image* vector at cosine **0.8956**.

Image-to-image and image-to-text both work; what is **not** reliable is ranking
whole page images against short text sections (measured spread 0.75–0.78 over
six sections, i.e. too flat to order). Use images to match images.

## 7. PDFs — there is no PDF input, so convert first

Nothing in this port reads PDF; it reads text and images. A PDF becomes usable
through one of two conversions, and which one you want depends on the file.

**Text route (preferred for any PDF with a text layer).** Extract, chunk,
embed:

```sh
pdftotext agreement.pdf - | jq -Rs '[split("\f")[] | select((gsub("\\s";"")|length) > 0)]' > pages.json
# then send pages.json's items in slices as in section 5
```

**Image route (for scans with no text layer).** Render pages and send them as
images per section 6:

```sh
pdftoppm -f 19 -l 19 -r 150 -png agreement.pdf page19
```

Rendering a full page costs ~470 soft tokens, so a 143-page image index is a
real GPU budget (~67k tokens), whereas the text route indexed the same document
in 19 s. Prefer text when a text layer exists — check with
`pdftotext file.pdf - | head`, and `pdfinfo file.pdf` for page count and size.

## 8. The whole workflow, measured on a real document

`Kollektivavtal2023_230607_0.pdf` (Riksavtalet 2023–2025, 143 pages, A5, Swedish
text layer) was indexed with `pdftotext` + the batched route, then queried. All
four queries returned the right section at rank 1:

| Query | Top page | Cosine | Section |
|---|---|---|---|
| Vilken uppsägningstid gäller vid egen uppsägning? | 109 | 0.8017 | § 17 Uppsägning |
| Hur hög är övertidsersättningen? | 73 | 0.7494 | § 9 Ersättning för övertidsarbete |
| Regler om semesterlön och semestertillägg | 78 | 0.8270 | § 11 Semester |
| Vad gäller vid sjukdom och sjuklön? | 102 | 0.8359 | § 15 Sjuklön m.m. |

The same test on six hand-picked sections, with the query and documents in
separate calls, also put the correct section first every time (0.7917, 0.7518,
0.7435, 0.8084), with a 0.04–0.09 margin over the runner-up. This is the honest
shape of the quality: on same-domain Swedish legal text the ordering is correct
and the margins are real but not huge.

## 9. How good is it? Measured, not asserted

**Live, on this machine, this session**

- Paraphrases against each other: `0.8628` ("The cat sat on the warm windowsill." vs "A feline rested on the sunny ledge."), `0.8620` (two ways of saying quarterly revenue rose).
- Unrelated sentences: `0.6232`–`0.6474`. So the gap between "same meaning" and "different topic" is about 0.22 — the model separates cleanly.
- Retrieval on the 143-page PDF: 4/4 correct at rank 1 (section 8).
- Image against its own page's text: `0.8956`.

**Committed gates in this repository** (the numbers the port is held to)

| Measurement | Result | File |
|---|---|---|
| CPU reference tower vs the frozen llama.cpp anchor, baseline weights (floor 0.9999) | mean cosine 0.99999681, min 0.99998727 | `docs/text-tower-gate4.txt` |
| The same tower on this 4-bit VQ artifact vs the anchor | mean 0.96007, min 0.82945 — this gap is the codec | `docs/text-tower-gate4.txt` |
| BEIR SciFact (300 queries / 283 docs / 339 pairs), 768 dims | baseline MRR@10 0.951778 / recall@10 0.983333; **VQ 0.935819 / 0.990000** | `docs/retrieval-cost-gate5.txt` |
| Device tower vs CPU tower, 300 items | min cosine 0.999999950, max rel RMS 3.149e-04 (bound 1e-03) | `docs/eg2vq-device-gate7.txt` |
| Tokenizer vs llama.cpp over the frozen set | 113 items, 7088 tokens, 0 id mismatches | `docs/text-tower-gate4.txt` |

Read the third row as the quality answer: on a standard retrieval benchmark the
4-bit codec costs about **1.6 points of MRR@10** against the full-precision
baseline (0.9358 vs 0.9518) and *gains* slightly on recall@10. The codec is not
free (the anchor agreement of 0.96 is the honest price) but on a retrieval task
it is close to indistinguishable.

## 10. Limits and gaps, stated plainly

- **No PDF input, no OCR.** Convert first (section 7). A scanned PDF with no
  text layer must go through the image route.
- **One image per item** (`per_item_max: 1`), and the image rules are strict:
  the text must contain exactly one contiguous `<|image|>` run where the soft
  block lands, or the request is refused by name.
- **The CLI embeds one item per process.** Batching is the HTTP route's job
  (section 5). Each CLI call reloads the artifact, so a 10k-item index through
  the CLI is the wrong tool.
- **The C CLI (`ds4-c`) refuses this artifact by name.** Its legacy vocab loader
  expects DeepSeek's special tokens while this family's `gemma4` BPE lives on the
  Rust side; the port's answer to that is `ds4-rs/ds4`, which is what this guide uses.
- **The bf16 baseline file is not loadable by the runtime reader.** It declares
  arch `gemma-embedding2` and is refused (`unsupported-arch gemma-embedding2`) —
  by design. It is consumed by the parity oracle (`tests/parity/`), not by
  `ds4`/`ds4-server`.
- **The audio tower is not ported.** 744 `audio_tower.*` tensors exist in the
  checkpoint and are out of scope for this port.
- **`--dims` is 128/256/512 only**, not an arbitrary width.

## 11. Verifying it yourself

```sh
cd /data/embeddinggemma2
tests/run.sh            # fast path: digests, container, loader gate, vision gates
                        # 22-27, the rule 19 QA gate — about 3 minutes
tests/run.sh --deep     # adds gate 1b, cargo test -p ds4-core, and gate 4
                        # (the CPU tower against the frozen anchor) — about 30 min
```

The fast path was run on this boundary and passed all eleven steps (exit 0,
164.8 s). The deep path's gate 4 is the measurement behind the first two rows of
the table in section 9.
