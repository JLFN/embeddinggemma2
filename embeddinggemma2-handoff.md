---
project: embeddinggemma2
plan_start_commit: d42fc8fdf8cef8dc0f6806d249160ae5472abe77
last_updated_commit: a2da7cd5d2f4369f5f50ce0947ce5b59daf1130e
branch: dev
remote: 
handoff_written_at_context_usage: 37%   (read from signals.json at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-20T12:02:21+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ in ds4-rs (rewritten 2026-10-20 at the llama-free-boundary: the landed phases are rotated out into git and the evidence docs, and section 3 is the ONE unit that is left)

Note on the frontmatter: it MUST be the first thing in the file, with
bare values (no inline comments after the value) — the handoff-hooks
parser reads it verbatim, and a commented or misplaced frontmatter
silently breaks the SessionStart probe readout. The scaffold writes this
exact shape; hand-written files must match it. <COMMIT> means the full
commit hash (plan_start_commit = the commit the current plan began from);
<PERCENT> = the context usage read from the session's signals.json
(target 25-35, hard cap 40).

0. Rules for writing THIS document (read before editing, not just before reading)

- Only write or replace this file at a clean, verified, committed boundary. Never checkpoint mid-work.
- Start drafting once context usage crosses ~25%. Finalize and commit before the auto-compact threshold — do not wait for it to begin writing.
- Every claim in "Current state" must be re-verified by running the relevant command in this session, not copied from the previous handoff. If you cannot re-run it, mark it unverified.
- Hard caps: "Current state" max 15 lines. "Guardrails and open items" max 15 lines. "Next unit plan" has no cap — it is the actual work spec.
- Do NOT: replay prior units' logs (git log is the record), re-explain global rules already in the loaded guardrails, restate anything the staleness check covers, or record acceptance evidence you did not observe this session.
- Section 3 holds ONLY the next unit. When a unit completes, fold its outcome into section 2 and REPLACE section 3 with the next unit's plan — history is not appended, it is rotated out into git and the evidence docs.
- This file is the plan of record. docs/ds4-gemma-port-phases.md holds the row-by-row work list behind it; where the two disagree, THIS file wins and the phase doc is corrected in the same commit.

1. Session protocol (read first)

STANDING OWNER INSTRUCTION (2026-10-19, verbatim in intent): the owner is AFK and
wants the session to CONTINUE AUTOMATICALLY after each handoff and each automatic
compaction. So: never stop to ask a question, never wait for a prompt, and never
end a turn with work left that the plan already specifies. Work unit after unit:
finish it, verify it, commit it locally, rotate this handoff, and continue. The one
thing that stays an offer is the PUSH (rule 23): record the offer in section 7 and
keep working; do not push on your own initiative.

SECOND OWNER DIRECTIVE (2026-10-20, verbatim): "We will not use llama it must be
ported all". Read as a boundary rule: the pin is the AUTHORITY the port reads and
cites (rule 0.2) and the one-shot producer of the frozen numbers, but no build, no
run and no default gate of this project executes it. Measured 2026-10-20 and closed
by commit 3729ec4: ds4-rs contains no llama call and links no llama library, gate 22
reads the mmproj back with the Rust host's own reader, and the two remaining
default-path runs of the pin (the vision harness, the tokenizer dump's regeneration)
are opt-in — tools/eg2_vision_anchor.sh and EG2_TOK_ANCHOR=1. Everything the pin
produced that matters is committed data.

1. Read context usage from the session's signals.json (contextTokensUsed /
   contextWindowTokens) under ~/.opengrok/sessions/<percent-encoded-cwd>/
   <session-id>/, newest session dir. If signals.json is absent, fall back to
   chat_history.jsonl bytes / 4 and say so in the session report.
2. If usage is below the threshold (~/.opengrok/config.toml
   [session].auto_compact_threshold_percent, this machine 60): continue
   immediately — write the next unit's plan into section 3 if it is not there,
   execute it, verify, commit, re-check usage, repeat.
3. At the threshold the harness compacts this session in place; nothing is
   launched. This handoff must be written and committed BEFORE the threshold.
4. Preferred cadence: once the finished unit's handoff, the project memory entry
   and the commits are in place, call the compact_session tool as the last action
   of that turn. The handoff write itself also parks a compaction.

2. Current state (2026-10-20, at the unit-25 close: the image path runs end to end — the vision tower on both legs, its soft tokens spliced into the text tower, and the engine entry that the bridge and the route call — and what is left of phase 8 is the WIRE decision)

- OWNER DIRECTIVE, 2026-10-19 (verbatim intent): "we need to add vision also see how the other model variant handle it". This answers the long-open phase-8 decision in favour of vision (the audio tower is NOT in scope; the audio decision stays open). It defers, and does not cancel, unit 21 (batching): its complete executable spec is parked at docs/eg2vq-batch21-plan.txt.
- WHAT THE ARTIFACT CARRIES (docs/tensor-inventory.tsv + out/artifact-kv.txt): of 1376 tensors, 208 are vision_tower.encoder.layers.*, 2 are vision_tower.patch_embedder.*, 744 are audio_tower.*; the text tower is 217 of the 461 vqblob payloads, the VISION TOWER IS DENSE BF16, and the two tie-in projections are in-file (embed_vision 512x768, embed_audio 512x1536). THE IMAGE PATH end to end is image -> preprocessing (gate 23's contract) -> the 16-block tower -> 3x3 pool + sqrt(768) -> weightless RMS -> 768-to-512 projection -> 88 soft tokens spliced into the text tower's image span -> the 768-dim pooled embedding. Graph: /data/llama.cpp @ 31288c9c tools/mtmd/models/gemma4v.cpp (133 lines), variant GEMMA4V (conversion/gemma.py:1001, :1055).
- THE ARTIFACT MUST NOT BE RE-ENCODED, and the owner's 2026-10-20 GPU-rebuild question is answered by RECIPE.md section 8 (read live): a CUDA build of the packer exists (-DEG2_CUDA with vq_gpu.o and the pinned-host path) but at this tier the kernels cannot run at all (81920 bytes of codebook against vq_qc.h's 49152-byte shared-memory gate) and where they can run they are 1.25-3x SLOWER here (89.6 s vs 58.8 s over a 92-tensor subset, identical cosines). The committed file records `encoder=cpu (eg2_pack)` and its digest defines the release.
- THE OWNER'S QUALITY FLOOR is RECORDED and MET (docs/retrieval-cost-gate5.txt section 9): recall@10 >= 0.983333, MRR@10 >= 98% of baseline (0.932742), recall@1 >= 0.855, mean cosine >= 0.95 — margins 0.006667 / 0.003077 / 0.006611 / 0.006040, all PASS. The floor covers the 768 width only.
- UNIT 23 (row P4.1) IS DONE: the vision tower on the CPU reference path (ds4-rs f459e924 + 0dd35319), gate 23 in the fast path, against the frozen dumps at pattern 0.10275560 / 0.008235 / 0.99978457 and gradient 0.06645030 / 0.005285 / 0.99988406.
- UNIT 24 (row P4.5) IS DONE: the DEVICE tower (ds4-rs 5183cd50 + 9e1e987a, outer 3a5f008 + 0225fe3), gate 24 in the fast path. Device vs the CPU leg 0.15841668 / 0.009006 / 0.99933243 worst; stages patch 1e-6, l0 7.8e-5, pooled 8.8e-3; device vs the frozen dumps 0.010428 / 0.005155 relrms inside gate 23's bound; 0.229 s per image against the CPU leg's 4.336 s (~19x). The 0.9% is the tower's own bf16-quantization-boundary band, measured with an input kick (1e-7 moves the output by EXACTLY ZERO); do not chase it.
- UNIT 25 (row P4.6) IS DONE except its WIRE half (section 3): the image SPLICE and the ENGINE ENTRY. ds4-rs d8e3aded (the splice in both legs, DS4_EG2VQ_IMAGE_TOKEN_ID 258880, the gemma_vq_vocab example), 6d07611d (the engine entries + the vocabulary case that made the engine openable at all), 5bd18880 (media_serial true); outer c124896 (gate 25) and 6765944 (this doc + the fast path). MEASURED by gate 25, four legs: the SPLICE on the BASELINE weights with the reference's own soft rows maxabs 0.00018850 / relrms 0.001579 / cosine 0.99999875 (pattern) and 0.00011922 / 0.001068 / 0.99999943 (gradient); the baseline with the PORT's soft rows cosine 0.99999091 / 0.99999824; the ARTIFACT with the port's rows 0.03693355 / 0.242372 / 0.97062788 and 0.02680407 / 0.202138 / 0.97957005 (the 4-bit tier's own loss, gate 5's 0.956 mean); the artifact's device leg vs its CPU leg 0.00057164 / 0.004348 / 0.99999055 and 0.00025823 / 0.002304 / 0.99999735. Engine entry: probe source 512x384 / content 512x384 / padded 528x384 / grid 11x8 / 88 tokens; encode_memory on the device leg vs the CPU leg 0.15841668 / 0.009006 / 0.99995945 and 0.04474682 / 0.004954 / 0.99998773. The text tower does NOT amplify its input (a 1e-2 kick on one soft value moves the pooled vector 6e-6 to 1.3e-5) — the opposite of the vision tower.
- TWO TRAPS THIS UNIT CAUGHT, recorded so they are not re-learned: (1) THE FROZEN SPLICES ARE BASELINE-MODEL VECTORS (the anchor's second argument is the baseline bf16 file), so comparing the shipped ARTIFACT's splice with them measures the tier (24% relrms / cosine 0.9706), not the splice — the splice itself lands at 0.16% on the baseline weights. (2) THE ENGINE COULD NOT OPEN THIS ARTIFACT AT ALL: the vocabulary's family chain had no gemma_vq case and demanded another family's tokens; the new case takes the artifact's own bos/eos and leaves the role ids absent. The CLI's generation refusal by name is unchanged, and the C tokenizer gap (the gemma4 BPE is Rust-side) is still exactly the open item it was.
- RULE 19 IS MECHANICAL: tests/qa-gate.sh + tests/run.sh (fast path: digests, DQVD container, scorer self-test, loader gate 3, gate 22, gate 23, gate 24, gate 25, QA gate; --deep adds gate 1b, cargo test -p ds4-core and gate 4). Unit 24's QA pass returned PASS with seven low findings (all dispositions in docs/eg2vq-vision-tower-gate24.txt section 9); unit 25's pass is the current report once it lands. F6 stands: the surface extractor still sees only `embed_caps`, so the QA report covers the new C surfaces by hand.
- Workspace: /data/embeddinggemma2, branch dev, HEAD = the commit that carries this handoff, tree clean at write time. THE UNIT-25 QA PASS WAS IN FLIGHT: if the tree carries a fresh qa-evidence/qa-report.md when the next session starts, that is it landing — commit it, read its findings, and only then begin unit 26. Unpushed: outer TWENTY-TWO commits over publish/dev d45394f (tip = this handoff chain); ds4-rs NINE over 29873445 (tip 5bd18880). Both pushes are OFFERS (section 7).
- WHAT IS LEFT, in order: (1) the WIRE (section 3): the /health vision row and the route accepting an image with a text item, through the engine entries that now exist; (2) unit 21, batching, DEFERRED with its spec at docs/eg2vq-batch21-plan.txt; (3) the C CLI's tokenizer gap — owner's call (the Rust host owns encoding); (4) the AUDIO tower — owner's call; (5) the two pushes, when the owner answers.
- Machine-facing facts RE-RUN this session (2026-10-20): tests/run.sh fast path GREEN in 171 s with gates 23, 24 and 25 in it (gate 23 unchanged at 0.10275560 / 0.008235 / 0.99978457 and 0.06645030 / 0.005285 / 0.99988406; gate 24 unchanged; gate 25 `EG2VQ_SPLICE_GATE25_PASS`, ~70 s warm with the engine section); `cargo test -q -p ds4-core` green; `make ds4.o` and `make ds4_cuda.o` clean; `sha256sum -c SHA256SUMS` all FOURTEEN entries OK. Carried over and NOT re-run here: the artifact sha256 5bcf2216.., CUDA_ARCH sm_89, rule 10 OSV (re-run before an actual push), gate 4/5/7/20's numbers.
- Nothing measured to date claims: an MTEB-comparable score; a serving-shaped latency or concurrency number for the image path; anything about the audio tower; or any quality verdict past the four declared floors.

3. Next unit plan — unit 26: the WIRE (the serving route and the /health row for the image path; unit 25's last half, phase-doc row P4.6's serving side)

Goal: a caller can send an image (with a text item) to the serving route and get the
spliced embedding back, and /health tells it what the family accepts — through the
one mechanism unit 20 built, not a second one.

STEP 0 — what unit 25 leaves, verbatim:

 1. The engine entry exists and is gated: ds4_engine_vision_probe (the decoding
    geometry: source, content, padded, the POOLED grid, token_count) and
    ds4_engine_vision_encode_memory (a ds4_vision_embedding of n_soft x 512 soft rows,
    freed by ds4_vision_embedding_free).  It runs the device leg when the graph
    backend is live, the CPU leg otherwise, and it takes the mmproj through
    opt->vision_path (the load switch's gemma_vq case).
 2. The splice exists and is gated: ds4_eg2vq_ref_embed_image / ds4_gpu_eg2vq_embed_image
    take (ids, n_ids, soft, n_soft) where the ids carry ONE contiguous run of
    DS4_EG2VQ_IMAGE_TOKEN_ID placeholders (258880) and the soft rows enter UNSCALED;
    the sequence the reference uses is [bos 2, boi 255999] + 88 placeholders +
    [eoi 258882, eos 1] = 92 positions (span [2, 90)).
 3. The measured tier facts, so the wire's numbers are stated rather than guessed:
    the SPLICE lands at 0.16% of the frozen baseline splice; the artifact at 3%
    (the tier's loss); the device at 0.43% of the CPU leg.  Gate 25 prints all four.
 4. The serving caps vocabulary: crates/ds4-core/src/serving.rs's family-keyed
    lookup (embed_caps) -> the server's /health render (crates/ds4-server/src/serve.rs
    around :1232) -> the route's refusal by name (:1618-1632).  GemmaVq now says
    media_serial = true (one vision workspace per engine).
 5. The wire's tokenizer is RUST-SIDE by design (the C CLI refuses the artifact by
    name): the marker ids are 255999 / "<|image>" and 258882 / "<image|>", the span's
    placeholder is 258880, bos 2, eos 1 -- read from the artifact's own vocabulary
    (crates/ds4-core/examples/gemma_vq_vocab.rs).  The wire must build the sequence
    there, not in C.

THE WORK, in the order it has to land:

 A. THE /health ROW.  Read how the "embed" object is rendered and enforced (serve.rs
    :1232, :1618) and add the image half the same way: the family accepts images, the
    soft-token ceiling (1120), the pixel ceiling (2580480), and media_serial's
    meaning.  A Rust test that pins the rendered JSON, in the shape unit 20's
    eg2vq_serving_caps.rs uses.
 B. THE ROUTE.  Decide and implement how an image reaches the embedding path: the
    bridge already carries vision_probe/encode_memory (native/bridge/ds4_bridge.c
    :462+), and the engine entries now answer for gemma_vq.  The route must (i) probe,
    (ii) encode the soft rows, (iii) build the 92-position sequence in Rust with the
    ids in STEP 0.5, (iv) call the embedding entry with the splice (a bridge entry for
    ds4_gpu_eg2vq_embed_image / ds4_eg2vq_ref_embed_image is the piece that does not
    exist yet -- name it in the bridge rather than inventing a second path), and
    (v) refuse by name an image past the ceiling, an item with no text around the
    image, and a text-only route that gets an image.
 C. THE GATE: an HTTP-level gate in the shape of the recorded POST /v1/embeddings
    one (docs/eg2vq-embed-entry-gate6.txt): start the server on the artifact with
    --vision, POST the two frozen fixtures with a fixed text item, and compare the
    returned vectors against the port's own CPU splice (not the frozen baseline rows:
    that comparison is gate 25's and is not the wire's).
 D. THE EVIDENCE + the fast path step + the rule 19 QA pass.

WHAT UNIT 26 MUST NOT DO: re-derive the splice or the vision tower, expose the image
on the text-only path in a way that changes a text request's answer, invent a second
caps mechanism, or claim a latency or concurrency number for the image path without
measuring it under the condition it claims.

4. Reference map — the exact repos and files (pinned; verified at this boundary)

| Role | Root and pin | What is read from it |
|---|---|---|
| Port base (all the work) | /data/embeddinggemma2/ds4-rs @ 70278c19, branch feat/ds41-dsml-carriage, gitignored by the outer repo, own git | ds4.c, ds4.h, ds4_eg2vq.{h,inc,graph.inc,gpu.cuh}, ds4_glmvq*.{h,inc}, cuda/eg2vq_{attn,ops,vq}.cuh, cuda/*.cuh, crates/ds4-core/src/**, crates/ds4-server/src/**, crates/ds4-sys/src/lib.rs, tests/parity/, the Makefile |
| Form and metadata authority | /data/llama.cpp @ 31288c9c (b11723, MIT, read-only, source only) | src/models/gemma-embedding2.cpp, src/llama-{arch,hparams,model,graph,vocab,batch}.cpp, ggml/include/gguf.h, conversion/{gemma,base}.py, gguf-py/gguf/**, examples/embedding/embedding.cpp, tools/mtmd/** (phase 8) |
| The VISION reference (phase 8) | /data/llama.cpp @ 31288c9c, the same pin | tools/mtmd/models/gemma4v.cpp (the graph, 133 lines: :5-117), tools/mtmd/clip.cpp (:337-450 build_vit's per-layer order and :144-147 the weightless V RMS, :254-256 the patch grid, :973-979 the builder dispatch, :1719-1736 the GEMMA4V hparams, :4419-4428 the image-token count, :5562-5568 pos_x/pos_y, :1342-1352 the mmproj keys), tools/mtmd/clip-impl.h:112-157 (the mmproj tensor names), tools/mtmd/mtmd.cpp:953-961 (dyn_size is the gemma4v preprocessor), mtmd-image.cpp:57-96 (the PAD_CEIL branch), :122-156 (the size solver), :204-390 (resize_pillow), :7-14 (the u8 -> f32 conversion), clip-model.h:66-68 (pad and pad-colour defaults), :204-211 (the token-to-pixel limits), mtmd-helper.cpp:404 (the decode, no EXIF), conversion/gemma.py:978-1063 (the projector variant and the tensor mapping) |
| The VISION weights file, and the frozen image reference | our own committed data | out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf (sha256 9ffb86e3.., 211 tensors, written by tools/eg2_mmproj.c from the checkpoint) and tests/vision/{eg2-vision-gradient-512x384.png,eg2-vision-pattern-512x384.png} with tests/vision/reference/*.{soft,text}.tsv (the pin's own numbers, produced once by tools/eg2_vision_anchor.sh) |
| The anchor build (phase 2) | /data/llama-anchor-build (CPU, plus an mtmd target built for the vision anchor) | bin/llama-embedding; the frozen text vectors it produced are phase 2's deliverable, and since 2026-10-20 nothing but tools/eg2_vision_anchor.sh touches this tree |
| Writer and payload precedent | /data/YoungAi (a4e6d31) | src/common/ds4_gguf_write.h, gguf-tools/quantize/v41_to_gguf.c, vq_fmt.h |
| Verification method | /data/ds4-dfm-rs (the live ds4 project; never modified from here) | AGENTS.md (the retained `-c` oracles), tests/parity/, Makefile (the false-green guard) |
| The anchor vectors | our own committed data | tests/anchor/anchor-items.tsv (113 items; three long enough to exercise the mask) and tests/expected/anchor-embeddinggemma2.jsonl (768/512/256/128 vectors, sha256 7561254c..); tools/eg2_anchor.sh / eg2_anchor_cmp.sh |
| The artifact, and the container behind it | our own committed recipe | out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf (sha256 5bcf2216.., 1376 tensors), out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf (0d84f1a9..), out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin (8902fb8a..), out/artifact-manifest.tsv, out/artifact-kv.txt |
| The family's contract, loader, tower | our own code | crates/ds4-core/src/gemma_vq.rs (+ the caps constants) and gemma_vq/layout.rs; tok/gemma_vq.rs; ds4_eg2vq_graph.inc + ds4_eg2vq_bind.inc; tools/eg2_{loader_check,phase4_gate,tok_check}.sh; docs/{loader-gate3,text-tower-gate4}.txt |
| The gate-5 harness and the frozen set | our own code and data | tools/eg2_retrieval_set.sh, tests/retrieval/scifact-test-{queries,pool,qrels}.tsv, tools/eg2_{phase5_gate,oracle_to_jsonl,retrieval_score,retrieval_score_selftest}.sh, docs/retrieval-cost-gate5.txt (sections 1-9) |
| The embedding entry and its caps (phase 6) | our own code | ds4.c's ds4_engine_embed_gemma_vq, ds4_bridge_eg2vq_*, crates/ds4-core/src/serving.rs (EmbedCaps/embed_caps), crates/ds4-core/tests/eg2vq_serving_caps.rs, fair route gate tools/eg2_embed_route_gate.sh (sections 1-5), docs/eg2vq-serving-caps-gate20.txt, docs/eg2vq-embed-entry-gate6.txt |
| The device path (phase 7) | our own code | cuda/eg2vq_*.cuh, ds4_eg2vq_gpu.cuh, tests/parity/eg2vq_embd_gpu_oracle.c, docs/eg2vq-device-gate7.txt, out/cuda-objects-backup-sm121a |
| The labelled data, and its origin | BEIR SciFact (Thakur et al., NeurIPS 2021 D&B) | the zip URL pinned in tools/eg2_retrieval_set.sh (ZIP_SHA 536e1444..); tests/retrieval/PROVENANCE.md records CC BY-NC 2.0 |
| The model | /data/embeddinggemma2/.models/EmbeddingGemma-2-914f7f8 (gitignored) | config.json, config_sentence_transformers.json (the 20 task prompts), 1_Pooling/config.json, tokenizer.json, README.md (the truncation contract) |
| Counter-example only | /data/llama-prism @ b10743 | never a source; read only to avoid its 1/sqrt(head_dim) scale |

5. Routes removed — do not resurrect any of them

- ONNX and onnxruntime in any role; PyTorch/transformers/sentence-transformers and
  any Python ML stack (owner, 2026-10-19: "remove this onnx i dont want it and no
  pytorch"; AGENTS.md P-1). An isolated venv is dead too.
- The DQVD container as the target format (owner's container decision); it remains
  only as phase 1's input and the byte-verified record of the quantization.
- The `eg2_pack reconstruct` fallback (DQVD-only, superseded by the GGUF).
- candle, burn, tch or any Rust tensor framework: the arithmetic is our own C/CUDA.
- Executing the pinned llama.cpp in any default path (owner directive 2026-10-20): no
  build, run or gate of this project links it. Its numbers are committed data and
  tools/eg2_vision_anchor.sh (opt-in) is the only thing that runs it.
- Metal and macOS: not a target here (Linux, NVIDIA only).
- Not in scope unless the owner says so: KV checkpoints (row X.3), ds4's
  distributed/agent surfaces, and the vision/audio towers (phase 8).

6. Project facts a fresh session cannot re-derive

- Workspace layout: RECIPE.md (recipe, quality, what the port implements, identity);
  RECIPE-deepseek41-vq.md (the ds4 VQ recipe mirrored); docs/ds4-gemma-port-phases.md
  (the row list — the authoritative work log); docs/ds4-gemma-port-reference.md;
  docs/{tensor-inventory.tsv,encode-plan.tsv,artifact-verify-summary.txt,
  ladder-embeddinggemma2.txt}; docs/{baseline-gguf-gate1a,anchor-gate2,
  artifact-gguf-gate1b,loader-gate3,text-tower-gate4,retrieval-cost-gate5,
  eg2vq-embed-entry-gate6,eg2vq-device-gate7,eg2vq-serving-caps-gate20}.txt;
  qa-evidence/qa-report.md (the latest rule 19 pass); tests/{anchor,expected,
  retrieval}/ + tests/{run.sh,qa-gate.sh}; tools/eg2_*.{c,sh} + bin/; out/ (all
  artifacts and logs, gitignored).
- The caps (unit 20): crates/ds4-core/src/gemma_vq.rs holds ITEM_TOKENS_MAX 2048
  (the device attention key plane) and BATCH_ROWS_MAX 8192 (a PORT policy: 108,548 B
  of workspace per row, 848.03 MiB at the cap), reached through
  `ds4_core::embed_caps(family)`; /health reports them under `embed`; the route
  refuses past either by name as 400; the CPU reference's rows are packed, so
  BATCH_ROWS_MAX is also the batch's row count.
- The tower (phase 4/7): the tokenizer is host-side (crates/ds4-core/src/tok/gemma_vq.rs,
  a gemma4 pre-tokenizer with the `<0xXX>` byte fallback; limits: invalid UTF-8,
  ignore_merges); the CPU reference is ds4_eg2vq_graph.inc (+ ds4_eg2vq_bind.inc,
  which decodes all 217 tower payloads into a 520 MiB f32 cache and carries BOTH
  weight legs by name); the device path is ds4_eg2vq_gpu.cuh over cuda/eg2vq_*.cuh,
  payloads consumed in the kernel (55.9 MiB plan, 137 MiB weights, workspace sized by
  max_tokens: 212.01 MiB at 2048 rows); the oracle is
  `tests/parity/eg2vq_embd_c_oracle <gguf> <ids.tsv>` (built by make).
- Shape to load: 24 layers, hidden 512, 4 q heads, 2 KV heads, head_dim 256; layers
  5/11/17/23 full (head_dim 512, 1 KV head); ffn 2048; rms eps 1e-6; vocab 262,144;
  out 768; the file's sliding_window is 1024 (the converter's doubling of the
  checkpoint's 512), so the symmetric mask is ±512; mean pool then L2 normalize.
- The vision tower (phase 8, new): in the SAME checkpoint as the text tower and DENSE
  BF16 (208 tensors under vision_tower.encoder.layers.*, 2 under
  vision_tower.patch_embedder.*), with config.json's vision_config as the shape
  authority: hidden 768, 16 layers, 12/12 heads, head_dim 64, ffn 3072, patch 16,
  pooling_kernel_size 3, position_embedding_size 10240, rope theta 100 with type
  axial, rms eps 1e-6, standardize false, use_clipped_linears false. The port does
  not read it from the checkpoint: `tools/eg2_mmproj.c` writes it to
  out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf (211 tensors, sha256 9ffb86e3..),
  and the engine opens THAT as its second model (`e->vision_model`, the companion
  GGUF pattern at ds4.c:72689, bound by name with required_tensor), exactly as the
  other families' vision GGUFs are bound (glm53_vision_weights_bind, ds4.c:9155+).
  The tie-in is embed_vision.embedding_projection.weight 512x768 (in the mmproj as
  mm.input_projection.weight [768,512]). The image markers are "<|image>" and
  "<image|>" (mtmd.cpp:955-956), image_token_id is 258880 with boi 255999 and eoi
  258882, and the audio side (audio_tower.*, audio_token_id 258881, embed_audio
  512x1536) is OUT of scope: the owner asked for vision.
- Machine: RTX 4070 SUPER (12,282 MiB, sm_89), CUDA 13.3, 28 CPUs. Rust is built with
  cargo + `--features native`; the C side via the tree's Makefile (`make ds4 ds4-server`
  rebuilds the Rust binaries in ~13 s incrementally); the canonical /data/build builder
  refuses this tree (recorded; rule 5's iterative exception applies). The arch record
  is ds4-rs/.ds4-cuda-config.mk (`CUDA_ARCH := sm_89`), so builds are card-native.
- Quality (gate 5): mean vector agreement 0.956039571 at 768 (min 0.898186175),
  MRR@10 0.951778 -> 0.935819, recall@1 0.888278 -> 0.861611, recall@10 0.983333 ->
  0.990000 on the frozen SciFact set; the owner's declared floor and the verdict are
  section 9 of docs/retrieval-cost-gate5.txt.
- ds4-rs remotes are shared with the live ds4 project (origin Baekpica/ds4-dfm-rs, fork
  JLFN/ds4-dfm-rs, publish JLFN/embeddinggemma2-ds4-rs); /data/ds4-dfm-rs is never
  modified from this workspace.
- Guardrails: global ~/.opengrok/AGENTS.md + this repo's mount at
  .opengrok/rules/project-work-rules.md + AGENTS.md (project P-1/P-2). .gitignore
  excludes .models/, ds4-rs/, out/, graphify-{out,rs-out}/ and *.o.

7. Guardrails and open items

- P-1 (no Python in the runtime, ever) and P-2 (references are citations or one-shot instruments; no ONNX/PyTorch) hold; all five P-1 checks return zero and `sha256sum -c SHA256SUMS` passes (re-verified at this boundary).
- Owner decisions in force: the container is GGUF + `gemma_vq` payload; the anchor run was approved; quality is paramount, "not less smart, just compact"; CUDA must actually be used (phase 7 did, 9.9x/18.1x per item); and (2026-10-19) the tier-tolerant quality floor recorded in section 9 of the gate-5 doc.
- CLOSED (rule 19 item 4, unit 20): the project now has tests/qa-gate.sh and tests/run.sh, so a push without fresh QA evidence fails MECHANICALLY. What it does NOT do: assert the REPORT's content beyond the surface names and the verdict line; the QA-tester's own judgement is still what makes the evidence worth anything.
- CLOSED (owner directive 2026-10-20, "We will not use llama it must be ported all"): measured that ds4-rs never called or linked the pin, and the two default-path executions were moved to opt-in — commit 3729ec4. Gate 22 is native (our reader), the anchor is `tools/eg2_vision_anchor.sh` (run once, green, 211/211 + all four frozen dumps reproduced), the tokenizer reference is committed data (`tests/expected/tok-anchor-ids.tsv`), and the fast path was 5.1 s at that boundary (171 s now, with gates 23, 24 and 25 in it). What is NOT claimed: that the pin is gone from the machine (it is still the authority we read, and the anchor still links it when asked) — only that nothing the port builds, runs or gates by default touches it.
- CLOSED (row P2.3's cap half, unit 20): the embedding-shaped caps exist, are advertised, are enforced by name and are parity-tested across the three files that carry the item ceiling. CLOSED (quality): the floor is declared and met.
- OPEN (owner, NEW): the C CLI's tokenizer gap — ds4-c refuses the artifact by name because the family's gemma4 BPE lives Rust-side, so porting it duplicates the encoder. The shipped binaries are Rust.
- DECIDED (owner, 2026-10-19): the vision tower (phase 8) IS wanted and is now the
  next work (section 3). DECIDED in the same message: the reference for it is the
  pattern the OTHER model variants use, which is the engine-level vision contract
  (probe/encode_memory over a vision weights table, ds4.h:297-310 with the GLM-5.3
  adapter as the worked example) — NOT the glmvq family, whose vision does not
  exist.
- STILL OPEN (owner): whether the AUDIO tower is wanted; whether KV checkpoints mean
  anything for an embeddings family.
- DEFERRED (owner's redirect, 2026-10-19): unit 21, batching several items into one
  tower call (phase-doc row P2.3's last slice). Its complete spec, plus the
  reference reading already done for it, is parked at docs/eg2vq-batch21-plan.txt.
  No code was written for it.
- OPEN (environment): the canonical builder cannot build this tree until it can pass `--features native`; fixing it changes a builder shared by every project on this machine, so it is PROPOSED, not done.
- OPEN (owner, NEW): the two pushes. Outer: `git push publish dev`, range d45394f..HEAD (TWENTY-TWO commits as of the commit that carries this handoff; read the log for the exact list: the vision fixtures and their frozen reference, the llama-free boundary, the vision tower on both legs, gates 23, 24 and 25, the splice and the engine entry, and the QA passes — no artifact: out/, .models/ and ds4-rs/ are gitignored; the tracked binaries in bin/ are rebuilt by the gates, and the anchor binary is deliberately NOT tracked because it links the pin). Target publish = https://github.com/JLFN/embeddinggemma2 (rule 6 wants a fresh `gh repo view` immediately before the push). Nested: `git push publish feat/ds41-dsml-carriage:refs/heads/dev`, range 29873445..HEAD (NINE commits as of 5bd18880: the caps, the unit-20 QA corrections, the vision tower CPU leg, the citation corrections, the device tower, its comment correction, the splice, the engine entry, and media_serial). Gates already run for both: rule 10 OSV 140 packages / 0 vulnerable (both roots, at an earlier boundary — re-run before an actual push), rule 19 QA fresh and PASS for unit 24 (unit 25's pass was IN FLIGHT when this handoff was written; its report and findings are qa-evidence/qa-report.md and must be read before the next unit), rule 5/12 as recorded. Rollback if a push must be undone: `git push --force-with-lease publish d45394f:refs/heads/dev` (outer) — a force-push, so the owner's call; both local histories are kept either way.
- STANDING (owner, 2026-10-19): the owner is AFK and wants automatic continuation after each handoff and compaction; pushes stay offers. Recorded in section 1.
- The unit-20 QA findings are all closed (four corrections committed, one recorded as informational); the unit-23 and unit-24 QA passes ran under the same recorded model (deepseek-v4.1-CC-flash) and their findings are in qa-evidence/qa-report.md (unit 24's dispositions are docs/eg2vq-vision-tower-gate24.txt section 9). Unit 25's QA pass was running when this handoff was written: its report is the current qa-evidence/qa-report.md when it lands, and any finding it raises is the next session's first read.
- Nothing is in flight in either repository. The outer repo's changes since the
  unit-22 boundary are the boundary unit alone (gate 22 rewritten native, the anchor
  script added and the harness renamed, the tokenizer gate's frozen ids, SHA256SUMS,
  tests/run.sh and the four docs), committed as 3729ec4. The nested ds4-rs checkout
  is untouched: everything through unit 22, and the boundary change, is outer-repo
  work.

8. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git log --grep "Unit: .* complete" d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git status --short && git branch -vv                 # clean, on dev, TWENTY-TWO unpushed commits
    tests/run.sh                                         # the fast path (171 s, gates 23, 24 and 25 included), ending in the rule 19 gate
    sha256sum -c SHA256SUMS                              # FOURTEEN entries
    tools/eg2_loader_check.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf out/artifact-manifest.tsv
    bash tests/qa-gate.sh                                # must be green before any push
    git -C ds4-rs log --oneline -3 && git -C ds4-rs status --short   # 5bd18880, clean
    cuobjdump --list-elf ds4-rs/ds4_cuda.o | tail -2     # ds4_cuda.sm_89.cubin = the arch record is real
    cat ds4-rs/.ds4-cuda-config.mk                       # CUDA_ARCH := sm_89
    tail -8 out/phase5-gate.log                          # the agreement + retrieval tables
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'
    # the deep gates, when a number is questioned (each log is in out/):
    #   gate 4   (~20 min, the token dump is reused): tools/eg2_phase4_gate.sh <artifact> <baseline> tests/anchor/anchor-items.tsv .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
    #   gate 5   (minutes, the legs are reused):      tools/eg2_phase5_gate.sh
    #   gate 20  (~15 s, needs ds4-rs/ds4-server built): tools/eg2_embed_route_gate.sh out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf
    #   gate 22  (seconds, NATIVE since 2026-10-20; ALSO step 5 of tests/run.sh):
    #            tools/eg2_mmproj_check.sh   -> EG2VQ_VISION_MMPROJ_PASS
    #            tests/vision/ holds the two fixtures and their four frozen dumps
    #   the anchor, OPT-IN and the only thing that runs the pin (~10 s, needs
    #   /data/llama-anchor-build): tools/eg2_vision_anchor.sh
    #   the tokenizer gate against the committed ids (no pin):
    #            tools/eg2_tok_check.sh <artifact> tests/anchor/anchor-items.tsv .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
    #   gate 23  (unit 23, in the fast path as step 6): tools/eg2_vision_gate23.sh
    #   gate 24  (unit 24, in the fast path as step 7; it rebuilds ds4_cuda.o and
    #            links the oracle when a device source changed, then runs both
    #            legs in one process plus four attribution flips):
    #            bash tools/eg2_vision_gate24.sh
    #   gate 25  (unit 25, in the fast path as step 8; it builds the splice oracle,
    #            runs the splice's four legs, the engine's own probe/encode_memory
    #            entries and the refusals, about 70 s warm):
    #            bash tools/eg2_splice_gate25.sh

Trust git over this document's headings: continue with the first unit whose work is
not in the diff, which at this boundary is section 3 (unit 26, the WIRE — the serving
route and the /health row for the image path — since the splice and the engine entry
now exist and are gated, gates 23, 24 and 25 are green, and the port executes no pin
code). Every phase through 7 is landed; unit 21 (batching) is DEFERRED with its spec
parked at docs/eg2vq-batch21-plan.txt. The outer repo's HEAD is HEAD with its
published dev at d45394f (TWENTY-TWO unpushed as of the handoff chain); ds4-rs is at HEAD with
its published dev at 29873445. Anything else in either unpushed range must be read
before sections 2-6 are trusted. The artifacts, the retrieval source, every out/ log and the CUDA objects
live under gitignored paths (out/, .models/, ds4-rs/), so they never appear in a
diff: they are identified by the digests in SHA256SUMS, by the pins inside the gate
scripts, and by the evidence docs' own digests.

This document is self-contained: a new session can execute the remaining unit from
this file alone. History is deliberately absent — git log, the phase doc and the
evidence docs are the historical record.
