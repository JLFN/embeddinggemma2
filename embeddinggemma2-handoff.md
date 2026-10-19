---
project: embeddinggemma2
plan_start_commit: d6da36680643c48c144536caf658e360b6d30d82
last_updated_commit: d6da36680643c48c144536caf658e360b6d30d82
branch: dev
remote: 
handoff_written_at_context_usage: 12%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-19T06:50:22+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ (2026-10-19, after the reference was pinned)

Note on the frontmatter: it MUST be the first thing in the file, with
bare values (no inline comments after the value) — the handoff-hooks
parser reads it verbatim, and a commented or misplaced frontmatter
silently breaks the SessionStart probe readout. The scaffold writes
this exact shape; hand-written files must match it. <COMMIT> means the
full commit hash (plan_start_commit = the commit the current plan
began from); <PERCENT> = the context usage read from the session's
signals.json (target 25-35, hard cap 40).

0. Rules for writing THIS document (read before editing, not just before reading)

- Only write or replace this file at a clean, verified, committed boundary. Never checkpoint mid-work.
- Start drafting once context usage crosses ~25%. Finalize and commit before the auto-compact threshold — do not wait for it to begin writing. The writing is a summarization task and degrades like any other task under context pressure. Treat 25-35 as the drafting target and the threshold as the hard cap. The threshold is the harness's own cut-over: [session] auto_compact_threshold_percent in ~/.opengrok/config.toml (this machine: 60), which makes it compact this session in place — no new session is opened.
- Every claim in "Current state" must be re-verified by running the relevant command in this session, not copied or paraphrased from the previous handoff. If you cannot re-run it, mark it unverified rather than asserting it.
- Hard caps: "Current state" max 15 lines. "Guardrails and open items" max 15 lines. "Next unit plan" has no cap — it is the actual work spec. If a section would exceed its cap, cut detail; do not compress into denser prose (denser prose is where drift hides).
- Do NOT: replay prior units' logs (git log is the record), re-explain global rules already in the loaded guardrails, restate anything the staleness check in section 6 already covers, or record acceptance evidence you did not personally observe this session.

1. Session protocol (read first)

This project is a sequence of units; each unit ends with a verified, locally
committed step of the plan, and a push is offered, never taken (no remote is
configured yet).

1. Read context usage from the session's signals.json (contextTokensUsed /
   contextWindowTokens) under ~/.opengrok/sessions/<percent-encoded-cwd>/
   <session-id>/, newest session dir. If signals.json is absent, fall back to
   chat_history.jsonl bytes / 4 and say in the session report that the number
   is a fallback, not the gauge.
2. If usage is below the threshold (~/.opengrok/config.toml
   [session].auto_compact_threshold_percent, this machine 60): continue
   immediately — write the next unit's plan into section 3, execute it,
   verify, commit, re-check usage, repeat.
3. At the threshold the harness compacts this session in place; nothing is
   launched. This handoff must be written and committed BEFORE the threshold,
   because a compaction carries a summary of the conversation, not unwritten
   plan prose.
4. Preferred cadence: once the finished unit's handoff, the project memory
   entry and the commits are in place, call the compact_session tool as the
   last action of that turn.

2. Current state (2026-10-19, after unit 5 — COMPLETE, verified this session)

- Unit 4 (unchanged): /data/embeddinggemma2/ds4-rs is a working copy of the ds4
  runtime (rsync of /data/ds4-dfm-rs @ eb67d493, minus target/ and misc/): 9,126
  files, 2.63 GB, gitignored here, its own history, 17 branches, both remotes.
  Evidence: docs/ds4-rs-copy.txt, commit ef0d6f1.
- Artifact (unchanged, unit 3): out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin,
  421,758,160 B from 1,488,743,984 B (3.53x), 1376 slots, every payload
  re-encoded and compared byte for byte; text-tower mean cosine 0.9859 against
  the DeepSeek bar 0.910-0.925 (RECIPE.md sections 5-7, SHA256SUMS).
- Unit 5 (new): AGENTS.md at the workspace root is the project guardrail file and
  open-grok inspect lists it (project, ~1337 tokens) alongside the global file
  and the mount. It carries P-1 (no Python in the runtime, with four zero-hit
  gate greps) and P-2 (references are citations or one-shot instruments, never
  dependencies). Commit fd63233.
- Unit 5 (new): docs/ds4-gemma-port-reference.md pins the porting reference —
  upstream llama.cpp b11723 (31288c9c, /data/llama.cpp) gemma-embedding2,
  cited line by line — with the shape contract, the 17-per-layer tensor name
  contract from our own inventory, the double-checked attention scale of 1.0,
  the weightless V RMS, the tanh GELU, the SWI/SWA split and the per-layer rope
  bases. Commit d6da366.

3. Next unit plan — run the artifact: the EmbeddingGemma-2 text tower in ds4-rs (2026-10-19)

Status: not started. Owner direction, this session: "we will use ONNX instead we
can port the parts from llama.cpp to this local /ds4-rs we need to add gemma to
this ds4-rs", and "i dont want no python you can put this in guard rail for this
project, we can use it as test script or script to optimize the code but it will
never be in final runtime". So: the reference is llama.cpp's gemma-embedding2
(read and cited, section 4 facts and docs/ds4-gemma-port-reference.md), the
measuring instrument is an ONNX run in Rust with the cached ort crates, and the
runtime stays Rust + C + CUDA with no Python anywhere.

Goal: a runtime that loads our DQVD artifact and produces the 768-dim embeddings
this model exists for, with the measurement that shows the port matches the
original.

The port template is the work spec: https://github.com/... is not needed — read
docs/deepseek41-port-plan.md and docs/deepseek41-port-inventory.md in ds4-rs
(the plan + inventory pair every prior family followed), plus the GLM VQ family
as the closest analog in shape (a VQ release artifact: Rust release reader, C
bind record table, C graph, CUDA kernels consuming the payload in-kernel).

Phases, each ending in a gate:

P0 — container and decode. Add a DQVD release reader as a third container
identity (identify.rs:199-207 routes a directory today; ours is one file, so the
routing is on the file path and a magic check). Validate the 64-byte header, the
recipe string, the slot table (name hash, alignment, payload ranges, kinds 0/1/2/3)
and reject anything unexpected, then bind a flat record table to C. Gate: a
ds4-side dump of all 1376 slots equals the encoder's own verify output
(bin/eg2_pack_cpu check + the per-slot re-encode comparison).
P1 — the text-tower forward on the CPU reference path (ds4vq_dequant_f32 where the
VQ payloads are decoded host-side; the kernel path is P3). 24 layers, hidden 512,
bidirectional attention with sliding window 512 on twenty layers and full
attention on 5/11/17/23 (head_dim 512 there, 1 KV head), the pre-layer input
block, per-layer residual block, layer_scalar, output norm and the 768-wide
embedding projection, then mean pool and L2 normalize. Gate, and it is also the
owner's behavioural proof: on a frozen set of at least 100 sentences the 768-dim
vectors agree with the reference instrument's — mean and minimum cosine, max
element difference, and the same three numbers under MRL truncation to 512/256/128.
P2 — the entry point: a CLI verb first, then /v1/embeddings with the family
registered through the five Rust dispatch tables and the C dispatch. The route
takes the task prompt (defaults from config_sentence_transformers.json). Gate: the
route returns the same vector as the CLI, byte for byte.
P3 — device path: a dim-4 VQ kernel (cuda/ds41_vq_row.cuh:55 refuses anything but
dim 8; ds4_cuda.cu:42201 is the existing bidirectional attention kernel but it is
scalar and capped at 256 rows at :42207, and it has no sliding window), then the
text tower on the GPU. Gate: GPU embeddings equal CPU embeddings within a stated
tolerance, plus a latency and a VRAM number.
P4 — vision and audio towers (16 and 12 layers; their graphs live in llama.cpp's
gemma4 code, and our inventory already carries embed_vision/embed_audio
projections), only if the owner wants them.

Measure order: P0, then the ONNX instrument check (docs/ds4-gemma-port-reference.md
section 6 lists the four checks; the instrument must be trustworthy before P1's
gate can mean anything), then P1.

Fallback kept from the previous revision of this plan: a reconstruct mode in
eg2_pack (eg2_pack_cpu reconstruct <hf-dir> <in.bin> <out-dir> writing a
safetensors directory with all 1376 tensors, verbatim slots byte-exact). It proves
the container has no unreachable slot and gives the ONNX route something loadable
in the original's own shape.

Known limit to state in any proof: the numbers say how close the quantized runtime
is to the original checkpoint, which is a different question from how good the
original is.

4. Project facts a fresh session cannot re-derive

- What exists in this repo: .models/EmbeddingGemma-2-914f7f8 (pinned checkpoint,
  1.5 GB, gitignored); tools/eg2_vq.c + bin/eg2_vq (single-matrix ladder tool);
  tools/eg2_pack.c + bin/eg2_pack_cpu + bin/eg2_pack (plan/encode/verify/check/
  merge); tools/eg2_gpuprobe.cu + bin/eg2_gpuprobe; docs/ds4-gemma-port-reference.md
  (the port map, unit 5); AGENTS.md (the project guardrails, unit 5).
- The two llama.cpp trees: /data/llama.cpp is upstream ggml-org/llama.cpp at
  b11723 (commit 31288c9c66e59377f7788d920678a565bb89dd37, 2026-10-18, 616 MB) and
  is THE reference — src/models/gemma-embedding2.cpp, arch "gemma-embedding2"
  (src/llama-arch.cpp:64). /data/llama-prism is the PrismML-Eng fork at b10743
  (adfffbe, 2026-09-25) with only the older 300m gemma-embedding, useful as a
  cross-check and as the "what not to copy" case (it scales attention by
  1/sqrt(head_dim), this model scales by 1.0). Both MIT, both read-only, neither
  built nor invoked by the port.
- Question that shapes the port: the model's own config.json gives the layer split
  (layer_types full_attention at 5/11/17/23) and per_layer_config (head_dim 512,
  1 KV head there); llama.cpp's set_swa_pattern(6, dense_first=false) is_swa =
  il % 6 < 5 gives the same split (src/llama-hparams.cpp:8-20). Rope base per
  layer: SWA layers 1e4, full layers 1e6 (config rope_parameters; llama.cpp
  get_rope_freq_base at src/llama-model.cpp:2748 selects the same way).
- Prompts live in the checkpoint: config_sentence_transformers.json
  (STS/SentenceSimilarity → "task: sentence similarity | query: ", Retrieval/query
  → "task: search result | query: ", document → "title: none | text: "). Mean pool
  then L2 normalize (1_Pooling/config.json pooling_mode mean, include_prompt true;
  modules.json has no Dense module).
- ds4-rs/ is the runner base: the ds4 runtime at eb67d493, 2.5 GB, nested and
  gitignored, with its own git. Rust side through the canonical builder
  (/data/build/linux/build.sh -p /data/embeddinggemma2/ds4-rs, rule 5); C/CUDA
  side through its Makefile, and the prebuilt objects carried over make the first
  build incremental. Its remotes are shared with the ds4 project (origin
  Baekpica/ds4-dfm-rs, fork JLFN/ds4-dfm-rs), so a push from here is shared state
  — confirm before doing it. /data/ds4-dfm-rs itself is the live ds4 project and
  is not to be modified from this workspace.
- Build commands for the tools are in each tool's header and in RECIPE.md section
  4. The reference headers emit harmless warnings (unused static plan-table paths).
- Reference toolchain pinned at /data/YoungAi revision a4e6d31. The payload rule
  the container's kind 0 obeys is ds41_vq_fmt.h:7-12, read in ds4-rs.
- Environment: RTX 4070 SUPER, 12282 MiB, sm_89, CUDA 13.3, 28 CPUs (i7-14700F);
  nvcc flags for vq_gpu.cu per the reference Makefile:197 (nvcc -O3 -arch=native
  -c, no -fmad=false — the reference validates the VQ GPU path by quality, not by
  bytes, vq_gpu.cu:7).
- transformers 5.14.1, torch 2.13.0+cu130 installed system-wide, and the installed
  transformers has NO embedding_gemma module, so it cannot load this model at all
  (measured: no file in the package matches embedding_gemma; its gemma family is
  diffusion_gemma, gemma, gemma2, gemma3, gemma3n, gemma4, gemma4_assistant,
  gemma4_unified, paligemma, recurrent_gemma, shieldgemma2, t5gemma, t5gemma2,
  vaultgemma). The checkpoint records transformers_version 5.18.0.dev0. This is why
  the instrument is ONNX, not transformers.
- Guardrails: global ~/.opengrok/AGENTS.md (core) + the mount
  .opengrok/rules/project-work-rules.md (symlink to the shared file, do not edit
  it for project-specific rules) + this repo's own AGENTS.md (project rules; the
  place for owner instructions). .gitignore excludes .models/, ds4-rs/,
  graphify-out/, graphify-rs-out/, out/ and *.o.

5. Guardrails and open items

- Owner rule in force (this repo's AGENTS.md): no Python in the runtime, ever;
  Python only as a test/optimization script or a one-time evidence producer, and
  never reachable from the runtime. Its four gate greps all return zero today.
- Reference policy: llama.cpp and any other upstream is a citation, never a
  dependency; a third-party conversion (ONNX, GGUF) is not ground truth until it
  is checked against the pinned source.
- OPEN (owner): which ONNX artifact becomes the instrument and whether its export
  includes the pre-layer input, the embedding projection, pooling and
  normalization (docs/ds4-gemma-port-reference.md section 6). Until that is
  checked, P1's gate has no denominator.
- OPEN (owner): whether the dim-4 VQ device kernel (P3) is in scope now or whether
  the device path waits for a dim-8 sibling artifact. Dim 4 / K=4096 is the
  higher-quality tier (text tower 0.9869 against 0.9218 at the same 12-bit index).
- OPEN (owner): whether P4 (vision and audio towers) is wanted at all.
- OPEN (owner): the scope reading — the VQ artifact covers all three towers, which
  still rests on a quality statement rather than an explicit list.
- OPEN: rule 12 graphify graph rebuild and rule 19 QA evidence before any push.
  Rule 10 applies to ds4-rs (it carries a Cargo.lock); this outer repo is not Rust.
- Dead item, removed this session: transformers 5.18.0.dev0 in an isolated venv.
  The owner chose ONNX, and the installed transformers cannot load the model
  anyway, so no venv is to be created.

6. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git diff --stat d6da36680643c48c144536caf658e360b6d30d82..HEAD
    git log --grep "Unit: .* complete" d6da36680643c48c144536caf658e360b6d30d82..HEAD
    ./bin/eg2_pack_cpu check out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin
    sha256sum -c SHA256SUMS
    git -C ds4-rs log --oneline -1
    open-grok inspect | sed -n '/Project Instructions/,/^$/p'

Trust git over this document's headings: continue with the first unit whose work
is not in the diff.

Staleness check at the time of writing: nothing but this handoff's own commit is
expected after d6da366 (the reference commit). Anything else must be read before
sections 2-5 are trusted.

This document is COMPLETE and self-contained: a new session can execute the
project end to end from this file alone. It carries the plan ahead, the facts
git cannot provide, and the guardrails. History is deliberately absent — git
log and the project memory entry are the historical record.
