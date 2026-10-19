---
project: embeddinggemma2
plan_start_commit: ef0d6f1da814bbfd28e18d754f633eee4f5727ea
last_updated_commit: ef0d6f1da814bbfd28e18d754f633eee4f5727ea
branch: dev
remote: 
handoff_written_at_context_usage: 38%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-19T06:28:59+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ (2026-10-19, after the ds4 runtime copy)

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

2. Current state (2026-10-19, after the runtime copy — COMPLETE, verified this session)

- Unit 4 outcome: /data/embeddinggemma2/ds4-rs is a working copy of the ds4
  runtime (rsync of /data/ds4-dfm-rs @ eb67d493, minus target/ and misc/): 9,126
  files, 2,631,391,376 B, gitignored here, with its own history, 17 branches and
  both remotes. Commit ef0d6f1; evidence in docs/ds4-rs-copy.txt.
- Verified on it: HEAD and the git ls-files -s hash match the source over 2,561
  files, tree clean, git fsck clean, make -n ds4 resolves inside the copy with
  zero references to the source path, the source's six worktrees are intact, and
  the 69 GB of cargo target/ was left behind because 16 fingerprint files bake
  /data/ds4-dfm-rs, so it rebuilds at the new path regardless.
- Unchanged from unit 3: the artifact is
  out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin, 421,758,160 B from
  1,488,743,984 B (3.53x), 1376 slots (461 VQ at dim 4 / K=4096, token table
  q8_0, 914 verbatim), every payload re-encoded and compared byte for byte,
  text-tower mean cosine 0.9859 against the DeepSeek bar 0.910-0.925. Detail in
  RECIPE.md, docs/allocation-plan.txt and SHA256SUMS.
- Owner direction: dim 4 ships because it measures 0.9869 against 0.9218 for dim
  8 at the same 12-bit index width — smaller must not mean dumber.

3. Next unit plan — run the artifact: the EmbeddingGemma-2 runner (2026-10-19, after the copy)

Status: not started. The owner asked for it in these words: "lets copy
/data/ds4-dfm-rs/ so we can make a code to run this model with". The copy is done
(unit 4); this unit is the code.

Goal: a runtime that loads the artifact and produces the 768-dim embeddings this
model exists for, with the measurement that shows it matches the original.

What the recon settled (read this session in ds4-rs, not recalled):
- Family registration is an enum plus five dispatch tables, no plugin system:
  ModelFamily/Variant (crates/ds4-core/src/shape.rs:20-35, :84-104),
  route_architecture (shape.rs:344-363), then arms in
  crates/ds4-core/src/validate.rs:1949-1974, bind.rs:990-1073,
  layout.rs:3782-3801, serving.rs:661-1050 and serving_host.rs:225-545, plus a
  native binder in ds4.c (ling3vl_bind :9916, step37_bind_common :9926) and a
  graph include (ds4.c:46292-46293). A port is 15-25 files, with ds4.c,
  ds4_gpu.h and the Makefile the heavy touch points.
- Weights come from mmap GGUF v3 (crates/ds4-core/src/gguf.rs:10-11) with custom
  type ids (crates/ds4-core/src/tensors.rs:105-112, e.g. (42,"vqblob",1,1)) — but
  a DIRECTORY path already routes to a different container reader
  (crates/ds4-core/src/identify.rs:199-207: ling31_vq::is_release /
  glm_vq::identify), so a DQVD reader as a third container identity fits a shape
  the runtime already has.
- Our payload is already the runtime's payload: ds41_vq_fmt.h:8-12 documents
  [u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols][codebook nc*dim f16][gains
  rows f16][index stream], "Value = codebook[idx][d] * g_r[row]", which is
  byte-for-byte what eg2_pack writes. The host decoder ds4vq_dequant_f32
  (ds41_vq_fmt.h:72-104) is dim-generic and width-generic (nbit derived from nc
  at :87), so it decodes our dim 4 / K=4096 tier as it stands.
- The device path is NOT compatible: cuda/ds41_vq_row.cuh:55 rejects every dim
  but 8 ("this kernel only writes dim 8 (the vq8 recipe)"), and the Rust device
  open rejects it too (crates/ds4-core/src/vq.rs:40-44 VQ_DIM, :246-247 BadDim).
- There is nothing to reuse for an encoder. No embeddings route: the server
  dispatches /v1/models, /metrics, /v1/stats, /v1/batch only
  (crates/ds4-server/src/serve.rs:1169-1240). No pooling: every family ends in a
  logits head (ds4_step37_graph.inc:309, ds4_ling3vl_graph.inc:690). Text
  attention is causal everywhere (ds4_gpu.h:3656/3669/5130, clamped in
  ds4_cuda.cu:19106/19163/19233). The one non-causal attention in the tree is the
  vision segment kernel qwen4exp_vision_attention_row (ds4_cuda.cu:29197-29237,
  used at ds4_step37_vision.inc:125 and ds4_ling3vl_vision.inc:399) — a template
  for the masking style, not a text-encoder body.

Phases, each ending in a gate, in the order the documented port template
prescribes (docs/deepseek41-port-plan.md:131-150; the plan + inventory pair to
copy is docs/deepseek41-port-plan.md and docs/deepseek41-port-inventory.md):

P0 — container and decode. Teach the copy to identify and read the DQVD container
(kind 0 through ds4vq_dequant_f32, kind 3 through ds4_deq_q8_0, kind 2 by copy)
and prove the decode: every slot must match what the encoder's own verify
produced. Gate: a ds4-side dump of all 1376 tensors equals the eg2_pack verify
output.
P1 — the text-tower forward for an encoder: bidirectional attention over the
sequence, 24 layers at hidden 512, head_dim 256, intermediate 2048, embedding_dim
768, mean pooling then L2 normalize. Gate, and it is also the proof the owner
asked for: on a frozen set of at least 100 sentences, the 768-dim embeddings agree
with the original checkpoint's — mean and minimum cosine, max element difference,
and the same three numbers under MRL truncation to 512/256/128. This needs the
reference runtime as well: transformers 5.18.0.dev0 in an isolated venv, still an
open owner decision (section 5).
P2 — an entry point: a CLI verb first, then the /v1/embeddings route with the
family registered through the five dispatch tables. Gate: the route returns the
same vector as the CLI.
P3 — device path: a dim-4 VQ kernel (or a dim-8 sibling artifact) and the text
tower on the GPU. Gate: GPU embeddings equal CPU embeddings within a stated
tolerance, plus a latency number. This is where VRAM and speed get answered by
measurement instead of estimate.
P4 — vision and audio towers (16 and 12 layers), only if the owner wants them;
their weights are already in the artifact.

Open owner decisions carried into this unit (rule 20): the shape of P1/P2 (a
standalone encoder binary inside the copy versus full family plus serving
integration), and whether P3's dim-4 kernel is in scope at all.

Fallback kept from the previous revision of this plan, still worth doing: a
reconstruct mode in eg2_pack (eg2_pack_cpu reconstruct <hf-dir> <in.bin> <out-dir>
writing a standard safetensors directory with all 1376 tensors, verbatim slots
byte-exact). It proves the container has no unreachable slot and gives the
reference runtime something loadable in the original's own shape.

Known limit to state in any proof: the numbers say how close the quantized runtime
is to the original checkpoint, which is a different question from how good the
original is.

4. Project facts a fresh session cannot re-derive

- What exists: .models/EmbeddingGemma-2-914f7f8 (pinned checkpoint, 1.5 GB,
  gitignored); tools/eg2_vq.c + bin/eg2_vq (single-matrix ladder tool);
  tools/eg2_pack.c + bin/eg2_pack_cpu + bin/eg2_pack (plan/encode/verify/check/
  merge, sharded encode); tools/eg2_gpuprobe.cu + bin/eg2_gpuprobe (GPU facts);
  bin/eg2_pack (the CUDA build) exists only as an A/B artifact — for the shipped
  tier the kernels cannot run (81920 B codebook vs the 49152 B gate), so the
  release binary is bin/eg2_pack_cpu and the container says encoder=cpu.
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
- transformers 5.14.1, torch 2.13.0+cu130 (torch sees CUDA). The checkpoint
  requires transformers 5.18.0.dev0. No auto_map / remote code in the checkpoint.
- Guardrails mount: /data/embeddinggemma2/.opengrok/rules/project-work-rules.md
  listed=yes, excluded via .git/info/exclude. .gitignore excludes .models/,
  ds4-rs/, graphify-out/, graphify-rs-out/, out/ and *.o.

5. Guardrails and open items

- Global guardrails: always-on core in ~/.opengrok/AGENTS.md (0, 0.1, 1, 2, 4, 8,
  15, 16, 17, 18, 20, 21, 22); project-work rules 3, 5, 6, 7, 9, 10, 12, 13, 14,
  19, 23 via the mount above.
- Repo hygiene: this repo has no remote, so rule 23's push offer is an offer to
  create one. The artifact (421 MB) and ds4-rs/ are gitignored; SHA256SUMS,
  RECIPE.md, docs/, tools/ and this handoff are committed.
- OPEN (owner): the shape of the runner — a standalone encoder binary inside
  ds4-rs versus the full family port with the serving surface (section 3, P1/P2).
- OPEN (owner): whether the dim-4 VQ device kernel (section 3, P3) is in scope, or
  whether the device path waits for a dim-8 artifact. The two are not equivalent:
  dim 4 / K=4096 is the higher-quality tier (text-tower cosine 0.9869 against
  0.9218 for dim 8 at the same 12-bit index).
- OPEN (owner): installing transformers 5.18.0.dev0 into an isolated venv
  (/data/embeddinggemma2/.venv), which P1 needs in order to have an original to
  compare against. Nothing else depends on it, but it is a dependency change and
  rule 20 wants it stated before it happens.
- OPEN (owner): whether to pursue gguf-tools/quantize/v41_vq.cu, the reference's
  device-resident encoder for dim 8 / K=4096, as the encoder-side GPU route — it
  would need its own payload layout and its own quality ladder.
- OPEN (owner): the scope reading (all three towers VQ'd) still rests on a
  quality statement rather than an explicit list.
- OPEN: rule 12 graphify graph rebuild and rule 19 QA evidence before any push.
  Rule 10 now applies to ds4-rs, which carries a Cargo.lock; this outer repo is
  not Rust and has none.

6. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline ef0d6f1da814bbfd28e18d754f633eee4f5727ea..HEAD
    git diff --stat ef0d6f1da814bbfd28e18d754f633eee4f5727ea..HEAD
    git log --grep "Unit: .* complete" ef0d6f1da814bbfd28e18d754f633eee4f5727ea..HEAD
    ./bin/eg2_pack_cpu check out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin
    sha256sum -c SHA256SUMS
    git -C ds4-rs log --oneline -1

Trust git over this document's headings: continue with the first unit whose work
is not in the diff.

Staleness check at the time of writing: nothing but this handoff's own commit is
expected after ef0d6f1 (the copy commit). Anything else must be read before
sections 2-5 are trusted.

This document is COMPLETE and self-contained: a new session can execute the
project end to end from this file alone. It carries the plan ahead, the facts
git cannot provide, and the guardrails. History is deliberately absent — git
log and the project memory entry are the historical record.
