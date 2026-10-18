---
project: embeddinggemma2
plan_start_commit: ed2fa0be41e80d3de2ccdd181a642f87b1552f5d
last_updated_commit: ed2fa0be41e80d3de2ccdd181a642f87b1552f5d
branch: dev
remote: 
handoff_written_at_context_usage: 34%   (recorded fact at this event; target 25-35, hard cap 40)
handoff_written_at: 2026-10-18T22:24:24+02:00
session_id: 01a1504c-af14-7c62-be24-2051df3b25e8
boundary_state: clean
---

Handoff + executable plan — EmbeddingGemma-2 VQ (2026-10-18, after unit 3)

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

2. Current state (2026-10-18, after unit 3 — COMPLETE, verified this session)

- Unit 3 outcome: the artifact exists.
  out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin, 421,758,160 B
  (421,604,557 B payload), from a 1,488,743,984 B bf16 checkpoint: 3.53x
  smaller. sha256 8902fb8a86ef7fdbd1dd079f08a0d1c4210237c1e9681d5545c983ccf0ce43fe.
  1376 slots: 461 VQ matrices at dim 4 / K=4096, the token table at q8_0, 914
  tensors verbatim.
- Quality measured from the merged file (decode every payload through the
  reference's reader): text-tower 216 matrices mean cos 0.9859 / 16.69% relL2,
  text-ple 0.9857 / 16.86, audio-attn 0.9992 / 3.64, audio-ffn 0.9993 / 3.35,
  audio-frontend 0.9936 / 9.53, vision-attn 0.9853 / 17.07, vision-ffn 0.9841 /
  17.76, token table 1.0000 / 0.55. Worst single matrix 0.9740 (vision encoder
  layer 0 attention o_proj); nothing lower. DeepSeek bar is 0.910-0.925.
- Verified: structure OK on read-back (1376 slots, names, hashes, order,
  alignment, ranges, reference byte accounts); 14 parallel verify runs covering
  all 1376 slots re-encoded and compared byte for byte, all identical; source
  bytes sum exactly to the checkpoint's tensor total.
- Cost: shards 19:42:39Z-20:02:16Z, merge 1.4 s, verifies to 20:21:45Z; 39 min
  wall on 14 cores against ~2 h single-process.
- GPU, all measured by tools/eg2_gpuprobe.cu: pageableMemoryAccess 0, so vq_gpu.cu's
  host-pointer kernels fault (cudaErrorIllegalAddress, 0/200000 indices written);
  the reference's vqg_host_alloc pinning fixes it (146x the CPU scalar loop on
  that stage); driven end to end the GPU path is 1.25-3x SLOWER per tensor on
  this card (89.6 s vs 58.8 s over a 92-tensor subset, identical cosines); and
  dim 4 / K=4096 cannot run at all (81920 B codebook vs the 49152 B gate).
- q8_K is not a weight format here (quants_p1.inc.c:55 can_quantize=false, no
  encoder, no ds4_deq_q8_K, engines use Q8_K for activations only). q8_0 is
  supported on both ends (quants_p3.inc.c:276, ds4_quantfmt.h:45).
- Owner decisions this session: asked for vq8 + q8_K, then "I want high quality";
  measurement showed dim 8 is the size path (0.9111-0.9495 at K=4096) and dim 4
  the quality path, so the artifact ships vq4x4096 with a q8_0 token table.

3. Next unit plan — the behavioural proof (2026-10-18, after unit 3)

Status: not yet started. The owner asked for it in these words: "when we are done
i want proof what this model can do". Weight-reconstruction cosine is not that
proof; this unit produces it.

Goal: show, end to end, what the artifact can do relative to the original
checkpoint, in one runtime on one input set.

Step 1 — reconstruct mode in eg2_pack (acceptance: `eg2_pack_cpu reconstruct
<hf-dir> <in.bin> <out-dir>` writes a standard safetensors directory containing
all 1376 tensors with the checkpoint's names, shapes and dtypes, and a byte
comparison of the verbatim slots against the source is exact).
- VQ slots decode through vq_unpack_dequant, q8_0 slots through ds4_deq_q8_0, and
  verbatim slots are copied. Write BF16 for tensors that were BF16 and F32 only
  where the source was F32; never silently widen, so the reconstruction is
  loadable by the same loading path as the original.
- This is also the strongest container proof available: if the reconstruction
  loads and runs, no slot can be missing, misnamed or mis-shaped. The safetensors
  header is JSON written by hand (write_safetensors: header length, then 8-byte
  aligned offsets) — the format is 8-byte-aligned little-endian data plus a JSON
  header, and it must be written in C, no Python.

Step 2 — a runtime for the ORIGINAL checkpoint (acceptance: the pinned checkpoint
loads and emits a 768-dim embedding in the venv, printed with its dtype).
- The checkpoint pins transformers 5.18.0.dev0 and this machine has 5.14.1, with
  no remote code in the checkpoint, so nothing here can load the model today.
  Install into an ISOLATED venv (e.g. /data/embeddinggemma2/.venv) with pip, not
  into the system python which other projects use. Record the exact versions
  (transformers, sentence-transformers, torch) in the recipe and in the proof log.
- This is a dependency change: state it to the owner in the session report before
  installing, per rule 20.

Step 3 — embedding agreement (acceptance: docs/proof-embeddings.txt carries, for a
fixed sentence set of at least 100 sentences, the mean and minimum cosine between
the original's embeddings and the reconstruction's, the max absolute element
difference, and the same three numbers under MRL truncation to 512/256/128).
- Inputs: the model card's own usage text, plus a small public set (a fixed slice
  of a public sentence-similarity or retrieval dataset). Freeze the exact list in
  the repo so the numbers are reproducible.
- Both models must be fed identical tokenization (same tokenizer.json), and the
  pooling/normalization composition from 1_Pooling + 2_Normalize.

Step 4 — behaviour (acceptance: docs/proof-retrieval.txt carries one retrieval or
ranking metric for BOTH models on the same labelled set, plus the delta).
- Prefer a small, deterministic task the two models can run in minutes: e.g.
  rank 100 queries against 1000 candidate passages and report recall@1/10 and MRR
  for the original and for the reconstruction.
- If the runtime handles the vision and audio towers, extend the same comparison
  to a handful of images and audio clips and say plainly if it does not.

Step 5 — record and commit: docs/proof-embeddings.txt, docs/proof-retrieval.txt,
the frozen input lists, the exact commands and the pinned runtime versions.
Anything that cannot be measured is written as unmeasured, with the reason.

Known limits to state in the proof: the artifact is verifiable but not runnable in
ds4-dfm-rs (no embedding_gemma2 family, no embeddings endpoint), so the only way
to run it today is the reconstruction in an HF runtime; and the numbers say how
close the quantized model is to the original, which is a different question from
how good the original is.

4. Project facts a fresh session cannot re-derive

- What exists: .models/EmbeddingGemma-2-914f7f8 (pinned checkpoint, 1.5 GB,
  gitignored); tools/eg2_vq.c + bin/eg2_vq (single-matrix ladder tool);
  tools/eg2_pack.c + bin/eg2_pack_cpu + bin/eg2_pack (plan/encode/verify/check/
  merge, sharded encode); tools/eg2_gpuprobe.cu + bin/eg2_gpuprobe (GPU facts);
  bin/eg2_pack (the CUDA build) exists only as an A/B artifact — for the shipped
  tier the kernels cannot run (81920 B codebook vs the 49152 B gate), so the
  release binary is bin/eg2_pack_cpu and the container says encoder=cpu.
- Build commands are in the header of each tool and in RECIPE.md section 4. The
  reference headers emit harmless warnings (unused static plan-table paths).
- Reference toolchain pinned at /data/YoungAi revision a4e6d31. The future loader
  is /data/ds4-dfm-rs @ 233dc7a9 (no gemma string anywhere, no embeddings
  endpoint; ds41_vq_fmt.h:7 is the DQVQ value rule the container's kind 0 obeys).
- Environment: RTX 4070 SUPER, 12282 MiB, sm_89, CUDA 13.3, 28 CPUs (i7-14700F);
  nvcc flags for vq_gpu.cu per the reference Makefile:197 (nvcc -O3 -arch=native
  -c, no -fmad=false — the reference validates the VQ GPU path by quality, not by
  bytes, vq_gpu.cu:7).
- transformers 5.14.1, torch 2.13.0+cu130 (torch sees CUDA). The checkpoint
  requires transformers 5.18.0.dev0. No auto_map / remote code in the checkpoint.
- Guardrails mount: /data/embeddinggemma2/.opengrok/rules/project-work-rules.md
  listed=yes, excluded via .git/info/exclude. .gitignore excludes .models/,
  graphify-out/, graphify-rs-out/, out/ and *.o.

5. Guardrails and open items

- Global guardrails: always-on core in ~/.opengrok/AGENTS.md (0, 0.1, 1, 2, 4, 8,
  15, 16, 17, 18, 20, 21, 22); project-work rules 3, 5, 6, 7, 9, 10, 12, 13, 14,
  19, 23 via the mount above.
- Repo hygiene: no remote configured, so rule 23's push offer is an offer to
  create one. The artifact (421 MB) is gitignored; SHA256SUMS, RECIPE.md, the
  docs and the tool sources are committed.
- OPEN (owner): installing a newer transformers in an isolated venv, which the
  proof unit needs. Nothing else in the project depends on it, but it is a
  dependency change and rule 20 wants it stated before it happens.
- OPEN (owner): whether to pursue gguf-tools/quantize/v41_vq.cu, the reference's
  device-resident encoder for dim 8 / K=4096, as the GPU route — it would need
  its own payload layout and its own quality ladder.
- OPEN (owner): the scope reading (all three towers VQ'd) still rests on a
  quality statement rather than an explicit list.
- OPEN: rule 12 graphify graph rebuild and rule 19 QA evidence before any push;
  rule 10 has no Cargo.lock to scan (this repo is not Rust).

6. Progress and staleness check (run this first, every session)

    cd /data/embeddinggemma2
    git log --oneline ed2fa0be41e80d3de2ccdd181a642f87b1552f5d..HEAD
    git diff --stat ed2fa0be41e80d3de2ccdd181a642f87b1552f5d..HEAD
    git log --grep "Unit: .* complete" ed2fa0be41e80d3de2ccdd181a642f87b1552f5d..HEAD
    ./bin/eg2_pack_cpu check out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin
    sha256sum -c SHA256SUMS

Trust git over this document's headings: continue with the first unit whose work
is not in the diff.

Staleness check:

    cd /data/embeddinggemma2 && git log --oneline 7ec9b56dfc0aa00ad5279d1f7b4c79a1a36eebfc..HEAD

Three commits carrying unit 3 (the tooling, the artifact, this handoff) are
expected; anything else must be read before sections 2-5 are trusted.

This document is COMPLETE and self-contained: a new session can execute the
project end to end from this file alone. It carries the plan ahead, the facts
git cannot provide, and the guardrails. History is deliberately absent — git
log and the project memory entry are the historical record.
