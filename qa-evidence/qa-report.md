QA report - unit 24: phase 8 row P4.2, the vision tower on the DEVICE (CUDA)

Unit: 24. Outer /data/embeddinggemma2, branch dev (no origin; the only remote is
publish): the unit's code commit 3a5f008 (HEAD, tests/run.sh + tools/eg2_vision_gate24.sh
+ docs/eg2vq-vision-tower-gate24.txt), last pushed publish/dev = d45394f (verified with
git rev-parse and git ls-remote --heads publish; 1691d13 is an unpushed local
docs(handoff) commit). Nested
/data/embeddinggemma2/ds4-rs, branch feat/ds41-dsml-carriage, HEAD 5183cd50 (the device
tower, Makefile, .gitignore, the C oracle's 2-line stages update); its last pushed
publish/dev = 29873445. Both trees were clean at the start of this pass except one file
neither this pass nor the unit wrote: embeddinggemma2-handoff.md carries an uncommitted
frontmatter refresh (last_updated_commit, context usage, timestamp) made at 08:49 by the
project's own session before this pass began; it is left as found. The only file this
pass wrote in either tree is this report. Scratch: /tmp/qa24, including a pristine
git-archive copy of ds4-rs at 5183cd50 (/tmp/qa24/ds4-rs) in which every falsification
edit was made, so no tracked source file was modified in place.

QA model: deepseek-v4.1-CC-flash (the model tests/qa-gate.sh records for rule 19).
Date: 2026-10-20. Host: RTX 4070 SUPER, sm_89 (the oracle's own stderr: "ds4: CUDA
backend initialized on NVIDIA GeForce RTX 4070 SUPER (sm_89)"), CUDA 13, nvcc at
/usr/local/cuda/bin/nvcc.
Method: shell, cc, nvcc/make, awk, sha256sum, cmp/grep. The pin /data/llama.cpp @
31288c9c66e59377f7788d920678a565bb89dd37 was READ ONLY at three citation sites
(rev-parse confirms the pin; sed only); it was not built or run by this pass, and there
is no Python, ONNX or PyTorch anywhere in this verification (P-1/P-2 respected).

COMMANDS RUN (all in /data/embeddinggemma2 unless noted)

  bash tools/eg2_vision_gate24.sh                                  (three times: once standalone,
                                                                   once inside each tests/run.sh run)
  ./ds4-rs/tests/parity/eg2vq_vision_cuda_oracle out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf \
      tests/vision/eg2-vision-pattern-512x384.png tests/vision/eg2-vision-gradient-512x384.png \
      --out-dir /tmp/qa24-dev
  env DS4_EG2VQ_VIS_KICK=<none|1e-7|3e-7|1e-6|3e-6|5e-6|1e-5|1e-4> <oracle> ... (my own sweep)
  env DS4_EG2VQ_VIS_{NO_ACT_BF16,CONV_BF16,NO_GELU_F16}=1 <oracle> ... and <oracle> --f32kv
  bash tools/eg2_vision_gate23.sh
  cc -O3 / -O2 (gate 23's exact flags) -o /tmp/qa24-cpu/oracle-o{3,2} tests/parity/eg2vq_vision_c_oracle.c
  /tmp/qa24-cpu/oracle-o3 soft ... / oracle-o2 soft ...   (per-row statistics)
  cd /tmp/qa24/ds4-rs && make tests/parity/eg2vq_vision_cuda_oracle   (from git archive; see F7)
  <copy oracle with the wrong rope pairing> <mmproj> <both fixtures>
  bash tests/run.sh                                                (timed)
  sha256sum -c SHA256SUMS
  the five P-1 greps of AGENTS.md, verbatim
  cd ds4-rs && cargo test -q -p ds4-core --test eg2vq_serving_caps
  bash tests/qa-gate.sh                                            (before and after writing this)

SURFACES COVERED

- ds4_gpu_eg2vq_vision_bind / _unbind / _workspace / _create / _free / _forward / _stage
  (ds4_eg2vq_vision.h:240-271, ds4_eg2vq_vision_gpu.cuh): all seven live through the
  oracle on the real mmproj. bind+workspace+create ran at start (printed "device plan
  bind 0.03 s, context 0.00 s"); forward ran 6 times (both fixtures, plus the oracle's
  own 1e-7 and 1e-5 kick re-runs) and my 8-run kick sweep; stage was read back three
  ways per fixture (patch/l0/pooled) and "soft"; free/unbind ran at exit on the PASS
  path (exit 0) and on the wrong-rope copy (exit 1, failure path exercised).
- ds4_eg2vq_vision_slot_table (ds4_eg2vq_vision_bind.inc:182-260): exercised on every
  bind; it filled all 211 slots from the file (the adapter's per-process census printed
  weights=289.88 MiB aux=62.70 MiB over 211 slots). Refusals by name were not injected
  (see WHAT I DID NOT TEST).
- cuda/eg2vq_vis.cuh primitives: every one ran in every forward (im2col, nine GEMMs,
  the two position adds, rope, the f16 K/V round, attention, gelu_mul, pool); act_round
  was exercised at both values (the NO_ACT_BF16 and CONV_BF16 flips), the gelu chain
  both ways (NO_GELU_F16), the K/V cast both ways (--f32kv). Their device entries are
  static inline, so no symbol is exported; the oracle is the driver.
- ds4_eg2vq_vision_forward with stages (ds4_eg2vq_vision_graph.inc:434-610): the stages
  path live in the CUDA oracle, the NULL path live in the C oracle and gate 23; grep
  shows exactly two callers in either tree, both oracles, so every shipping-shaped call
  passes NULL as the evidence says.
- tools/eg2_vision_gate24.sh: run standalone (exit 0, EG2VQ_VISION_GATE24_PASS) and as
  fast-path step 7 of tests/run.sh.
- Makefile rules and the two .gitignore entries: the oracle was built by the tree's own
  rule in place (no-op, current) and from scratch in /tmp/qa24 (F7).
- embed_caps (crates/ds4-core/src/serving.rs:206): the only surface the rule 19 mount
  extracts for this window; unchanged by this unit (no crates/ file in either commit's
  stat) and re-verified live here: cargo test -q -p ds4-core --test eg2vq_serving_caps
  -> "3 passed; 0 failed". The mount's coverage limit for this unit is F6.

CLAIM 1 - gate 24 passes and prints the claimed numbers: PASS

bash tools/eg2_vision_gate24.sh: exit 0, EG2VQ_VISION_GATE24_PASS, wall 58.91 s with the
oracle already current (no rebuild); the bound (0.30 / 0.025 / 0.998, stages 0.010) is
printed before the first run. Output, against the evidence doc:

  device vs CPU, worst of the two fixtures
      maxabs 0.15841668 (doc 0.15841668), relrms 0.009006 (0.009006),
      min row cosine 0.99933243 (0.99933243).                      IDENTICAL
      pattern 0.15841668 / 0.009006 / 0.99933243; gradient 0.04474682 / 0.004954 /
      0.99993688 (the commit message's pair, reproduced exactly).
  stages, device vs CPU
      patch 0.000001 / 0.000000; l0 0.000078 / 0.000034; pooled 0.008827 / 0.005011
      (doc: patch 1e-6; l0 7.8e-5 / 3.4e-5; pooled 8.8e-3 / 5.0e-3).   IDENTICAL
  device vs the frozen dumps (gate 23's comparison and bound)
      pattern  0.22244230 / 0.010428 / 0.99861157 (doc/commit the same);
      gradient 0.05363610 / 0.005155 / 0.99992922 (commit the same).   IDENTICAL
  geometry  both fixtures: canvas 528x384, region 512x384, offset 8,0, grid 33x24,
      patches 792, soft 88.
  the four flips: NO_ACT_BF16 0.014936 (1.66x), CONV_BF16 0.029459 (3.27x),
      NO_GELU_F16 0.009729 (1.08x), f32_kv 0.008678 (0.96x) - identical to the doc.
- A direct oracle run (not through the gate's output filter) reproduces the same
  numbers and the probe lines the gate filters out: weights 289.88 MiB, aux 62.70 MiB,
  workspace 566.64 MiB at max_patches 10080, vram delta 922.0 MiB of 11893.9 MiB, cpu
  f32 dense cache 576.0 MiB (doc section 7: the same five figures).
- The rerun cost: 58.91 s warm here; the commit's "about 35 s warm" is F3.

CLAIM 2 - the bound's honesty: it is a recorded finding whose justification reproduces;
the superseded constant itself is not verifiable in git (F1/F2 are documentation drift)

(a) The recorded reason is real. The oracle's internal probe prints, per fixture:
    a 1e-7 relative kick on ONE canvas value moves the device output maxabs 0.00000000,
    relrms 0.000000 (both fixtures); a 1e-5 kick moves it 0.04770827 / 0.005481 (pattern)
    and 0.03242344 / 0.004198 (gradient) - the doc's numbers verbatim. My own sweep,
    setting DS4_EG2VQ_VIS_KICK from outside on the pattern fixture's device-vs-CPU
    comparison, shows the absorption is even wider than claimed and the crossing is
    sharp:
        kicks 0, 1e-7, 3e-7, 1e-6, 3e-6, 5e-6 -> 0.15841668 / 0.009006 / 0.99933243,
            i.e. bit-identical output at eight printed decimals (the kick is deleted by
            the first block's bf16 activation quantizer);
        1e-5 -> 0.15357840 / 0.009240 / 0.99938695; 1e-4 -> 0.07516777 / 0.008301 /
            0.99986893.
    So "absorbs below the quantization granularity, amplifies above it" is a measured
    property of the tower, not an explanation invented to justify a bound.
(b) No post-hoc tuning in the file: git log -p on tests/parity/eg2vq_vision_cuda_oracle.c
    shows ONE commit (5183cd50, the file's addition) and no later edit; the constants
    (VIS_SOFT_MAXABS 0.30, VIS_SOFT_RELRMS 0.025, VIS_SOFT_MINROWCOST 0.998,
    VIS_STAGE_RELRMS 0.010) have never moved since. The first, tighter declaration
    (0.15 / 0.010 / 0.9990) exists only as the author's record in the oracle's comment
    and the doc's section 5 - it is not in git, so its existence is unverifiable; what
    is verifiable is that it is exactly consistent with the live measurements: today's
    worst maxabs 0.15841668 exceeds 0.15 while today's relrms 0.009006 and mincos
    0.99933243 pass 0.010 and 0.9990. The widening moved precisely the one number that
    fails, which is what a widened-to-pass bound looks like; the case for it is the
    envelope, and the envelope checks out: the gate's device-vs-CPU bound IS gate 23's
    own declared bound for one leg against the frozen dumps (tools/eg2_vision_gate23.sh
    lines 55-57: 0.30 / 0.025 / 0.998, the same three constants), and the noise floor is
    independently reproduced: my -O2 vs -O3 builds of the SAME CPU source, same flags,
    differ by 0.007415 / 0.005039 relrms (pattern/gradient), the same size as the
    device-vs-CPU distance (0.009006 / 0.004954). Two implementations of this tower
    cannot be held closer than each is to a recompilation of itself.
(c) The discrimination rows are real; I re-ran all four outside the gate:
    NO_ACT_BF16 0.014936 (1.66x), CONV_BF16 0.029459 (3.27x), NO_GELU_F16 0.009729
    (1.08x), f32_kv 0.008678 (0.96x) against the port's 0.009006 - identical to the
    gate's own output and the doc's table. The gate requires strict ordering (a flip
    strictly further than the port) for the three env flips and prints the f32_kv row
    without requiring it. Note the honest scale: only CONV_BF16 (0.0295) is outside the
    0.025 bound itself; NO_ACT_BF16 (0.0149) is inside it. The gate is a regression
    bound, and the doc's section 6 says so ("What the gate therefore does NOT claim:
    that one of its fine-grained flips can be seen through this bound. It cannot").
(d) The f32_kv row is recorded honestly: the script prints "0.96x" and does not fail on
    it; the commit message and doc section 6 both call it NOT discriminating and name
    the reason (below the noise floor; the port implements the f16 cast because
    clip.cpp:787-790 says so, not because this gate can see it). Verified by reading
    the script's awk (no failure clause for f32kv) and by re-measuring the number.
- As a whole: the bound is not a quiet widening-to-pass. It is gate 23's declared
  envelope, its measured justification reproduces on this host, and the bound does
  catch a real defect (see CLAIM 3's copy: l0 3.36% > 0.010, soft 1.06 > 0.025).

CLAIM 3 - the stage attribution and the rope defect: PASS

- The same object on both sides, read in the code: the CPU forward copies
  stages->patch from h immediately after the conv and the two position adds, before
  the block loop (graph.inc:496-511); the device copies L.stage_patch from its own h
  at exactly the same point, after eg2vq_vis_pos_add and before the first block
  (gpu.cuh:439-443). The h buffer is the tower's input row the first block reads
  (ln1(x = RMS(h))); patch is therefore defined identically on both sides. l0: CPU
  copies after the il==0 block's FFN residual (graph.inc:511, :589-592); device copies
  at the same place (gpu.cuh:507-509). pooled: CPU copies after the pooler and before
  the embedder's weightless RMS (graph.inc:594-607); the device's stage reads
  L.pooled, which the device pooler wrote and which the embedder's RMS does not touch
  (the RMS reads pooled and writes z, gpu.cuh:512-518). Same three objects, same
  production points.
- The rope fix is right, from both sides' code: the CPU leg calls eg2vq_rope(v, 32,
  pos_x) then eg2vq_rope(v+32, 32, pos_y) (graph.inc:326-331), and eg2vq_rope pairs
  (i, i + n_rot/2) = (i, i+16) with theta starting at the position and multiplied by
  theta_scale = 100^(-2/32) each step (ds4_eg2vq_graph.inc:66-76). The device kernel
  pairs (i, i+16) inside dims [0,32) against the X table and inside [+32,+64) against
  the Y table, both halves reading the same 16-angle host table row
  (cuda/eg2vq_vis.cuh:322-358, table :300-316). The pin says the same: gemma4v.cpp:46-73
  is two ggml_rope_ext calls each with n_dims = n_dim/2 = 32 (second one offset by 32),
  rotate_pairs pairs (i, i + n_offset) (ops.cpp:6379-6393), and theta_scale is
  powf(freq_base, -2.0f/n_dims) (ops.cpp:6455). The old (i, i+32) pairing with a
  second read at +32 was the wrong form and reached past the 64-dim head.
- Empirical falsification from the /tmp/qa24 copy (only the one line changed:
  half = HEAD_DIM / 2 instead of / 4, reproducing the recorded wrong pairing), oracle
  run over both fixtures:
      patch  relrms 0.000001 / 0.000000  (IDENTICAL to the fixed build - the patch
             stage cannot see a rope error, which is inside the blocks)
      l0     relrms 0.033569 / 0.011768  (the recorded first-run signature: 3.2% / 1.2%)
      pooled relrms 1.107384 / 0.918820; soft relrms 1.056567 / 0.859375, maxabs
             5.99 / 5.33, min row cosine 0.165 / 0.187
      exit 1, with stage FAILs on l0 and pooled and soft FAILs on all three numbers.
  So: a wrong rope pairing shows up ONLY from l0 onward, never in patch, and the gate as
  declared fails it. The doc's account (patch 1e-6 while l0 was 3.2%, which localized
  the bug in one step) is exactly this behavior; my copy reproduces it without the
  author's kernel, from the recorded description alone.

CLAIM 4 - the mechanisms, in one named place each, defaults = the reference: PASS

Read in the code, not the comments:
- bf16 activation quantization: the GEMM rounds x at the shared-tile load only when
  act_round != 0 (cuda/eg2vq_vis.cuh:210-219); the adapter passes act_round (=1 unless
  the flip) for q, k, v, o, ffn_gate, ffn_up, ffn_down and the tie-in projection, and
  conv_round (=0 unless the flip) for the patch conv alone (gpu.cuh:432-433, :440, :466-468,
  :489, :496-497, :502, :516-517). The CPU leg's eg2vq_vision_matvec applies the same
  bf16 to the activation of the seven matrices and the projection, and its conv worker
  does not (graph.inc:420-432, :273-296). Same rounding expression on both sides
  (gpu.cuh's eg2vq_vis_bf16 is the CPU's eg2vq_vision_bf16 bit for bit,
  graph.inc:96-103). With no env vars the defaults are rounding ON for the matmuls and
  OFF for the conv, which is the reference's own behaviour.
- The flip is exactly what it says: NO_ACT_BF16 removes only the activation rounding
  (the switch gates the one if (act_round) line; nothing else reads ctx->flags for the
  GEMMs); CONV_BF16 changes only the conv call's round flag; neither touches the other.
- gelu_quick f16 chain: eg2vq_vis_gelu_quick is f16(x) -> x*(1/(1+expf(-1.702x))) ->
  f16(y) (cuda/eg2vq_vis.cuh:148-152), the same four conversions in the same order as
  the CPU's eg2vq_vision_gelu_quick (graph.inc:83-88). NO_GELU_F16 substitutes one
  kernel that computes the same expression WITHOUT the two f16 round trips
  (gpu.cuh:346-353): only the f16 chain is removed.
- f16 K/V cast: eg2vq_vis_f16_round runs on k and v after the rope and after the
  weightless V RMS (gpu.cuh:480-484), exactly where the CPU calls f16_round_inplace_cpu
  (graph.inc:549-552); --f32kv sets opts.f16_kv = 0 and removes only that cast.
- No mask, kq_scale exactly 1.0, 12 heads over 12 KV heads, head_dim 64: the device
  attention kernel has no mask operand and no scale factor (the score is
  "(a0 + a1) + (a2 + a3)", cuda/eg2vq_vis.cuh:442), one warp per head over 12 heads,
  and the key range is the whole item; the CPU is the same (graph.inc:335-380). The pin
  says kq_scale = 1.0f (gemma4v.cpp:75) and the executed flash branch casts K and V to
  f16 (clip.cpp:787-790, read at the pin).
- Named device-side differences, which I confirm are named rather than hidden: the GEMM
  and the attention accumulate in f32 where the CPU's matvec and attention accumulate in
  double (kernel header, gpu.cuh:89-96 comment; verified in code); the rope table's
  trig is computed in double on the host and rounded to f32 where the CPU uses cosf/sinf
  (the kernel comment names the table and --use_fast_math as the reason; verified); the
  pool's division is (float)(acc/9.0)*scale on the device against (float)acc/9.0f*scale
  on the CPU - a one-ulp expression difference, inside the measured pooled band
  (8.8e-3), not separately measured by me.

CLAIM 5 - the CPU leg is unchanged where it must be: PASS

bash tools/eg2_vision_gate23.sh: exit 0, EG2VQ_VISION_GATE23_PASS, 19.36 s.
    pattern  maxabs 0.10275560, relrms 0.008235, min row cosine 0.99978457
    gradient maxabs 0.06645030, relrms 0.005285, min row cosine 0.99988406
  - the doc's "unchanged ones" to the last digit, and identical to the values recorded
  in docs/eg2vq-vision-tower-gate23.txt and the handoff before this unit existed. The
  gate also printed the anchor sums of squares 65828.280406 / 69223.788734 against the
  anchor's printed 65827.217181 / 69219.023404.
- The only CPU-leg change is the stages parameter; the C oracle (the leg's own driver)
  passes NULL (git show 5183cd50 -- tests/parity/eg2vq_vision_c_oracle.c: one line,
  &g, err -> &g, NULL, err), and the two callers in the tree are the two oracles.
  Independent confirmation: my own -O3 build of the C oracle (gate 23's flags) run over
  both fixtures prints 0.10275560 / 0.008235 / 0.99978457 and 0.06645030 / 0.005285 /
  0.99988406 - byte-identical to the gate, from a binary this pass built itself.

CLAIM 6 - the fast path: PASS

bash tests/run.sh: exit 0, "fast path complete", wall 84.10 s (this host, this pass);
gate 24 is step 7 of the script (read: steps 1-8, gate 24 then the rule 19 gate at step
8). All steps green: the fourteen digests, the DQVD container, the scorer self-test,
loader gate 3, gate 22 (EG2VQ_VISION_MMPROJ_PASS), gate 23 (EG2VQ_VISION_GATE23_PASS),
gate 24 (EG2VQ_VISION_GATE24_PASS), the QA gate. The fast path was 25.3 s before this
unit (the handoff's number) and is now 84.1 s here: gate 24 is the step that moved it,
as the commit message says.

CLAIM 7 - the project's constraints: PASS

- P-1's five checks, run verbatim in /data/embeddinggemma2: all zero hits (the first,
  second and fifth greps return rc=1 with no output; ldd | grep -ic returns 0; find for
  '*.py' prints nothing). The unit's own changed files contain no python/onnx/torch
  either (the two oracles are C, the kernels CUDA C++, the gates sh).
- The pin was neither built nor run by this unit: the outer commit touches only
  tools/eg2_vision_gate24.sh, tests/run.sh and the evidence doc; the nested commit
  touches the tower's C/CUDA files, the Makefile, .gitignore and the C oracle; grep for
  llama in both diffs finds only comment citations of the revision; gate 24's script
  builds nothing of the pin (read: it runs make in ds4-rs and the tree's own oracle);
  the oracle includes ds4.c and the tree's CUDA objects, nothing of /data/llama.cpp.
  The pin executions in the tree remain opt-in only: tools/eg2_vision_anchor.sh (the
  vision reference regeneration, never called by tests/run.sh or any gate) and
  tools/eg2_anchor.sh (the text anchor, reachable only from tools/eg2_tok_check.sh's
  EG2_TOK_ANCHOR-gated regeneration branch). The QA brief says "the only pin use
  remains tools/eg2_vision_anchor.sh"; that is slightly wider than the tree - the text
  anchor script also runs a pin-built binary when invoked, but only through the same
  opt-in regeneration, and nothing in this unit calls either.
- sha256sum -c SHA256SUMS: all fourteen entries OK (including the mmproj the gate binds
  and both vision fixtures).
- Tree state: the nested tree is clean; the outer tree holds only the pre-existing
  handoff frontmatter refresh and, after this pass, this report. My scratch work is
  under /tmp/qa24 and /tmp/qa24-*.

CLAIM 8 - the cheap numbers: PASS, with two timings that are host-state dependent

- MiB figures: the adapter's own report matches the doc exactly - weights 289.88 MiB,
  aux 62.70 MiB, workspace 566.64 MiB at max_patches 10080 (= 2580480/256), vram delta
  922.0 MiB, cpu f32 dense cache 576.0 MiB. Note for precision: the CPU leg's total f32
  weight bytes are 579.75 MiB, because patch_embd (2.25 MiB) and input_projection
  (1.5 MiB) have their own allocations beside the 16x36.0 MiB dense cache; the doc's
  "576.0 MiB" is the dense-cache number, which is what the oracle prints.
- Timing ratio: device 0.211-0.234 s per image, CPU 4.62-5.00 s in my runs (doc: 0.229/
  0.226 and 4.336/4.323), so the ratio is about 21x here against the doc's about 19x.
  Timings on
  this shared host are not a contract; the order is the same and the device figure is
  stable.
- Per-row statistics (computed by me with an awk comparison over all 88 rows):
      CPU -O3 vs the frozen dumps: mean row 0.76% / 0.49%, worst row 2.08% / 1.55%
        (doc: exactly these);
      DEVICE vs the frozen dumps: mean row 0.82% / 0.48%, worst row 5.29% / 1.20%
        (doc: exactly these);
      rows at or below 0.1%: 0 on both legs and both fixtures (doc: "every row of both
        legs deviates by more than 0.1%").
- Sums of squares: device dumps 65836.197907 / 69215.637590, the frozen TSVs' own
  65827.217146 / 69219.023429, the anchor's printed 65827.217181 / 69219.023404, the
  CPU -O3 leg's 65828.280406 / 69223.788734 - all three legs agree at the 1e-4 relative
  level, consistent with the doc's use of the anchor digest as a magnitude check.
- The doc's "0.78% build-to-build spread": reproduced as a class - my -O2-vs-O3 rebuild
  of the CPU leg (same flags both) measures 0.007415 / 0.005039 (0.74% / 0.50%) on
  pattern/gradient, and the unit-23 QA report recorded 0.78% / 0.06609. The unit's
  original bullet (0.064%) was -O2 WITHOUT -ffast-math, a different configuration; the
  doc keeps both, and both are consistent with my measurement. The exact doc pair
  0.78% / 0.0661 comes from the unit-23 QA pass (recorded there), not re-derived here;
  my 0.74% / 0.0822 is the same configuration class.

WHAT I DID NOT TEST / COULD NOT VERIFY

- The superseded first bound's physical existence (0.15 / 0.010 / 0.9990): not in git
  (the file was added once); only the consistency of the recorded failure with the live
  numbers is verifiable, and it holds.
- The device bind's and slot table's refusal paths (a mis-named, mis-typed or mis-sized
  tensor), the workspace-too-small and image-too-large refusals: not injected. The
  mechanisms are read (they are the CPU binder's own required_tensor/tensor_expect_layout
  calls); gate 3/23's tamper tests cover the reader, not this table.
- Leaks: no valgrind/ASan; free/unbind are called and every run exited cleanly.
- The engine entry and the caps accounting: not built (F4).
- Unit 25's subject (the splice) and the *.text.tsv dumps.
- The pin's execution (not allowed in this pass); I read three citation sites only.
- The doc's specific single-session wall clocks for the CPU/device legs and the
  "device plan bind 0.04-0.22 s" range (mine: 0.03 s); the doc's 19x ratio (mine ~21x).
- The "0.78% / 0.0661" pair as such (see CLAIM 8; reproduced as a class, not digit for
  digit).

FINDINGS (none blocks the unit)

F1 (low; the gate script's own header still describes the superseded bound as current).
tools/eg2_vision_gate24.sh lines 12-14 say the device leg "is held to 1.0e-2 relative
RMS of the CPU leg, 0.15 max |difference| and 0.9990 minimum row cosine" while the
executed constants at lines 58-60 are 0.025 / 0.30 / 0.998 and the printed bound is the
latter. The doc's section 5 and the oracle record the tighter declaration as REPLACED;
the script's header was not updated with it. No gate-behaviour impact (the executed and
printed bound is the envelope), but the unit's own artifact contradicts the unit's own
evidence text.

F2 (low; the oracle's justification comment overstates its own flips).
eg2vq_vision_cuda_oracle.c lines 73-74 say "the discrimination flips in
tools/eg2_vision_gate24.sh are 5x to 100x further out"; the same gate prints 1.66x,
3.27x, 1.08x, 0.96x. The 5x-100x scale belongs to gate 23's table against the frozen
dumps; the doc's section 6 has the correct gate-24 ratios. Comment only; the gate
requires a > 1x ordering and prints the actual numbers.

F3 (low; a timing claim in the commit message and the run.sh header is not reproduced).
"about 35 s warm" for gate 24: measured 58.91 s warm (oracle current) and 84.10 s for
the whole fast path. The gate's own parts account for it: six oracle runs, each with a
~4.6-5.0 s CPU leg per fixture, plus the device legs. The evidence doc states no wall
time, so nothing in the doc is contradicted; the figure to carry is ~59 s warm here
(plus ~90 s nvcc and ~40 s link when the CUDA object is stale).

F4 (medium; the handoff's unit-24 plan item 4 was deferred, visibly).
The handoff's unit-24 section lists four items; item 4 ("WIRE IT: an engine entry shaped
like ds4_engine_embed_gemma_vq ... and the caps question") is not built. The evidence
doc section 8 records the deferral to unit 25 and the code agrees: the only caller of
ds4_gpu_eg2vq_vision_forward in either tree is the CUDA oracle, so nothing in the
runtime reaches the device tower yet, and embed_caps was not extended for the vision
workspace. Recorded rather than hidden; unit 25's plan must carry it.

F5 (low; the handoff's row naming and the phase doc disagree, and the unit did not
correct the phase doc).
The handoff calls this unit "phase 8, row P4.2"; docs/ds4-gemma-port-phases.md has no
phase-8 row table and its P4.2 row is "Audio tower forward: 12 layers, hidden 1024"
(OPEN), with P4.3 the tie-in projections. The handoff is the plan of record and its own
precedence rule (AGENTS.md, "Where the plan and the record live") requires the
disagreeing document to be corrected in the same commit; neither 3a5f008 nor 5183cd50
touches the phase doc. Documentation follow-up, no runtime impact.

F6 (informational; the rule 19 mount still cannot see this unit's surfaces).
bash tests/qa-gate.sh demands {embed_caps} only: the nested scan paths are
"crates ds4.c cuda" with a C-definition pattern, so ds4_eg2vq_vision_gpu.cuh,
ds4_eg2vq_vision_bind.inc and ds4_eg2vq_vision_graph.inc (root of ds4-rs) are invisible,
and cuda/eg2vq_vis.cuh's primitives are eg2vq_vis_* (no ds4_ prefix). This report
therefore names and covers the unit's real surfaces explicitly (SURFACES COVERED); the
mount should widen QA_SURFACE_PATHS/QA_DB_RE in a follow-up. The mount's other three
checks (exists, fresh, operative verdict) are satisfied by this report.

F7 (informational; inherited, not introduced by this unit).
The oracle's make rule compiles ds4_cuda.cu, which needs the gitignored
.ds4-cuda-config.mk (CUDA_ARCH := sm_89): a pristine git-archive copy without it fails
nvcc with six "incomplete type nvcuda::wmma::fragment<... __nv_bfloat16 ...>" errors
rather than a clear CUDA_ARCH message. The project records the arch step in
docs/cuda-sm89-port.txt and the tree here carries the file, so the in-place gate is
unaffected; only fresh-checkout reproducibility needs the step.

post-write confirmation: bash tests/qa-gate.sh -> QA GATE: overall PASS (report exists,
fresh, operative verdict, covers embed_caps), exit 0; bash tests/run.sh -> fast path
complete in 83.17 s, exit 0, gate 23 EG2VQ_VISION_GATE23_PASS and gate 24 again
EG2VQ_VISION_GATE24_PASS on the fresh report, QA gate overall PASS.

verdict: overall PASS
