QA report - unit 23: phase 8 row P4.1, the vision tower on the CPU reference path

Unit: 23. Outer /data/embeddinggemma2, branch dev: the unit's code commit 2450b9a,
HEAD at the end of this pass fb3b819 (the unit-23-close handoff rotation, committed
by the project's own session during the pass); last pushed publish/dev = d45394f.
Nested /data/embeddinggemma2/ds4-rs, branch feat/ds41-dsml-carriage, HEAD f459e924,
byte-identical before and after this pass; last pushed publish/dev = 29873445. Both
trees were clean at the start; the only file this pass wrote in either tree is this
report. Scratch: /tmp/qa23, including a pristine git-archive copy of ds4-rs at
f459e924 (/tmp/qa23/rs) in which every falsification edit was made, so no tracked
source file was modified in place.

QA model: deepseek-v4.1-CC-flash (the model tests/qa-gate.sh records for rule 19).
Date: 2026-10-20. Host: RTX 4070 SUPER (sm_89); this unit's path is CPU-only.
Method: shell, cc, make, awk, sha256sum, cmp/diff, the project's own gguf_inventory
reader. The pin /data/llama.cpp @ 31288c9c66e59377f7788d920678a565bb89dd37 was READ
ONLY (citations); it was not built or run, tools/eg2_vision_anchor.sh was not run and
there is no Python, ONNX or PyTorch anywhere in this verification (P-1/P-2 respected).

SURFACES COVERED

- ds4_eg2vq_vision_prepare (ds4_eg2vq_vision.h:148-155, graph.inc:147-257): live on
  both frozen fixtures via the oracle's prep mode in gate 23 (geometry, the
  structural padding 1*8 0*512 1*8 / 0*384, the positions recomputed from the rule).
- ds4_eg2vq_vision_forward (h:156-164, graph.inc:434-610): live on both fixtures in
  gate 23 and in seven oracle builds/runs of mine (baseline, three flips, the matvec
  kick, three -O2 builds), each a full bind+forward over the real mmproj.
- ds4_eg2vq_vision_bind (bind.inc:61-141) and ds4_eg2vq_vision_free (bind.inc:143-150):
  bind runs in every oracle soft run (211 tensors by name); free is called by the
  oracle (oracle.c:170) and every run exited 0. No leak check (no valgrind/ASan).
- qwen4exp_resize_rgb_pillow (ds4_resize_pillow.inc): byte-identity against the
  pre-move revision plus a dynamic old-vs-new harness through the bicubic path
  (CLAIM 3).
- tools/eg2_vision_gate23.sh: run standalone (exit 0, 19.8 s) and inside tests/run.sh;
  its output compared with the doc's section 6.
- embed_caps: the only surface the rule 19 mount extracts for this window; unchanged
  by this unit (no crates/ file in either commit's stat), still exported at
  crates/ds4-core/src/serving.rs:206, its unit-20 live evidence stands. The mount's
  coverage limits for this unit are FINDING F4.

CLAIM 1 - gate 23 is green and the doc's section 6 is its output verbatim: PASS

- bash tools/eg2_vision_gate23.sh: exit 0, EG2VQ_VISION_GATE23_PASS. pattern maxabs
  0.10275560 (bound 0.30), relrms 0.008235 (0.025), min row cosine 0.99978457
  (0.998); gradient 0.06645030 / 0.005285 / 0.99988406. All six equal the doc to the
  last digit, as do the sums of squares 65828.280406 / 69223.788734 against the
  anchor's printed 65827.217181 / 69219.023404 (1.6e-5 / 6.9e-5 relative). The bound
  is a constant printed before the run and never widened.
- Wall time 19.8 s (the doc says ~19 s); the only fields that differ from the doc's
  captured output are the wall-clock bind/tower seconds (0.17/4.75 and 0.17/4.72
  versus 0.19/4.38 and 0.19/4.57), which cannot be byte-stable between runs.
- Independent reproduction: my own baseline oracle (pristine copy, exact gate flags)
  prints relrms 0.008235, maxabs 0.10275560, min row cosine 0.99978457, sumsq
  65828.280406 on pattern - identical to the gate's own awk numbers.

CLAIM 2 - every load-bearing pin citation is TRUE (read in /data/llama.cpp @ 31288c9c,
tree clean): PASS (the non-load-bearing pointers are FINDING F1)

- vec.h:47 GGML_GELU_QUICK_FP16 defined unconditionally - TRUE (no guard; the f32
  path is the #ifdef at :1047); vec.h:1047-1057 gelu_quick_f32 through the f16 table
  - TRUE (:1048-1055 quantize to f16, look up, widen back); vec.h:964 and :1031 the
  formula.
- ggml-cpu.c:395-400 the BF16 entry declares .vec_dot_type = GGML_TYPE_BF16 - TRUE
  (:398); ggml-cpu.c:1328-1340 mul_mat converts src1 to vec_dot_type, widening only
  when it is F32 - TRUE; ggml-cpu.c:3884-3891 the table's construction - TRUE.
- ggml-impl.h:623-635 the bf16 rounding - TRUE (:634 the exact add-0x7fff-plus-LSB
  expression the port copies); ggml.c:4785-4789 conv2d = mul_mat(im2col, weight)
  with the im2col F32 for a bf16 weight - TRUE (the conv exception).
- clip.cpp:1414-1429 the ffn_op assignment: FFN_GELU_QUICK when neither key is set -
  TRUE (:1428); clip-model.h:91 the FFN_GELU default - TRUE.
- clip.cpp:787-790 the flash branch's f16 cast of K and V - TRUE (:787 the branch,
  the casts at :790-791); clip.cpp:3928-3961 AUTO -> ENABLED - TRUE; clip.cpp:5562-5568
  the pos_x/pos_y fill - TRUE.
- gemma4v.cpp:9, :12-16, :26-43, :46-73, :75, :83-99, :104-114 - all TRUE against the
  numbered file (the *2-1, the conv with no bias, the two table views with tbl_y at
  pos_size*nb1, the two rope calls and set_offset, kq_scale 1.0, the pooler with
  sqrt(n_embd), the embedder RMS and 768->512 projection).
- Also TRUE: ggml-cpu/common.h:9 (Q_TILE 64); ops.cpp:9171 and :9285-9350 (tiled
  flash, K tiles f16->f32 at :9343-9346); ops.cpp:6455 (theta_scale, op's own n_dims);
  mtmd-helper.cpp:404 (stbi_load_from_memory, no EXIF); mtmd-image.cpp:7-14 (:9
  dst.from_u8 - the /255 itself is clip-impl.h:783), :57-96, :122-156, :767-791,
  :775-782; clip-model.h:67 and :207-212; clip.cpp:1726, :1734, :2202-2230, :254-256,
  :440-465, :457-478, :481-483, :519-529, :337.
- The executed-facts citations are also in the shipped anchor log: load_hparams ffn_op
  gelu_quick; warmup: flash attention is enabled; image_tokens->nx = 88; clip_encode:
  copying image 1/1 to input buffer (nx=528, ny=384).

CLAIM 3 - the resampler move is behaviour preserving: PASS

- git show f459e924^:ds4.c lines 26635-26778 and ds4_resize_pillow.inc lines 20-163
  are 144 lines each with the SAME sha256 2936b637...3ce867: the typedef and the five
  functions are byte-for-byte the moved bodies. (The include's ":26636-26779" pointer
  is off by one: FINDING F1.)
- The frozen fixtures take only the resampler's identity branch (region == source), so
  a scratch harness compiled the old bytes and the new .inc over one deterministic
  700x530 pseudo-random image at five sizes (identity; 512x384; 1024x768; 513x385;
  49x48): identical FNV-1a hashes and byte sums (TOTAL 72561f5339715edd both).
- ds4.c calls qwen4exp_resize_rgb_pillow at :26783 and :26916, after the :18450
  include; grep finds no remaining definition in ds4.c - exactly one implementation.

CLAIM 4 - the builds and the suite: PASS

- make ds4.o: the target was up to date, so it was forced with make -W ds4.c ds4.o,
  which runs the CUDA host recipe unchanged (cc -O3 -ffast-math -g -march=native
  -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -c -o ds4.o ds4.c):
  exit 0 in 40.7 s, pre-existing warnings only, and the rebuilt object's sha256
  9a3607d4...cb8266 is identical to the object it replaced - compiles, deterministic.
- bash tests/run.sh (fast path): exit 0, "fast path complete", 24.3 s (pre-report) and
  24.1 s (post-report); steps 1-7 green, including gate 22 (211/211 tensors, 19/19
  metadata keys, digest 9ffb86e3...6838) and gate 23.
- Nothing of the pin is built or linked: the oracle includes ds4.c with DS4_NO_GPU.

CLAIM 5 - the discrimination table: the rows' PURPOSE is confirmed; two of the three
rows I re-measured differ in exact digits (F2); two noise bullets are off 7-12x (F3)

Measured on pattern, each flip applied in the pristine copy, rebuilt with the gate's
exact flags, then restored and cmp'd:

  baseline (unchanged)                    maxabs 0.10275560  relrms 0.82%  mincos 0.99978457
  tanh GELU drop-in (eg2vq_gelu)          1.38684  7.55%   (doc 1.409 / 7.60%)
  tanh GELU inside the f16 round trips    1.41403  7.54%
  position slices swapped                 4.51476  43.74%  (doc 4.527 / 47.20%)
  fill rules swapped                      5.69762  103.39%
  rope axes swapped (pos_y first)         5.87766  103.51% (doc 5.889 / 103.50%)
  every matvec output scaled by (1+1e-5)  0.08070  0.83%   (doc 0.069%)

So the rope row reproduces digit for digit, and every flip is 3x-100x beyond the
2.5% bound while the port sits at the table's minimum: the gate discriminates.

CLAIM 6 - the doc's supporting claims: PASS except the two bullets in F3

- SHA256SUMS carries all four tests/vision/reference dumps and preproc-facts.txt
  (lines 10-14 of 14 entries); sha256sum -c SHA256SUMS reports OK on every entry
  (measured 1.1 s on this machine, not the ~25 s the QA brief expected).
- Gate 23 is fast-path step 6 of tests/run.sh (read in the file and observed); the
  doc's "~19 s for the whole gate" measured 19.8 s; the fast path 24.1-24.3 s.
- The dense cache is 576.0 MiB: the oracle prints dense_mb=576.0 on every run, and
  the binder's arithmetic agrees exactly - 16 x (4*768^2 + 3*768*3072) x 4 B =
  603,979,776 B. patch_embd and input_projection have their own allocations, which
  is what the doc's "the 16 blocks' matrices" says.

CLAIM 7 - shapes against the artifact's own metadata: PASS

Read back through the project's own reader (gguf_inventory --kv-values) from the
mmproj: 211 tensors, 19 keys; embedding_length 768; feed_forward_length 3072;
block_count 16 (blk.0-15 present, no blk.16); attention.head_count 12;
attention.head_count_kv 12; attention.head_dim 64; layer_norm_epsilon 1e-6;
patch_size 16; projection_dim 512; projector.scale_factor 3. Tensor shapes:
v.position_embd.weight [768,10240,2] (10240 rows per axis), v.patch_embd.weight
[16,16,3,768], mm.input_projection.weight [768,512], attn_q_norm/k_norm [64]. Every
value equals the constant in ds4_eg2vq_vision.h. The reference's log prints the same
hparams (n_embd 768, n_head 12, n_ff 3072, n_layer 16, ffn_op gelu_quick,
projection_dim 512, patch_size 16, n_merge 3, image_min/max_pixels 161280/2580480
= 70/1120 x 2304).

WHAT I DID NOT TEST

- The doc's first perturbation bullet (a 1e-7 relative perturbation of the position
  contribution -> 0.068% RMS, saturating): not measured.
- The anchor's own execution (forbidden in this pass); the shipped log is the only
  witness for the executed-reference facts, and gate 23 cross-checks its two lines
  against preproc-facts.txt.
- The CUDA vision tower, the text/vision splice and the pooled *.text.tsv dumps
  (units 24/25), and the other discrimination rows (kq_scale, weightless V RMS,
  gate/up exchange, the bf16 rows, f32 K/V).
- The Qwen4exp GPU path (covered statically: same bytes, same include) and leaks of
  bind/free (no valgrind or ASan run; free is called and every run returned 0).

FINDINGS (none blocks the unit)

F1 (low; off or stale file:line pointers in the port's comments; every mechanism
they name is real and the load-bearing citations of CLAIM 2 are exact).
- vision.h:78 cites clip.cpp:1723 for ROPE_THETA; the assignment is clip.cpp:1725
  (1723 is a case label). vision.h:79 cites clip-model.h:93 for eps; it is :95.
- graph.inc:27 cites clip.cpp:379 for the input RMS; the build_norm call is :376
  (379 is a comment). graph.inc:28 cites clip.cpp:436-451 for the q/k/v projections;
  the matmuls are :424, :429, :434. graph.inc:37 cites clip.cpp:534 for the FFN
  residual; the add is :538 (534 is the ls_2 callback), and :35's 486-500 stops two
  lines before the attention residual at :501-502.
- graph.inc:305 cites gemma4v.cpp:71 for ggml_rope_set_offset; line 71 is blank and
  the call is :70.
- vision.h:11 and bind.inc:4 cite ds4.c:72689 for the GLM-5.3 companion open; at HEAD
  that model_open is ds4.c:72566 (72689 is DFlash cleanup). 72689 is exactly the
  parent revision's line, shifted by the resampler move - not refreshed.
- resize_pillow.inc:7 says the bodies "were at ds4.c:26636-26779"; the byte-identical
  144 lines are ds4.c:26635-26778 (one line earlier at both ends), and :4's "block
  that opens at ds4.c:18444" is the guard at ds4.c:18452 at HEAD (18444 is the new
  include's own comment; the parent's guard was :18443).
- Loose but pointing at the right builder: build_vit :337-560 (the function ends
  :568), build_norm :607-628 (def at :603, body ends :626).

F2 (low; two discrimination-table rows do not reproduce digit for digit).
- "tanh GELU (eg2vq_gelu)": doc 7.60% / 1.409. Measured 7.55% / 1.38684 with the
  literal drop-in and 7.54% / 1.41403 with the tanh formula kept inside the f16
  round trips; neither reading gives the doc's pair.
- "position slices swapped": doc 47.20% / 4.527. Measured 43.74% / 4.51476 by
  swapping the two slices in the embedding add; the fill-swap reading measures
  103.39% / 5.69762, i.e. the doc's rope-axes row territory. The rope row itself
  reproduces (103.51% / 5.87766 vs 103.50% / 5.889), so the table is not
  systematically stale - these two rows are simply not exactly reproducible as
  written. Impact: none; every measured flip is >=7.5% against a 2.5% bound and the
  port is at 0.82%.

F3 (low; two noise-floor bullets in the doc's section 5 understate the measured
sensitivity by ~7-12x).
- "the SAME source built at -O2 (no -ffast-math) versus -O3 -ffast-math
  -march=native differs by 0.064% RMS (max-abs 0.009)": measured 0.78% / 0.06609 on
  pattern and 0.46% / 0.03393 on gradient. Three builds tried: -O2 plain, -O2
  -march=native (same output), -O2 -ffast-math -march=native (0.80% / 0.06796). All
  -O2 builds still pass the gate bounds themselves (0.79% residual on pattern).
- "scaling every matvec's output by (1 + 1e-5) ... gives 0.069%": measured 0.83% /
  max-abs 0.08070 (applied to every eg2vq_vision_matvec output).
- The doc's own summary sentence "the same port moves 0.06-0.4% under a pure
  recompilation" is likewise below the measured 0.46-0.80%.
- Impact: none on the verdict - the port's residual (0.82%/0.53%) and its measured
  noise (<=0.83%) stay far inside the declared bound - but the "saturates /
  decision-boundary" narrative rests on numbers this pass could not reproduce; read
  the summary range as ~0.5-0.8% for a recompile.

F4 (informational; the mechanical QA mount cannot see this unit's surfaces).
tests/qa-gate.sh printed {embed_caps}, not the unit's C entry points: the nested scan
paths are "crates ds4.c cuda" and the C-definition pattern needs a source line ending
in "(", so neither ds4_eg2vq_vision.h nor the new .inc files enter the list, and the
outer tools/ scan uses the same C pattern, so tools/eg2_vision_gate23.sh is invisible
too. The fast path was green before this report existed, on the unit-20 report being
fresh by mtime. Not a unit defect; the unit's real surfaces were QA'd by hand above.

F5 (informational; tree state at the end of this pass). The nested tree is clean and
byte-identical (the forced ds4.o rebuild reproduced its own sha256). During the pass
the project's own session committed the uncommitted unit-23-close handoff rotation
it had left in the working tree (fb3b819, a docs(handoff) commit directly on dev per
rules 9/23); the outer tree now shows only this report as modified. The handoff was
not written by this QA pass.

Post-write confirmation (recorded after this report was written): bash
tests/qa-gate.sh -> four PASS (exists; fresh; operative verdict; covers embed_caps),
"QA GATE: overall PASS", exit 0; bash tests/run.sh -> fast path complete, exit 0,
24.1 s.

verdict: overall PASS
