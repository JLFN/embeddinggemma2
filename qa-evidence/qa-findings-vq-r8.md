QA findings — the VQ payload GEMM unit (engine 8d618ecb, completed by 793386db)
================================================================================

The rule-19 QA-tester (deepseek-v4.1-CC-flash, this project's recorded model) ran
over the unit at its boundary. First verdict: FAIL, with no code defect among the
findings. Its second pass, after the fixes below, is recorded in qa-report.md.

The findings and their dispositions. Every "fixed" line names the artifact that
now carries the evidence; nothing here is a promise.

F1 (record, moderate) The committed identity/ablation probes can no longer
demonstrate what the record says they demonstrate: probe-r8.cu compares a
templated R=8 kernel against whatever the header says at compile time, which
after the change is the same R=8 shape, so it self-compares; the record
presented it as the old-vs-new check.
FIXED. probe-r8.cu and probe-big.cu are left as the instruments they were, and
the record now states their scope (section 3 and the reproduce section) and how
to compile them against the pre-change header. The unit's identity claim no
longer rests on them at all: it rests on qa-identity/, committed, which feeds the
same bytes to the old and the new kernel in one process.

F2 (non-issue, the unit's central claim confirmed independently) Bit-identity
old vs new: the tester built its own two-translation-unit driver from the parent
header and compared raw float bytes on 15 cases.
Confirmed: 12,492,416 floats compared, 0 differing, maxabs 0.000e+00 on every
case, including n_tok 1, ragged tails, rows exactly 64 and t_stride > cols. That
driver is now committed as qa-identity/ so the claim is reproducible rather than
resting on one session.

F3 (non-issue) Memory safety of the ragged path, and specifically whether
removing the t_ok fast path changed tail behaviour: compute-sanitizer memcheck on
the identity driver, with the activation allocated at exactly n_tok rows.
Confirmed: 0 errors; loads remain predicated by t0 + a < n_tok.

F4 (record, minor) The probe output probe-r8-rows.txt was not reproducible with
the command the record implied: the exact build line needs -Xcompiler
-march=native (it changes the last bits of the host-generated test data, so the
sampled outputs shift by an ulp while the identity outcome is unchanged).
FIXED: the reproduce section carries the full build line for every probe.

F5, F6 (non-issue) The two gates. make test-eg2vq-vq -> EG2VQ_VQ_PASS, mincos
1.000000000 on all six cases, wrong-geometry payload refused, output untouched.
tests/parity/eg2vq_embd_gpu_oracle -> EG2VQ_GPU_PASS, min cos 0.999999950, max
rel_rms 3.150e-04, i.e. numerically identical to the attention unit's record.
Note the tester's own warning, which stands: the engine binaries in the tree were
stale relative to the header, so it rebuilt before every gate run. The committed
record now carries the gate-7 raw output.

F7 (non-issue, record audit) The tester traced the nsys table, the 66-row bench
table, the four corpus CSVs, the ablation table, the rejected-geometry numbers
and the register/shared figures to their artifacts and found them consistent,
including the file mtimes that prove the interleaved order (old, new, old, new).

F8 (record) "the second round agrees with the first to within 1% on every row"
was false: three rows differ by 1.4-2.7%.
FIXED: the claim is gone; the measured spread is published instead.

F9 (record, moderate) The published speedup range used only the more favourable
old round. Round-paired, round 1 spans 1.26x-1.79x and round 2 spans 1.23x-1.75x,
because the first binary to run in the sitting is up to 10.5% slower on its first
rows (a warm-up effect on the old side, not on the change).
FIXED: both pairings are published and the conservative one is the headline.

F10 (record) The V3 upper bound was quoted as "27-67 TFLOP/s"; the committed
artifact reads 26.5-67.2.
FIXED: the record now quotes the artifact.

F11 (record) The claim that the register-hoisting variant "spilled and ran 5-7x
slower" had no artifact behind it.
FIXED: the variant is now measured as V6 in probe-ablation.txt (0.14x-0.19x of
baseline on the large shapes, 0.52x on the smallest, STACK:4096 in cuobjdump),
and the record quotes that.

F12 (record, minor) The gate-7 corroboration pair (1,499.6 -> 2,048.6 tokens/s)
had no raw file in the repository.
FIXED: gate7-embd-gpu-oracle.txt is committed; the before value is cited to
attention-occupancy.md, the file that recorded it; the after value is quoted with
the three readings of that morning (2,008.9 / 2,048.6 / 2,060.9 tokens/s).

F13 (record, moderate) The reproduce command for the ablation probe crashed when
rebuilt against the current header: the probe took its tile from the header, so
after the change it built an R=8 body with the old 16x16 launcher and read past
its own tiles (illegal memory access).
FIXED: probe-vq.cu pins its own PVQ_T/PVQ_R and carries the reason; the tester
re-ran the documented command and confirmed it works and reproduces.

F14 (non-issue, two wording notes) The launcher's block dimension did change
(16x16 -> 16x8), so "the launcher and its guard are unchanged" was loose; and the
"16/R bytes per FMA" argument counted a four-wide group as one multiply-add.
FIXED: the record says the guard and grid math are what is unchanged and gives
the block dimension as the one launcher change; the argument is stated per scalar
multiply-add (4/R bytes) in the record and in the kernel comment.

F15 (non-issue) Edge cases: n_tok = 1, ragged tails, rows exactly 64, t_stride >
cols, and the launcher's refusal path (9 of 9 invalid shapes refused by BOTH the
old and the new kernel) all pass, with no out-of-bounds access.

F16 (process) The tester first ran the engine repository's own tests/qa-gate.sh,
which is an inherited stale mount from the ds4-dfm-rs lineage, with a
qa-evidence/qa-report.md dated 2026-10-18 belonging to that project. That gate is
red and unrelated to this unit. This project's rule-19 mount is the outer
/data/embeddinggemma2/tests/qa-gate.sh, which scans both repositories against
their publish/dev bases; its result for this unit is recorded in qa-report.md.

What the tester could NOT verify, kept visible: it did not re-measure the
speedups in its own window (it verified the artifacts trace and are internally
consistent, and that the direction survives the least favourable round pairing);
the nsys single-run deltas prove nothing on their own (the record says so, and
the byte-identical attention kernel demonstrates why); no hardware counters exist
on this machine; and its own identity driver is its own instrument, though its
strength is that it feeds identical bytes to both kernels in one process.
