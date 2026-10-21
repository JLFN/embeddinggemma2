QA report - the embedding-path performance unit, second half: the VQ payload GEMM
================================================================================

Unit and repositories at verification time

Engine /data/embeddinggemma2/ds4-rs, branch feat/eg2vq-attn-perf, HEAD 0f9ab8cf,
three commits over dev 10e73aa0 (which is publish/dev; dev is untouched):

    8d618ecb  perf(vq): give each thread eight output rows in the payload GEMM
    793386db  docs(qa): complete the unit's evidence and fix the defects its QA
              pass found
    0f9ab8cf  docs(qa): the second QA round's three one-line corrections

Outer /data/embeddinggemma2, branch feat/eg2vq-attn-perf, equal to its dev
a43479d plus this unit's record commits.  Nothing in either repository has been
pushed.  Lineage, measured not assumed: `git rev-list --count publish/dev..dev`
is 0 in the engine repository, so dev carries none of this work.

What the unit changes

cuda/eg2vq_vq.cuh, one kernel: eg2vq_vq_gemm_kernel, the dim-4 DQVQ payload GEMM
that consumes the tower's weights.  The block becomes 16x8 (was 16x16) and each
thread owns 8 output rows (was 4), 32 accumulators.  Unchanged: the 64-token by
64-row tile, the 8.5 KB index slab, 41,472 B of static shared memory, the
launcher's guard (rows % 64, cols % 256, t_stride >= cols) and grid math, and the
accumulation order per output element.  The two commits after the kernel are
evidence and record only; the tester verified with its own filter that the header
diff of 0f9ab8cf has no non-comment lines, so the object code it gated is the
object code that shipped.

Surfaces added by this unit (publish/dev..HEAD in both repositories)

None.  The rule-19 mount's own extractor names no new surface for this unit -
no new `ds4_*` C entry point, no new /v1/ route, no new `pub fn` - which is what
the mount reports: "QA GATE: ALL PASS (no new runtime, route or public-API
surfaces relative to publish/dev and ds4-rs/publish/dev)", exit 0.  The unit's
whole surface is one device kernel whose signature, launcher and call site are
unchanged, plus its measurement artifacts.  ALL NEW SURFACES is therefore empty
by the extractor's own reading, and the QA pass below was run over the unit
anyway, by hand, exactly as the previous unit's was.

How the QA pass was run, and what it found

The project's recorded QA model (deepseek-v4.1-CC-flash) ran three passes over
this unit at its boundary.  All three are recorded with their commands and
observed output; qa-findings-vq-r8.md carries the finding-by-finding disposition.

Pass 1, verdict FAIL, no code defect among the findings.  It confirmed
independently that the change is bit-identical (its own two-translation-unit
driver, old kernel from the parent header and the shipped kernel in one process,
12.49 million floats compared, zero differing), that compute-sanitizer reports no
memory error with the activation allocated at exactly n_tok rows (the strongest
out-of-bounds probe for the removed fast path), that both correctness gates pass
on rebuilt binaries (EG2VQ_VQ_PASS with mincos 1.000000000 on every case;
EG2VQ_GPU_PASS with min cos 0.999999950 and max rel_rms 3.150e-04, numerically
identical to the previous unit's record), and that the edge cases and the
launcher's refusal path are intact.  What it rejected was the record and its
instruments: one committed probe had become self-comparing and the other crashed
when rebuilt from the documented command, one quoted figure had no artifact, the
published bench range used only the more favourable round, three numbers did not
trace to any file, one sentence claimed a round agreement that three rows
contradict, and there was no fresh rule-19 report for the unit.

Pass 2, after the fixes, verdict FAIL, on three one-line problems and two digits
of the same class: a corpus sentence claiming a pairing agreement batch 1 does
not have, a gate-7 "before" figure cited to a line that does not contain it, and
a bytes-per-multiply-add sentence that contradicted itself by a factor of four.
It confirmed by rebuilding and running - not by reading - that the pinned
ablation probe is alive and reproduces its artifact's shape with a genuine
register-spill row, that the identity driver rebuilds from the committed sources
to byte-identical output, and that both bench round pairings recompute from the
raw files.

Pass 3, narrow confirmation of the corrected lines, verdict PASS.  It recomputed
all sixteen seconds cells and all twelve ratio cells of the corpus table from the
four committed CSVs, opened docs/eg2vq-device-gate7.txt:461 and found the
1,499.0 tokens/s before-figure there, confirmed every number attributed to the
committed gate-7 raw output, checked both conventions of the activation-bytes
argument are now named and self-consistent in the record and in the kernel
comment, and recomputed the attention share (116,671,381/287,804,881 = 40.5%).
Its one remaining note, a one-word provenance slip ("handoff" for "QA report"),
was fixed in the record and does not touch any cited number.

Per-surface detail

FUNCTION eg2vq_vq_gemm_kernel (cuda/eg2vq_vq.cuh, the unit's only runtime change)
  verified by: make test-eg2vq-vq -> EG2VQ_VQ_PASS, mincos 1.000000000 on all
  six cases, wrong-geometry payload refused with the output untouched;
  tests/parity/eg2vq_embd_gpu_oracle -> EG2VQ_GPU_PASS (raw output committed as
  docs/benchmarks/eg2-embedding-2026-10-20/gate7-embd-gpu-oracle.txt);
  docs/benchmarks/eg2-embedding-2026-10-20/qa-identity/ -> QA_IDENTITY_OK,
  12,492,416 floats compared byte for byte old against new, 0 differing,
  9 of 9 invalid shapes refused by both; compute-sanitizer memcheck, 0 errors.

ENDPOINT (none added) and CLIENT METHOD (none added) - the extractor's reading,
re-verified by the mount's own output above.

What this unit's QA does NOT establish

The three passes did not re-measure the speedups in their own windows: they
verified that every number in the record traces to a committed artifact and is
internally consistent, and that the direction survives the least favourable round
pairing (kernel 1.23x-1.75x, corpus 1.20x-1.25x).  The single-run nsys profiles
prove nothing on their own, and the record says so: a byte-identical attention
kernel reads 3.33 ms per launch in one profile and 2.43 ms in another.  No
hardware counters exist on this machine (ncu refuses with ERR_NVGPUCTRPERM), so
the ablation brackets the bottleneck without naming the mechanism.  The identity
driver is the tester's own instrument; its strength is that it feeds identical
bytes to both kernels in one process, and its limits are that it does not run the
tower end to end - the kernel gate and the tower oracle above do, and both pass.

Blocks.  No block remains: all three passes' findings are fixed in the commits
listed at the top, the gate is green, and the record's own section 5 states what
the measurement does not prove.

verdict: overall PASS
