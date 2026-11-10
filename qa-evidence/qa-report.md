QA report - the embedding-path performance unit, third: the attention kernel's
grouped score pass
============================================================================

Unit and repositories at verification time

Engine /data/embeddinggemma2/ds4-rs, branch feat/eg2vq-attn-perf, HEAD 803decb7,
five commits over 1c9938ba (the previous, already-QA'd unit's tip):

    72ef3ac1  perf(attn): have LAB lanes share one key row in attention's score
              pass
    9e74b2ed  docs(bench): the attention unit's record, its ablation and its
              probes
    947d15f2  docs(bench): the forward's split after the change, from one fresh
              profile
    f0639289  docs(qa): the attention unit's first QA round -- E1 and eight
              record defects
    58ea1da5  docs(qa): the attention unit's second QA round -- six text items
              the fix commit introduced
    803decb7  docs(bench): rewrap the two lines the third QA round called cosmetic

Only 72ef3ac1 touches code (the kernel and the gate's two new cases); every
commit after it is evidence or comments, and the QA tester verified that with its
own filter of the diffs.

Outer /data/embeddinggemma2, branch feat/eg2vq-attn-perf, equal to its dev
a43479d plus the previous unit's record commits plus this unit's record commits.
Nothing in either repository has been pushed.  Lineage, measured not assumed:
`git rev-list --count publish/dev..dev` is 0 in the engine repository, so dev
carries none of this work.

This report supersedes the previous unit's (the VQ payload GEMM's), which is
preserved at the revision that carried it: `git show 1c9938ba \
docs/benchmarks/eg2-embedding-2026-10-20/vq-gemm-rows.md` cites it, and the file
itself is in that commit's history under the same path this one occupies.

What the unit changes

cuda/eg2vq_attn.cuh, one kernel: eg2vq_attn_kernel, the gemma_vq text tower's
attention.  Pass 1 goes from one lane owning one whole K row to LAB =
head_dim/64 lanes cooperating on one key, each owning 16 float4 chunks of it
4*LAB elements apart, with the lane's 64-float slice of the query row in
registers.  The kernel becomes template <int LAB>; the launcher dispatches on
head_dim == 64*LAB for LAB in {2,4,8,16} and to LAB = 0 (the unchanged
lane-per-key body) otherwise.  Unchanged: the window arithmetic, the score plane
and its 32 KB static allocation, the row maximum, pass 2, pass 3, the scale, the
mask, the launcher's guards, and every gate's declared bound.

tests/test_eg2vq_attn.cu gains two cases, head_dim 384 and 320, because the
dispatcher's LAB = 0 arm -- the instantiation every other shape takes, and the
only path that executes the scalar bodies -- was otherwise reachable and gated by
nothing.

Surfaces added by this unit (publish/dev..HEAD in both repositories)

None.  The rule-19 mount's own extractor names no new surface for this unit: no
new `ds4_*` C entry point, no new /v1/ route, no new `pub fn`.  The mount's
reading at this boundary is "QA GATE: ALL PASS (no new runtime, route or
public-API surfaces relative to publish/dev and ds4-rs/publish/dev)", exit 0,
and that gate is step 11 of the fast path, which exits 0 as a whole.  The unit's
whole surface is one device kernel whose launcher signature, call site and
behaviour are unchanged, plus its measurement artifacts.  ALL NEW SURFACES is
therefore empty by the extractor's own reading, and the QA pass below was run
over the unit anyway, by hand, exactly as the previous two were.

What the unit claims, and where each claim's evidence is

1. The kernel is 1.94x-2.61x faster on the tower's nine shapes, both round
   pairings of an interleaved A/B against the pre-change binary.
   Evidence: attn-ab-{old,new}-r{1,2}.txt (raw bench output, 30 repeats per
   row), attn-before.cuh.txt (the header the old binary is built from), and the
   table in attn-kernel-grouped.md section 4.1, which is recomputable from the
   two files by joining on the shape name and dividing the ms_med columns.
2. The corpus is 1.32x faster end to end (8.42 s -> 6.40 s at batch 8, 9,490-9,761
   -> 12,507-12,720 tokens/s) over 143 pages / 80,747 tokens.
   Evidence: the six CSVs under speed-bench/ and their _health.json beside them,
   joined in attn-ab-sweep.txt; the runs are old r1, old r2, new r1, old r3,
   new r2, old r4 in wall-clock order, so each new run sits between old runs.
3. The K access is what the kernel was paying for, and the exponentials and the
   score plane are free.
   Evidence: attn-ablation.txt, the output of probes/probe-attn.cu (V1, V2, V3,
   V5, V6, V7, V8, V9 rows), with the probe's own V0 measured twice per shape in
   the same run.
4. The candidate form was measured before it was landed, and the shipped form
   agrees with it to cosine 1.000000000 and 2.4e-06-6.5e-06 maximum element-wise
   relative difference.
   Evidence: attn-grouped-probe.txt, the output of probes/probe-attn-grp.cu,
   whose ON=0 instantiation is also checked byte for byte against the shipped
   launcher in the same run.
5. The gates pass: make test-eg2vq-attn (nine arithmetic cases, mincos
   1.000000000, rel 3.31e-07 to 9.73e-07), compute-sanitizer memcheck 0 errors,
   gate 7 EG2VQ_GPU_PASS at min cos 0.999999950 / max rel_rms 3.149e-04,
   tests/run.sh exit 0 with gates 23-27 and the QA gate green.
   Evidence: gate7-embd-attn-grouped.txt and the gate's own printed lines; the
   rest is reproduced by the commands in the record's section 7.
6. One `nsys` profile of a batch-8 forward now reads attention at 46.0 ms of
   197.5 ms of kernel time, against the previous unit's profile of the older
   kernel at 116.7 ms of 287.8 ms.
   Evidence: nsys-kern-sum-b8-attn-grouped.txt.  The record labels this split as
   the profile's contribution and explicitly not as evidence of the delta.

The QA pass's verdict: THREE ROUNDS, FAIL-FAIL-PASS, NO CODE DEFECT IN ANY OF
THEM. Round 1 verified every gate, every number in the record and an independent
nine-case driver of its own, and failed on one blocking evidence defect (both
probes' documented commands, which after this unit no longer reproduce their
committed output without the pre-change header) and eight record defects. Round 2
verified all ten fixed and failed on six text items the fix commit itself
introduced. Round 3 verified those six fixed by recomputation and live run and
returned PASS with no finding open, naming one cosmetic byproduct (fixed in
803decb7). The findings and their dispositions are in
qa-findings-attn-grouped.md.

Open at this boundary

- The unit was NOT pushed (rule 23 offer, recorded in the handoff).
- Pass 3's V loads (measured at 1.29x-1.48x by V4) are untouched by this unit.
- The engine repository's inherited stale QA mount (ds4-rs/tests/qa-gate.sh and
  ds4-rs/qa-evidence/qa-report.md, from the ds4-dfm-rs lineage) is still there;
  the project's gate is the outer tests/qa-gate.sh.  This is an owner decision
  and was open before this unit too.
- The deep path (tests/run.sh --deep), the loader digest check's own script and
  the OSV scan were not re-run at this boundary; the fast path (which includes
  the fourteen digests and gates 23-27) was, exit 0, and the QA pass did not
  re-run it either — its own verification covered the gates named above.
