QA findings — the attention kernel's grouped score pass
=======================================================

The rule-19 QA-tester (deepseek-v4.1-CC-flash, this project's recorded model) ran
over the unit at its boundary, twice: once over engine revision 9e74b2ed/947d15f2
(verdict FAIL, no code defect) and once over f0639289, the commit that fixes what
the first pass found. This file lists every finding and its disposition. Every
"fixed" line names the artifact or the commit that now carries the evidence;
nothing here is a promise.

The first pass's own summary, worth keeping: it rebuilt and re-ran the gate (nine
arithmetic cases, mincos 1.000000000, rel 3.31e-07 to 9.73e-07), the sanitizer
(0 errors), and gate 7 (EG2VQ_GPU_PASS, 0.999999950 / 3.149e-04), confirmed the
five kernel instantiations with cuobjdump (LAB 0 REG:48, 2/8/16 REG:98, 4 REG:100,
all STACK:0 LOCAL:0) and the one-instantiation pre-change binary, verified the
occupancy arithmetic against the device limits, checked that both nsys summaries
of the "same" forward match launch for launch (434 GEMM, 40+8 attention, 242 rms,
...), recomputed all 45 ablation ratios, all 18 kernel-bench ratios, all 30 corpus
wall times and the section 4.4 rows, and wrote its OWN driver
(/tmp/qa-attn-independent/drv.cu, calling the shipped launcher over head dims
128/256/320/384/512/1024 plus a packed two-item case against an independent f64
CPU implementation of the documented semantics) — all nine cases cosine
1.000000000, max element-wise deviation 6.6e-07 to 3.2e-06. It found no code
defect, and it did not accept a single claim without checking it.

E1 (evidence defect, blocking — the class the previous unit's QA treated as
blocking). Section 7's probe commands failed as written: both probes anchor their
baseline to the header on the include path, so after this unit they compare a
lane-per-key baseline against a grouped shipped launcher, print "V0 IS NOT THE
SHIPPED KERNEL" / "ON=0 IS NOT THE SHIPPED KERNEL" on every shape, exit 1, and
their committed outputs are unreproducible (the tester measured batch8x565_f V0a
14.9480 against the committed 16.3915).
FIXED. Section 7 now stages the pre-change header (`attn-before.cuh.txt`) into
/tmp/ab-attn and compiles BOTH probes with `-I/tmp/ab-attn` ahead of the repo, in
its own block with its own documented working directory; the paragraph after the
block states the failure mode and why the probes cannot follow the shipped header
— the same requirement the previous unit's record carries for probe-r8.cu and
probe-big.cu. The pin is verified: the parent session re-ran both documented
commands at f0639289 and got PROBE_ATTN_OK and PROBE_ATTN_GRP_OK, with the
agreement columns reproducing the committed files (2.3e-06 to 6.5e-06).

R1 (record defect). The cross-window product quoted "1.63 (attention)" as an
"interleaved delta"; no committed file carried 1.63, and the attention pair is
not interleaved (its own record says corpus rows that far apart are not
comparable).
FIXED. The sentence now recomputes the attention factor from the committed CSVs
(1.61x: eg2_embed_vizzio.csv 18.799776866 s against 11.654060797 / 11.717910747 /
11.658361879 s in the three eg2_embed_vizzio_attn-vec4 runs), drops the word
"interleaved", and publishes both the product (1.61 x 1.25 x 1.32 = 2.66x) and
the endpoint ratio (18.7998 -> 6.40 = 2.94x) with the drift that separates them,
naming today's own 8.42-vs-8.89 s old-build discrepancy as the measure of that
drift.

R2 (record defect). The paired corpus ratios were labelled with numerator and
denominator swapped ("new r1 / mean(old r2, r3) = 8.365/6.456").
FIXED in the record and in attn-ab-sweep.txt: "mean(old r2, old r3) / new r1".

R3 (record defect, minor). The candidate probe's agreement range starts at
2.3e-06 (page565_s G4x16), not 2.4e-06.
FIXED in the record (two places) and in cuda/eg2vq_attn.cuh (two places).

R4 (record defect, minor). "869-1,496 GFLOP/s before" took round 1's maximum and
dropped round 2's 1,498.4, and the sentence cited a "GFLOP/s column" the markdown
table does not have.
FIXED: "869-1,498 GFLOP/s before (the maxima are 1,496.7 in round 1 and 1,498.4
in round 2)", recomputed from the two raw runs' own column.

R5 (record defect, minor). The corpus speedup column mixed truncation with
rounding (batch 8 read 1.31x where the CSVs give 1.31551).
FIXED: batch 8 is 1.32x in the record and in attn-ab-sweep.txt, and the record
states the convention (mean of four old runs over mean of two new runs, rounded to
two decimals) with every unrounded ratio printed beside it. The unit's headline
corpus figure is therefore 1.32x, not 1.31x.

R6 (record defect, minor). The spanbug narrative named the wrong pairings as the
fast-and-wrong ones: the rows that looked fast were (LAB=4,GS=16) 2.41x-2.60x and
(LAB=8,GS=16) 2.12x-2.59x, while (LAB=16,GS=8) read 1.43x-1.90x; the cosine span
is 0.02-0.24.
FIXED in both places the narrative appears.

R7 (cosmetic). "These are the same numbers the previous unit's record carries"
overstated: this run prints 3.149e-04 where the previous records carry 3.150e-04.
FIXED: "the same to three significant figures", with both values and both
records' scopes named, plus the tester's own re-run tokens/s (2,144.7 / 2,146.5)
as the window-dependence note.

R8 (cosmetic). The kernel comment called gate 7's measured 0.999999950 /
3.149e-04 its "declared" bounds.
FIXED: the comment now names the declared bounds (cos >= 0.999999, rel_rms <=
1e-03), the measured values on this change, and the test's own bounds
(1 - 1e-6 and 1e-5).

R9 (cosmetic, documentation). Section 7's block stated no working directory and
could not run from any single one.
FIXED: the block is split into "from the ENGINE ROOT" and "from the BENCHMARK
DIRECTORY" groups with the paths adjusted for each.

N1 (clarification). Section 2's `xS` column divides by the first shipped run
(`S-a`), not by the mean of the two.
FIXED: the text says so and gives the mean-based value for the worst row (1.62x
against 1.66x).

N3 (clarification). All six committed health readouts are the same caps payload
byte for byte and identify no build.
FIXED: the file list now says what they identify (the port and the caps), which is
what the sweep writes.

N2 (state, resolved). Two numbers the tester flagged were fixed in the working
tree while its pass ran (the V1 range 2.05x-2.25x -> 1.86x-2.25x, and section
4.4's small-ops/total 24.3 ms / 197.3 ms -> 24.5 ms / 197.5 ms, recomputed from
nsys-kern-sum-b8-attn-grouped.txt). Both are committed in f0639289.

S1 (state, resolved). Section 8 cited a findings file that did not exist yet, and
the report's revision header was one commit stale. Resolved at this boundary: this
file exists, and qa-report.md now names the final revision.

Round 2's verdict and any remaining findings follow below.

ROUND 2

Round 2 verified all ten round-1 findings fixed — E1 by running the new probe
block exactly as documented (both probes green, agreement columns reproducing the
committed file), R1-R9/N1/N3 by reading each new statement against the artifact or
the arithmetic, N2 as committed and consistent — and returned FAIL on six items
the fix commit itself introduced. All six are text-level; the tester again found
no code defect and every executable check passed. Fixed in 58ea1da5.

F1 (record defect, minor). The N1 note's "differs by 1-4%" understated the range:
recomputed row by row, the mean-based value differs by under 0.3% on the two short
shapes and by 2% to 6.4% on the long ones.
FIXED: "under 0.3% on the two short shapes and by 2% to 6.4% on the long ones
(max1414_s's (8,8) reads 1.66x this way and 1.62x against the mean; batch8x565_s's
(8,8) reads 1.37x and 1.46x)".

F2 (record defect, minor). The N3 note said the health readouts identify the port.
They do not: the payload carries no port field and no build field, and the
record's own trap note says the health probe cannot tell which server answered.
FIXED: "the same caps payload, byte for byte, and it carries no port and no build
field: what identifies a run is the CSV beside it and the server's own `listening
on` line in the sweep log, never this file".

F3 (cosmetic). "the 2.7x the previous record published" cited the wrong source:
neither previous record carries a 2.7x; it was this record's own earlier
revision's product.
FIXED: "the 2.7x this record's own earlier revision published".

F4 (cosmetic). One of the three attention ratios was truncated where its siblings
were rounded.
FIXED: 1.6132 / 1.6044 / 1.6126, all rounded.

F5 (cosmetic). The documented counter-failure quoted one message for both probes.
FIXED: the record now attributes the right message to each — probe-attn: "V0 IS
NOT THE SHIPPED KERNEL", five shapes; probe-attn-grp: "ON=0 IS NOT THE SHIPPED
KERNEL", six shapes — verified live in round 3.

F6 (cosmetic). Section 7's A/B redirects wrote to the engine root instead of the
record directory the committed raw files live in.
FIXED: the block defines `R=docs/benchmarks/eg2-embedding-2026-10-20` and both
run lines redirect into it.

ROUND 3

VERDICT: PASS. The tester verified F1-F6 fixed by recomputation and live run
(F1 and F4 recomputed from attn-grouped-probe.txt and the four CSVs, F2 grepped
against the payload, F3 grepped against both records and the earlier revision, F5
by building and running both probes against the current header, F6 by resolving
`$R` from the block's stated working directory), found nothing new introduced by
the fix, and reported: "No finding of mine remains open." It named exactly one
byproduct it did not count as a defect — a word orphaned onto its own line in the
F5 parenthetical — which is fixed in 803decb7.

Across all three rounds: no code defect, one blocking evidence defect (E1) and
fourteen text-level items, every one of them fixed by a change the next round
verified. The lesson for the next unit is the round-2 shape: a fix commit is new
text and can carry new defects, so the round that verifies the fixes has to read
what the fix wrote rather than what it meant to write.
