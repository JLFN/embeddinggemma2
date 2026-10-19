# QA report — unit 15, "gate 5: what the VQ codec costs"

- Unit: 15 (phase 5, row X.7) — outer-repo commits `07e0ccd` (the work, trailer
  "Unit: 15 complete") and `99efb30` (the handoff).
- QA model: deepseek-v4.1-CC-flash (independent QA-tester, global rule 19).
- Date: 2026-10-19. Host: /data/embeddinggemma2, outer repo at branch dev, HEAD 99efb30.
- Nested ds4-rs: verified detached/clean at 29873445, `git status --short` empty —
  the unit did not touch it, as claimed.
- Scope: independent re-verification of the unit's eight claim groups from committed
  inputs (`tests/retrieval/`, `tools/eg2_*.sh`, docs) plus the gitignored `out/`
  artifacts, using only shell, awk, jq, cut, sha256sum, cmp; no Python, no ONNX, no
  PyTorch anywhere in this verification (P-1/P-2 respected). No source, tool, doc or
  data was modified; the only writes were under /tmp and this report.

## Surfaces covered

TOOL tools/eg2_phase5_gate.sh
TOOL tools/eg2_retrieval_score.sh
TOOL tools/eg2_oracle_to_jsonl.sh
TOOL tools/eg2_retrieval_set.sh
TOOL tools/eg2_retrieval_score_selftest.sh
TOOL tools/eg2_parity_check.awk
DOC docs/retrieval-cost-gate5.txt (sections 4, 5, 5a, 5b, 5c, 6, 7, 7a, 7b, 8)
DOC RECIPE.md (section 4e command 24, section 5b, section 6 item 3, the three new
section-7 rows)
DOC docs/ds4-gemma-port-phases.md (row X.7, section 9, section 10)
DOC embeddinggemma2-handoff.md (spot-check of the unit's numbers)

## Claim 1 — set provenance (PASS)

Ran:

    sha256sum tests/retrieval/scifact-test-{queries,pool,qrels}.tsv
    grep -n '_SHA=' tools/eg2_phase5_gate.sh

Observed: 1247b1f4d25a1875404480961095df8266424323eb310d9d821d3231b69a404d /
caf498d9db8aca3648c7e688edb970226b7bee70ad009a8f05c24c11473b73b0 /
864141f92383b8b7140ebbaf61cb9dac40ae522796690063ffa1a114f973476c. The gate pins
(eg2_phase5_gate.sh:78-80) and docs/retrieval-cost-gate5.txt:117-119 print the same
three values, digit for digit.

Re-derivation and refusal, demonstrated live in a /tmp sandbox (gate copy at
/tmp/qa-p5/sandbox/tools, copies of the three tsv in its tests/retrieval, real
`out/beir/scifact` as EG2_BEIR_DIR, a fake fast oracle so no 16-minute leg ran):
the gate printed "the builder reproduces all three frozen files byte for byte from
out/beir/scifact"; with a tampered corpus in the source dir it exited 2 with
"scifact-test-pool.tsv is not what the builder produces from
/tmp/qa-p5/beir-tamper/scifact"; with a tampered frozen copy it exited 2 the same
way; and the builder itself refuses a bad scifact.zip digest (exit 2,
"f3c16f52... != 536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165").
The zip is 2,816,079 B with sha256 536e1444..., as docs section 3 states.

## Claim 2 — the legs (PASS, one wording nuance)

Ran:

    wc -l out/phase5-baseline-raw.txt out/phase5-artifact-raw.txt
    cat out/phase5-baseline.meta out/phase5-artifact.meta
    sha256sum out/phase5-ids.tsv out/eg2vq_embd_c_oracle_o3
    awk -F'\t' '{n++; t+=$2} END {print n, t}'  (each raw)

Observed: 583 lines each; both metas record ids_sha
9464c27272399e21895b49b6e031f7a2c1e626ba0d842d5bb4e1f49fb67d5b4c (equal to the
current ids file) and oracle_sha
a9f74d0b7a66920abba7a82de7eee022ede301b61e78bb9692f952db8fbe7940; the only file in
the tree matching that sha is out/eg2vq_embd_c_oracle_o3 (exists, executable),
which my own rebuild of the RECIPE command-24 line also reproduced bit for bit;
each raw sums to 583 items / 107029 tokens. The raws' own shas match docs section
5b (baseline acbc0f692f083b6339ecf7c83d806e9d0c8d0cc83e0c3f90817d2fcaeac4ff01,
artifact 3e9ace37211450750911ff9df808c27fa1371e49588b57067ac649434856bb06) and the
items file matches doc line 160
(66b956ad1fab62b2996343ae9578b9e66d3dddfba9735f11a936cb47c8bfca03).

Nuance: the .meta files name no binary path, only the sha (by design, section 7a
says "each leg's .meta recording the binary's own sha256"); the claim's phrase
"the oracle binary named in the .meta" holds only via the sha match, reported as
finding F4.

## Claim 3 — the 5b/5c numbers (PASS, every digit)

Ran exactly the prescribed chain:

    tools/eg2_oracle_to_jsonl.sh out/phase5-baseline-raw.txt > /tmp/qa-base.jsonl
    tools/eg2_oracle_to_jsonl.sh out/phase5-artifact-raw.txt > /tmp/qa-art.jsonl
    tools/eg2_retrieval_score.sh /tmp/qa-base.jsonl /tmp/qa-art.jsonl \
        tests/retrieval/scifact-test-qrels.tsv 768 512 256 128

Observed, and compared digit by digit against docs sections 5b and 5c:

- the two rebuilt jsonl files are byte-identical to the gate's own recorded outputs
  (d53ec48325fd135a7d42a6268daae448adb9cda9cc567d4b78f6c42f699d24c8 and
  76b5ac7eccab2d360b79df4c533c7640f15c148de2df7cd9629a862ed62f5551);
- agreement table 5b: 768 means/min/p1/p5/p25/p50/p75/worst =
  0.956039571 / 0.898186175 / 0.922665454 / 0.937516751 / 0.951416471 /
  0.957530714 / 0.962775245 / 6421792, and the 512/256/128 rows and the 128 mean
  0.972493258 — all identical to the document;
- side split and tails: 0.958537658 / 0.953391422 with 583, 124, 1 at 768; 583,
  104, 1 at 512; 583, 72, 1 at 256; 0.973980059 / 0.970917143 with 583, 7, 0 at
  128 — identical;
- retrieval table 5c, all eight rows and all deltas: recall@1 0.888278 -> 0.861611,
  recall@5 0.979333 -> 0.975333, recall@10 0.983333 -> 0.990000, MRR@10 0.951778
  -> 0.935819 (-0.015959) at 768; 512: 0.886611 -> 0.864944, 0.949861 -> 0.936944
  (-0.012917); 256: 0.872611 -> 0.833278, 0.935167 -> 0.914671 (-0.020496); 128:
  0.829278 -> 0.805944, recall@5 0.954667 -> 0.915889, 0.908902 -> 0.879488
  (-0.029414) — identical;
- top-1 changed / mean margin: 23/0.085722, 22/0.085825, 29/0.081908, 37/0.050978
  — identical;
- invariants: 583, 2332, 300, 283, 300, 283, 283, 339, 1.144e-07, 1.430e-07,
  5.000e-08 — identical.

Bonus, same rigor: section 5a's phase-4 pair, re-run as
`tools/eg2_retrieval_score.sh out/phase4-baseline.jsonl out/phase4-artifact.jsonl - 768 512 256 128`
(0.9 s), reproduces the 5a table exactly, including the tail counts 112/24/1,
112/19/1, 112/11/1, 107/1/1 and the invariants. Not a single digit differed
anywhere in sections 5a, 5b or 5c.

## Claim 4 — the second instrument and the rel_rms identity (PASS)

Ran:

    awk -F'\t' -f tools/eg2_parity_check.awk out/phase5-baseline-raw.txt \
        out/phase5-artifact-raw.txt

Observed: "items cpu=583 device=583  min cos 0.898186179 (item 6421792)  max
rel_rms 4.513e-01 (item 6421792)  malformed 0" — identical to the document's
quoted line and to out/phase5-parity-check.txt.

Identity check (my own awk over both raws, items 1, 3, 5, 13, 36 and 6421792):
rel_rms against sqrt(2(1-cos)) differs by at most 1.6e-08 (floating-point
rounding); item 6421792 gives cos 0.898186179 and sqrt(2(1-cos)) = 4.512512e-01,
matching the document's 0.45125. The document's "exactly" holds at instrument
precision.

## Claim 5 — the -O3 equivalence (PASS for the instruments and hashes; timings re-measured only at small scale)

Rebuilt both builds from the current tree, into /tmp (no tree file written):

    (cd ds4-rs && cc -O3 -ffast-math -g -march=native -Wall -Wextra -std=c99 \
       -D_GNU_SOURCE -fno-finite-math-only -O0 -DDS4_NO_GPU -ffunction-sections \
       -fdata-sections -Wno-unused-function -I. -o /tmp/qa-p5/o0 \
       tests/parity/eg2vq_embd_c_oracle.c -Wl,--gc-sections -lm -pthread)
    cc -O3 -ffast-math -march=native ... -I ds4-rs -o /tmp/qa-p5/o3 \
       ds4-rs/tests/parity/eg2vq_embd_c_oracle.c -Wl,--gc-sections -lm -pthread

Observed: my -O0 build's sha256 is 22324867... = ds4-rs/tests/parity/eg2vq_embd_c_oracle;
my -O3 build's sha256 is a9f74d0b7a66920abba7a82de7eee022ede301b61e78bb9692f952db8fbe7940
= out/eg2vq_embd_c_oracle_o3, the exact binary both .meta files record. The RECIPE
command-24 line is therefore the line that produced the binary the tables came from.

Ran both on `head -3 out/phase5-ids-sub24.tsv` (ids 1, 3, 5) over the artifact
GGUF:

    /tmp/qa-p5/o0 <artifact> /tmp/qa-p5/ids3.tsv > /tmp/qa-p5/o0-3.raw   (3.21 s)
    /tmp/qa-p5/o3 <artifact> /tmp/qa-p5/ids3.tsv > /tmp/qa-p5/o3-3.raw   (0.59 s)

Observed: both 3-item raws are byte-identical to the first three lines of the
committed out/phase5-sub24-o0-raw.txt / -o3-raw.txt; the independent instrument on
my pair prints "min cos 1.000000000 (item 5)  max rel_rms 9.000e-07 (item 5)
malformed 0"; the scorer on the converted pair prints min 0.999999945, mean
1.000000040, 0 items below 0.99. From the committed 24-item pair I also re-ran
both instruments myself: instrument 1.000000000 / 9.000e-07 (item 5), scorer min
0.999999926, mean 1.000000022, 0 of 24 below 0.99 — exactly the document's and the
log's numbers.

Timings, stated precisely: the 24-item 149.17 s (-O0) / 21.87 s (-O3) / 6.8x
figures were READ from the committed out/phase5-oracle-o0-vs-o3.log, not re-run
at 24 items (the task asked for a cheap small-subset re-measure). My small-subset
runs give 3.21 s vs 0.59 s (5.4x, startup-dominated), consistent with the recorded
6.8x at 24 items. The ~80/~12 minutes-per-leg extrapolations to 583 items check
out arithmetically: 107029/3297 x 149.17 s = 4842 s (80.7 min); x 21.87 s = 710 s
(11.8 min). The 716% CPU side-claim is addressed in finding F3.

## Claim 6 — reuse logic and refusals (PASS, all demonstrated live)

In the /tmp sandbox with a fake fast oracle (logs its invocations, emits a canned
583-line raw), the real one was never risked; the real out/ raws are unchanged
(re-checked: acbc0f69..., 3e9ace37..., 9464c272...):

(a) 580-line raw not reused: artifact raw truncated to 580 lines, matching meta
    -> run 3 printed baseline "reused", did NOT reuse the artifact (one fake-oracle
    invocation for the artifact gguf), re-ran and rewrote it to 583 lines.
(b) different oracle sha forces a re-run: artifact .meta oracle_sha overwritten
    with zeros -> run 4 left baseline "reused" and re-ran the artifact (one new
    invocation); the new .meta records the new binary's sha.
(c) pinned-set mismatch exits 2: a tampered qrels passed as the 5th argument ->
    run 5a exited 2 with "eg2_phase5_gate.sh:
    tests/retrieval/scifact-test-qrels.tsv sha256 f3eca4c1... != the pinned
    864141f9...".
Controls: run 1 (no raws) re-ran both legs (2 invocations, gate completed in
15.4 s, set check printed the byte-for-byte line); run 2 re-ran nothing (0 new
invocations, both legs "reused"). Two further refusals were also exercised live:
the builder-vs-frozen cmp branch (exit 2, "is not what the builder produces") and
the builder's scifact.zip digest refusal (exit 2).

## Claim 7 — self-test and standing gates (PASS)

- `bash tools/eg2_retrieval_score_selftest.sh`: 8 cases, "selftest: PASSED (8
  cases, both fixtures)", exit 0 (including "B numeric tie-break: rank 2, not 1").
- `sha256sum -c SHA256SUMS`: 5 entries, all OK, exit 0.
- The five P-1 greps from AGENTS.md in /data/embeddinggemma2: grep 1 (python
  spawn/pyo3/libpython/cpython in ds4-rs/crates/*/src and ds4-rs/native) zero
  hits; grep 2 (pyo3/cpython/libpython/candle/tch/torch/transformers in
  ds4-rs/Cargo.toml and Cargo.lock) zero hits; ldd ds4-rs/ds4-server | grep -ic
  'python|torch' = 0; find ... -name '*.py' = no files; grep 5 (onnx/ort-sys)
  zero hits.

## Claim 8 — document integrity (PASS)

- `grep -n PENDING docs/retrieval-cost-gate5.txt` -> no match.
- docs/ds4-gemma-port-phases.md:230 — row X.7 is a single-line table row
  (awk -F'|': NF=6, 5 pipes), with the unit's numbers inline and DONE 2026-10-19;
  section 9's "P1 gate -> X.7 (DONE...)" and section 10 item 5 carry the same
  digits (0.956039571, 0.951778 -> 0.935819, 0.983333 -> 0.990000, -0.029414).
- RECIPE.md: heading "### 4e. What the codec costs, measured on a task (phase 5)"
  at line 288 contains command 24 with the gate invocation and the -O3 build
  line; section 5b's table matches doc 5c digit for digit; the three new
  section-7 rows (Retrieval, Device tower, Entry point) are at lines 509-511,
  contiguous table rows with no blank line before them (the preceding table rows
  are immediately above; a blank line follows the last one).
- Spot-check: the handoff's unit-15 paragraph, status line and PHASE 5 section
  carry the same numbers as the document (0.956039571 / 0.898186175 /
  0.972493258 / 0.951778 -> 0.935819 / 0.983333 -> 0.990000 / 23 of 300 /
  0.085722 / -0.029414); its "1000 s wall" is consistent with the file mtimes
  (20:47:46 items -> 21:04:27 gate-log end = 1001 s).

## Findings

F1 (severity: low — overstatement, comment and doc). The claim that the awk body
is byte-identical to the phase-4 gate's to_jsonl() function is false as written:
tools/eg2_oracle_to_jsonl.sh:11-12 and docs/retrieval-cost-gate5.txt:17-18. The
bodies differ in the error string ("eg2_oracle_to_jsonl: line" vs "gate: line"),
in the id print ("%s%s" plus a prefix argument vs "%s"), and in the shell wrapper
(-v prefix="$PREFIX" plus the RAW/PREFIX setup). Output for the no-prefix case is
identical — my replay of the phase-4 body over out/phase5-baseline-raw.txt is
byte-identical to the committed out/phase5-baseline.jsonl (cmp clean) — so no
number is affected. Introduced in 8ddd057; untouched by 07e0ccd.

F2 (severity: low — false claim, no such code). "the phase-5 gate diffs the two
bodies and reports the result" (tools/eg2_oracle_to_jsonl.sh:12-13, repeated at
docs/retrieval-cost-gate5.txt:18): nothing in tools/ diffs the two awk bodies. A
repo-wide grep finds the converter referenced only as a command to run
(tools/eg2_phase5_gate.sh:67, 174-178); no diff of any body exists. Again no
measurement is affected.

F3 (severity: low — one-off figure not reproducible exactly, no recorded source).
The "716% CPU on a four-item run" statement (docs/retrieval-cost-gate5.txt:171
and :398, tools/eg2_phase5_gate.sh:27, and the commit message) was not reproduced
in five independent /usr/bin/time -f '%P' runs of the two rebuilt binaries:
558%, 568%, 597% (-O3, 4 items), 592% (-O3, 3 items), 671% (-O0, 4 items), 670%
(-O0, 3 items). The claim's operative content is confirmed — the oracle is
multi-threaded through the pooled tower, about six to seven cores in flight, not
single-threaded — but the exact 716% is a one-off machine reading I could not
reproduce on 4-item runs, and I found no committed log that records it (searched
docs/, RECIPE.md, the handoff and out/phase5-* text and log files).

F4 (severity: informational — wording). The .meta record binds the oracle by
sha256 only; it names no path (tools/eg2_phase5_gate.sh:146-149, content of
out/phase5-{baseline,artifact}.meta). The sha matches out/eg2vq_embd_c_oracle_o3,
so the binding is sound; the claim's "named in the .meta" is loose wording.

F5 (severity: informational — environment, not the unit). At QA start (about
21:35) `git status --short` in the outer repo was empty; at QA end
embeddinggemma2-handoff.md showed a working-tree modification (mtime 21:42:22,
diff = the state card's push-boundary text updated to mention "the QA-evidence
commit"). This QA session wrote nothing into the repo (writes were /tmp only and
this report), so the change came from the working session/hook while QA ran. It
is excluded from the two reviewed commits, both of which are unchanged; flagging
it only so the tree state at commit time is honest.

No other discrepancy was found: every number in sections 4, 5, 5a, 5b, 5c, 7a,
7b and 8 that can be recomputed from committed inputs recomputed exactly; the
agreement, retrieval, tail, invariant, token, timing (except F3), hash and
refusal claims all held.

## What I did not test

- The full 583-item oracle legs and the full 24-item timing pair were not re-run
  (deliberately; the task asked for a cheap small-subset re-measure). The 24-item
  149.17 s / 21.87 s figures, the 977.48 s / 967.81 s leg times and the
  109.5/110.6 tokens/s were read from the committed logs, and the arithmetic was
  cross-checked, but not re-measured at scale.
- The retrieval quality numbers were reproduced with the project's own scorer
  only; I did not write an independent ranking implementation (the scorer's
  hand-fixture self-test is its cross-check).
- EG2_PHASE5_SERIAL=1, EG2_PHASE5_WIDTHS overrides, and the gate's other
  missing-file/exit-2 paths were not exercised; the three required refusal paths
  were.
- The 716% origin measurement was not found and could not be reproduced exactly
  (F3); the four-item 101-token CPU numbers above are my own replacement
  measurements, at a different time on a shared host.
- The device, serving and other later-unit rows that the documents mention as
  context (phase 6/7 numbers in RECIPE section 7, docs section 8 items 2-3) belong
  to units 16-19 and were not re-verified here beyond noting that the text is
  consistent with those units' own evidence files.
- BEIR's upstream distribution was not re-downloaded; the local zip's size and
  sha256 were checked instead.

verdict: overall PASS
