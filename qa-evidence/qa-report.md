QA report - the embedding-path performance unit, first half: the attention kernel
================================================================================

Unit and repositories at verification time

Engine /data/embeddinggemma2/ds4-rs, branch feat/eg2vq-attn-perf, HEAD
31285531, based on dev 10e73aa0 (which is publish/dev; the branch carries
exactly one commit and dev is untouched).  Outer /data/embeddinggemma2, branch
feat/eg2vq-attn-perf (equal to its dev a43479d plus this unit's record commits).
Nothing in this unit has been pushed.  The owner asked for a feature branch
explicitly ("dont tought dev we need a new Feat branch for this"), and both
repositories comply: zero commits on dev in either.

What the unit changes

cuda/eg2vq_attn.cuh: pass 1 of the device attention reads the K rows as float4
and pass 3 reads V and writes the output as float4, when head_dim % 128 == 0 and
q/k/v/out are 16-byte aligned; the scalar loops are retained as the fallback. The
accumulation order is unchanged element for element, so the vector path is
bit-identical to the scalar one.  A guard returns without writing when a row needs
more keys than the plane holds.  tests/bench_eg2vq_attn.cu is new, with a make
target.  A per-call score-plane sizing was implemented, measured, rejected and
reverted WITHIN the unit; nothing of it remains in the diff.

Rule 19 surface extractor

`QA_BASE=publish/dev bash tests/qa-gate.sh` prints
`QA GATE: ALL PASS (no new runtime, route or public-API surfaces relative to
publish/dev and ds4-rs/publish/dev)` and exits 0: the unit adds no
`ds4_*(` engine entry point, no HTTP route and no `pub fn`, so the extractor
names no surface.  That verdict is about NEW surfaces only, which is why this
pass was run by hand anyway -- the unit rewrites the body of a kernel the whole
tower's arithmetic depends on, and the two passes below are what actually
establish it.

The QA-tester passes

Two passes, both model deepseek-v4.1-CC-flash, both ending VERDICT: PASS, both
driving the code rather than reading it.  Their findings and dispositions are
recorded in qa-evidence/qa-findings-attn-perf.md.

Pass 1, on the intermediate revision (the item-sized plane).  It reproduced the
correctness gates, wrote its own probe proving the vector path byte-identical to
the scalar path (22,335,488 floats, no differing byte), drove the plane guard and
the caller's cap, and measured the plane-size curve with plane size as the only
variable.  That measurement is what killed the plane-sizing half of the unit: the
curve is non-monotonic, and the call-sized plane was slower end to end on every
corpus row.

Pass 2, on this revision.  It re-ran every gate, re-proved the bit-identity and
the guard (a 2200-token item leaves the output untouched; 2048 tokens writes it
fully), confirmed the removed plumbing leaves nothing dangling (no hits for
keys_cap/plane_keys/attn_keys, one declaration of `plane`, live make targets),
reproduced 9/9 rows of the committed bench table (within 0.3% on most), rebuilt
the REJECTED kernel from its commit and reproduced the plane curve 5/5 rows, and
built the static/dynamic twin the record calls a wash (<=0.9% both ways, so the
wash holds).  It also found four defects in the RECORD, all fixed in this commit:
the item-plane CSV had been overwritten with the ceiling policy's numbers; three
table rows had no committed artifact; the bench's stability was overstated at ~1%
against a measured ~10% error bar; and two line-number citations were stale.  It
found no code defect.

Gates run against this revision (all by the parent session, after the amend)

  make test-eg2vq-attn              EG2VQ_ATTN_PASS, min cosine 1.000000000 on
                                    every case (26 to 1414 tokens, both layer
                                    geometries, the mask edge, two items packed,
                                    the 5-head refusal leaving the output alone)
  tests/parity/eg2vq_embd_gpu_oracle  EG2VQ_GPU_PASS, 300 items / 8059 tokens,
                                    min cos(device,cpu) 0.999999950, max
                                    rel_rms 3.150e-04 against the declared
                                    cos >= 0.999999 / rel_rms <= 1e-03;
                                    device steady state 1499.6 tok/s
  make bench-eg2vq-attn             9 shapes, one kernel launch each, 30 reps:
                                    2.98x to 3.88x over the dev kernel, every
                                    row reproduced by the QA pass
  make ds4 ds4-server               builds; the C CLI and the route both serve
                                    the artifact afterwards (the route was driven
                                    by the QA pass with its own server: a mixed
                                    5-item request whose 980-token item matched
                                    its solo vector cos 1.000000000)
  GET /health                       byte-identical to speed-bench/
                                    eg2_embed_vizzio_health.json (the unit's own
                                    capture of the shipped build): item_tokens_max
                                    2048, batch_rows_max 8192, image soft
                                    70..1120, pixels 161280..2580480, embd 512,
                                    per_item_max 1, serial true

What this pass does NOT cover

- No hardware counters were taken (ncu refuses with ERR_NVGPUCTRPERM on this
  machine); every number in the record is a wall or kernel duration.
- The nsys attributions in the record's section 1 were not re-run by the QA pass;
  they were measured by the parent session and their reports are reproducible by
  the commands the record gives.
- The corpus sweeps' window ordering rests on the session log, not on the CSVs
  (they carry no timestamp); the record says so.
- The VQ GEMM, now the largest item left, is untouched by this unit and its
  limiter is not established.  That is the next unit.

verdict: overall PASS
