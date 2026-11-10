QA findings - the attention performance unit (two passes, both overall PASS)
============================================================================

Both passes were run by the AI QA-tester on deepseek-v4.1-CC-flash, the model
this project records, against the engine tree with the kernels built and driven
live on the RTX 4070 SUPER.  Their verdicts are in qa-evidence/qa-report.md.
This file records what they FOUND, including the finding that changed the unit
and the ones that changed only its record.

Pass 1 (intermediate revision, the item-sized plane)

F1.  THE PLANE-SIZING HALF OF THE UNIT WAS WRONG.  Severity: changed the code.
     Evidence: with the same vec4=1 kernel and the same data, interleaved 30
     reps, plane size as the only variable: page565_s 1.53 / 1.38 / 1.09 ms at
     planes 565 / 1025 / 2048, and page294_s 0.2578 / 0.4167 / 0.3011 --
     non-monotonic, and reproduced across sessions, data sets, and with the
     shared/L1 carveout pinned.  The record's claim that a smaller plane is what
     makes the kernel fast was therefore incomplete at best.
     Disposition: the parent session reproduced the curve independently (its own
     probe, 25 reps: 565 -> 1.378 ms, 1025 -> 1.232, 2048 -> 0.977 at 565 tokens;
     294 -> 0.258, 1025 -> 0.419, 2048 -> 0.302 at 294 tokens), then measured the
     policies end to end over the 143-page corpus: the item-sized plane 13.09 s
     at batch 8 against 11.05 and 10.60 s for the fixed ceiling plane.  The
     caller plumbing and the per-call plane were removed; the plane is the item
     ceiling again and only the vectorization ships.  The curve is recorded as
     measured and unexplained, and the record says plainly that the occupancy
     model does not predict it.

F2.  EG2VQ_ATTN_SHARED_MAX was dead (nothing referenced it).
     Disposition: with the dynamic plane gone there is one shared constant again
     (EG2VQ_ATTN_SHARED), referenced by the kernel's own declaration.

F3.  The plane guard drops a row's output silently by design.
     Disposition: documented in the kernel; the honest caller never trips it (the
     cap comes from DS4_EG2VQ_MAX_TOKENS) and the QA probe confirms the drop is
     clean rather than a partial write.

F4.  Not driven in pass 1: the nsys attributions and the corpus sweep rows.
     Disposition: recorded as UNVERIFIED in that pass; the parent session's
     reports and CSVs are committed with the unit.

F5.  The committed before/after rows came from one session whose clocks could not
     be reproduced exactly (the QA's long-shape after rows came out 11-13%
     faster).
     Disposition: the record now carries a window caveat, and the QA pass
     subsequently rebuilt the dev kernel from its own sources and matched the
     before column within 7.4%.

Pass 2 (this revision: static ceiling plane plus vectorized loads and stores)

F6.  No code defect found.  The load-bearing claims were driven independently:
     the vector path is byte-identical to the scalar path (22,335,488 floats
     compared, zero differing bytes, and identical when the fallback is forced by
     unaligned pointers), the guard leaves a 2200-token item's output untouched
     while a 2048-token item is written fully and correctly, the launcher refuses
     every head count but 4, and the removed plumbing leaves nothing dangling.

F7.  THE ITEM-PLANE CSV WAS MISLABELLED.  Severity: record-only, but severe.
     Evidence: speed-bench/eg2_embed_vizzio_attn-itemplane.csv held the fixed
     ceiling plane's numbers (11.0454 s at batch 8), not the rejected item plane's
     (13.0896 s); read literally, the two committed CSVs would have shown the
     rejected variant FASTER than the shipped build, the opposite of the record's
     central end-to-end claim.  The real item-plane CSV existed only on the
     revision that was amended away.
     Disposition: the correct CSV is committed under that name, and every policy
     now has its own committed artifact (item-plane, 2048 and 1536 with their two
     runs each, the shipped build with three), each identified by its own
     srv_batch_8 row.

F8.  Three rows of the end-to-end table had no committed artifact (the second
     ceiling sweep and both 1536 sweeps).
     Disposition: all are committed now, under the names the table cites.

F9.  The bench's stability was overstated ("did not move by more than ~1%").
     Evidence: the same static binary, minutes apart, measured page294_s 0.3021
     vs 0.2724 ms and batch8x565_s 7.6520 vs 6.9530 ms -- about 10%.
     Disposition: the record and the kernel's measurement note now say the
     bench's practical error bar is ~10% on any single row, and that the claim is
     the before/after ratio, which is 3.0x-3.9x.

F10. Nits, all fixed: a tokens/s figure off by 0.04%; ds4_eg2vq.h:180 citing
     cuda/eg2vq_attn.cuh:90 for a constant at :93; and the two committed health
     JSONs being identical, so they identified no build (there is now one health
     payload, captured from the shipped build, and the record says the caps did
     not move).

F11. Provenance limit, recorded rather than fixed: the corpus CSVs carry no
     timestamp, so the window ordering the tables rely on comes from the session
     log.  The record says so where the tables are.

F12. Not re-driven in pass 2: the nsys attributions, and the corpus sweeps
     themselves (the QA reproduced the kernel-level table and audited the
     committed CSVs instead).  The per-shape table it did reproduce landed within
     0.3% on most rows.
