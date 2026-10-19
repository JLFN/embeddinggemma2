# Project rules — EmbeddingGemma-2 VQ runner

Scope: this file adds to this project's loaded guardrails: the always-on core in
`~/.opengrok/AGENTS.md` and the mounted project-work rules in
`.opengrok/rules/project-work-rules.md`. Nothing here weakens those; where this
file is more specific, it wins for `/data/embeddinggemma2` and everything under
it. A rule below is binding on any session in this workspace, and a fresh agent
with nothing but this file must be able to execute it — that is why every rule
carries its own verification command.

Build and test commands stay in the handoff and in `RECIPE.md`; guardrails live
here.

## P-1. No Python in the runtime — ever (owner instruction, 2026-10-19)

Owner's words: "i dont want no python you can put this in guard rail for this
project, we can use it as test script or script to optimize the code but it will
never be in final runtime".

The runtime is Rust, C and CUDA. That is the whole delivery: every binary this
project builds or ships — the `ds4-rs` port and its crates (`ds4`, `ds4-server`,
`ds4-cli`, `ds4-bench`, `ds4-agent`, any library the port produces), any
launcher, service or wrapper added here, and anything derived from them.

Forbidden in the runtime, with no exception and no "temporary" allowance:

1. A `.py` file the runtime reads, imports, executes or ships.
2. Starting an interpreter: `std::process::Command` to `python`/`python3`,
   `system("python3 ...")`, `popen`, `exec`/spawn of any interpreter, or a shell
   wrapper that does so.
3. Embedding an interpreter: `pyo3`, `cpython`, `libpython`, maturin/PyO3
   builds, `ctypes`/`CDLL`/`dlopen` of libpython, or any JIT that runs Python.
4. A Python-side ML stack as the compute path: `torch`, `transformers`,
   `sentence-transformers`, Python `onnxruntime`, Python `numpy`.
5. A general-purpose tensor framework as the compute path in Rust: `candle`,
   `burn`, `tch`. The arithmetic is this project's own C/CUDA kernels, extended
   where the port needs it. Owner, 2026-10-19, on `candle`: "are you see we use
   cuda and no candle 0.11 its to slow see how the rest of the code work".

Allowed, and normal here:

- Python as a test harness, gate script or driver of a compiled binary: it runs
  the binary and compares its output.
- Python as a one-off measurement, probing or optimization script for the codec
  and kernel work.
- Python to produce evidence once (for example a reference-vector file): the
  output is committed as data, and the runtime never reads the script, re-runs
  it, or depends on it.
- A pinned upstream implementation written in another language may be read and
  cited line by line as the source of the mathematics, or run once outside the
  runtime. It is a document, never a runtime dependency.

Placement: Python allowed by this rule lives only under `tests/` or `tools/`,
is named so a reader sees it is a script, and is never referenced from the
runtime's build, source or startup path. (`ds4-rs` is a copied upstream tree and
keeps its own 164 harness scripts under `tests/`, `docs/benchmarks/` and
`dir-steering/`; those are harnesses, not runtime.)

Verification — run these in `/data/embeddinggemma2` and report the result. Each
must come back with zero hits:

    grep -rn 'Command::new("python\|python3\|pyo3\|libpython\|cpython' ds4-rs/crates/*/src ds4-rs/native
    grep -in 'pyo3\|cpython\|libpython\|candle\|tch \|torch\|transformers' ds4-rs/Cargo.toml ds4-rs/Cargo.lock
    ldd ds4-rs/ds4-server | grep -ic 'python\|torch'
    find ds4-rs/crates/*/src ds4-rs/native -name '*.py'
    grep -rin 'onnx\|ort-sys' ds4-rs/crates/*/src ds4-rs/native ds4-rs/Cargo.toml ds4-rs/Cargo.lock

This project's own tooling is C and shell — the writer, the encoder, the checkers
and the anchor generator (`tools/*.c`, `tools/eg2_*.sh`), which is the faster and
more durable choice when awk, sed, jq, od and dd already do the job. There is
exactly one Python file in the tree, `tools/ref_vocab_probe.py`, and it is the
kind the allowed list names: a one-off evidence producer that runs the pinned
reference's own `SpecialVocab` over the checkpoint to print the GGUF keys it would
write, whose output is frozen into `docs/anchor-gate2.txt`. Nothing in the runtime
reads, imports or invokes it, and nothing else here needs Python.

All five returned zero on 2026-10-19, measured in this workspace. The ldd check is
not vacuous: `ds4-rs/ds4-server` is a built 76 MB ELF linked against
`libcudart.so.13` and `libcublas.so.13`, and `ldd` on it reports no python and no
torch. A hit means
the change is rejected before merge and before any push, and it is not a style
preference: global rule 0 makes an owner instruction binding, and this check is
how a fresh session proves the rule still holds.

Why: the runner must be as fast and as optimal as the hardware allows. An
interpreter is a dependency, a version, and a failure mode the model does not
need.

## P-2. Reference sources are citations or one-shot instruments, never dependencies

The runtime is only ever this project's own Rust host, C graph, and C/CUDA
kernels. Everything else is evidence:

1. A reference implementation (upstream model code, a sibling engine, a
   conversion a third party published) is read first-hand and cited by file and
   line in the port's code comments and in the evidence file that lands with the
   change — global rule 0.2.
2. A reference may be executed once, outside the runtime, to freeze numbers into
   a committed evidence file (reference vectors, a reconstruction table, a
   latency baseline). After that it is data.
3. If a reference route needs a Python stack to run, the route still may not
   enter the runtime, and the frozen output must state which revision produced
   it. Prefer a route that needs no Python at all.
4. A third-party conversion (a published export, a GGUF conversion) is never
   ground truth by itself: before its numbers are used as the denominator of a
   quality claim, the conversion is checked against the pinned source and that
   check is recorded.
5. No ONNX and no PyTorch in this project at all — not in the runtime, not in the
   verification chain, not as a measuring instrument (owner, 2026-10-19: "remove
   this onnx i dont want it and no pytorch see how /data/ds4-dfm-rs works"). The
   verification is the ds4-dfm-rs method: the project's own native C oracle
   against the Rust candidate, committed golden snapshots, and device parity,
   with the upstream implementation read and cited as the authority
   (docs/ds4-gemma-port-reference.md section 6).

Verification: any quality or parity number in a commit, evidence file or
handoff must name the artifact and revision that produced it, and the runtime
tree must still pass the five P-1 checks.

## Where the plan and the record live (rule 9 pointers)

Read these before doing port work in this workspace:

1. `embeddinggemma2-handoff.md` — the plan of record: the phased work spec with the
   exact reference files per phase, the pinned repos, the routes removed, the
   guardrails and the staleness check. Where it disagrees with any other plan
   document, it wins; correct the other document in the same commit.
2. `docs/ds4-gemma-port-phases.md` — the row-by-row work list behind the handoff
   (P0..P4 plus the X rows), each row with its authority citation and its gate.
3. `docs/ds4-gemma-payload-contract.md` — the `gemma_vq` contract the writer and
   the reader are both written against: the metadata keys with their GGUF types,
   the tensor naming and types, the payload byte layout, the recipe string, where
   shape and rank live, and the host/device split. It also settles, by
   measurement, that the encoder's payloads are DQVQ payloads byte for byte and
   that the artifact GGUF step is a re-wrap rather than a re-encode.
4. `docs/baseline-gguf-gate1a.txt` — the phase-1a evidence: the baseline GGUF's
   identity, the two-sided check (writer's manifest vs the port's own reader, then
   every tensor's bytes against the checkpoint) and the reference's own acceptance
   of the file, with the four measured corrections to the contract.
5. `docs/anchor-gate2.txt` — the phase-2 evidence: the fifth correction (the three
   tokenizer keys the reference's own `SpecialVocab` writes and our writer was
   missing, found by running that reference code), the frozen 113-item input set,
   the anchor's provenance and digests, and the measurements that settle the eos
   and prompt-prefix decisions against their alternatives.
6. `docs/artifact-gguf-gate1b.txt` — the phase-1b evidence: the artifact GGUF
   written by re-wrapping the DQVD container (arch `gemma_vq`, 461 `vqblob`
   payloads, one `q8_0` token table, 914 `bf16` tensors), gate 1b's three-sided
   check, the metadata read back value by value through the port's own reader, the
   sixth correction (the file's rank is never 0), and the fact that ds4's private
   type 42 is why the reference loader accepts the baseline and refuses the
   artifact.
7. `docs/ds4-gemma-port-reference.md` — the port map: shape, tensor naming,
   arithmetic, and the four-leg native verification method.
8. `RECIPE.md` — how the artifacts in `out/` were made, and their measured quality;
   `RECIPE-deepseek41-vq.md` — the ds4 VQ recipe this project follows.
9. Cross-session memory for this workspace lives under
   `~/.opengrok/memory/embeddinggemma2-1200866d/MEMORY.md`; a fresh session's
   memory_search for "embeddinggemma2" must surface it.
