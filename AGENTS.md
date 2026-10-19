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

All four returned zero on 2026-10-19 (measured in this workspace). A hit means
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
4. A third-party conversion (for example an ONNX export, a GGUF conversion) is
   never ground truth by itself: before its numbers are used as the denominator
   of a quality claim, the conversion is checked against the pinned source and
   that check is recorded.

Verification: any quality or parity number in a commit, evidence file or
handoff must name the artifact and revision that produced it, and the runtime
tree must still pass the four P-1 checks.
