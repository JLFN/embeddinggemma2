#!/usr/bin/env python3
# One-shot evidence producer. NOT part of the runtime, never imported by it.
#
# Why this exists: tools/eg2_gguf.c hand-writes the checkpoint's tokenizer
# metadata, reproducing llama.cpp's conversion/vocab path (conversion/gemma.py
# set_vocab -> special_vocab.add_to_gguf). This probe runs that SAME reference
# code (gguf-py from the pinned /data/llama.cpp) over the same checkpoint and
# dumps the GGUF keys it would write, so the hand-written key set can be
# compared against the reference instead of assumed. The output is frozen into
# docs/ as evidence; the runtime never reads this file (project rule P-1 allows
# Python as a one-off measurement/evidence producer under tools/, and P-2 item 2
# allows a reference to be executed once outside the runtime).
#
# Usage:
#   python3 tools/ref_vocab_probe.py <checkpoint_dir> <out_gguf>
#
# The out_gguf is a GGUF containing ONLY the keys SpecialVocab.add_to_gguf()
# writes. Dump it with the port's own reader:
#   cargo run -q -p ds4-core --release --example gguf_inventory -- <out_gguf>
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/llama.cpp/gguf-py")
import gguf  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    ckpt = Path(sys.argv[1])
    out = Path(sys.argv[2])

    sv = gguf.SpecialVocab(ckpt, load_merges=True)
    print("repr             :", repr(sv))
    print("special_token_ids:", json.dumps(sv.special_token_ids, sort_keys=True))
    print("add_special_token:", json.dumps(sv.add_special_token, sort_keys=True))
    print("n_merges         :", len(sv.merges))
    print("merges_head      :", sv.merges[:2] if sv.merges else None)
    print("chat_template    :", None if sv.chat_template is None else f"{len(sv.chat_template)} chars")
    print("normalizer nfc   :", sv.normalizer_nfc)
    print("normalizer lower :", sv.normalizer_lowercase)
    print("normalizer strip :", sv.normalizer_strip_accents)

    out.parent.mkdir(parents=True, exist_ok=True)
    gw = gguf.GGUFWriter(out, arch="gemma-embedding2")
    sv.add_to_gguf(gw)
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print("wrote            :", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
