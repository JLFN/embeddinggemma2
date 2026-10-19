# Provenance and license of the retrieval set in this directory

The three frozen files next to this note are a redistribution of a labelled
retrieval dataset.  Their own headers carry the source URL and the archive's
sha256; this note records the terms they are redistributed under, which the
files themselves cannot state.

## Source

    scifact-test-queries.tsv   300 queries
    scifact-test-pool.tsv      283 documents (title and abstract text)
    scifact-test-qrels.tsv     339 labelled query-document pairs

built by `tools/eg2_retrieval_set.sh` from BEIR's own SciFact distribution:

    https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
    sha256 536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165

## License

SciFact is released under **Creative Commons Attribution-NonCommercial 2.0**
(CC BY-NC 2.0), as stated on the dataset owner's own card
(allenai/scifact, "The SciFact dataset is released under the CC BY-NC 2.0").
Redistribution of these files is therefore permitted with attribution and for
non-commercial use; a commercial reuse of this repository cannot reuse them.

The query and document text belongs to the dataset and its authors, not to this
project.  The scoring harnesses (`tools/eg2_retrieval_score.sh`, its self-test)
and the gate that consumes the set (`tools/eg2_phase5_gate.sh`) are this
project's own work and are not covered by the dataset's terms.

## Why this note is a separate file

The gate pins each of the three files' sha256 and builds the item list from
their non-comment lines, so editing a header in place would change the pinned
digests and the item/ids account the phase-5 numbers are measured on.  A note
beside them records the terms without touching the measured inputs.
