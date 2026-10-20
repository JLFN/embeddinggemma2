/* eg2vq_vision_anchor.cpp — the ANCHOR that produced the frozen reference for
 * EmbeddingGemma-2's IMAGE path (phase 8, unit 22; the instrument
 * docs/eg2vq-vision-oracle-gate22.txt describes).  Renamed from
 * eg2vq_vision_oracle.cpp on 2026-10-20, under the owner's directive "We will
 * not use llama it must be ported all": this file is the repository's only
 * remaining link to the pin, it is opt-in (tools/eg2_vision_anchor.sh), and
 * neither the port, its build, its run nor any default gate touches it.  The
 * numbers it froze in tests/vision/reference/ are committed data that the PORT
 * must reproduce on its own (gate 23, unit 23).
 *
 * Why it exists: for text, this project's reference is a shipped binary
 * (llama-embedding at the pin). No shipped binary at the pin embeds an image —
 * examples/embedding/embedding.cpp has no mtmd support and tools/mtmd/mtmd-cli.cpp
 * has no embedding or pooling mode — so the reference has to be an instrument we
 * build. This one links the pinned mtmd (built in /data/llama-anchor-build) and
 * dumps, for one image:
 *
 *   (i)  the projected image soft tokens: the output of the vision tower, its
 *        3x3 average pooler, the sqrt(n_embd) scale, the weightless RMS and the
 *        768->512 tie-in projection, i.e. exactly what mtmd_get_output_embd
 *        returns after one mtmd_encode_chunk. This is what unit 23's CPU port and
 *        unit 24's device port are gated against.
 *   (ii) the pooled TEXT-T0WER embedding of the whole input (marker + text +
 *        image + text), produced by mtmd_helper_eval_chunks, which is the
 *        reference's OWN implementation of the splice our unit 25 must mirror:
 *        encode the image chunk, take its embeddings, and decode them as
 *        embeddings in place of the image tokens. Dumping it is what makes the
 *        image path gated end to end. Its L2 normalization is applied here with
 *        the same rule the project's own reference forward uses (mean pool, then
 *        L2 with the zero-norm guard), so the number is comparable.
 *
 * Everything the harness needs from the reference is a public mtmd/llama API; it
 * contains no arithmetic of its own, which is the point: a reference that
 * computes anything is a second implementation to debug.
 *
 * Build (recorded in tools/eg2_vision_anchor.sh; the anchor tree is the pin's own
 * CPU build):
 *   g++ -O2 -std=c++17 -I/data/llama.cpp/include -I/data/llama.cpp/ggml/include \
 *       -I/data/llama.cpp/tools/mtmd -o bin/eg2vq_vision_anchor \
 *       tests/parity/eg2vq_vision_anchor.cpp \
 *       -L/data/llama-anchor-build/bin -lmtmd -llama -lggml -lggml-base -lggml-cpu \
 *       -Wl,-rpath,/data/llama-anchor-build/bin
 *
 * Usage: eg2vq_vision_anchor <mmproj.gguf> <text-model.gguf> <image file> <out prefix>
 *   writes <out>.soft.tsv   the image soft tokens, one row per token
 *   writes <out>.text.tsv   the pooled embedding, one row, plus the L2 norm
 *   prints the preprocessing facts (grid, token counts) to stdout
 */
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void die(const char *m) { std::fprintf(stderr, "eg2vq_vision_anchor: %s\n", m); std::exit(1); }

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <mmproj.gguf> <text-model.gguf> <image> <out-prefix>\n", argv[0]);
        return 2;
    }
    const char *mmproj_path = argv[1];
    const char *model_path  = argv[2];
    const char *image_path  = argv[3];
    const std::string out_prefix = argv[4];

    llama_backend_init();

    // 1. the text model, CPU only: the reference's own loader, so a refusal here
    //    is the reference's verdict on the file rather than ours.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.vocab_only  = false;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) die("the text model did not load");
    std::fprintf(stdout, "[model] n_embd_inp %d, n_embd %d, n_embd_out %d, vocab %d\n",
                 llama_model_n_embd_inp(model), llama_model_n_embd(model),
                 llama_model_n_embd_out(model), llama_vocab_n_tokens(llama_model_get_vocab(model)));

    // 2. the mmproj: THIS is the acceptance check of the file tools/eg2_mmproj.c
    //    writes — the pinned reference either resolves it or it does not.
    mtmd_context_params cparams = mtmd_context_params_default();
    cparams.use_gpu    = false;
    cparams.print_timings = true;
    mtmd_context *ctx = mtmd_init_from_file(mmproj_path, model, cparams);
    if (!ctx) die("the reference refused the mmproj (mtmd_init_from_file returned NULL)");
    const int n_mmproj_embd = llama_model_n_embd_inp(model);
    /* The width of one soft token is the text model's input width: mtmd itself
     * refuses a pair that disagrees ("mismatch between text model (n_embd = %d)
     * and mmproj (n_embd = %d)", mtmd.cpp:652-659), so after a successful init
     * this IS the mmproj's projection dim. */
    std::fprintf(stdout, "[mmproj] accepted, projection dim %d (== text n_embd_inp)\n", n_mmproj_embd);

    // 3. the image, through the reference's own preprocessing. The printed grid
    //    and token count are the contract our port's preprocessing must meet.
    mtmd_helper_bitmap_wrapper bmp =
        mtmd_helper_bitmap_init_from_file(ctx, image_path, false, mtmd_helper_init_opt_default());
    if (!bmp.bitmap) die("the reference refused the image");
    std::fprintf(stdout, "[image] %ux%u pixels\n",
                 mtmd_bitmap_get_nx(bmp.bitmap), mtmd_bitmap_get_ny(bmp.bitmap));

    /* The prompt carries the mtmd marker (mtmd_default_marker() = "<__media__>",
     * mtmd.cpp:590-591, mtmd.cpp:1245-1256): mtmd splits the text on it and
     * substitutes the model's own begin marker, its image tokens and its end
     * marker ("<|image>" / "<image|>" for GEMMA4V, mtmd.cpp:955-958). */
    const char *marker = mtmd_default_marker();
    mtmd_input_text text;
    text.text          = marker;
    text.text_len      = std::strlen(marker);
    text.add_special   = true;
    text.parse_special = true;

    const mtmd_bitmap *bitmaps[1] = { bmp.bitmap };
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) != 0) die("mtmd_tokenize failed");

    // 4. (i) the image soft tokens, and the preprocessing facts.
    const mtmd_input_chunk *img_chunk = nullptr;
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    std::fprintf(stdout, "[chunks] %zu\n", n_chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk *c = mtmd_input_chunks_get(chunks, i);
        const enum mtmd_input_chunk_type t = mtmd_input_chunk_get_type(c);
        const size_t n = mtmd_input_chunk_get_n_tokens(c);
        std::fprintf(stdout, "[chunk %zu] type %d, n_tokens %zu\n", i, (int)t, n);
        if (t == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            img_chunk = c;
            const mtmd_image_tokens *it = mtmd_input_chunk_get_tokens_image(c);
            std::fprintf(stdout, "[image tokens] nx %zu, ny %zu, n_pos %d\n",
                         mtmd_image_tokens_get_nx(it), mtmd_image_tokens_get_ny(it),
                         (int)mtmd_image_tokens_get_n_pos(it));
        }
    }
    if (!img_chunk) die("the tokenizer produced no image chunk");

    if (mtmd_encode_chunk(ctx, img_chunk) != 0) die("the reference's vision encode failed");
    const size_t n_img_tok = mtmd_input_chunk_get_n_tokens(img_chunk);
    const float *soft = mtmd_get_output_embd(ctx);
    if (!soft) die("the reference produced no image embeddings");
    {
        std::string path = out_prefix + ".soft.tsv";
        FILE *f = std::fopen(path.c_str(), "wb");
        if (!f) die("cannot write the soft-token dump");
        std::fprintf(f, "# %s: %s, %zu soft tokens x %d dims, 7 decimals\n",
                     image_path, "mtmd_get_output_embd after one mtmd_encode_chunk",
                     n_img_tok, n_mmproj_embd);
        double sumsq = 0.0;
        for (size_t t = 0; t < n_img_tok; t++) {
            std::fprintf(f, "%zu", t);
            for (int d = 0; d < n_mmproj_embd; d++) {
                const float v = soft[t * (size_t)n_mmproj_embd + d];
                std::fprintf(f, "\t%.7f", v);
                sumsq += (double)v * (double)v;
            }
            std::fprintf(f, "\n");
        }
        std::fprintf(stdout, "[soft] %zu x %d written to %s (sum of squares %.6f)\n",
                     n_img_tok, n_mmproj_embd, path.c_str(), sumsq);
        std::fclose(f);
    }

    // 5. (ii) the whole path, as ONE batch.
    //
    // NOT through mtmd_helper_eval_chunks: that helper evaluates chunk by chunk,
    // and for an embeddings model the pooling is per batch, so the pooled vector
    // it leaves behind describes the LAST TEXT CHUNK alone. Measured, not
    // assumed: with the helper, two visibly different images produced byte-
    // identical pooled vectors (the soft tokens differed, the pooled did not).
    // NOT through the public llama_batch either: this version's compat conversion
    // (src/llama-batch.cpp:1410-1484) stores embeddings only when the batch has
    // NO token array, so a mixed batch built that way loses the image rows and
    // segfaults (measured: exit 139 inside the first encode).
    //
    // The splice is therefore built with the reference's own llama_batch_ext,
    // which is what its image path uses (mtmd-helper.cpp's decode_embd_batch) and
    // which carries a token id and/or an embedding per entry: the text tokens go
    // in as tokens, the image span as embedding entries, all in sequence 0 with
    // their own positions, and one llama_process call encodes the lot so the
    // graph pools over the whole sequence.
    llama_context_params lparams = llama_context_default_params();
    lparams.n_ctx        = 4096;
    lparams.n_batch      = 4096;
    lparams.n_ubatch     = 4096;
    lparams.embeddings   = true;
    lparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    llama_context *lctx = llama_init_from_model(model, lparams);
    if (!lctx) die("the llama context did not initialize");
    if (llama_pooling_type(lctx) != LLAMA_POOLING_TYPE_MEAN) die("the model does not declare mean pooling");

    /* The final sequence is the chunks' own order: the leading text chunk, then
     * the image span where its markers were, then the trailing text chunk. The
     * image span's rows are marked with LLAMA_TOKEN_NULL so the fill loop knows
     * where the embeddings go. */
    std::vector<llama_token> seq_tokens;
    size_t image_start = 0, n_img_span = 0;
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk *c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t nt = 0;
            const llama_token *tt = mtmd_input_chunk_get_tokens_text(c, &nt);
            for (size_t k = 0; k < nt; k++) seq_tokens.push_back(tt[k]);
        } else {
            image_start = seq_tokens.size();
            n_img_span  = mtmd_input_chunk_get_n_tokens(c);
            for (size_t k = 0; k < n_img_span; k++) seq_tokens.push_back(LLAMA_TOKEN_NULL);
        }
    }
    const size_t n_total = seq_tokens.size();
    std::fprintf(stdout, "[splice] %zu positions: the image span is [%zu, %zu)\n",
                 n_total, image_start, image_start + n_img_span);

    llama_batch_ext *be = llama_batch_ext_init(lctx);
    if (!be) die("llama_batch_ext_init failed");
    for (size_t i = 0; i < n_total; i++) {
        int32_t idx;
        if (n_img_span && i >= image_start && i < image_start + n_img_span) {
            llama_embd e;
            e.data   = soft + (i - image_start) * (size_t)n_mmproj_embd;
            e.n_rows = 1;
            e.n_embd = (size_t)n_mmproj_embd;
            idx = llama_batch_ext_add_embd(be, 0, e);
        } else {
            idx = llama_batch_ext_add_token(be, 0, seq_tokens[i]);
        }
        if (idx < 0) die("a batch entry was refused");
        const llama_pos pos = (llama_pos)i;
        if (!llama_batch_ext_set_pos(be, idx, &pos)) die("a batch position was refused");
        if (!llama_batch_ext_set_output_embd(be, idx, true)) die("an output flag was refused");
    }

    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, be) != 0) die("the single-batch process failed");
    const float *pooled = llama_get_embeddings_seq(lctx, 0);
    if (!pooled) die("no pooled embedding came out");
    const int n_out = llama_model_n_embd_out(model);

    double ss = 0.0;
    for (int i = 0; i < n_out; i++) ss += (double)pooled[i] * (double)pooled[i];
    const double norm = std::sqrt(ss);
    {
        std::string path = out_prefix + ".text.tsv";
        FILE *f = std::fopen(path.c_str(), "wb");
        if (!f) die("cannot write the pooled dump");
        std::fprintf(f, "# %s through ONE llama_process of %zu positions (image span [%zu, %zu)): "
                        "pooled %d dims, then L2 with the zero-norm guard\n",
                     image_path, n_total, image_start, image_start + n_img_span, n_out);
        std::fprintf(f, "raw");
        for (int i = 0; i < n_out; i++) std::fprintf(f, "\t%.7f", pooled[i]);
        std::fprintf(f, "\n");
        std::fprintf(f, "l2");
        const double inv = norm > 0.0 ? 1.0 / norm : 0.0;
        for (int i = 0; i < n_out; i++) std::fprintf(f, "\t%.7f", (double)pooled[i] * inv);
        std::fprintf(f, "\n");
        std::fclose(f);
    }
    std::fprintf(stdout, "[text] pooled %d dims, norm %.9f, %zu positions in one batch; written to %s.text.tsv\n",
                 n_out, norm, n_total, out_prefix.c_str());

    llama_batch_ext_free(be);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp.bitmap);
    mtmd_free(ctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
