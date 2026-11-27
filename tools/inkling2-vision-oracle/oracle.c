// Inkling-2 vision oracle: llama.cpp b12780 CPU (mtmd + libllama).
// usage: oracle <llm.gguf> <mmproj.gguf> <image.ppm|none> <prompt.txt> <outdir> <n_gen> <threads> [dllDir]
// Writes: outdir/embd.f32 (image embeddings, n_tok x n_embd), outdir/logits.f32 ((n_gen+1) x n_vocab),
//         outdir/meta.txt (chunk layout, generated ids). The prompt marker is "<__media__>".
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

static unsigned char * read_ppm(const char * path, int * w, int * h) {
    FILE * f = fopen(path, "rb"); if (!f) return NULL;
    char magic[3] = {0}; int maxv;
    if (fscanf(f, "%2s %d %d %d", magic, w, h, &maxv) != 4 || strcmp(magic, "P6")) { fclose(f); return NULL; }
    fgetc(f);
    size_t n = (size_t)(*w) * (*h) * 3;
    unsigned char * d = malloc(n);
    if (fread(d, 1, n, f) != n) { fclose(f); return NULL; }
    fclose(f);
    return d;
}

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage\n"); return 2; }
    const char * llm = argv[1], * mmproj = argv[2], * img = argv[3], * promptf = argv[4], * outdir = argv[5];
    int n_gen = atoi(argv[6]), threads = atoi(argv[7]);
    if (argc > 8) ggml_backend_load_all_from_path(argv[8]); else ggml_backend_load_all();
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(llm, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    struct llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    struct mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = false; mcp.n_threads = threads; mcp.print_timings = false;
    mcp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; mcp.warmup = false;
    mtmd_context * mctx = mtmd_init_from_file(mmproj, model, mcp);
    if (!mctx) { fprintf(stderr, "mtmd init failed\n"); return 1; }

    FILE * pf = fopen(promptf, "rb"); fseek(pf, 0, SEEK_END); long pl = ftell(pf); fseek(pf, 0, SEEK_SET);
    char * prompt = calloc(pl + 1, 1); fread(prompt, 1, pl, pf); fclose(pf);

    char path[1024];
    snprintf(path, sizeof path, "%s/meta.txt", outdir);
    FILE * meta = fopen(path, "w");

    mtmd_bitmap * bmp = NULL; int w = 0, h = 0;
    const mtmd_bitmap * bitmaps[1];
    size_t nb = 0;
    if (strcmp(img, "none")) {
        unsigned char * px = read_ppm(img, &w, &h);
        if (!px) { fprintf(stderr, "ppm read failed\n"); return 1; }
        bmp = mtmd_bitmap_init(w, h, px);
        bitmaps[0] = bmp; nb = 1;
    }
    struct mtmd_input_text txt = { prompt, strlen(prompt), true, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    int32_t rc = mtmd_tokenize(mctx, chunks, &txt, bitmaps, nb);
    if (rc) { fprintf(stderr, "tokenize rc=%d\n", rc); return 1; }
    size_t nc = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < nc; i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        enum mtmd_input_chunk_type t = mtmd_input_chunk_get_type(c);
        if (t == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n; const llama_token * tk = mtmd_input_chunk_get_tokens_text(c, &n);
            fprintf(meta, "text %zu", n);
            for (size_t j = 0; j < n; j++) fprintf(meta, " %d", tk[j]);
            fprintf(meta, "\n");
        } else {
            size_t n = mtmd_input_chunk_get_n_tokens(c);
            fprintf(meta, "image %zu n_pos %d w %d h %d\n", n, (int)mtmd_input_chunk_get_n_pos(c), w, h);
            if (mtmd_encode_chunk(mctx, c)) { fprintf(stderr, "encode failed\n"); return 1; }
            float * e = mtmd_get_output_embd(mctx);
            int ne = llama_model_n_embd_inp(model);
            snprintf(path, sizeof path, "%s/embd.f32", outdir);
            FILE * ef = fopen(path, "wb"); fwrite(e, sizeof(float), n * ne, ef); fclose(ef);
            fprintf(meta, "embd_dim %d\n", ne);
        }
    }
    llama_pos n_past = 0;
    rc = mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, 512, true, &n_past);
    if (rc) { fprintf(stderr, "eval rc=%d\n", rc); return 1; }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int nv = llama_vocab_n_tokens(vocab);
    fprintf(meta, "n_vocab %d n_past %d\n", nv, (int)n_past);
    snprintf(path, sizeof path, "%s/logits.f32", outdir);
    FILE * lf = fopen(path, "wb");
    float * lg = llama_get_logits_ith(lctx, -1);
    fwrite(lg, sizeof(float), nv, lf);
    fprintf(meta, "gen");
    for (int g = 0; g < n_gen; g++) {
        int best = 0;
        for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
        fprintf(meta, " %d", best);
        llama_token tok = best;
        struct llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(lctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        lg = llama_get_logits_ith(lctx, -1);
        fwrite(lg, sizeof(float), nv, lf);
    }
    fprintf(meta, "\n");
    fclose(lf); fclose(meta);
    return 0;
}
