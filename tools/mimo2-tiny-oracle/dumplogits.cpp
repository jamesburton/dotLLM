// dumplogits model.gguf ids.txt out.f32 [decode]
// Loads a GGUF with llama.cpp (CPU, no offload), decodes the token ids in ONE batch with logits for every position and writes
// n_tokens x n_vocab float32 logits. With the optional 4th arg "decode" it instead feeds tokens one at a time (KV-cache decode
// path) and writes the same layout.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage\n"); return 2; }
    bool one_by_one = argc > 4 && std::string(argv[4]) == "decode";
    std::vector<llama_token> ids;
    { std::ifstream f(argv[2]); long v; while (f >> v) ids.push_back((llama_token) v); }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr }; mp.devices = devs;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.no_perf = true; cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n = (int) ids.size();
    std::vector<float> out((size_t) n * n_vocab);
    if (!one_by_one) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) { b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        b.n_tokens = n;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int i = 0; i < n; i++) memcpy(&out[(size_t) i * n_vocab], llama_get_logits_ith(ctx, i), n_vocab * sizeof(float));
        llama_batch_free(b);
    } else {
        for (int i = 0; i < n; i++) {
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = ids[i]; b.pos[0] = i; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
            if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
            memcpy(&out[(size_t) i * n_vocab], llama_get_logits_ith(ctx, 0), n_vocab * sizeof(float));
            llama_batch_free(b);
        }
    }
    FILE * f = fopen(argv[3], "wb"); fwrite(out.data(), sizeof(float), out.size(), f); fclose(f);
    fprintf(stderr, "wrote %d x %d\n", n, n_vocab);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
