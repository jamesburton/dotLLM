# llama.cpp IQ4_XS logits sidecar

This directory holds the optional sidecar consumed by
`CudaLlamaCppLogitsParitySidecarTests.IQ4XS_LastTokenLogits_MatchLlamaCppSidecar_Cuda`.
Do not commit GGUF model files here.

Default sidecar path:

```text
tests/DotLLM.Tests.Unit/Cuda/References/llamacpp/iq4-xs-logits-sidecar.json
```

The test also accepts:

```text
DOTLLM_IQ4_XS_GGUF_PATH=<path-to-IQ4_XS.gguf>
DOTLLM_CUDA_IQ4_GGUF=<path-to-IQ4_XS.gguf>
DOTLLM_LLAMA_CPP_IQ4_XS_SIDECAR_PATH=<path-to-sidecar.json>
```

`DOTLLM_IQ4_XS_GGUF_PATH` should be preferred when the sidecar was generated
from a specific IQ4_XS file. `DOTLLM_CUDA_IQ4_GGUF` is accepted as the shared
real-IQ4 fixture variable used by the CUDA smoke and perf tests.

## Schema

```json
{
  "schema": "llama.cpp-logits-v1",
  "source": "llama.cpp llama-eval-callback",
  "model_path": "C:/models/example-IQ4_XS.gguf",
  "quantization": "IQ4_XS",
  "prompt": "The capital of France is",
  "input_ids": [510, 5765, 302, 6181, 349],
  "vocab_size": 49152,
  "last_token_logits": [0.0],
  "argmax_token_id": 0,
  "max_abs_tolerance": 2.0,
  "mean_abs_tolerance": 0.25
}
```

`last_token_logits` must contain exactly `vocab_size` float values from llama.cpp
after evaluating the full `input_ids` prompt. Keep the sidecar small enough for
review; do not include model weights or tensor dumps.

## Generation

Build llama.cpp with the `llama-eval-callback` diagnostic tool, then capture the
last-token logits for the same IQ4_XS GGUF and prompt:

```powershell
cmake -S C:\src\llama.cpp -B C:\src\llama.cpp\build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build C:\src\llama.cpp\build --config Release --target llama-eval-callback

$env:DOTLLM_IQ4_XS_GGUF_PATH = "C:\models\example-IQ4_XS.gguf"
C:\src\llama.cpp\build\bin\Release\llama-eval-callback.exe `
  --model $env:DOTLLM_IQ4_XS_GGUF_PATH `
  --prompt "The capital of France is" `
  --logits-json tests\DotLLM.Tests.Unit\Cuda\References\llamacpp\iq4-xs-logits-sidecar.json
```

If your llama.cpp checkout names the binary or output flag differently, use the
same schema above and write the JSON to the default path, or point
`DOTLLM_LLAMA_CPP_IQ4_XS_SIDECAR_PATH` at the generated file.
