#Requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'
$repo = "E:/Development/dotLLM"
$outDir = "$repo/.perf-runs/ncu-2026-07-28"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.1\ncu.bat"
$env:DOTLLM_NVCC_ALLOW_UNSUPPORTED_COMPILER = "1"
Set-Location $repo

Write-Host "=== [1/3] attention_f16 vs attention_f16_dyn (issue #213 follow-up) ===" -ForegroundColor Cyan
$testDll = "$repo/tests/DotLLM.Tests.Unit/bin/Release/net10.0/DotLLM.Tests.Unit.dll"
& $ncu --set full `
    --kernel-name "regex:attention_f16" `
    --launch-count 12 `
    --export "$outDir/attn-dyn-vs-scalar" --force-overwrite `
    -- dotnet test tests/DotLLM.Tests.Unit -c Release --no-build `
       --filter "FullyQualifiedName~CudaAttentionDynVsScalarPerfTest" `
    *>&1 | Tee-Object -FilePath "$outDir/attn-dyn-vs-scalar.txt"

Write-Host "=== [2/3] I2_S ragged GEMV (bitnet_b1_58-xl ffn_down, k=5460, issue #206 follow-up) ===" -ForegroundColor Cyan
$xlModel = "E:/Development/bitnet-tests/models/bitnet_b1_58-xl/ggml-model-i2_s.gguf"
& $ncu --set full `
    --kernel-name "regex:i2_s_gemv.*ragged|dequant_i2_s_f16_ragged" `
    --launch-count 8 `
    --export "$outDir/i2s-ragged-gemv" --force-overwrite `
    -- dotnet run --project src/DotLLM.Cli -c Release --no-build -- `
       bench "$xlModel" --device cuda -p 8 -n 16 -r 1 `
    *>&1 | Tee-Object -FilePath "$outDir/i2s-ragged-gemv.txt"

Write-Host "=== [3/3] BitNet-2B-4T full decode-step kernel breakdown (fresh, post-#206/#207/#212/#213) ===" -ForegroundColor Cyan
$btModel = "E:/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T-gguf/snapshots/a1f2f1c765812aa8af3f6eda4a313707064bba15/ggml-model-i2_s.gguf"
& $ncu --set full `
    --launch-skip 20 --launch-count 30 `
    --export "$outDir/bitnet-2b4t-decode" --force-overwrite `
    -- dotnet run --project src/DotLLM.Cli -c Release --no-build -- `
       bench "$btModel" --device cuda -p 8 -n 12 -r 1 `
    *>&1 | Tee-Object -FilePath "$outDir/bitnet-2b4t-decode.txt"

Write-Host "=== ALL DONE ===" -ForegroundColor Green
Write-Host "Reports in $outDir"
Write-Host "Press Enter to close..."
Read-Host
