#!/usr/bin/env bash
# Linux x64 test only: simulated HIP runtime, real production loading path/libc O_DIRECT reads.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$script_dir/../../.." && pwd)"
out="$repo/.docs/linux-hip-qualification"
rocm="${ROCM_PATH:-/opt/rocm}"
dotnet="${DOTNET:-dotnet}"
export DOTNET_PROCESSOR_COUNT=4
mkdir -p "$out/co" "$out/stub" "$out/kernels"
cd "$repo"
"$rocm/bin/hipcc" --version > "$out/toolchain.log"
# Build through the production script in scratch, preserving tracked Windows objects.
cp native/hip/build.sh "$out/build.sh"
cp native/hip/kernels/*.hip "$out/kernels/"
HIP_COMPILER="$rocm/bin/hipcc" HIP_ARCHS="gfx1030 gfx1100 gfx1151" bash "$out/build.sh" > "$out/native-build.log" 2>&1
g++ -std=c++17 -shared -fPIC -I"$rocm/include" "$script_dir/hip_stub.cpp" -o "$out/stub/libamdhip64.so" > "$out/abi-build.log" 2>&1
"$dotnet" build "$script_dir/HipLinuxProbe.csproj" -c Release --artifacts-path "$out/artifacts" -p:HipCoOutputDir="$out/co" -m:4 --nologo > "$out/managed-build.log" 2>&1
LD_LIBRARY_PATH="$out/stub" "$dotnet" "$out/artifacts/bin/HipLinuxProbe/release/HipLinuxProbe.dll" > "$out/probe.log" 2>&1
"$rocm/llvm/bin/clang-offload-bundler" --list --type=o --input="$out/co/dense_f16.co" > "$out/targets.txt"
for arch in gfx1030 gfx1100 gfx1151; do
  "$rocm/llvm/bin/clang-offload-bundler" --unbundle --type=o --input="$out/co/dense_f16.co" --targets="hipv4-amdgcn-amd-amdhsa--$arch" --output="$out/dense_f16-$arch.hsaco"
  "$rocm/llvm/bin/llvm-objdump" --syms "$out/dense_f16-$arch.hsaco" > "$out/manifest-$arch.txt"
done
sha256sum "$out/co/"*.co "$out/stub/libamdhip64.so" > "$out/hashes.txt"
cat "$out/probe.log"
