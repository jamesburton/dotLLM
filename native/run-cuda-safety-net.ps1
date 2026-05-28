param(
    [ValidateSet("all", "memcheck", "initcheck", "racecheck")]
    [string]$Tool = "all",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$project = "tests\DotLLM.Tests.Unit\DotLLM.Tests.Unit.csproj"
$filter = "Category=GPU&FullyQualifiedName~CudaKernelTests"

dotnet test $project -c $Configuration --filter $filter

$testCommand = "dotnet test $project -c $Configuration --no-build --filter `"$filter`""

if ($Tool -in @("all", "memcheck")) {
    & compute-sanitizer --tool memcheck --error-exitcode 1 cmd /c $testCommand
}

if ($Tool -in @("all", "initcheck")) {
    & compute-sanitizer --tool initcheck --error-exitcode 1 cmd /c $testCommand
}

if ($Tool -in @("all", "racecheck")) {
    & compute-sanitizer --tool racecheck --error-exitcode 1 cmd /c $testCommand
}
