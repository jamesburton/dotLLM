# usage: launch.ps1 <outdir> <logfile> <train args...>   (run on the T5500 from powershell/cmd)
param([string]$Out, [string]$Log, [string]$ArgFile)
$git = 'C:\Program Files\Git\bin\bash.exe'
$cmd = "@echo off`r`n`"$git`" /c/littlebit/watcher.sh`r`n"
Set-Content -Path C:\littlebit\watcher.cmd -Value $cmd -Encoding ascii
$cmd2 = "@echo off`r`n`"$git`" /c/littlebit/run_train.sh $Out $Log $ArgFile`r`n"
Set-Content -Path C:\littlebit\run_train.cmd -Value $cmd2 -Encoding ascii
(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c C:\littlebit\run_train.cmd'}).ProcessId
Start-Sleep 2
(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c C:\littlebit\watcher.cmd'}).ProcessId
