# Open a .rgp in Radeon GPU Profiler, dump the UI Automation text of every tab, and PrintWindow a screenshot of each.
# Usage: pwsh scripts/rgp-uia-dump.ps1 <abs path to .rgp> <abs out dir>
param([string]$Rgp, [string]$Out)
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Drawing
Add-Type @'
using System; using System.Runtime.InteropServices;
public class W { [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint f);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr a, int x, int y, int cx, int cy, uint f);
  public struct RECT { public int L, T, R, B; } }
'@
New-Item -ItemType Directory -Force $Out | Out-Null
$exe = 'C:\Development\tools\RadeonDeveloperToolSuite\RadeonGPUProfiler.exe'
$p = Start-Process $exe -ArgumentList "`"$Rgp`"" -PassThru
$root = $null
for ($i = 0; $i -lt 60 -and -not $root; $i++) {
    Start-Sleep -Seconds 1
    $p.Refresh()
    if ($p.MainWindowHandle -ne 0) { $root = [System.Windows.Automation.AutomationElement]::FromHandle($p.MainWindowHandle) }
}
if (-not $root) { throw 'no RGP window' }
[W]::SetWindowPos($p.MainWindowHandle, [IntPtr]::Zero, 0, -3000, 1600, 1100, 0x0040) | Out-Null
Start-Sleep -Seconds 8
function Shot($name) {
    $r = New-Object W+RECT; [W]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
    $bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
    $g = [System.Drawing.Graphics]::FromImage($bmp); $hdc = $g.GetHdc()
    [W]::PrintWindow($p.MainWindowHandle, $hdc, 2) | Out-Null
    $g.ReleaseHdc($hdc); $g.Dispose(); $bmp.Save("$Out\$name.png"); $bmp.Dispose()
}
function Dump($name) {
    $all = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)
    $lines = foreach ($e in $all) { $c = $e.Current; if ($c.Name) { "{0}|{1}" -f $c.ControlType.ProgrammaticName, $c.Name } }
    $lines | Set-Content "$Out\$name.txt"
    Shot $name
}
Dump 'initial'
function SelectTab($nm) {
    $t = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty, $nm))) |
        Where-Object { $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::TabItem } | Select-Object -First 1
    if ($t) { $t.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select(); Start-Sleep -Seconds 4; return $true }
    return $false
}
foreach ($nm in @('OVERVIEW', 'EVENTS', 'Event timing', 'Pipeline state', 'Instruction timing')) {
    if (SelectTab $nm) { Dump ('tab-' + ($nm -replace '[^A-Za-z0-9]', '_')) } else { "missing $nm" | Add-Content "$Out	abs.txt" }
}
# Instruction table is virtualised and exposes no ScrollPattern: focus a row via UIA and post PageDown key messages to the window.
Add-Type @'
using System; using System.Runtime.InteropServices;
public class K { [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l); }
'@
if (SelectTab 'Instruction timing') {
    $rowsEl = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::TreeItem)))
    try { $rowsEl[3].SetFocus() } catch { "setfocus failed: $_" | Add-Content (Join-Path $Out 'tabs.txt') }
    $acc = New-Object System.Collections.Generic.List[string]
    for ($pg = 0; $pg -lt 16; $pg++) {
        foreach ($e in $root.FindAll([System.Windows.Automation.TreeScope]::Descendants,
            (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::TreeItem)))) {
            $acc.Add('TreeItem|' + $e.Current.Name)
        }
        $acc.Add('---PAGE---')
        [K]::PostMessage($p.MainWindowHandle, 0x0100, [IntPtr]0x22, [IntPtr]0x00000001) | Out-Null   # WM_KEYDOWN VK_NEXT
        [K]::PostMessage($p.MainWindowHandle, 0x0101, [IntPtr]0x22, [IntPtr]0xC0000001) | Out-Null  # WM_KEYUP
        Start-Sleep -Milliseconds 900
    }
    $acc | Set-Content (Join-Path $Out 'instr-all.txt')
}
Stop-Process -Id $p.Id -Force
