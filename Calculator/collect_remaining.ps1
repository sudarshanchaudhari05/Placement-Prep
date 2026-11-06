$ErrorActionPreference = 'Continue'
$adb = "C:\Users\sudar\Downloads\OS\platform-tools-latest-windows\platform-tools\adb.exe"
$serial = "a9efe51e"
$outDir = Join-Path $PSScriptRoot "redmi_diagnostics"

function Run-AdbCmd($name, $cmd, $targetFile) {
    Write-Host "Collecting $name..."
    & $adb -s $serial shell $cmd > (Join-Path $outDir $targetFile)
    Start-Sleep -Milliseconds 400
}

Run-AdbCmd "Battery details" "dumpsys battery" "dumpsys_battery.txt"
Run-AdbCmd "Packages with UIDs" "pm list packages -U" "packages_uids.txt"
Run-AdbCmd "Top CPU snapshot" "top -b -n 1 -m 25" "top_processes.txt"
Run-AdbCmd "Process list" "ps -ef" "ps_ef.txt"
Run-AdbCmd "Activity processes" "dumpsys activity processes" "dumpsys_activity_processes.txt"
Run-AdbCmd "Battery stats summary" "dumpsys batterystats --daily" "dumpsys_batterystats.txt"

Write-Host "Supplementary collection finished!"
