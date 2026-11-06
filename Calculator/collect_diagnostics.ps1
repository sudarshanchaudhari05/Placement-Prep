$ErrorActionPreference = 'Continue'
$adb = "C:\Users\sudar\Downloads\OS\platform-tools-latest-windows\platform-tools\adb.exe"
$outDir = Join-Path $PSScriptRoot "redmi_diagnostics"

if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

Write-Host "Diagnostic Target Directory: $outDir"

Write-Host "1/10: Collecting device properties..."
& $adb shell getprop > (Join-Path $outDir "getprop.txt")

Write-Host "2/10: Collecting storage information..."
& $adb shell df -h /data > (Join-Path $outDir "storage_data.txt")
& $adb shell df -h > (Join-Path $outDir "storage_all.txt")

Write-Host "3/10: Collecting RAM and memory statistics..."
& $adb shell free -h > (Join-Path $outDir "free.txt")
& $adb shell cat /proc/meminfo > (Join-Path $outDir "proc_meminfo.txt")
& $adb shell dumpsys meminfo > (Join-Path $outDir "dumpsys_meminfo.txt")

Write-Host "4/10: Collecting package lists..."
& $adb shell pm list packages -3 -f -i -u > (Join-Path $outDir "packages_third_party.txt")
& $adb shell pm list packages -s -f -i -u > (Join-Path $outDir "packages_system.txt")
& $adb shell pm list packages -d > (Join-Path $outDir "packages_disabled.txt")
& $adb shell pm list packages -e > (Join-Path $outDir "packages_enabled.txt")

Write-Host "5/10: Collecting running processes..."
& $adb shell dumpsys activity processes > (Join-Path $outDir "dumpsys_activity_processes.txt")

Write-Host "6/10: Collecting running services..."
& $adb shell dumpsys activity services > (Join-Path $outDir "dumpsys_activity_services.txt")

Write-Host "7/10: Collecting battery details..."
& $adb shell dumpsys battery > (Join-Path $outDir "dumpsys_battery.txt")

Write-Host "8/10: Collecting appops and background execution permissions..."
& $adb shell dumpsys appops > (Join-Path $outDir "dumpsys_appops.txt")

Write-Host "9/10: Collecting package list with UID/installer..."
& $adb shell pm list packages -U > (Join-Path $outDir "packages_uids.txt")

Write-Host "10/10: Collecting top process snapshot..."
& $adb shell top -b -n 1 -m 20 > (Join-Path $outDir "top_processes.txt")

Write-Host "Diagnostics collection finished successfully!"
