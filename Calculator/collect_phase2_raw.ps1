$ErrorActionPreference = 'Continue'
$adb = "C:\Users\sudar\Downloads\OS\platform-tools-latest-windows\platform-tools\adb.exe"
$serial = "a9efe51e"
$outDir = "C:\Users\sudar\Documents\Collage\Placement prep\Calculator\redmi_diagnostics\phase2"

if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

function Run-Adb($name, $cmd, $filename) {
    Write-Host "Running: $name..."
    $targetPath = Join-Path $outDir $filename
    try {
        & $adb -s $serial shell "$cmd" > $targetPath
    } catch {
        $_ | Out-File -FilePath $targetPath -Encoding utf8
    }
    Start-Sleep -Milliseconds 250
}

# Task 1: Memory commands
Run-Adb "cat /proc/meminfo" "cat /proc/meminfo" "proc_meminfo.txt"
Run-Adb "dumpsys meminfo" "dumpsys meminfo" "dumpsys_meminfo.txt"
Run-Adb "dumpsys meminfo -a" "dumpsys meminfo -a" "dumpsys_meminfo_a.txt"
Run-Adb "cat /sys/block/zram0/mm_stat" "cat /sys/block/zram0/mm_stat" "zram_mm_stat.txt"
Run-Adb "cat /sys/block/zram0/stat" "cat /sys/block/zram0/stat" "zram_stat.txt"
Run-Adb "cat /proc/pressure/memory" "cat /proc/pressure/memory" "psi_memory.txt"
Run-Adb "cat /proc/pressure/cpu" "cat /proc/pressure/cpu" "psi_cpu.txt"
Run-Adb "cat /proc/pressure/io" "cat /proc/pressure/io" "psi_io.txt"
Run-Adb "dumpsys activity processes" "dumpsys activity processes" "dumpsys_activity_processes.txt"
Run-Adb "ps -A" "ps -A -o USER,PID,PPID,VSZ,RSS,WCHAN,ADDR,S,NAME" "ps_A.txt"
Run-Adb "audio hal smaps" "cat /proc/$(pidof android.hardware.audio.service)/smaps_rollup" "audio_smaps_rollup.txt"
Run-Adb "audio hal status" "cat /proc/$(pidof android.hardware.audio.service)/status" "audio_status.txt"
Run-Adb "dmesg LMK / memory events" "dmesg | grep -iE 'lowmemorykiller|lmkd|oom|killed process' | tail -n 50" "dmesg_lmk.txt"
Run-Adb "dumpsys activity lru" "dumpsys activity lru" "dumpsys_activity_lru.txt"

# Task 2: Storage commands
Run-Adb "df -h" "df -h" "df_h.txt"
Run-Adb "df -k" "df -k" "df_k.txt"
Run-Adb "dumpsys diskstats" "dumpsys diskstats" "dumpsys_diskstats.txt"

# Task 3: Package inventory
Run-Adb "pm list packages system" "pm list packages -f -i -s" "packages_system.txt"
Run-Adb "pm list packages third party" "pm list packages -f -i -3" "packages_third_party.txt"
Run-Adb "pm list packages disabled" "pm list packages -d" "packages_disabled.txt"
Run-Adb "pm list packages enabled" "pm list packages -e" "packages_enabled.txt"

Write-Host "Raw data collection finished successfully!"
