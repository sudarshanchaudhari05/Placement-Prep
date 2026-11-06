import subprocess
import os
import time

adb_path = r"C:\Users\sudar\Downloads\OS\platform-tools-latest-windows\platform-tools\adb.exe"
serial = "a9efe51e"
out_dir = r"C:\Users\sudar\Documents\Collage\Placement prep\Calculator\redmi_diagnostics\phase2"

os.makedirs(out_dir, exist_ok=True)

def run_adb(cmd, timeout=30):
    full_cmd = [adb_path, "-s", serial, "shell"] + cmd
    try:
        res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT"
    except Exception as e:
        return "", str(e)

commands = [
    # Task 1
    ("proc_meminfo.txt", ["cat", "/proc/meminfo"]),
    ("dumpsys_meminfo.txt", ["dumpsys", "meminfo"]),
    ("zram_mm_stat.txt", ["cat", "/sys/block/zram0/mm_stat"]),
    ("zram_stat.txt", ["cat", "/sys/block/zram0/stat"]),
    ("psi_memory.txt", ["cat", "/proc/pressure/memory"]),
    ("psi_cpu.txt", ["cat", "/proc/pressure/cpu"]),
    ("psi_io.txt", ["cat", "/proc/pressure/io"]),
    ("dumpsys_activity_processes.txt", ["dumpsys", "activity", "processes"]),
    ("ps_A.txt", ["ps", "-A", "-o", "USER,PID,PPID,VSZ,RSS,WCHAN,ADDR,S,NAME"]),
    ("dumpsys_activity_lru.txt", ["dumpsys", "activity", "lru"]),
    # Task 2
    ("df_h.txt", ["df", "-h"]),
    ("df_k.txt", ["df", "-k"]),
    ("dumpsys_diskstats.txt", ["dumpsys", "diskstats"]),
    # Task 3
    ("packages_system.txt", ["pm", "list", "packages", "-f", "-i", "-s"]),
    ("packages_third_party.txt", ["pm", "list", "packages", "-f", "-i", "-3"]),
    ("packages_disabled.txt", ["pm", "list", "packages", "-d"]),
    ("packages_enabled.txt", ["pm", "list", "packages", "-e"]),
]

for filename, cmd in commands:
    print(f"Collecting: {filename} ({' '.join(cmd)})...")
    out, err = run_adb(cmd, timeout=25)
    filepath = os.path.join(out_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(out)
        if err:
            f.write(f"\n--- STDERR / ERROR ---\n{err}")
    time.sleep(0.3)

print("Checking audio HAL pid and memory details...")
pid_out, _ = run_adb(["pidof", "android.hardware.audio.service"])
audio_pid = pid_out.strip()
if audio_pid:
    print(f"Audio HAL PID: {audio_pid}")
    smaps_out, _ = run_adb(["cat", f"/proc/{audio_pid}/smaps_rollup"])
    status_out, _ = run_adb(["cat", f"/proc/{audio_pid}/status"])
    mem_out, _ = run_adb(["dumpsys", "meminfo", audio_pid])
    with open(os.path.join(out_dir, "audio_hal_details.txt"), "w", encoding="utf-8") as f:
        f.write(f"PID: {audio_pid}\n")
        f.write("=== /proc/<pid>/smaps_rollup ===\n")
        f.write(smaps_out)
        f.write("\n=== /proc/<pid>/status ===\n")
        f.write(status_out)
        f.write("\n=== dumpsys meminfo <pid> ===\n")
        f.write(mem_out)
else:
    with open(os.path.join(out_dir, "audio_hal_details.txt"), "w", encoding="utf-8") as f:
        f.write("android.hardware.audio.service PID not found via pidof\n")

print("Now testing dumpsys meminfo -a with generous timeout (45s)...")
out_a, err_a = run_adb(["dumpsys", "meminfo", "-a"], timeout=45)
with open(os.path.join(out_dir, "dumpsys_meminfo_a.txt"), "w", encoding="utf-8") as f:
    f.write(out_a)
    if err_a:
        f.write(f"\n--- STDERR / ERROR ---\n{err_a}")

print("Phase 2 collection complete.")
