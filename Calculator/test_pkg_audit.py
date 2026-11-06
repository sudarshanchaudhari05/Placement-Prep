import subprocess
import os
import csv
import re

adb_path = r"C:\Users\sudar\Downloads\OS\platform-tools-latest-windows\platform-tools\adb.exe"
serial = "a9efe51e"
out_dir = r"C:\Users\sudar\Documents\Collage\Placement prep\Calculator\redmi_diagnostics\phase2"

def run_adb(cmd, timeout=15):
    full_cmd = [adb_path, "-s", serial, "shell"] + cmd
    try:
        res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return res.stdout.strip()
    except Exception as e:
        return ""

# Check running processes
ps_out = run_adb(["ps", "-A", "-o", "NAME"])
running_procs = set(line.strip() for line in ps_out.splitlines())

# Check launcher activities
dumpsys_launcher = run_adb(["cmd", "package", "query-activities", "-c", "android.intent.category.LAUNCHER"])

# Parse all packages
pkg_info = {}

# pm list packages -f -i -u
pm_out = run_adb(["pm", "list", "packages", "-f", "-i", "-u"])
for line in pm_out.splitlines():
    m = re.search(r"package:(.*?)=([a-zA-Z0-9_\.]+)\s+installer=(.*)", line)
    if m:
        path = m.group(1).strip()
        pkg = m.group(2).strip()
        installer = m.group(3).strip()
        pkg_info[pkg] = {
            "path": path,
            "installer": installer,
            "is_system": not path.startswith("/data/app/"),
            "is_updated_system": path.startswith("/data/app/") and any(x in installer for x in ["mipicks", "discover", "android.vending", "system"]),
            "is_running": any(p.startswith(pkg) for p in running_procs),
            "has_launcher": (pkg in dumpsys_launcher)
        }

print(f"Total packages discovered: {len(pkg_info)}")
