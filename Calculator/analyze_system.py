import os
import re

out_dir = r"redmi_diagnostics"

def read_file(fname):
    p = os.path.join(out_dir, fname)
    if not os.path.exists(p):
        return ""
    for enc in ["utf-16", "utf-16le", "utf-8", "latin1"]:
        try:
            with open(p, "r", encoding=enc) as f:
                content = f.read()
                if content and ("\x00" not in content[:50]):
                    return content
        except Exception:
            continue
    return ""

meminfo = read_file("dumpsys_meminfo.txt")

print("=== TOP 25 RAM CONSUMING PROCESSES (dumpsys meminfo) ===")
in_pss = False
pss_lines = []
for line in meminfo.splitlines():
    if "Total PSS by process:" in line:
        in_pss = True
        continue
    if in_pss:
        if line.strip().startswith("Total PSS by OOM"):
            break
        if line.strip():
            pss_lines.append(line.strip())

for l in pss_lines[:25]:
    print(l)

print("\n=== SYSTEM PACKAGES - XIAOMI SPECIFIC ===")
sys_pkgs = [l.strip() for l in read_file("packages_system.txt").splitlines() if l.strip()]
xiaomi_pkgs = []
google_pkgs = []
android_pkgs = []
other_sys_pkgs = []

for line in sys_pkgs:
    pkg = line.split("=")[-1].strip() if "=" in line else line.replace("package:", "").strip()
    if any(k in pkg for k in ["xiaomi", "miui", "duokan", "mipicks"]):
        xiaomi_pkgs.append(pkg)
    elif "google" in pkg or "android.apps" in pkg:
        google_pkgs.append(pkg)
    elif "android" in pkg or "qti" in pkg or "qualcomm" in pkg:
        android_pkgs.append(pkg)
    else:
        other_sys_pkgs.append(pkg)

print(f"Total Xiaomi/MIUI System Packages: {len(xiaomi_pkgs)}")
print(f"Total Google System Packages: {len(google_pkgs)}")
print(f"Total Android/SoC/Core Packages: {len(android_pkgs)}")
print(f"Other Packages: {len(other_sys_pkgs)}")

print("\nNotable Xiaomi bloatware/analytics candidates present:")
candidates = [
    "com.miui.msa.global", "com.miui.analytics", "com.xiaomi.mipicks",
    "com.miui.daemon", "com.miui.hybrid", "com.miui.hybrid.accessory",
    "com.miui.bugreport", "com.miui.miservice", "com.xiaomi.midrop",
    "com.miui.videoplayer", "com.miui.player", "com.miui.yellowpage",
    "com.miui.micloudsync", "com.miui.cloudservice", "com.miui.cloudservice.sysbase",
    "com.miui.cloudbackup", "com.xiaomi.payment", "com.xiaomi.simactivate.service",
    "com.mi.health", "com.miui.compass", "com.miui.cleanmaster",
    "com.android.browser", "com.miui.notes", "com.miui.weather2"
]
for c in candidates:
    found = any(c in p for p in sys_pkgs) or any(c in p for p in read_file("packages_third_party.txt").splitlines())
    print(f"  {c}: {'PRESENT' if found else 'NOT FOUND'}")

print("\n=== TOP PROCESSES (top) ===")
top_content = read_file("top_processes.txt")
for line in top_content.splitlines()[:20]:
    print(line)
