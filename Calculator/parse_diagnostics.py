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

getprop = read_file("getprop.txt")
props = {}
for line in getprop.splitlines():
    m = re.match(r"\[(.*?)\]: \[(.*?)\]", line.strip())
    if m:
        props[m.group(1)] = m.group(2)

print("=== DEVICE SPECIFICATIONS ===")
keys = [
    "ro.product.model", "ro.product.device", "ro.product.brand", "ro.product.marketname",
    "ro.build.version.release", "ro.build.version.security_patch", "ro.miui.ui.version.name",
    "ro.miui.ui.version.code", "ro.build.version.incremental", "ro.build.fingerprint",
    "ro.soc.model", "ro.board.platform", "ro.product.cpu.abi", "ro.product.manufacturer"
]
for k in keys:
    print(f"{k}: {props.get(k, 'N/A')}")

print("\n=== STORAGE (/data) ===")
print(read_file("storage_data.txt").strip())

print("\n=== STORAGE (All Partitions Summary) ===")
for line in read_file("storage_all.txt").splitlines()[:15]:
    print(line)

print("\n=== RAM & MEMORY ===")
print("--- free -h ---")
print(read_file("free.txt").strip())

print("\n--- proc/meminfo highlights ---")
for line in read_file("proc_meminfo.txt").splitlines():
    if any(line.startswith(x) for x in ["MemTotal:", "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "SwapTotal:", "SwapFree:", "Zram:"]):
        print(line)

print("\n--- dumpsys meminfo highlights ---")
meminfo = read_file("dumpsys_meminfo.txt")
for line in meminfo.splitlines():
    if any(term in line.lower() for term in ["total ram:", "free ram:", "used ram:", "lost ram:", "tuning:", "zram:"]):
        print(line)

print("\n=== THIRD-PARTY PACKAGES ===")
third_party = [l.strip() for l in read_file("packages_third_party.txt").splitlines() if l.strip()]
print(f"Total Third-Party Packages: {len(third_party)}")
for p in third_party:
    print(p)

print("\n=== DISABLED PACKAGES ===")
disabled = [l.strip() for l in read_file("packages_disabled.txt").splitlines() if l.strip()]
print(f"Total Disabled Packages: {len(disabled)}")
for p in disabled:
    print(p)

print("\n=== SYSTEM PACKAGES SUMMARY ===")
sys_pkgs = [l.strip() for l in read_file("packages_system.txt").splitlines() if l.strip()]
print(f"Total System Packages: {len(sys_pkgs)}")

print("\n=== BATTERY STATE ===")
print(read_file("dumpsys_battery.txt").strip())
