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

def parse_pkg_line(line):
    # e.g.: package:/system/app/...=com.foo installer=bar
    m = re.search(r"=([a-zA-Z0-9_\.]+)\s+installer=(.*)", line)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"package:(?:.*=)?([a-zA-Z0-9_\.]+)", line)
    if m:
        return m.group(1), "unknown"
    return line.strip(), "unknown"

third_party_raw = [l.strip() for l in read_file("packages_third_party.txt").splitlines() if l.strip()]
system_raw = [l.strip() for l in read_file("packages_system.txt").splitlines() if l.strip()]
disabled_raw = [l.strip() for l in read_file("packages_disabled.txt").splitlines() if l.strip()]

third_party = [parse_pkg_line(l) for l in third_party_raw]
system_pkgs = [parse_pkg_line(l) for l in system_raw]
disabled_pkgs = [parse_pkg_line(l)[0] for l in disabled_raw]

print(f"Total Third-Party: {len(third_party)}")
print(f"Total System: {len(system_pkgs)}")
print(f"Total Disabled: {len(disabled_pkgs)}")

all_pkgs = {}
for pkg, inst in third_party:
    all_pkgs[pkg] = {"type": "3rd-party", "installer": inst}
for pkg, inst in system_pkgs:
    all_pkgs[pkg] = {"type": "system", "installer": inst}

print("\n=== COMPLETE THIRD-PARTY PACKAGES LIST ===")
for p, inst in third_party:
    print(f"- {p} (installer: {inst})")

print("\n=== XIAOMI PACKAGES IN SYSTEM ===")
xiaomi_sys = [p for p in all_pkgs if any(x in p for x in ["miui", "xiaomi", "duokan", "milink", "mipicks"])]
print(f"Count: {len(xiaomi_sys)}")
for p in sorted(xiaomi_sys):
    print(f"- {p} [{all_pkgs[p]['type']}] (installer: {all_pkgs[p]['installer']})")

print("\n=== GOOGLE PACKAGES ===")
google_all = [p for p in all_pkgs if "google" in p or "android.apps" in p]
print(f"Count: {len(google_all)}")
for p in sorted(google_all):
    print(f"- {p} [{all_pkgs[p]['type']}]")
