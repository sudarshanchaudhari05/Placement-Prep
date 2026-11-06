import subprocess
import os
import re

adb_path = r"C:\Users\sudar\Downloads\OS\platform-tools-latest-windows\platform-tools\adb.exe"
serial = "a9efe51e"

# We can query launcher activities to get application labels for many packages
cmd = [adb_path, "-s", serial, "shell", "cmd", "package", "query-activities", "-c", "android.intent.category.LAUNCHER"]
res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

labels = {}
# format in query-activities:
# Activity #X:
#   Priority=...
#   Action: ...
#   Category: ...
#   AutoVerify=...
#   package=com.foo
#   labelRes=... or nonLocalizedLabel=...
for block in res.stdout.split("Activity #"):
    pkg_m = re.search(r"package=([a-zA-Z0-9_\.]+)", block)
    lbl_m = re.search(r"nonLocalizedLabel=(.+)", block)
    if pkg_m and lbl_m:
        labels[pkg_m.group(1)] = lbl_m.group(1).strip()

print(f"Discovered {len(labels)} launcher labels:")
for k, v in list(labels.items())[:15]:
    print(f"  {k}: {v}")
