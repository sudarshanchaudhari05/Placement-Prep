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

ps_content = read_file("ps_ef.txt")
app_processes = []
for line in ps_content.splitlines():
    parts = line.split()
    if len(parts) >= 8:
        uid = parts[0]
        pid = parts[1]
        cmd = parts[7]
        if uid.startswith("u0_"):
            app_processes.append((uid, pid, cmd))

print(f"Total User App Processes currently running: {len(app_processes)}")
print("\nUser App Processes:")
for u, pid, cmd in sorted(app_processes, key=lambda x: x[2]):
    print(f"PID {pid:>5} | UID {u:<8} | {cmd}")
