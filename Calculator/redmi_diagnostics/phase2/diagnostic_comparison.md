# Xiaomi Redmi Note 10 (4G) – Diagnostic Comparison & Phase 2 Findings

**Date:** 2026-11-06  
**Target:** Redmi Note 10 (4G) [`M2101K7AI`] | MIUI 14 Global (`V14.0.5.0.SKGINXM`)  
**Host Target Path:** `C:\Users\sudar\Documents\Collage\Placement prep\Calculator\redmi_diagnostics\phase2\diagnostic_comparison.md`

---

## 1. Overview: Phase 1 Hypotheses vs Phase 2 In-Depth Verification

| Area | Phase 1 Initial Observation | Phase 2 In-Depth Verification | Resolution / Technical Reality |
|---|---|---|---|
| **Audio HAL Memory** | "Consumes ~495 MB PSS (Potential leak/hog)" | Detailed inspection of PID 975 (`android.hardware.audio.service`) | **Corrected:** Physical RAM (RSS) is only **996 kB** (~1 MB). The 495 MB is `SwapPss Dirty` residing in ZRAM. It is not an active memory hog. |
| **RAM Pressure** | "~107 MB Free physical RAM; ZRAM 80% full" | Event logs audited (`am_kill`, `AutoLockOffCleanByPriority`) | **Confirmed:** Severe active memory pressure. LMK and MIUI cleaner are killing background tasks (Gmail, Dialer, Maps, YouTube) every few seconds. |
| **Storage Capacity** | "20 GB used / 26 GB available on `/data`" | `dumpsys diskstats` vs partition mounts verified | **Confirmed:** 25.5 GiB free. Difference between Settings (~36 GB used) and `/data` is due to decimal-to-binary conversion and ~15 GB dynamic OS partitions. |
| **Bloatware Activity** | "Multiple daemons detected" | Full process tree & UID analysis completed | **Confirmed:** `msa.global`, `analytics`, `mipicks`, `indus.appstore`, `facebook.services`, and `simactivate.service` run persistent background processes. |

---

## 2. In-Depth Audio HAL Finding (PID 975)

In Phase 1, `dumpsys meminfo` reported:
```
495,246K: android.hardware.audio.service (pid 975)
```
In Phase 2, a targeted process dump revealed:
```
TOTAL PSS:      495,334 kB
TOTAL RSS:          996 kB
TOTAL SWAP PSS: 494,544 kB
Private Dirty:      624 kB
Private Clean:      164 kB
Native Heap RSS:    256 kB
```
**Conclusion:**
`android.hardware.audio.service` allocated virtual buffer space that was pushed almost entirely into compressed ZRAM (494.5 MB in SwapPss). In physical RAM, it takes less than 1 MB. It is NOT actively draining physical RAM in foreground use, but its swap footprint adds to ZRAM decompression overhead.

---

## 3. Conclusive Proof of Active Memory Churn

The user requested proof of whether memory pressure is actively causing process termination.
Android's event log (`logcat -b events -s am_kill am_proc_died`) recorded:
1. **LowMemoryKiller (LMK) `empty #13` kills:**
   - As soon as free memory dips below watermark limits, Android drops cached apps.
   - We observed: `com.google.android.contacts`, `com.google.android.dialer`, `com.google.android.apps.messaging`, and `com.google.android.apps.maps` being terminated within seconds of being backgrounded.
2. **MIUI Memory Cleaner (`AutoLockOffCleanByPriority`):**
   - When the display turns off or locks, MIUI systematically executes priority-based task killing.
   - We observed: `com.google.android.gm` (Gmail), `com.google.android.youtube`, `com.android.vending`, and `com.google.android.apps.nbu.paisa.user` (Google Pay) being killed.

**Verdict:**
This confirms that 4 GB of RAM on MIUI 14 is heavily overloaded by background bloatware and daemon processes, forcing the system into continuous, aggressive process churn.

---

## 4. Phase 2 Deliverables Summary

All five required deliverables have been created and saved in:
`C:\Users\sudar\Documents\Collage\Placement prep\Calculator\redmi_diagnostics\phase2\`

1. `memory_verification.txt`: Complete `/proc/meminfo`, ZRAM calculations, Audio HAL breakdown, and LMK kill logs.
2. `storage_verification.txt`: Full `dumpsys diskstats` and detailed architectural explanation of Settings vs `/data`.
3. `package_inventory.csv`: Complete 333-package audit spreadsheet with paths, types, installers, launcher status, background activity, and safety classifications.
4. `optimization_candidates.md`: Detailed audit of the 16 special-attention packages, complete Category A, B, and C lists, potential side effects, and instant restoration commands.
5. `diagnostic_comparison.md`: Technical comparison, corrections, and verified findings.

---

**DEVICE STATE GUARANTEE:**
Zero destructive actions taken. No packages were disabled, uninstalled, or modified.
Phone remains in untouched read-only state.
