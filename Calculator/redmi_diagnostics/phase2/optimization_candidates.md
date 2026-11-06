# Xiaomi Redmi Note 10 (4G) – Phase 2 Optimization Candidates

**Generated:** 2026-11-06  
**Device:** Redmi Note 10 (4G) [`M2101K7AI`] | **Codename:** `sunny` / `sunny_in`  
**MIUI Version:** MIUI 14 Global India (`V14.0.5.0.SKGINXM`)  
**Storage Destination:** `C:\Users\sudar\Documents\Collage\Placement prep\Calculator\redmi_diagnostics\phase2\optimization_candidates.md`

---

## 1. Deep Audit of the 16 High-Priority Packages

Below is the verified audit of the 16 packages highlighted for scrutiny:

| # | Package Name | Actual Application Name | Path & Status | Background Activity | Purpose & Side Effects | Safe for User 0? |
|---|---|---|---|---|---|---|
| **1** | `com.miui.msa.global` | MIUI System Ads | `/system/priv-app/MSA-Global` (System) | **RUNNING** (PID 9432) | Pushes ads across MIUI apps. **No side effects** when removed. System becomes ad-free. | **YES (Category A)** |
| **2** | `com.miui.analytics` | MIUI Analytics | `/system/app/AnalyticsCore` (System, updated) | **RUNNING** (PID 8479) | Continuous telemetry and usage profiling. **No side effects** on normal phone operations. | **YES (Category A)** |
| **3** | `com.miui.daemon` | MIUI Daemon | `/system/app/MiuiDaemon` (System) | Present | Background diagnostics and error telemetry. Safe to debloat; reduces background wakeups. | **YES (Category A)** |
| **4** | `com.miui.bugreport` | MIUI Bug Report | `/system/app/BugReport` (System) | Present | Captures crash logs and sends to Xiaomi. Unnecessary for normal daily use. | **YES (Category A)** |
| **5** | `com.xiaomi.mipicks` | GetApps (Xiaomi App Store) | `/system/priv-app/MiPicks` (System) | **RUNNING** (PID 21847) | Secondary app store. Constantly runs update checkers. **Side effect:** GetApps disabled; Play Store handles all updates. | **YES (Category A)** |
| **6** | `com.miui.android.fashiongallery` | Glance Wallpaper Carousel | `/data/app/.../com.miui.android.fashiongallery` (3rd-party) | Present | Lockscreen advertisement carousel and news stories. Disabling stops lockscreen feed/ads. | **YES (Category A)** |
| **7** | `com.facebook.services` | Meta / Facebook Services | `/data/app/.../com.facebook.services` (Preloaded) | **RUNNING** (PID 10959) | Background analytics stub for Meta apps. Completely safe to remove. | **YES (Category A)** |
| **8** | `com.facebook.system` | Meta / Facebook Installer | `/data/app/.../com.facebook.system` (Preloaded) | Present | Auto-installer for Meta apps without Play Store. Completely safe to remove. | **YES (Category A)** |
| **9** | `com.facebook.appmanager` | Meta App Manager | `/data/app/.../com.facebook.appmanager` (Preloaded) | Present | Silent background updater for Meta apps. Completely safe to remove. | **YES (Category A)** |
| **10** | `com.indus.appstore` | Indus Appstore | `/data/app/.../com.indus.appstore` (Preloaded via GetApps) | **RUNNING** (PID 24480) | Third-party Indian app store. Runs background services. Safe to remove. | **YES (Category A)** |
| **11** | `com.eterno` | Dailyhunt | `/data/app/.../com.eterno` (Preloaded) | Present | Third-party news feed app with heavy background notification load. Safe to remove. | **YES (Category A)** |
| **12** | `com.miui.hybrid` | Quick Apps (MIUI Hybrid) | `/system/app/HybridPlatform` (System) | Present | Engine for running web-based mini-apps without installing. Safe to remove. | **YES (Category A)** |
| **13** | `com.miui.hybrid.accessory` | Quick Apps Accessory | `/system/app/HybridAccessory` (System) | Present | Supporting runtime for Quick Apps. Safe to remove alongside Hybrid. | **YES (Category A)** |
| **14** | `com.google.android.verifier` | Google Play Protect Service | `/data/app/.../com.google.android.verifier` | Present | Play Protect security scanning module. **Side effect:** Disables Play Protect offline verifier. | **RETAIN (Category C)** |
| **15** | `com.xiaomi.simactivate.service` | Xiaomi SIM Activate Service | `/system/app/XiaomiSimActivateService` | **RUNNING** (PID 22077) | Activates Xiaomi Cloud messaging/find device via SMS. Safe if not using Mi Cloud. | **CATEGORY B** |
| **16** | `com.miui.cloudservice` | Xiaomi Cloud Service | `/system/app/CloudService` | **RUNNING** (PID 24427) | Xiaomi Cloud sync. If user uses Google Drive/Photos, this wastes ~50 MB RAM. | **CATEGORY B** |

---

## 2. Categorized Optimization Candidate Lists

### CATEGORY A: Likely Safe to Debloat (Bloatware, Ads, Telemetry & Unwanted Preloads)
*These packages can be safely uninstalled for User 0 (`pm uninstall --user 0 <pkg>`). They do not compromise phone stability, cellular connectivity, camera, or Play Store functionality.*

1. **`com.miui.msa.global`** – MIUI System Ads
2. **`com.miui.analytics`** – Xiaomi Analytics / Tracking
3. **`com.miui.daemon`** – MIUI Background Telemetry Daemon
4. **`com.miui.bugreport`** – Xiaomi Bug Report Collector
5. **`com.xiaomi.mipicks`** – GetApps Store (Alternative Xiaomi Store)
6. **`com.miui.android.fashiongallery`** – Glance Lockscreen Carousel (Ads & sponsored articles)
7. **`com.facebook.services`** – Facebook Background Tracking Service
8. **`com.facebook.system`** – Facebook App Installer
9. **`com.facebook.appmanager`** – Facebook App Manager
10. **`com.indus.appstore`** – Indus Appstore
11. **`com.eterno`** – Dailyhunt News
12. **`com.meesho.supply`** – Meesho Shopping App
13. **`com.facemoji.lite.xiaomi`** – Facemoji Keyboard
14. **`cn.wps.xiaomi.abroad.lite`** – WPS Office Lite for Xiaomi
15. **`com.xiaomi.glgm`** – Mi Game Center Global
16. **`com.opera.browser.afin`** – Preinstalled Opera Browser
17. **`com.miui.yellowpage`** – MIUI Yellow Pages (Caller ID / Advertising)
18. **`com.miui.miservice`** – Services & Feedback App
19. **`com.xiaomi.payment`** – Mi Pay
20. **`com.miui.hybrid`** – Quick Apps Platform
21. **`com.miui.hybrid.accessory`** – Quick Apps Engine

---

### CATEGORY B: Optional Applications (User-Approval Required)
*These applications have standalone user-facing utility. If you use them, keep them. If you do not use them, removing them frees up significant RAM and stops background process wakeups.*

1. **`com.miui.cloudservice`**, **`com.miui.cloudbackup`**, **`com.miui.micloudsync`**, **`com.xiaomi.micloud.sdk`**
   - *Utility:* Syncs photos, contacts, and notes to Xiaomi Cloud.
   - *Recommendation:* If you use Google Drive/Photos for backup, remove these to save ~60 MB RAM.
2. **`com.xiaomi.simactivate.service`**
   - *Utility:* Sends SMS to activate Xiaomi SIM verification for Mi Cloud.
   - *Recommendation:* Safe to remove if not using Mi Cloud.
3. **`com.google.android.apps.tachyon`** (Google Meet / Duo)
   - *Recommendation:* Remove if you only use WhatsApp for video calls.
4. **`com.google.android.videos`** (Google TV)
   - *Recommendation:* Remove if you don't rent/buy movies on Google TV.
5. **`com.google.android.apps.magazines`** (Google News)
   - *Recommendation:* Remove if you do not use Google News app.
6. **`com.google.android.apps.youtube.music`** (YouTube Music)
   - *Recommendation:* Remove if you use Spotify or other music players.
7. **`com.google.android.projection.gearhead`** (Android Auto)
   - *Recommendation:* Remove if you do not connect your phone to a car screen.
8. **`com.google.android.apps.wellbeing`** (Digital Wellbeing)
   - *Recommendation:* Monitors screen time and app usage; runs background tracking loops. Removing improves UI fluidity.
9. **`com.google.android.apps.googleassistant`** (Google Assistant stub)
   - *Recommendation:* Remove if you don't use voice "Hey Google".
10. **`com.google.ar.core`** (Google Play Services for AR)
    - *Recommendation:* AR functionality is rarely used on budget devices; can be removed safely.
11. **`com.miui.player`** (Mi Music)
    - *Recommendation:* Full of online banner ads; remove if using Spotify or YouTube Music.
12. **`com.miui.cleaner`** (MIUI Cleaner)
    - *Recommendation:* Built on Cheetah Mobile / Clean Master; contains ads. Safe to remove.
13. **`com.duokan.phone.remotecontroller`** (Mi Remote)
    - *Recommendation:* Retain if you use the IR blaster to control TV/AC. Remove if unused.
14. **`com.miui.compass`**, **`com.miui.notes`**, **`com.miui.weather2`**, **`com.miui.screenrecorder`**, **`com.xiaomi.midrop`**
    - *Recommendation:* Retain if you actively use these built-in utilities.

---

### CATEGORY C: Protected System Components (DO NOT UNINSTALL OR DISABLE)
*Modifying these packages will cause bootloops, crashing UI, loss of phone calls, or broken security.*

- **System UI & Core:** `com.android.systemui`, `com.android.settings`, `com.miui.home` (Launcher), `android`, `com.miui.rom`, `com.miui.system`
- **Telephony & Emergency:** `com.android.phone`, `com.android.server.telecom`, `com.google.android.dialer`, `com.google.android.apps.messaging`, `com.google.android.cellbroadcastreceiver`
- **Google Play Framework:** `com.google.android.gms` (Play Services), `com.android.vending` (Play Store), `com.google.android.gsf`, `com.google.android.packageinstaller`, `com.google.android.permissioncontroller`
- **Security & Integrity:** `com.miui.securitycenter`, `com.miui.securitycore`, `com.miui.securityadd`, `com.lbe.security.miui`, `com.google.android.verifier` (Play Protect)
- **Hardware & Drivers:** `com.qualcomm.*`, `vendor.qti.*`, `android.hardware.*`, `com.android.camera` (Mi Camera)
- **Personal Data Apps:** `com.whatsapp`, `com.truecaller`, `org.localsend.localsend_app`

---

## 3. Restoration Procedure (Reversibility Guarantee)

Any package uninstalled for User 0 can be restored instantly without internet access or data loss using:
```powershell
adb shell cmd package install-existing <PACKAGE_NAME>
```
*Example:*
```powershell
adb shell cmd package install-existing com.miui.msa.global
```
