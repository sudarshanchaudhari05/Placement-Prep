# Xiaomi Redmi Note 10 (4G) – Phase 1 Diagnostic Report

**Report Generated:** 2026-11-06  
**Target Device Serial:** `a9efe51e`  
**Diagnostics Directory on Host:** `c:\Users\sudar\Documents\Collage\Placement prep\Calculator\redmi_diagnostics`  

---

## 1. Device Hardware & Build Identification

| Parameter | Value | Details |
|---|---|---|
| **Device Model** | `M2101K7AI` | Redmi Note 10 (India Market) |
| **Codename** | `sunny` / `sunny_in` | Board: `sm6150` |
| **SoC / Chipset** | Qualcomm Snapdragon 678 (SM6150) | Octa-core Kryo 460 (up to 2.2 GHz), Adreno 612 GPU |
| **Architecture** | `arm64-v8a` (64-bit) | 64-bit kernel & userspace |
| **Android Version** | **Android 12 (API 31)** | Build: `SKQ1.210908.001` |
| **MIUI Version** | **MIUI 14 Global** | Incremental: `V14.0.5.0.SKGINXM` |
| **Security Patch** | `2023-10-01` | Final stable security update baseline |
| **Bootloader / Root** | Locked / Unrooted | Enforcing SELinux |

---

## 2. Storage Partition Breakdown

- **Total Physical Flash:** 64 GB (UFS 2.2)
- **Userdata Partition (`/data`):**
  - **Total Allocated Size:** 45.0 GB
  - **Space Used:** 20.0 GB (44% capacity utilized)
  - **Available / Free Space:** **26.0 GB**
- **Other Mounted Partitions:**
  - System (`/`): 2.9 GB (Read-only)
  - Product (`/product`): 1.6 GB (Read-only)
  - Vendor (`/vendor`): 1.1 GB (Read-only)
  - Cust (`/cust`): 976 MB (187 MB used, 763 MB free)
- **Diagnostic Finding:** Storage capacity is healthy with 26 GB free. The primary cause of system sluggishness is **not** storage exhaustion, but intense RAM pressure and background CPU cycles.

---

## 3. RAM & Memory Architecture Analysis

- **Physical RAM:** **3.5 GB (Nominal 4.0 GB LPDDR4x)**
- **Current RAM Allocation Snapshot:**
  - **Total System RAM:** 3,677,132 kB (~3.59 GB)
  - **Used RAM:** **3,613,115 kB (~3.52 GB)**
  - **Free Physical RAM:** **107 MB (Critically Low - 2.9% free buffer!)**
  - **Available (Cached + Free):** ~891 MB
- **Swap / ZRAM Configuration:**
  - **Total Swap / ZRAM:** 2,306,044 kB (~2.2 GB)
  - **Used Swap:** **1,846,732 kB (~1.8 GB - 80% full!)**
  - **ZRAM Physical Compression Overhead:** 395,764 kB (~386 MB physical RAM dedicated just to hold compressed swapped pages)
- **Impact on System Performance:**
  Because the 4 GB RAM is virtually saturated, Android's Low Memory Killer (LMK) and ZRAM compression engine run constantly in the background. Every time you switch apps or open a new screen, the CPU must compress/decompress pages and kill cached processes, causing micro-stutters, frame drops, and severe app relaunch delays.

---

## 4. Top Memory Consuming Processes

From `dumpsys meminfo` (PSS breakdown):

1. **`android.hardware.audio.service`**: ~495 MB *(Extremely high for an audio HAL)*
2. **`system` (System Server)**: ~300 MB
3. **`com.google.android.gms` (Google Play Services)**: ~182 MB
4. **`com.android.systemui` (MIUI System UI)**: ~173 MB
5. **`surfaceflinger` (Display Compositor)**: ~130 MB
6. **`com.miui.face` (MIUI Face Recognition Service)**: ~118 MB
7. **`com.android.vending` (Google Play Store)**: ~115 MB (active) + 72 MB (background)
8. **`com.whatsapp`**: ~110 MB
9. **`com.google.android.tts`**: ~96 MB
10. **`com.google.android.inputmethod.latin` (Gboard)**: ~94 MB
11. **`com.google.android.apps.photos`**: ~88 MB
12. **`com.google.android.googlequicksearchbox:googleapp`**: ~84 MB
13. **`com.miui.home` (MIUI System Launcher)**: ~83 MB
14. **`com.android.phone` (Telephony stack)**: ~77 MB
15. **`com.miui.miwallpaper` (Wallpaper service)**: ~63 MB
16. **`com.miui.aod` (Always-On Display daemon)**: ~55 MB
17. **`com.miui.securitycenter.remote`**: ~53 MB
18. **`com.google.android.projection.gearhead:car` (Android Auto)**: ~50 MB

---

## 5. Active Bloatware & Telemetry Running in Background

Diagnostics revealed multiple non-essential packages actively loaded in memory:

1. **`com.miui.msa.global` (MIUI System Ads)**: Actively fetching advertisements and push recommendations.
2. **`com.miui.analytics`**: Collecting behavioral and telemetry data continuously.
3. **`com.xiaomi.mipicks` (GetApps Store)**: Xiaomi alternate app store running background update checkers.
4. **`com.indus.appstore`**: Indus App Store running in the background.
5. **`com.facebook.services` / `com.facebook.system` / `com.facebook.appmanager`**: Embedded Meta/Facebook analytics and background tracking stubs.
6. **`com.miui.android.fashiongallery` (Glance / Wallpaper Carousel)**: Pushes lockscreen stories, sponsored articles, and media feeds.
7. **`com.xiaomi.simactivate.service`**: Repeatedly checks SIM activation status for Xiaomi Cloud.
8. **`com.miui.cloudservice`**: Background syncing for Mi Cloud (unnecessary if you use Google Drive/Photos).
9. **`com.google.android.projection.gearhead:car`**: Android Auto car service running even when disconnected.
10. **`com.eterno` (Dailyhunt)**: News feed app with high notification and background refresh frequency.
11. **`com.facemoji.lite.xiaomi`**: Preinstalled third-party keyboard telemetry/service.

---

## 6. Installed Package Inventory Summary

- **Total Installed Packages:** 333
  - **Third-Party Packages:** 22
  - **System Packages:** 311
  - **Currently Disabled Packages:** 2 (`com.android.nfc` [hardware not present on Indian sunny], `com.google.android.ims`)

### Complete List of Third-Party Packages (22):
1. `com.truecaller` (Truecaller)
2. `cn.wps.xiaomi.abroad.lite` (WPS Office Lite for Xiaomi)
3. `com.whatsapp` (WhatsApp)
4. `com.google.android.verifier` (Play Protect Verifier)
5. `com.google.ar.core` (AR Core)
6. `org.localsend.localsend_app` (LocalSend)
7. `com.google.android.apps.tachyon` (Google Meet)
8. `com.miui.android.fashiongallery` (Glance Wallpaper Carousel)
9. `com.duokan.phone.remotecontroller` (Mi Remote)
10. `com.google.android.apps.nbu.files` (Files by Google)
11. `com.google.android.apps.docs` (Google Drive)
12. `com.indus.appstore` (Indus Appstore)
13. `com.google.android.contactkeys` (Google Contact Keys)
14. `com.xiaomi.glgm` (Mi Game Center Global)
15. `com.google.android.videos` (Google TV)
16. `com.google.android.apps.photos` (Google Photos)
17. `com.eterno` (Dailyhunt)
18. `com.miui.mediaeditor` (MIUI Gallery Editor)
19. `com.google.android.apps.youtube.music` (YouTube Music)
20. `com.google.android.apps.magazines` (Google News)
21. `com.facemoji.lite.xiaomi` (Facemoji Keyboard for Xiaomi)
22. `com.meesho.supply` (Meesho)

### Notable Preinstalled Bloatware Found in System Packages:
- `com.facebook.services`, `com.facebook.system`, `com.facebook.appmanager` (Meta background telemetry)
- `com.opera.browser.afin` (Preinstalled Opera browser)
- `com.miui.msa.global` (MIUI System Ads)
- `com.miui.analytics` (MIUI Analytics)
- `com.miui.daemon` (MIUI Telemetry Daemon)
- `com.miui.hybrid` & `com.miui.hybrid.accessory` (Quick Apps / HTML5 Web apps engine)
- `com.miui.yellowpage` (Caller ID / Yellow pages advertising)
- `com.miui.miservice` (Services & Feedback telemetry)
- `com.xiaomi.mipicks` (GetApps)
- `com.xiaomi.payment` (Mi Pay)
- `com.miui.player` (Mi Music with online ads)
- `com.miui.bugreport` (Bug report logger)
- `com.miui.cleaner` (Clean Master engine)

---

## 7. Battery State

- **Current Level:** 80% (Charging via USB)
- **Battery Health:** Good (Code `2`)
- **Battery Voltage:** 4232 mV (~4.23V)
- **Battery Temperature:** 33.0 °C (Normal thermal state)
- **Technology:** Lithium-Polymer (5,000 mAh capacity)

---

## 8. Current Window & Animation Parameters

- `window_animation_scale`: `1.0`
- `transition_animation_scale`: `1.0`
- `animator_duration_scale`: `1.0`
*(MIUI 14 blur effects and 1.0x animation scales on 4 GB RAM Snapdragon 678 cause visible perceived delay).*

---

## 9. Phase 1 Conclusion & Bottleneck Diagnosis

1. **Storage is not the bottleneck:** 26 GB of free storage on `/data` is sufficient.
2. **RAM saturation is the critical issue:** With only ~107 MB of uncompressed free physical RAM and 1.8 GB of swapped ZRAM, the CPU is under constant memory management stress.
3. **Background daemon proliferation:** Multiple advertising, telemetry, and preinstalled store daemons (`msa.global`, `analytics`, `mipicks`, `indus.appstore`, `facebook.services`, `fashiongallery`) run background sync loops that drain battery and consume precious RAM slots.
4. **Zero destructive actions taken:** The device remains untouched, operating in read-only diagnostic inspection.
