# Automated Conveyor Belt and Product Sorting System
## CODESYS V3.5 PLC Implementation & Visualization Guide

---

### Project Overview
This project implements a fully automated industrial conveyor belt and 3-way product sorting system in **CODESYS V3.5** compliant with **IEC 61131-3**.

Key capabilities included:
- **Latching Circuit**: Start/Stop memory rung with Emergency Stop and safety overrides.
- **Operating Modes**: Dual-mode architecture (**MANUAL** and **AUTOMATIC**) selectable via interactive Visu Radio Buttons.
- **Mandatory Timers Implemented**:
  - `TON` (On-Delay Timer): Object detection debouncing (150 ms) and travel delay tracking from inspection station to Sorter 1 (1.65 s) and Sorter 2 (3.30 s).
  - `TP` (Pulse Timer): Fixed pneumatic actuator stroke (1.2 s) for diverter pushers.
  - `TOF` (Off-Delay Timer): Controlled stopping sequence (3.0 s) allowing the conveyor to clear in-flight items after STOP is pressed.
- **Safety Interlocking Logic**: Pusher motion interlock (cannot fire if belt is stationary), cross-interlock between pushers, bin capacity limit detection, and immediate E-Stop overrides.
- **Interactive Visualization**: Complete HMI screen showing the conveyor, real-time moving objects with coordinate translation, sensor beacons, pneumatic diverters, bin metrics, and alarm banners.
- **Trace Element**: Real-time oscilloscope-style plotting of `rObjectPos`, `rConveyorSpeed`, `bMotorRun`, and pusher signals against time.

---

### Directory Files Created
All ready-to-use files are saved in:
`C:\Users\sudar\Documents\Collage\Placement prep\Calculator\CODESYS_Conveyor_Sorting\`

1. **`Conveyor_Sorting_PLCopen.xml`**: Universal PLCopen XML file. Can be imported directly into CODESYS in 10 seconds.
2. **`PRG_Conveyor.st`**: Complete IEC 61131-3 Structured Text logic.
3. **`GVL.st`**: Global Variable List defining all physical I/O, internal flags, timers, and Visu tags.
4. **`DUT_Enums.st`**: Data types and structures (`E_OperationMode`, `E_ObjectType`, `ST_BinStatus`).

---

## 1. Quick Setup in CODESYS (Under 2 Minutes)

### Option A: Direct 1-Click PLCopen XML Import (Recommended)
1. Open **CODESYS V3.5** (installed at `C:\Program Files\CODESYS 3.5.22.40\CODESYS\Common\CODESYS.exe`).
2. Go to **File -> New Project -> Standard Project**.
3. Name it `Conveyor_Sorting_System` and click **OK**.
4. In the Standard Project dialog:
   - Device: `CODESYS Control Win V3 x64` (or `CODESYS Control Win V3`)
   - PLC_PRG in: `Structured Text (ST)`
5. In the Devices tree, right-click **Application** -> **Import...** (or click menu **Project -> Import PLCopen XML...**).
6. Browse and select:
   `C:\Users\sudar\Documents\Collage\Placement prep\Calculator\CODESYS_Conveyor_Sorting\Conveyor_Sorting_PLCopen.xml`
7. Click **OK**. CODESYS will automatically import `DUT_Enums`, `GVL`, and `PRG_Conveyor`!
8. Open the existing `PLC_PRG` and add one line:
   ```pascal
   PRG_Conveyor();
   ```

---

## 2. System Logic & Timer Architecture

```
Conveyor Infeed (0mm) ----> S1 (100mm) ----> S2 Inspection (250mm) ----> Pusher 1 (500mm) ----> Pusher 2 (750mm) ----> Bin 3 (950mm)
                                                    |                         |                         |                    |
                                            [TON Debounce]             [TON Delay 1.65s]         [TON Delay 3.30s]      (Defective / End)
                                                    |                         |                         |
                                            Identify Material         [TP Pusher Pulse]         [TP Pusher Pulse]
                                            (Metal / Plastic)                 |                         |
                                                                        [Bin 1: Metal]          [Bin 2: Plastic]
```

### 1. Latching Circuit (Start / Stop)
- **NO Start Button** (`bStartBtn`) latches `bRunLatch := TRUE`.
- **NC Stop Button** (`bStopBtn`) breaks the latch.
- **Emergency Stop** (`bEmergencyStop`) breaks the latch and instantly cuts all motor and pusher outputs without delay.
- Software equation:
  ```pascal
  IF GVL.bEmergencyStop THEN
      GVL.bRunLatch := FALSE;
  ELSIF GVL.bStopBtn THEN
      GVL.bRunLatch := FALSE;
  ELSIF rtrigStart.Q AND NOT GVL.bAlarmActive THEN
      GVL.bRunLatch := TRUE;
  END_IF;
  ```

### 2. Mandatory Timers Breakdown
| Timer Type | Instance Name | Parameter | Industrial Purpose & Function |
|---|---|---|---|
| **TON** (On-Delay) | `tonDetectionFilter` | `PT := T#150MS` | **Sensor Debounce**: Filters optical noise/chatter. Sensor must remain covered continuously for 150ms before the object is confirmed. |
| **TON** (On-Delay) | `tonDelayToBin1` | `PT := T#1650MS` | **Sorting Travel Delay 1**: Times travel from inspection station (250 mm) to Sorter 1 (500 mm) at 150 mm/s. Triggers Pusher 1 upon arrival. |
| **TON** (On-Delay) | `tonDelayToBin2` | `PT := T#3300MS` | **Sorting Travel Delay 2**: Times travel from inspection station (250 mm) to Sorter 2 (750 mm) at 150 mm/s. Triggers Pusher 2 upon arrival. |
| **TP** (Pulse) | `tpPusher1Actuate` | `PT := T#1200MS` | **Pneumatic Cylinder Pulse**: Holds Pusher 1 extended for exactly 1.2s to cleanly push the metal workpiece into Bin 1. |
| **TP** (Pulse) | `tpPusher2Actuate` | `PT := T#1200MS` | **Pneumatic Cylinder Pulse**: Holds Pusher 2 extended for exactly 1.2s to cleanly push the plastic workpiece into Bin 2. |
| **TOF** (Off-Delay) | `tofMotorControlledStop` | `PT := T#3000MS` | **Controlled Stopping (Runout)**: Belt continues running for 3.0s after operator presses Stop, ensuring in-transit objects clear before shutdown. |

### 3. Safety & Interlocking Rules
1. **Belt Stationary Lock**: Pushers can only actuate if `bMotorRun = TRUE`. If the belt is stopped, any manual or automatic pusher command is discarded to prevent mechanical jamming.
2. **Pusher Cross-Interlock**: `bPusher1` blocks `bPusher2` and vice versa, preventing simultaneous extension.
3. **Bin Capacity Lock**: When `stBin1.nCount >= 10`, `stBin1.bFull := TRUE`. Automatic sorting to that bin is locked out, raising a visual alarm banner.
4. **Emergency Stop Override**: Instantly zeroes `bMotorRun`, `rConveyorSpeed`, and clears all pusher commands, bypassing the TOF runout timer.

---

## 3. Visualization Screen Layout & Design

Right-click **Application** -> **Add Object -> Visualization** (name it `Visualization`).

### Screen Layout Blueprint

```
+-------------------------------------------------------------------------------------------------------+
|  AUTOMATED CONVEYOR BELT & PRODUCT SORTING SYSTEM                               [ E-STOP BUTTON ]     |
|  Status: [ SYSTEM RUNNING - CONVEYOR ACTIVE                                        ] [ RESET ALARMS ] |
+-------------------------------------------------------------------------------------------------------+
|  OPERATOR CONTROLS           |  MANUAL CONTROLS                 |  PRODUCT SPAWN SIMULATION           |
|  [ START ]   [ STOP ]        |  [ JOG MOTOR FWD ]               |  Spawn Type: (o) Metal  (o) Plastic |
|  Mode:                       |  [ FIRE PUSHER 1 (Bin 1) ]       |              (o) Defective          |
|  ( ) MANUAL   (*) AUTOMATIC  |  [ FIRE PUSHER 2 (Bin 2) ]       |  [ SPAWN OBJECT NOW ]               |
+-------------------------------------------------------------------------------------------------------+
|                                                                                                       |
|   Infeed            S1 (Detect)    S2 (Inspection)          Pusher 1 (Metal)       Pusher 2 (Plastic) |
|   [===]                 (O)              (O)                      [ | ]                  [ | ]        |
|  ==================================================================================================== |
|      >>>>>               [ PRODUCT ]                                                         >>>>>    |
|  ==================================================================================================== |
|                                                                     |                      |     |    |
|                                                                     v                      v     v    |
|                                                              +--------------+       +--------------+  |
|                                                              | BIN 1 (Metal)|       |BIN 2(Plastic)|  |
|                                                              | Count:  3/10 |       | Count:  5/10 |  |
|                                                              +--------------+       +--------------+  |
|                                                                                              [BIN 3]  |
+-------------------------------------------------------------------------------------------------------+
|  REAL-TIME PROCESS TRACE (Oscilloscope View)                                                          |
|  +-------------------------------------------------------------------------------------------------+  |
|  |  Position (mm) / Speed (mm/s) / Actuator States vs Time (s)                                     |  |
|  +-------------------------------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------------------------------+
```

### Exact Element Configuration Table

| Element Name | CODESYS Toolbox Item | Position (X, Y, W, H) | Key Properties & Variable Bindings | Visual Behavior |
|---|---|---|---|---|
| **E-Stop Button** | Push Switch / Button | (850, 15, 90, 45) | **Variable**: `GVL.bEmergencyStop`<br>**Toggle**: Checked | Bright Red Mushroom; instantly stops system |
| **Start Button** | Button (Common Controls) | (30, 85, 90, 40) | **Variable**: `GVL.bStartBtn`<br>**Tap variable**: `GVL.bStartBtn` | Green; Initiates latch circuit |
| **Stop Button** | Button (Common Controls) | (130, 85, 90, 40) | **Variable**: `GVL.bStopBtn`<br>**Tap variable**: `GVL.bStopBtn` | Red; Initiates 3s TOF controlled stopping |
| **Mode Radio Buttons** | Radio Button (Common) | (30, 140, 190, 60) | **Variable**: `GVL.iModeSelection`<br>Number of items: `2`<br>Texts: `[0] Manual`, `[1] Automatic` | Selects operating mode (Value 1 or 2) |
| **Jog Motor Fwd** | Button (Common Controls) | (260, 85, 130, 35) | **Tap variable**: `GVL.bManualJogFwd`<br>**State variables -> Enabled**: `GVL.bModeManualActive` | Active only in Manual mode |
| **Manual Pusher 1** | Button (Common Controls) | (260, 130, 130, 35) | **Tap variable**: `GVL.bManualPusher1`<br>**Enabled**: `GVL.bModeManualActive` | Fires Diverter 1 manually |
| **Manual Pusher 2** | Button (Common Controls) | (400, 130, 130, 35) | **Tap variable**: `GVL.bManualPusher2`<br>**Enabled**: `GVL.bModeManualActive` | Fires Diverter 2 manually |
| **Spawn Button** | Button (Common Controls) | (600, 130, 140, 35) | **Tap variable**: `GVL.bSpawnNewObjectBtn` | Manually injects test object |
| **Conveyor Frame** | Rectangle (Basic) | (80, 240, 740, 60) | **Fill color**: Dark Slate (`16#303030`) | Represents mechanical conveyor frame |
| **Conveyor Belt** | Rectangle (Basic) | (90, 250, 720, 40) | **Fill color**: Black (`16#1A1A1A`)<br>**Frame color**: `16#00E676` when `GVL.bMotorRun` | Indicates active belt motion |
| **Moving Product** | Rectangle (Basic) | (120, 255, 30, 30) | **Position -> X**: `120`<br>**Motion absolute -> X**: `GVL.rVisuObjectX`<br>**Color -> Fill color**: `GVL.dwVisuObjectColor`<br>**State variables -> Invisible**: `NOT GVL.bVisuObjectVisible` | Moves dynamically across the screen. Color changes by type. |
| **Entry Sensor S1** | Lamp / Circle (Lamps) | (180, 215, 22, 22) | **Variable**: `GVL.bSensorEntry`<br>**Color on**: Bright Yellow | Lights when object enters infeed |
| **Inspection S2** | Lamp / Circle (Lamps) | (270, 215, 22, 22) | **Variable**: `GVL.bSensorDetectFiltered`<br>**Color on**: Bright Cyan | Lights when material inspected |
| **Pusher 1 Diverter** | Rectangle (Basic) | (420, 185, 20, 65) | **Fill color**: `16#FF9800` (Orange)<br>**Motion absolute -> Y**: `185 + (BOOL_TO_INT(GVL.bPusher1) * 35)` | Visually extends downwards into belt when actuated |
| **Pusher 2 Diverter** | Rectangle (Basic) | (570, 185, 20, 65) | **Fill color**: `16#2196F3` (Blue)<br>**Motion absolute -> Y**: `185 + (BOOL_TO_INT(GVL.bPusher2) * 35)` | Visually extends downwards into belt when actuated |
| **Bin 1 (Metal)** | Rectangle (Basic) | (400, 320, 90, 80) | **Text**: `Bin 1 (Metal)\n%d / 10`<br>**Text variable**: `GVL.stBin1.nCount`<br>**Color**: Light Gray | Collects Metal workpieces |
| **Bin 2 (Plastic)** | Rectangle (Basic) | (550, 320, 90, 80) | **Text**: `Bin 2 (Plastic)\n%d / 10`<br>**Text variable**: `GVL.stBin2.nCount`<br>**Color**: Soft Blue | Collects Plastic workpieces |
| **Bin 3 (Reject)** | Rectangle (Basic) | (770, 320, 90, 80) | **Text**: `Bin 3 (Reject)\n%d / 10`<br>**Text variable**: `GVL.stBin3.nCount`<br>**Color**: Soft Coral | Collects Defective workpieces |
| **Alarm Banner** | Text Field (Common) | (30, 48, 650, 28) | **Text**: `%s`<br>**Text variable**: `GVL.sSystemStatusMessage`<br>**Color toggle**: `GVL.bAlarmActive` | Displays live status and fault codes |

---

## 4. Real-Time Process Trace Configuration

The Trace object plots high-speed PLC process variables cyclically, exactly matching modern SCADA and digital commissioning tools.

### Step-by-Step Trace Setup
1. In the CODESYS Devices tree, right-click **Application -> Add Object -> Trace**.
2. Name it `Trace_Conveyor` and click **Add**.
3. Under **Trace Configuration** (right-click the trace area -> **Configuration**):
   - **Task**: Select `MainTask` (Cyclical, default 20ms).
   - **Record buffer**: 500 samples (provides 10 seconds of continuous display).
4. Click **Add Variable** for each of the following:
   - `PRG_Conveyor.rObjectPos` (Color: **Cyan**, Min: 0, Max: 1000) -> Object position along the belt.
   - `PRG_Conveyor.rConveyorSpeed` (Color: **Green**, Min: 0, Max: 200) -> Conveyor belt speed.
   - `GVL.bMotorRun` (Color: **Lime**, Min: 0, Max: 1) -> Motor running digital state.
   - `GVL.bPusher1` (Color: **Orange**, Min: 0, Max: 1) -> Diverter 1 stroke pulse.
   - `GVL.bPusher2` (Color: **Blue**, Min: 0, Max: 1) -> Diverter 2 stroke pulse.
5. In your `Visualization`, drag a **Trace** element from the **Special Control** category into the bottom section of your screen:
   - In its properties, bind **Trace** to `Application.Trace_Conveyor`.
6. When online in CODESYS, click **Trace -> Download Trace -> Start Trace**. The waveforms will draw live as objects move and sort!

---

## 5. Verification & Testing Procedure

Follow this checklist to demonstrate the system:

1. **Enable Simulation**:
   - Menu: **Online -> Simulation** (check that "Simulation" appears in the bottom status bar).
   - Menu: **Build -> Generate Code** (confirm 0 Errors, 0 Warnings).
   - Menu: **Online -> Login (Alt+F8)** -> Click **Yes** to download application.
   - Menu: **Debug -> Start (F5)**.

2. **Test Latching & Controlled Stop (TOF)**:
   - Ensure mode is set to **AUTOMATIC**.
   - Click **START** button:
     - `bRunLatch` activates, motor runs, and conveyor starts moving at 150 mm/s.
   - Click **STOP** button:
     - Notice that `bRunLatch` drops immediately, but the conveyor **continues to run for 3.0 seconds** (TOF timer `tofMotorControlledStop` active) before halting smoothly!

3. **Test Automatic Sorting Cycle (TON & TP)**:
   - Click **START**.
   - An object spawns at `0 mm` and travels forward:
     - When crossing `100 mm`, `bSensorEntry` flashes.
     - When crossing `250 mm`, `tonDetectionFilter` confirms the object, triggering the appropriate TON travel timer.
     - **Metal Object (Type 1)**: Reaches `500 mm` after 1.65s delay -> `bPusher1` fires with a 1.2s `TP` pulse -> Object pushed into **Bin 1** -> `Bin 1 Count` increments to 1!
     - **Plastic Object (Type 2)**: Reaches `750 mm` after 3.30s delay -> `bPusher2` fires with a 1.2s `TP` pulse -> Object pushed into **Bin 2** -> `Bin 2 Count` increments to 1!
     - **Defective Object (Type 3)**: Passes both sorters and drops into **Bin 3** at `950 mm` -> `Bin 3 Count` increments to 1!

4. **Test Interlocking Safety Logic**:
   - **Emergency Stop**: Press E-Stop while an item is in motion. The motor and pushers shut down immediately without waiting for any TOF delay.
   - **Stationary Pusher Lock**: Switch to **MANUAL** mode while the motor is stopped. Click **FIRE PUSHER 1**. The pusher does NOT actuate, protecting the belt against mechanical jams.
   - **Bin Full Lock**: When any bin reaches capacity (10 items), its "FULL" flag trips and further sorting into that bin is blocked, generating an alarm.
