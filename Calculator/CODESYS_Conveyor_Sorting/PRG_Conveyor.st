// ====================================================================
// PROGRAM: PRG_Conveyor
// Architecture: IEC 61131-3 Structured Text (ST)
// System: Automated Conveyor Belt and Product Sorting System
// Target: CODESYS V3.5
// ====================================================================

PROGRAM PRG_Conveyor
VAR
    // --- TIMERS (Mandatory TP, TON, TOF implementations) ---
    // 1. TON: Object Detection Debounce & Verification Filter
    tonDetectionFilter      : TON;
    tDebounceTime           : TIME := T#150MS;
    
    // 2. TON: Sorting Travel Delays (from detection sensor to sorters)
    tonDelayToBin1          : TON;
    tTravelTimeBin1         : TIME := T#1650MS; // Time to travel from Sensor (250mm) to Sorter 1 (500mm)
    
    tonDelayToBin2          : TON;
    tTravelTimeBin2         : TIME := T#3300MS; // Time to travel from Sensor (250mm) to Sorter 2 (750mm)
    
    // 3. TP: Pulse Timers for Pneumatic Sorting Pusher Actuation
    tpPusher1Actuate        : TP;
    tPusher1PulseWidth      : TIME := T#1200MS; // Duration pusher cylinder remains extended
    
    tpPusher2Actuate        : TP;
    tPusher2PulseWidth      : TIME := T#1200MS; // Duration pusher cylinder remains extended
    
    tpEntryAlert            : TP;
    tEntryPulseWidth        : TIME := T#500MS;  // Beacon pulse when new object detected
    
    // 4. TOF: Off-Delay Timer for Controlled Stopping
    tofMotorControlledStop  : TOF;
    tRunoutDuration         : TIME := T#3000MS; // Belt continues running for 3s after stop command to clear items
    
    // --- EDGE DETECTORS & TRIGGERS ---
    rtrigStart              : R_TRIG;
    rtrigPusher1Fired       : R_TRIG;
    rtrigPusher2Fired       : R_TRIG;
    rtrigObjectAtBin3       : R_TRIG;
    rtrigSpawn              : R_TRIG;
    
    // --- INTERNAL CONTROL VARIABLES ---
    bRawObjectDetected      : BOOL := FALSE;
    bTargetBin1Armed        : BOOL := FALSE;    // Armed when Metal detected
    bTargetBin2Armed        : BOOL := FALSE;    // Armed when Plastic detected
    bAutoPusher1Cmd         : BOOL := FALSE;
    bAutoPusher2Cmd         : BOOL := FALSE;
    
    // --- SIMULATION TIMING & KINEMATICS ---
    tCurrentCycleTime       : TIME;
    tLastCycleTime          : TIME;
    rDeltaTime              : REAL := 0.02;     // 20ms default task cycle (s)
    bCycleInitDone          : BOOL := FALSE;
    rBeltLength             : REAL := 1000.0;   // Total conveyor length in mm
    rSpawnIntervalTimer     : REAL := 0.0;      // Auto re-spawn delay counter
    
    // Auto object sequence generator (rotates 1 -> 2 -> 3)
    iSpawnSequenceCounter   : INT := 1;
END_VAR

// ====================================================================
// 1. CYCLE TIME CALCULATION FOR SMOOTH POSITION SIMULATION
// ====================================================================
tCurrentCycleTime := TIME();
IF NOT bCycleInitDone THEN
    tLastCycleTime := tCurrentCycleTime;
    bCycleInitDone := TRUE;
    rDeltaTime := 0.02; // initial 20 ms
ELSE
    // Calculate delta time in seconds
    rDeltaTime := TIME_TO_REAL(tCurrentCycleTime - tLastCycleTime) / 1000.0;
    // Clamp delta time to guard against initial wrap or pause
    IF rDeltaTime <= 0.001 OR rDeltaTime > 0.1 THEN
        rDeltaTime := 0.02;
    END_IF;
    tLastCycleTime := tCurrentCycleTime;
END_IF;

// ====================================================================
// 2. OPERATING MODE DECODING & MUTUAL INTERLOCKING
// ====================================================================
// Radio button selector binds to GVL.iModeSelection (1 = Manual, 2 = Automatic)
IF GVL.iModeSelection = 1 THEN
    GVL.bModeManualActive := TRUE;
    GVL.bModeAutoActive   := FALSE;
ELSE
    GVL.iModeSelection    := 2; // Default to Auto
    GVL.bModeManualActive := FALSE;
    GVL.bModeAutoActive   := TRUE;
END_IF;

// ====================================================================
// 3. LATCHING CIRCUIT FOR CONVEYOR SYSTEM START / STOP
// ====================================================================
// Standard industrial Start/Stop latch with safety overrides.
// Interlocking conditions that prevent running or drop the latch:
// - Emergency stop pressed (bEmergencyStop)
// - Targeted bin full in automatic mode (bInterlockBinFull)
rtrigStart(CLK := GVL.bStartBtn);

IF GVL.bEmergencyStop THEN
    GVL.bRunLatch := FALSE;
    GVL.sSystemStatusMessage := 'EMERGENCY STOP ENGAGED! System halted.';
ELSIF GVL.bStopBtn THEN
    GVL.bRunLatch := FALSE;
    GVL.sSystemStatusMessage := 'Stop requested. Controlled deceleration in progress.';
ELSIF rtrigStart.Q THEN
    IF NOT GVL.bAlarmActive THEN
        GVL.bRunLatch := TRUE;
        GVL.sSystemStatusMessage := 'System Running - Conveyor Active.';
    ELSE
        GVL.sSystemStatusMessage := 'Cannot start: Clear active alarms first!';
    END_IF;
END_IF;

// Reset Fault Button logic
IF GVL.bResetFaultBtn THEN
    GVL.bAlarmActive            := FALSE;
    GVL.bInterlockBinFull       := FALSE;
    GVL.bInterlockPusherBlocked := FALSE;
    GVL.stBin1.bFull            := FALSE;
    GVL.stBin2.bFull            := FALSE;
    GVL.stBin3.bFull            := FALSE;
    GVL.stBin1.nCount           := 0;
    GVL.stBin2.nCount           := 0;
    GVL.stBin3.nCount           := 0;
    GVL.iTotalSorted            := 0;
    GVL.sSystemStatusMessage    := 'Alarms & Bin counts reset. System ready.';
END_IF;

// ====================================================================
// 4. CONTROLLED STOPPING USING TOF TIMER (Off-Delay Timer)
// ====================================================================
// When bRunLatch transitions from TRUE to FALSE (operator presses STOP),
// the TOF timer keeps its output Q TRUE for tRunoutDuration (3 seconds).
// This clears any product remaining on the belt safely.
// Note: EMERGENCY STOP bypasses the TOF to shut down instantaneously.
tofMotorControlledStop(IN := GVL.bRunLatch, PT := tRunoutDuration);

IF GVL.bEmergencyStop THEN
    // Instant safety cutoff
    GVL.bMotorRun := FALSE;
    GVL.rConveyorSpeed := 0.0;
ELSIF GVL.bModeAutoActive THEN
    // Automatic Mode: Motor runs while Latch is active OR during TOF runout phase
    GVL.bMotorRun := tofMotorControlledStop.Q;
    IF GVL.bMotorRun THEN
        GVL.rConveyorSpeed := GVL.rSetSpeed;
    ELSE
        GVL.rConveyorSpeed := 0.0;
    END_IF;
ELSIF GVL.bModeManualActive THEN
    // Manual Mode: Motor runs only while operator holds Manual Jog button
    GVL.bMotorRun := GVL.bManualJogFwd;
    IF GVL.bMotorRun THEN
        GVL.rConveyorSpeed := GVL.rSetSpeed * 0.6; // 60% creep speed for manual jog
    ELSE
        GVL.rConveyorSpeed := 0.0;
    END_IF;
END_IF;

// ====================================================================
// 5. OBJECT SPAWNING & CONVEYOR KINEMATIC SIMULATION
// ====================================================================
rtrigSpawn(CLK := GVL.bSpawnNewObjectBtn);

// Auto-spawn logic in Automatic Mode if belt is empty and running
IF GVL.bModeAutoActive AND GVL.bMotorRun AND NOT GVL.bObjectPresent THEN
    rSpawnIntervalTimer := rSpawnIntervalTimer + rDeltaTime;
    IF rSpawnIntervalTimer >= 1.5 THEN // 1.5 second gap between items
        GVL.bObjectPresent := TRUE;
        GVL.rObjectPos := 0.0;
        // Alternate product types: 1 = Metal, 2 = Plastic, 3 = Defective
        GVL.iCurrentObjectType := iSpawnSequenceCounter;
        iSpawnSequenceCounter := iSpawnSequenceCounter + 1;
        IF iSpawnSequenceCounter > 3 THEN
            iSpawnSequenceCounter := 1;
        END_IF;
        rSpawnIntervalTimer := 0.0;
    END_IF;
ELSIF rtrigSpawn.Q AND NOT GVL.bObjectPresent THEN
    // Operator manually spawned an object via UI button
    GVL.bObjectPresent := TRUE;
    GVL.rObjectPos := 0.0;
    GVL.iCurrentObjectType := GVL.iNextObjectTypeSelect;
    IF GVL.iCurrentObjectType < 1 OR GVL.iCurrentObjectType > 3 THEN
        GVL.iCurrentObjectType := 1;
    END_IF;
END_IF;

// Object position integration along the belt
IF GVL.bObjectPresent AND GVL.bMotorRun THEN
    GVL.rObjectPos := GVL.rObjectPos + (GVL.rConveyorSpeed * rDeltaTime);
END_IF;

// Clamp position to belt end
IF GVL.rObjectPos > rBeltLength THEN
    GVL.rObjectPos := rBeltLength;
END_IF;

// ====================================================================
// 6. SENSOR SIMULATION & OBJECT DETECTION (TON Timer Debounce)
// ====================================================================
// Position map:
// 0 mm   : Infeed / Spawn
// 100 mm : Entry Presence Sensor (bSensorEntry)
// 250 mm : Material & Height Inspection Sensor (bSensorMetal, bSensorHeight)
// 500 mm : Bin 1 (Metal) Station (bSensorBin1Station) & Sorter 1
// 750 mm : Bin 2 (Plastic) Station (bSensorBin2Station) & Sorter 2
// 950 mm : Bin 3 (Reject) Station (bSensorBin3Station)

// Simulated Raw Sensor Inputs based on physical position
GVL.bSensorEntry := GVL.bObjectPresent AND (GVL.rObjectPos >= 90.0 AND GVL.rObjectPos <= 140.0);

// Inspection Station (Pos: 230mm - 270mm)
IF GVL.bObjectPresent AND (GVL.rObjectPos >= 230.0 AND GVL.rObjectPos <= 270.0) THEN
    bRawObjectDetected := TRUE;
    GVL.bSensorMetal   := (GVL.iCurrentObjectType = 1); // Metal sensor triggers
    GVL.bSensorHeight  := (GVL.iCurrentObjectType = 2); // Optical height classifier triggers
ELSE
    bRawObjectDetected := FALSE;
    GVL.bSensorMetal   := FALSE;
    GVL.bSensorHeight  := FALSE;
END_IF;

// Station Arrival Sensors
GVL.bSensorBin1Station := GVL.bObjectPresent AND (GVL.rObjectPos >= 480.0 AND GVL.rObjectPos <= 520.0);
GVL.bSensorBin2Station := GVL.bObjectPresent AND (GVL.rObjectPos >= 730.0 AND GVL.rObjectPos <= 770.0);
GVL.bSensorBin3Station := GVL.bObjectPresent AND (GVL.rObjectPos >= 930.0 AND GVL.rObjectPos <= 970.0);

// --- TON IMPLEMENTATION 1: Object Detection Debounce / Verification ---
// Ensures sensor signal is stable and genuine (filters false optical reflections)
tonDetectionFilter(IN := bRawObjectDetected, PT := tDebounceTime);
GVL.bSensorDetectFiltered := tonDetectionFilter.Q;

// Arm sorting routes when object is confirmed at inspection point
IF GVL.bSensorDetectFiltered AND GVL.bModeAutoActive THEN
    IF GVL.iCurrentObjectType = 1 THEN
        bTargetBin1Armed := TRUE;
    ELSIF GVL.iCurrentObjectType = 2 THEN
        bTargetBin2Armed := TRUE;
    END_IF;
END_IF;

// Entry alert pulse timer (TP)
tpEntryAlert(IN := GVL.bSensorEntry, PT := tEntryPulseWidth);

// ====================================================================
// 7. SORTING DELAYS USING TON TIMERS (Travel Tracking)
// ====================================================================
// --- TON IMPLEMENTATION 2: Travel Delay from Inspection to Pusher 1 ---
// Distance = 500mm - 250mm = 250mm. At 150mm/s, travel time = ~1.66s.
tonDelayToBin1(IN := bTargetBin1Armed AND GVL.bMotorRun, PT := tTravelTimeBin1);

// --- TON IMPLEMENTATION 3: Travel Delay from Inspection to Pusher 2 ---
// Distance = 750mm - 250mm = 500mm. At 150mm/s, travel time = ~3.33s.
tonDelayToBin2(IN := bTargetBin2Armed AND GVL.bMotorRun, PT := tTravelTimeBin2);

// Auto-trigger sorting pusher when travel delay elapses
IF tonDelayToBin1.Q THEN
    bAutoPusher1Cmd  := TRUE;
    bTargetBin1Armed := FALSE; // Disarm once reached
END_IF;

IF tonDelayToBin2.Q THEN
    bAutoPusher2Cmd  := TRUE;
    bTargetBin2Armed := FALSE; // Disarm once reached
END_IF;

// ====================================================================
// 8. INTERLOCKING LOGIC (Safety & Mechanical Protection)
// ====================================================================
// Rule 1: Pusher Actuation Interlock
// Pushers must NEVER fire if conveyor is stationary (avoids jams & damage)
IF NOT GVL.bMotorRun THEN
    IF GVL.bManualPusher1 OR GVL.bManualPusher2 OR bAutoPusher1Cmd OR bAutoPusher2Cmd THEN
        GVL.bInterlockPusherBlocked := TRUE;
    END_IF;
    bAutoPusher1Cmd := FALSE;
    bAutoPusher2Cmd := FALSE;
ELSE
    GVL.bInterlockPusherBlocked := FALSE;
END_IF;

// Rule 2: Pusher Cross-Interlock
// Pusher 1 and Pusher 2 cannot fire simultaneously (pneumatic cross-lock)
// Rule 3: Bin Full Interlock
// If a bin has reached maximum capacity, diversion to that bin is blocked!
IF GVL.stBin1.nCount >= GVL.stBin1.nMaxCapacity THEN
    GVL.stBin1.bFull := TRUE;
    bAutoPusher1Cmd := FALSE; // Block sorting to Bin 1
    GVL.bInterlockBinFull := TRUE;
    GVL.bAlarmActive := TRUE;
    GVL.sSystemStatusMessage := 'ALARM: Bin 1 is FULL! Empty bin and reset.';
END_IF;

IF GVL.stBin2.nCount >= GVL.stBin2.nMaxCapacity THEN
    GVL.stBin2.bFull := TRUE;
    bAutoPusher2Cmd := FALSE; // Block sorting to Bin 2
    GVL.bInterlockBinFull := TRUE;
    GVL.bAlarmActive := TRUE;
    GVL.sSystemStatusMessage := 'ALARM: Bin 2 is FULL! Empty bin and reset.';
END_IF;

IF GVL.stBin3.nCount >= GVL.stBin3.nMaxCapacity THEN
    GVL.stBin3.bFull := TRUE;
    GVL.bInterlockBinFull := TRUE;
    GVL.bAlarmActive := TRUE;
    GVL.sSystemStatusMessage := 'ALARM: Reject Bin 3 is FULL!';
END_IF;

// ====================================================================
// 9. PUSHER ACTUATION USING TP TIMERS (Pulse Timers)
// ====================================================================
// --- TP IMPLEMENTATION 1: Sorter 1 (Bin 1 - Metal) ---
// Actuated by TON sorting delay in Auto, or manual button in Manual mode.
// The TP timer guarantees a fixed 1.2-second physical pneumatic stroke.
tpPusher1Actuate(
    IN := (bAutoPusher1Cmd OR (GVL.bModeManualActive AND GVL.bManualPusher1)) 
          AND GVL.bMotorRun 
          AND NOT GVL.bPusher2 // Cross-interlock with Pusher 2
          AND NOT GVL.stBin1.bFull 
          AND NOT GVL.bEmergencyStop,
    PT := tPusher1PulseWidth
);
GVL.bPusher1 := tpPusher1Actuate.Q;
IF GVL.bPusher1 THEN
    bAutoPusher1Cmd := FALSE; // Clear trigger flag
END_IF;

// --- TP IMPLEMENTATION 2: Sorter 2 (Bin 2 - Plastic) ---
// Fixed 1.2-second physical pneumatic stroke
tpPusher2Actuate(
    IN := (bAutoPusher2Cmd OR (GVL.bModeManualActive AND GVL.bManualPusher2)) 
          AND GVL.bMotorRun 
          AND NOT GVL.bPusher1 // Cross-interlock with Pusher 1
          AND NOT GVL.stBin2.bFull 
          AND NOT GVL.bEmergencyStop,
    PT := tPusher2PulseWidth
);
GVL.bPusher2 := tpPusher2Actuate.Q;
IF GVL.bPusher2 THEN
    bAutoPusher2Cmd := FALSE; // Clear trigger flag
END_IF;

// ====================================================================
// 10. PRODUCT SORTING EXECUTION & BIN ACCOUNTING
// ====================================================================
rtrigPusher1Fired(CLK := GVL.bPusher1);
rtrigPusher2Fired(CLK := GVL.bPusher2);
rtrigObjectAtBin3(CLK := (GVL.rObjectPos >= 950.0 AND GVL.bObjectPresent));

// 1. Sorted into Bin 1 (Metal)
IF rtrigPusher1Fired.Q AND GVL.bObjectPresent AND (GVL.rObjectPos >= 450.0 AND GVL.rObjectPos <= 550.0) THEN
    GVL.stBin1.nCount  := GVL.stBin1.nCount + 1;
    GVL.iTotalSorted   := GVL.iTotalSorted + 1;
    GVL.bObjectPresent := FALSE; // Item ejected into Bin 1
    GVL.rObjectPos     := 0.0;
    bTargetBin1Armed   := FALSE;
    GVL.sSystemStatusMessage := 'Object (Metal) sorted into Bin 1 successfully.';
END_IF;

// 2. Sorted into Bin 2 (Plastic)
IF rtrigPusher2Fired.Q AND GVL.bObjectPresent AND (GVL.rObjectPos >= 700.0 AND GVL.rObjectPos <= 800.0) THEN
    GVL.stBin2.nCount  := GVL.stBin2.nCount + 1;
    GVL.iTotalSorted   := GVL.iTotalSorted + 1;
    GVL.bObjectPresent := FALSE; // Item ejected into Bin 2
    GVL.rObjectPos     := 0.0;
    bTargetBin2Armed   := FALSE;
    GVL.sSystemStatusMessage := 'Object (Plastic) sorted into Bin 2 successfully.';
END_IF;

// 3. Reached Bin 3 / Reject Bin (End of Conveyor)
IF rtrigObjectAtBin3.Q THEN
    GVL.stBin3.nCount  := GVL.stBin3.nCount + 1;
    GVL.iTotalSorted   := GVL.iTotalSorted + 1;
    GVL.bObjectPresent := FALSE; // Drops off belt into Bin 3
    GVL.rObjectPos     := 0.0;
    bTargetBin1Armed   := FALSE;
    bTargetBin2Armed   := FALSE;
    IF GVL.iCurrentObjectType = 3 THEN
        GVL.sSystemStatusMessage := 'Defective item ejected to Bin 3 (Reject).';
    ELSE
        GVL.sSystemStatusMessage := 'Item passed through to Bin 3.';
    END_IF;
END_IF;

// ====================================================================
// 11. VISUALIZATION COORDINATES & COLOR BINDING
// ====================================================================
// Converts simulated mm position (0-1000mm) to screen pixel coordinates
// Belt visual screen bounds: X = 120px to X = 720px (600px total length)
GVL.rVisuObjectX       := 120.0 + (GVL.rObjectPos * 0.6);
GVL.bVisuObjectVisible := GVL.bObjectPresent;

// Dynamic object color:
// Type 1 (Metal): Silver/Light Slate Gray (16#FF9E9E9E)
// Type 2 (Plastic): Sky Blue (16#FF2196F3)
// Type 3 (Defective): Coral Red (16#FFE91E63)
CASE GVL.iCurrentObjectType OF
    1: GVL.dwVisuObjectColor := 16#FF9E9E9E; // Metal
    2: GVL.dwVisuObjectColor := 16#FF2196F3; // Plastic
    3: GVL.dwVisuObjectColor := 16#FFE91E63; // Defective
ELSE
    GVL.dwVisuObjectColor := 16#FFFFFFFF;
END_CASE;

END_PROGRAM
