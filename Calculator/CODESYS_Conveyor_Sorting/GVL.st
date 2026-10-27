// ====================================================================
// GLOBAL VARIABLE LIST (GVL)
// Project: Automated Conveyor Belt and Product Sorting System
// Target: CODESYS V3.5
// ====================================================================

VAR_GLOBAL
    // --- OPERATOR PUSHBUTTONS & CONTROLS ---
    bStartBtn               : BOOL := FALSE;    // Momentary Start Pushbutton (NO)
    bStopBtn                : BOOL := FALSE;    // Momentary Stop Pushbutton (NC logic or software trigger)
    bEmergencyStop          : BOOL := FALSE;    // Emergency Stop Button (NC / Latched)
    bResetFaultBtn          : BOOL := FALSE;    // Fault Reset Button
    
    // --- MODE SELECTION (Radio Buttons in Visu) ---
    // In CODESYS Radio Button group, bind this INT variable:
    // Option 1 = Manual Mode (Value = 1)
    // Option 2 = Automatic Mode (Value = 2)
    iModeSelection          : INT := 2;         // Default to AUTOMATIC (2)
    bModeManualActive       : BOOL;             // Status flag for Manual Mode
    bModeAutoActive         : BOOL;             // Status flag for Automatic Mode
    
    // --- MANUAL CONTROLS (Active in Manual Mode only) ---
    bManualJogFwd           : BOOL := FALSE;    // Manual Jog Conveyor Forward
    bManualPusher1          : BOOL := FALSE;    // Manual Extend Diverter 1 (Bin 1)
    bManualPusher2          : BOOL := FALSE;    // Manual Extend Diverter 2 (Bin 2)
    
    // --- SYSTEM LATCH & MOTOR STATUS ---
    bRunLatch               : BOOL := FALSE;    // Latching circuit memory bit
    bMotorRun               : BOOL := FALSE;    // Conveyor Motor Output (Contactor / Inverter run)
    rConveyorSpeed          : REAL := 0.0;      // Current speed (mm/s), used in Trace
    rSetSpeed               : REAL := 150.0;    // Target operating speed (mm/s)
    
    // --- OBJECT POSITION & SIMULATION ---
    rObjectPos              : REAL := 0.0;      // Current object position along belt (0.0 to 1000.0 mm)
    iCurrentObjectType      : INT := 1;         // 1 = Metal, 2 = Plastic, 3 = Defective
    bObjectPresent          : BOOL := FALSE;    // True if an object is currently on the belt
    iNextObjectTypeSelect   : INT := 1;         // Visu dropdown / selection for next item to spawn
    bSpawnNewObjectBtn      : BOOL := FALSE;    // Trigger to spawn next item manually or in auto
    
    // --- SENSORS (Simulated & Physical) ---
    bSensorEntry            : BOOL := FALSE;    // Optical Entry Sensor (Pos: 100 mm)
    bSensorDetectFiltered   : BOOL := FALSE;    // Debounced presence detection (TON filtered)
    bSensorMetal            : BOOL := FALSE;    // Inductive Metal Proximity Sensor (Pos: 250 mm)
    bSensorHeight           : BOOL := FALSE;    // Optical Height / Color Classifier Sensor (Pos: 250 mm)
    bSensorBin1Station      : BOOL := FALSE;    // Arrival sensor at Sorter 1 (Pos: 500 mm)
    bSensorBin2Station      : BOOL := FALSE;    // Arrival sensor at Sorter 2 (Pos: 750 mm)
    bSensorBin3Station      : BOOL := FALSE;    // Arrival sensor at Reject Bin 3 (Pos: 950 mm)
    
    // --- ACTUATOR OUTPUTS (Pneumatic Diverters / Pushers) ---
    bPusher1                : BOOL := FALSE;    // Sorter 1 (Diverts Metal into Bin 1)
    bPusher2                : BOOL := FALSE;    // Sorter 2 (Diverts Plastic into Bin 2)
    
    // --- COLLECTION BINS & METRICS ---
    stBin1                  : ST_BinStatus := (nCount := 0, nMaxCapacity := 10, bFull := FALSE);
    stBin2                  : ST_BinStatus := (nCount := 0, nMaxCapacity := 10, bFull := FALSE);
    stBin3                  : ST_BinStatus := (nCount := 0, nMaxCapacity := 10, bFull := FALSE);
    iTotalSorted            : INT := 0;         // Total items successfully sorted
    
    // --- SAFETY & INTERLOCKING ALARMS ---
    bInterlockPusherBlocked : BOOL := FALSE;    // Active if pusher actuation blocked due to motor stopped
    bInterlockBinFull       : BOOL := FALSE;    // Active if targeted bin is full
    bAlarmActive            : BOOL := FALSE;    // Master Alarm indicator
    sSystemStatusMessage    : STRING(80) := 'System Ready';
    
    // --- VISUALIZATION HELPERS ---
    rVisuObjectX            : REAL := 120.0;    // Belt start X-coord in pixels + (rObjectPos * scale)
    rVisuObjectY            : REAL := 220.0;    // Belt Y-coord
    bVisuObjectVisible      : BOOL := FALSE;    // Controls visibility of product on belt
    dwVisuObjectColor       : DWORD := 16#FFB0B0B0; // Hex ARGB color (Gray = Metal, Blue = Plastic, Red = Reject)
END_VAR
