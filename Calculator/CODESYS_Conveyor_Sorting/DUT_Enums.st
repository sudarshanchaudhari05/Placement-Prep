// ====================================================================
// DATA TYPES & ENUMERATIONS (DUT)
// Project: Automated Conveyor Belt and Product Sorting System
// Target: CODESYS V3.5
// ====================================================================

TYPE E_OperationMode :
(
    MODE_NONE       := 0,
    MODE_MANUAL     := 1,   // Manual jog and individual actuator control
    MODE_AUTOMATIC  := 2    // Fully automated sensor & timer-driven cycle
) INT;
END_TYPE

TYPE E_ObjectType :
(
    OBJ_NONE        := 0,   // No object on the belt
    OBJ_METAL       := 1,   // Metal object -> To be sorted into Bin 1
    OBJ_PLASTIC     := 2,   // Plastic object -> To be sorted into Bin 2
    OBJ_DEFECTIVE   := 3    // Defective object -> Passes to Bin 3 (Reject)
) INT;
END_TYPE

TYPE E_SystemState :
(
    STATE_IDLE      := 0,   // Ready, waiting for Start latch
    STATE_RUNNING   := 1,   // Conveyor motor running, actively processing
    STATE_STOPPING  := 2,   // Controlled stopping (TOF runout phase)
    STATE_ESTOP     := 3,   // Emergency Stop engaged
    STATE_FAULT     := 4    // Bin full or interlocking alarm
) INT;
END_TYPE

TYPE ST_BinStatus :
STRUCT
    nCount          : INT;      // Current number of items sorted into this bin
    nMaxCapacity    : INT;      // Max capacity before bin full alarm
    bFull           : BOOL;     // Bin full sensor / flag
END_STRUCT
END_TYPE
