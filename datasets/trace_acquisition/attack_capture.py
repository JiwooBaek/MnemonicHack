import sys
import time
import os
import numpy as np
import phywhisperer.usb as pw

# Load LScope library
try:
    from lecroy_osc.lscope import LScope
    print("[System] LScope library loaded.")
except ImportError:
    print("[Error] 'lecroy_osc' package not found.")
    sys.exit(1)

# ========= CONFIGURATION =========
SCOPE_IP = "XXX.XXX.XXX.XXX"  # <-- Set the LeCroy oscilloscope IP address here
# Target USB pattern for trigger (HOST TO DEVICE, getAdress() request)
SHORT_PATTERN = [0x00, 0x3f, 0x23, 0x23] 
SHORT_MASK    = [0x00, 0xFF, 0xFF, 0xFF]

def run_c1_c2_capture():
    print(f"[Scope] Connecting to {SCOPE_IP}...")
    
    # [A] Oscilloscope Setup
    try:
        scope = LScope(num_of_channels=4) 
        scope.connect(SCOPE_IP)
        
        # Channel Configuration
        # C1: PhyWhisperer (Trigger Source & USB State)
        scope.send("C1:TRACE ON"); scope.send("C1:VOLT_DIV 1.0"); scope.send("C1:OFFSET -1.5")
        
        # C2: EM Probe (Measurement Target)
        scope.send("C2:TRACE ON")

        # Trigger Settings
        scope.send("TRIG_SELECT EDGE,SR,C1")   
        scope.send("C1:TRIG_LEVEL 1.0")
        scope.send("TRIG_SLOPE POSITIVE")
        
        # Timebase and Delay
        # 5ms total window (500us/div * 10)
        scope.send("TIME_DIV 5000E-6")
        # Offset to capture specific operation window
        scope.send("TRIG_DELAY -415E-3") 
        
        print("[Scope] Setup complete.")
        print("        Channels  : C1 (Trigger), C2 (EM)")
        print("        Timebase  : 500us/div")
        print("        Trig Delay: -415ms") 

    except Exception as e:
        print(f"[Error] Scope setup failed: {e}")
        return

    # [B] PhyWhisperer Setup
    try:
        phy = pw.Usb()
        phy.con()
        phy.set_power_source("host")
        time.sleep(0.5)
        # Trigger on specific USB pattern
        phy.set_pattern(pattern=SHORT_PATTERN, mask=SHORT_MASK)
        phy.set_trigger(delay=0, width=100)
    except Exception as e:
        print(f"[Error] PhyWhisperer setup failed: {e}")
        return

    # [C] Capture Execution
    try:
        scope.send("TRMD SINGLE")
        print("\n[READY] Waiting for trigger...")
        
        # Arm PhyWhisperer to detect pattern
        phy.arm()
        
        wait_start = time.time()
        triggered = False
        
        while True:
            # Poll for scope STOP state (indicates trigger event)
            if scope.check_is_stop():
                print(f"\n[Event] Capture triggered.")
                triggered = True
                break
                
            if time.time() - wait_start > 100:
                print("\n[Timeout] No trigger detected.")
                break
            time.sleep(0.1)

        # [D] Data Retrieval
        if triggered:
            print("[System] Downloading waveforms...")
            
            # Retrieve C1 (USB State/Trigger)
            res_c1 = scope.get_waveform(channel=1)
            if res_c1 and res_c1[1] is not None: 
                np.save("usb_data.npy", res_c1[1])
                print(f"  >> Saved C1 (USB data)")

            # Retrieve C2 (EM Trace)
            res_c2 = scope.get_waveform(channel=2)
            if res_c2 and res_c2[1] is not None: 
                np.save("EM_trace.npy", res_c2[1])
                print(f"  >> Saved C2 (EM trace)")
                
    except Exception as e:
        print(f"[Error] Capture/Save error: {e}")
    finally:
        try:
            scope.disconnect()
            phy.close()
            print("[System] Disconnected.")
        except:
            pass

if __name__ == "__main__":
    run_c1_c2_capture()