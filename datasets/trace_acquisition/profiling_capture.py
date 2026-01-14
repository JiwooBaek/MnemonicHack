# %% Oscilloscope Capture Script for Trezor EM (Profiling Phase)
# Please set Sampling Rate to 500 MS/s and time window to 50.0 μs on the oscilloscope before running.
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from tqdm import tqdm

# Setup colored error output for visibility during long captures
try:
    from colorama import init as _cinit, Fore, Style
    _cinit()
    def _ERR(msg: str):
        tqdm.write(Fore.RED + msg + Style.RESET_ALL, file=sys.stderr)
except Exception:
    def _ERR(msg: str):
        tqdm.write("\033[31m" + msg + "\033[0m", file=sys.stderr)

# Assumes lecroy_osc.py is in the same directory
from lecroy_osc import LScope

# ========= CONFIGURATION =========
HOST = "XXX.XXX.XXX.XXX"  # <-- Set the LeCroy oscilloscope IP address here

BIT_DEPTH = "BYTE"      # 'BYTE' (int8) or 'WORD' (int16)
TARGET_CH = 2           # EM measurement channel
TARGET_CH_TRIG = 1      # Trigger channel (used for length validation)

MAX_CAPTURES = 20000    # Total number of traces to acquire
IDLE_TIMEOUT_S = 5.0    # Max wait time for subsequent triggers
POLL_INTERVAL_S = 0.02  # Polling rate for STOP state
REARM_GUARD_S = 2.0     # Force re-arm if no trigger within this window

OUT_PATH = Path("../traces/trace_A_a_20000.npy") # <-- Device A, Unique Key (mnemonic_code_a_20000.txt)   : Training
# OUT_PATH = Path("../traces/trace_B_b_20000.npy") <-- Device B, Unique Key (mnemonic_code_b_20000.txt)   : Training
# OUT_PATH = Path("../traces/trace_C_c_20000.npy") <-- Device C, Unique Key (mnemonic_code_c_20000.txt)   : Validation 
# OUT_PATH = Path("../traces/trace_D_d_20000.npy") <-- Device D, Unique Key (mnemonic_code_d_20000.txt)   : Test
# OUT_PATH = Path("../traces/trace_A_common_20000.npy") <-- Device A, Common Key (mnemonic_code_common_20000.txt) : Training
# OUT_PATH = Path("../traces/trace_B_common_20000.npy") <-- Device B, Common Key (mnemonic_code_common_20000.txt) : Training
# =================================

def _dtype_from_bitdepth(bit_depth: str):
    return np.int8 if bit_depth == "BYTE" else np.int16


def _capture_once(lscope: LScope,
                  ch: int,
                  ch_trig: int,
                  expected_len: Optional[int],
                  bit_depth: str,
                  idle_timeout_s: Optional[float]) -> Optional[np.ndarray]:
    """
    Arms the scope and waits for a single trigger event.
    Returns the waveform data or None on timeout.
    """
    want = _dtype_from_bitdepth(bit_depth)
    idle_start = time.monotonic()

    while True:
        try:
            lscope.arm(wait_time=None)
            t0 = time.monotonic()

            while True:
                # 1. Check global timeout for this capture slot
                if idle_timeout_s is not None and (time.monotonic() - idle_start) >= idle_timeout_s:
                    return None

                # 2. Poll scope status (Check for STOP state)
                if lscope.check_is_stop():
                    t = lscope.get_waveform(channel=ch)
                    tt = lscope.get_waveform(channel=ch_trig)

                    if t is None or t[1] is None:
                        _ERR("[ERROR] get_waveform returned None. Retrying...")
                        break  # Break inner loop to re-arm

                    data = t[1]
                    data_trig = tt[1] if (tt is not None and tt[1] is not None) else None

                    # Validate sample length consistency
                    if expected_len is not None and data.shape[0] != expected_len:
                        _ERR(f"[ERROR] Length mismatch ({data.shape[0]} vs {expected_len}). Retrying.")
                        break

                    if data_trig is not None and expected_len is not None and data_trig.shape[0] != expected_len:
                        _ERR(f"[ERROR] Trigger length mismatch ({data_trig.shape[0]} vs {expected_len}). Retrying.")
                        break

                    # Ensure correct dtype
                    if data.dtype != want:
                        data = data.astype(want, copy=False)

                    return data

                # 3. Re-arm guard: if stuck in ARM for too long without trigger, reset
                if (time.monotonic() - t0) > REARM_GUARD_S:
                    break 

                time.sleep(POLL_INTERVAL_S)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            _ERR(f"[ERROR] Exception during capture: {e!r}. Clearing buffer...")
            try:
                lscope.clear_recv_buf()
            except Exception:
                pass
            time.sleep(0.1)


def main():
    lscope = LScope()
    traces: List[np.ndarray] = []
    samples = 0

    try:
        # --- Connection ---
        print(f"[INFO] Connecting to LeCroy at {HOST}...")
        lscope.connect(host=HOST)
        lscope.set_waveform_dtype(BIT_DEPTH)
        lscope.stop()
        try:
            lscope.clear_recv_buf()
        except Exception:
            pass
        time.sleep(0.2)

        # Get initial reference length
        samples = lscope.get_current_waveform_length()
        print(f"[INFO] Reference waveform length: {samples}")
        
        bar = tqdm(total=MAX_CAPTURES, desc="Capturing", ncols=100, ascii=True, leave=True)

        for idx in range(MAX_CAPTURES):
            bar.set_postfix_str(f"idx={idx}, status=waiting...")

            # Sync logic: First trigger waits indefinitely, subsequent ones use timeout
            # to detect end of operation or device hang.
            timeout = None if idx == 0 else IDLE_TIMEOUT_S

            em_wave = _capture_once(
                lscope,
                ch=TARGET_CH,
                ch_trig=TARGET_CH_TRIG,
                expected_len=samples if samples else None,
                bit_depth=BIT_DEPTH,
                idle_timeout_s=timeout,
            )

            if em_wave is None:
                _ERR(f"\n[INFO] Timeout reached at idx={idx}. Stopping capture loop.")
                break

            traces.append(em_wave.copy())
            bar.update(1)
            bar.set_postfix_str(f"idx={idx}, status=captured")

        bar.close()

        if not traces:
            _ERR("[WARN] No traces captured. Exiting without save.")
            return

        # --- Save Data ---
        # Stack to (N_traces, N_samples)
        traces_arr = np.vstack(traces)
        np.save(OUT_PATH, traces_arr)
        print(f"[INFO] Captured {len(traces)} traces, shape={traces_arr.shape}")
        print(f"[INFO] Saved to {OUT_PATH}")

    except KeyboardInterrupt:
        _ERR("\n[INFO] Interrupted by user.")
    except Exception as e:
        _ERR(f"\n[FATAL] Unexpected error: {e!r}")
    finally:
        try:
            lscope.disconnect()
            print("[INFO] Scope disconnected.")
        except Exception:
            pass


if __name__ == "__main__":
    main()