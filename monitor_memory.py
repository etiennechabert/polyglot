"""
GPU Memory Monitor for Polyglot
Continuously monitors GPU memory usage while the app is running
"""
import time
import subprocess
import sys
from datetime import datetime

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        used, total = result.stdout.strip().split(',')
        return float(used.strip()), float(total.strip())
    except Exception as e:
        print(f"Error reading GPU memory: {e}")
        return None, None

def monitor_memory(interval=2, duration=None):
    """
    Monitor GPU memory usage

    Args:
        interval: Time between measurements in seconds
        duration: Total monitoring duration in seconds (None = indefinite)
    """
    print("=" * 80)
    print("GPU Memory Monitor for Polyglot")
    print("=" * 80)
    print(f"Monitoring interval: {interval}s")
    if duration:
        print(f"Duration: {duration}s")
    else:
        print("Duration: Indefinite (press Ctrl+C to stop)")
    print()
    print(f"{'Timestamp':<20} {'Used (GB)':<12} {'Total (GB)':<12} {'Usage %':<10} {'Delta (MB)':<12}")
    print("-" * 80)

    start_time = time.time()
    previous_used = None
    peak_used = 0

    try:
        while True:
            current_time = time.time()

            # Check if we've exceeded duration
            if duration and (current_time - start_time) >= duration:
                break

            used_mb, total_mb = get_gpu_memory()

            if used_mb is not None:
                used_gb = used_mb / 1024
                total_gb = total_mb / 1024
                usage_pct = (used_mb / total_mb) * 100

                # Track peak
                if used_mb > peak_used:
                    peak_used = used_mb

                # Calculate delta
                if previous_used is not None:
                    delta_mb = used_mb - previous_used
                    delta_str = f"+{delta_mb:.0f}" if delta_mb > 0 else f"{delta_mb:.0f}"
                else:
                    delta_str = "N/A"

                timestamp = datetime.now().strftime("%H:%M:%S")

                # Highlight significant changes (>100MB)
                if previous_used is not None and abs(used_mb - previous_used) > 100:
                    marker = " *** SPIKE ***" if delta_mb > 0 else " *** DROP ***"
                else:
                    marker = ""

                print(f"{timestamp:<20} {used_gb:<12.2f} {total_gb:<12.2f} {usage_pct:<10.1f} {delta_str:<12}{marker}")

                previous_used = used_mb

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Monitoring stopped by user")

    print("=" * 80)
    if peak_used > 0:
        print(f"Peak GPU Memory Usage: {peak_used / 1024:.2f} GB ({peak_used:.0f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    # Default: monitor every 2 seconds indefinitely
    interval = 2
    duration = None

    if len(sys.argv) > 1:
        try:
            interval = float(sys.argv[1])
        except ValueError:
            print(f"Invalid interval: {sys.argv[1]}, using default: 2s")

    if len(sys.argv) > 2:
        try:
            duration = float(sys.argv[2])
        except ValueError:
            print(f"Invalid duration: {sys.argv[2]}, monitoring indefinitely")

    monitor_memory(interval, duration)
