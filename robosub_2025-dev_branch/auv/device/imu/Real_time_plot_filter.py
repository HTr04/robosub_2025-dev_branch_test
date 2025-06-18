import time
import threading
from serial import Serial
import numpy as np
from scipy.signal import iirdesign, sosfiltfilt
import matplotlib.pyplot as plt
from datetime import datetime
import csv

# === FILTER SETTINGS ===
SAMPLE_RATE = 40
FP = 5
FSB = 7.5
GPASS = 1
GSTOP = 60

# --- MATLAB-equivalent Butterworth filter ---
wp = FP / (SAMPLE_RATE / 2)
ws = FSB / (SAMPLE_RATE / 2)
sos = iirdesign(wp, ws, GPASS, GSTOP, ftype='butter', output='sos')

class VN100:
    def __init__(self, port: str = "COM3"):
        self.__port = port
        self.__bps = 115200
        self.__ser = Serial(port=self.__port, baudrate=self.__bps, timeout=1)
        self.yaw_list = []
        self.pitch_list = []
        self.roll_list = []
        self.time_list = []
        self.iso_time_list = []
        self.lock = threading.Lock()
        self.read_thread = threading.Thread(target=self.read, daemon=True)
        self.read_thread.start()
        time.sleep(2)

    def read(self):
        while True:
            time.sleep(1 / SAMPLE_RATE)
            try:
                data_line = self.__ser.readline().decode()
                data_list = data_line.split(',')
                yaw = (float(data_list[1]) + 90) % 360
                pitch = float(data_list[3])
                roll = float(data_list[2])
                now = time.time()
                now_iso = datetime.now().isoformat()
                with self.lock:
                    self.yaw_list.append(yaw)
                    self.pitch_list.append(pitch)
                    self.roll_list.append(roll)
                    self.time_list.append(now)
                    self.iso_time_list.append(now_iso)
            except Exception:
                pass

    def get_data(self):
        with self.lock:
            return (
                np.array(self.yaw_list),
                np.array(self.pitch_list),
                np.array(self.roll_list),
                np.array(self.time_list),
                list(self.iso_time_list)
            )

if __name__ == "__main__":
    sensor = VN100()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_{timestamp}.csv"

    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    lines = []
    labels = ['Yaw', 'Pitch', 'Roll']
    for ax, label in zip(axs, labels):
        ax.set_ylabel(label)
        ax.grid(True)
        l1, = ax.plot([], [], 'k-', label='Raw')
        l2, = ax.plot([], [], 'r-', label='Filtered')
        lines.append((l1, l2))
        ax.legend()
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Live Yaw, Pitch, Roll (ALL DATA): Raw vs Filtered')

    print("Recording & plotting... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1 / SAMPLE_RATE)
            raw_yaw, raw_pitch, raw_roll, t_buf, iso_buf = sensor.get_data()
            if len(t_buf) < 7:
                continue  # Wait until enough data for filter

            t_plot = t_buf - t_buf[0]

            # Apply filter to full buffers (like MATLAB "filtfilt")
            filt_yaw = sosfiltfilt(sos, raw_yaw)
            filt_pitch = sosfiltfilt(sos, raw_pitch)
            filt_roll = sosfiltfilt(sos, raw_roll)

            # Update plots
            for i, (l1, l2) in enumerate(lines):
                l1.set_data(t_plot, [raw_yaw, raw_pitch, raw_roll][i])
                l2.set_data(t_plot, [filt_yaw, filt_pitch, filt_roll][i])
                axs[i].relim()
                axs[i].autoscale_view()
            axs[-1].set_xlim([0, t_plot[-1]])
            fig.canvas.flush_events()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nExiting and saving data to CSV...")

        # Final filter on all data (to ensure matching CSV output)
        raw_yaw, raw_pitch, raw_roll, t_buf, iso_buf = sensor.get_data()
        filt_yaw = sosfiltfilt(sos, raw_yaw)
        filt_pitch = sosfiltfilt(sos, raw_pitch)
        filt_roll = sosfiltfilt(sos, raw_roll)

        with open(filename, mode="w", newline="") as csvfile:
            fieldnames = ["timestamp", "Raw_Yaw", "Raw_Pitch", "Raw_Roll",
                          "Filt_Yaw", "Filt_Pitch", "Filt_Roll"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(iso_buf)):
                writer.writerow({
                    "timestamp": iso_buf[i],  # Keep in ISO format for uniformity
                    "Raw_Yaw": raw_yaw[i],
                    "Raw_Pitch": raw_pitch[i],
                    "Raw_Roll": raw_roll[i],
                    "Filt_Yaw": filt_yaw[i],
                    "Filt_Pitch": filt_pitch[i],
                    "Filt_Roll": filt_roll[i]
                })
        print(f"Data saved to {filename}")

    except Exception as e:
        print(f"Generic exception caught: {e}")
