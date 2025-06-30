import psutil
import time
import glob

def read_temp_zones():
    temps = {}
    zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
    for z in zones:
        try:
            with open(z, "r") as f:
                temp = int(f.read().strip()) / 1000
            with open(z.replace("temp", "type"), "r") as f:
                label = f.read().strip()
            temps[label] = temp
        except:
            continue
    return temps

print("🧪 Mulai monitoring Jetson Orin Nano (Ctrl+C untuk keluar)")
try:
    while True:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        temps = read_temp_zones()

        print("\n[⚙️  CPU]:", f"{cpu:.1f}%")
        print("[🧠 RAM]:", f"{ram:.1f}%")
        print("[🌡️  Temperature]")
        for k, v in temps.items():
            print(f"  {k}: {v:.1f}°C")
        time.sleep(2)
except KeyboardInterrupt:
    print("✅ Monitoring dihentikan.")

