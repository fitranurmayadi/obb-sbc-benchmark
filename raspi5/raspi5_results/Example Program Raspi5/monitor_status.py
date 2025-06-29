import psutil
import time
import subprocess

# Fungsi baca suhu CPU
def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return -1.0

# Fungsi baca SoC temp dari output `sensors`
def read_soc_temp():
    try:
        output = subprocess.check_output(["sensors"]).decode()
        lines = output.splitlines()
        in_rp1_section = False
        for line in lines:
            if "rp1_adc" in line.lower():
                in_rp1_section = True
            elif in_rp1_section and "temp1" in line.lower():
                temp_str = line.split('+')[1].split('°')[0]
                return float(temp_str.strip())
    except Exception:
        return -1.0
    return -1.0

print("\n📟 Monitoring Raspberry Pi 5 (Tekan CTRL+C untuk berhenti)\n")

try:
    while True:
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        cpu_temp = read_cpu_temp()
        soc_temp = read_soc_temp()

        print(f"[STATUS]")
        print(f"  CPU Usage : {cpu_usage:.1f}%")
        print(f"  RAM Usage : {ram_usage:.1f}%")
        print(f"  CPU Temp  : {cpu_temp:.1f}°C")
        print(f"  SoC Temp  : {soc_temp:.1f}°C")
        print(f"  GPU Usage : -")
        print(f"  GPU Temp  : -")
        print(f"  NPU Temp  : -")
        print("-" * 40)

        time.sleep(1)

except KeyboardInterrupt:
    print("⏹️ Dihentikan.")
