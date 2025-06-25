import logging
import subprocess
import time
import threading
from prometheus_client import Gauge, start_http_server

# Setup logging to a file called script.log
LOG_FILE = "script.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Disk-related metrics from iostat
io_read_rate = Gauge('io_read_rate', 'Disk read rate (reads per second)', ['device'])
io_write_rate = Gauge('io_write_rate', 'Disk write rate (writes per second)', ['device'])
io_tps = Gauge('io_tps', 'Disk transactions per second', ['device'])
io_read_bytes = Gauge('io_read_bytes', 'Disk read bytes per second', ['device'])
io_write_bytes = Gauge('io_write_bytes', 'Disk write bytes per second', ['device'])

# CPU usage metrics (skipping 'steal')
cpu_modes = ['user', 'nice', 'system', 'iowait', 'idle']
cpu_metric = Gauge('cpu_avg_percent', 'CPU average percentage', ['mode'])

# Memory metrics (dynamically created)
meminfo_metrics = {}


def collect_iostat_metrics():
    """Collects disk and CPU stats from iostat and updates Prometheus gauges"""
    while True:
        try:
            # Run iostat command and parse output
            result = subprocess.run(['iostat'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout
            lines = output.splitlines()

            cpu_header_index = None
            device_header_index = None

            # Locate section headers
            for i, line in enumerate(lines):
                if line.startswith("avg-cpu:"):
                    cpu_header_index = i
                if line.startswith("Device"):
                    device_header_index = i

            # Process CPU usage metrics
            if cpu_header_index is not None:
                cpu_values = lines[cpu_header_index + 1].split()
                for i, mode in enumerate(cpu_modes):
                    # Skip 'steal' without breaking the order
                    if mode != "idle":
                        cpu_metric.labels(mode=mode).set(float(cpu_values[i]))
                    else:
                        cpu_metric.labels(mode=mode).set(float(cpu_values[i + 1]))

            # Process device-specific metrics
            if device_header_index is not None:
                for device_line in lines[device_header_index + 1:]:
                    if device_line.strip() == '':
                        continue  # Skip empty lines
                    parts = device_line.split()
                    device_name = parts[0]
                    tps = float(parts[1])
                    read_rate = float(parts[2])
                    write_rate = float(parts[3])
                    read_bytes = float(parts[5])
                    write_bytes = float(parts[6])

                    # Update Prometheus gauges for each device
                    io_tps.labels(device=device_name).set(tps)
                    io_read_rate.labels(device=device_name).set(read_rate)
                    io_write_rate.labels(device=device_name).set(write_rate)
                    io_read_bytes.labels(device=device_name).set(read_bytes)
                    io_write_bytes.labels(device=device_name).set(write_bytes)

            logging.info("iostat metrics collected successfully")

        except Exception as e:
            logging.error(f"Error collecting iostat metrics: {e}")

        time.sleep(1)  # Collect every second


def normalize_meminfo_key(key):
    """Convert meminfo keys into Prometheus-friendly metric names"""

    # Manual rename for selected Mem fields
    manual_map = {
        'MemFree': 'free_memory',
        'MemTotal': 'total_memory',
        'MemAvailable': 'available_memory'
    }

    if key in manual_map:
        new_key = manual_map[key]
    else:
        new_key = key

    # Replace '(' and ')' with underscores
    new_key = new_key.replace('(', '_').replace(')', '')

    # Insert underscores before internal uppercase letters or digits
    converted_key = new_key[0]
    for i in range(1, len(new_key)):
        if ((new_key[i].isupper() and i < len(new_key) - 1 and new_key[i + 1].islower()) or
            (new_key[i].isdigit() and new_key[i - 1].isalpha())) and new_key[i - 1] != '_':
            converted_key += '_' + new_key[i]
        else:
            converted_key += new_key[i]

    converted_key = converted_key.lower()

    return 'meminfo_' + converted_key


def collect_meminfo_metrics():
    """Reads /proc/meminfo and exports all fields as Prometheus metrics"""
    global meminfo_metrics

    while True:
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.split()
                    key = parts[0].rstrip(':')
                    value = float(parts[1])  # All values are in kB

                    metric_name = normalize_meminfo_key(key)

                    # Create a new gauge if it doesn't exist yet
                    if metric_name not in meminfo_metrics:
                        meminfo_metrics[metric_name] = Gauge(metric_name, f"Metric from /proc/meminfo: {key}")

                    # Update the metric value
                    meminfo_metrics[metric_name].set(value)

            logging.info("meminfo metrics collected successfully")

        except Exception as e:
            logging.error(f"Error collecting meminfo metrics: {e}")

        time.sleep(1)  # Collect every second


if __name__ == "__main__":
    try:
        # Start Prometheus exporter on port 18000
        start_http_server(18000)
        logging.info("Prometheus exporter successfully running on port 18000")

        # Start background threads for iostat and meminfo collection
        threading.Thread(target=collect_iostat_metrics, daemon=True).start()
        threading.Thread(target=collect_meminfo_metrics, daemon=True).start()

        # Keep the main thread alive to allow KeyboardInterrupt
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nPrometheus exporter script stopped gracefully")
        logging.info("Prometheus exporter script stopped gracefully")

    except Exception as e:
        logging.error(f"Prometheus exporter did not run as expected: {e}")
