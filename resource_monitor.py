import psutil
import GPUtil
import time

def get_cpu_memory_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    return cpu_percent, memory

def get_gpu_memory_usage():
    gpu_info = GPUtil.getGPUs()
    if gpu_info:
        gpu = gpu_info[0]
        gpu_percent = gpu.memoryUsed/gpu.memoryTotal*100
        return gpu_percent, gpu.memoryUsed, gpu.memoryTotal
    else:
        return None

def main():
    try:
        while True:
            cpu_percent, memory = get_cpu_memory_usage()
            gpu_memory,gpu_used,gpu_total = get_gpu_memory_usage()

            print(f"CPU Usage: {cpu_percent}%")
            print(f"Memory Usage: {memory.percent}%")
            
            if gpu_memory is not None:
                print(f"GPU Memory Usage: {gpu_memory} %")
                print(f"GPU Memory Used: {gpu_used}/{gpu_total} MB")

            print("-" * 40)
            time.sleep(1)

    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    main()
