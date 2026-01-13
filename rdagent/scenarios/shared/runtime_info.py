import json
import platform
import subprocess
import sys
from importlib.metadata import distributions


def get_runtime_info():
    return {
        "python_version": sys.version,
        "os": platform.system(),
        "os_release": platform.release(),
    }


def get_gpu_info():
    gpu_info = {}
    try:
        import torch

        if torch.cuda.is_available():
            print("\n=== GPU Info (via PyTorch) ===")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                gpu_name_list = []
                gpu_total_mem_list = []
                gpu_allocated_mem_list = []
                gpu_cached_mem_list = []

                for i in range(torch.cuda.device_count()):
                    gpu_name_list.append(torch.cuda.get_device_name(i))
                    gpu_total_mem_list.append(torch.cuda.get_device_properties(i).total_memory)
                    gpu_allocated_mem_list.append(torch.cuda.memory_allocated(i))
                    gpu_cached_mem_list.append(torch.cuda.memory_reserved(i))

                for i in range(torch.cuda.device_count()):
                    print(f"  - GPU {i}: {gpu_name_list[i]}")
                    print(f"    Total Memory: {gpu_total_mem_list[i] / 1024**3:.2f} GB")
                    print(f"    Allocated Memory: {gpu_allocated_mem_list[i] / 1024**3:.2f} GB")
                    print(f"    Cached Memory: {gpu_cached_mem_list[i] / 1024**3:.2f} GB")
                print("  - All GPUs Summary:")
                print(f"    Total Memory: {sum(gpu_total_mem_list) / 1024**3:.2f} GB")
                print(f"    Total Allocated Memory: {sum(gpu_allocated_mem_list) / 1024**3:.2f} GB")
                print(f"    Total Cached Memory: {sum(gpu_cached_mem_list) / 1024**3:.2f} GB")
            else:
                print("No CUDA GPU detected (PyTorch)!")
        else:
            gpu_info["source"] = "pytorch"
            gpu_info["message"] = "No CUDA GPU detected"
    except ImportError:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                gpu_info["source"] = "nvidia-smi"
                lines = result.stdout.strip().splitlines()
                gpu_info["gpus"] = []
                for line in lines:
                    name, mem_total, mem_used = [x.strip() for x in line.split(",")]
                    gpu_info["gpus"].append(
                        {
                            "name": name,
                            "memory_total_mb": int(mem_total),
                            "memory_used_mb": int(mem_used),
                        }
                    )
            else:
                gpu_info["source"] = "nvidia-smi"
                gpu_info["message"] = "No GPU detected or nvidia-smi not available"
        except FileNotFoundError:
            gpu_info["source"] = "nvidia-smi"
            gpu_info["message"] = "nvidia-smi not installed"
    return gpu_info


if __name__ == "__main__":
    info = {
        "runtime": get_runtime_info(),
        "gpu": get_gpu_info(),
    }
    print(json.dumps(info, indent=4))
