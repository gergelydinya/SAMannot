import psutil
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetName
import tkinter as tk
from visual_widget_utils import *
class SystemMonitor:
    def __init__(self):
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()
        self.device_handles = []
        self.device_names = []
        self.used_ram = 0
        self.total_ram = 0
        self.gpu_util = []
        self.gpu_memory = []
        for idx in range(self.device_count):
            handle = nvmlDeviceGetHandleByIndex(idx)
            self.device_handles.append(handle)
            self.device_names.append(nvmlDeviceGetName(handle))
            self.gpu_memory.append(0)
            self.gpu_util.append(0)
        self.update_system_info()
    def update_system_info(self):
        virtual_memory = psutil.virtual_memory()
        self.used_ram = virtual_memory.used
        self.total_ram = virtual_memory.total
        for idx in range(self.device_count):
            self.gpu_util[idx] = nvmlDeviceGetUtilizationRates(self.device_handles[idx]).gpu
            gpu_mem = nvmlDeviceGetMemoryInfo(self.device_handles[idx])
            self.gpu_memory[idx] = (gpu_mem.used,gpu_mem.total)
    def __del__(self):
        nvmlShutdown()
class SystemMonitorWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(bg="#f5f6f7")
        self.title("System Resources")
        self.info_frame = tk.Frame(self, bg="#f5f5f5")
        self.info_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        self.is_open = True
        # RAM
        ram_frame = tk.Frame(self.info_frame, bg="#f5f5f5")
        ram_frame.grid(row=0, column=0, padx=15)
        ram_label = tk.Label(ram_frame, text="RAM", bg="#f5f5f5", fg="#222222", font=("Segoe UI", 14, "bold"))
        ram_label.pack(pady=(0, 5))
        self.circular_progress_bar_ram = CircularProgress(self.info_frame, size=200)
        self.circular_progress_bar_ram.grid(row=1,column=0, sticky="ew")
        # GPU Util
        gpu_frame = tk.Frame(self.info_frame, bg="#f5f5f5")
        gpu_frame.grid(row=0, column=1, padx=15)
        gpu_label = tk.Label(gpu_frame, text="GPU", bg="#f5f5f5", fg="#222222", font=("Segoe UI", 14, "bold"))
        gpu_label.pack(pady=(0, 5))
        self.circular_progress_bar_gpu = CircularProgress(self.info_frame, size=200)
        self.circular_progress_bar_gpu.grid(row=1,column=1, sticky="ew")
        # VRAM
        vram_frame = tk.Frame(self.info_frame, bg="#f5f5f5")
        vram_frame.grid(row=0, column=2, padx=15)
        vram_label = tk.Label(vram_frame, text="VRAM", bg="#f5f5f5", fg="#222222", font=("Segoe UI", 14, "bold"))
        vram_label.pack(pady=(0, 5))
        self.circular_progress_bar_vram = CircularProgress(self.info_frame, size=200)
        self.circular_progress_bar_vram.grid(row=1,column=2, sticky="ew")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
    def hide(self):
        self.withdraw()
        self.is_open = False
    def show(self):
        self.deiconify()
        self.lift()
        self.focus_force()
        self.is_open = True
    def on_close(self):
        self.hide()