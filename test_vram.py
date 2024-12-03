import io
from flux_pipeline import FluxPipeline
import time
from threading import Thread, Event
import torch

class VRamMonitor:
    def __init__(self):
        self._stop_flag = Event()
        self._vram_usage = 0
        self._thread = Thread(target=self._monitor)
        self._thread.start()
    
    def _monitor(self):
        while not self._stop_flag.is_set():
            if torch.cuda.is_available():
                # Track both allocated and reserved memory
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                current_usage = max(allocated, reserved)
                self._vram_usage = max(self._vram_usage, current_usage)
            self._stop_flag.wait(0.001)  # 1ms sampling rate for better accuracy
    
    def complete(self) -> int:
        self._stop_flag.set()
        self._thread.join()
        return self._vram_usage

def format_memory(bytes_value: int) -> str:
    return f"{bytes_value / 1024 / 1024:.2f} MB ({bytes_value / 1024 / 1024 / 1024:.2f} GB)"

# Start VRAM monitoring before loading
print("Starting model load...")
vram_monitor = VRamMonitor()

# Load pipeline
start_load = time.time()
pipe = FluxPipeline.load_pipeline_from_config_path(
    "configs/config-schnell-cuda0.json"
)
load_time = time.time() - start_load

# Get peak VRAM during loading
peak_load_vram = vram_monitor.complete()
print(f"Model loading time: {load_time:.2f} seconds")
print(f"Peak VRAM during loading: {format_memory(peak_load_vram)}")

# Start new monitoring for inference
print("\nStarting inference...")
vram_monitor = VRamMonitor()

# Run inference
start = time.time()
torch.cuda.reset_peak_memory_stats()

output_jpeg_bytes: io.BytesIO = pipe.generate(
    prompt="a bird",
    width=1024,
    height=1024,
    num_steps=4,
    guidance=0.0,
    seed=2203,
    strength=0.8,
)
infer_time = time.time() - start

# Get peak VRAM during inference
peak_infer_vram = vram_monitor.complete()
print(f"Inference time: {infer_time:.2f} seconds")
print(f"Peak VRAM during inference: {format_memory(peak_infer_vram)}")

# Save output
with open("output.jpg", "wb") as f:
    f.write(output_jpeg_bytes.getvalue())

# Print comparison
print(f"\nVRAM usage difference:")
print(f"Loading vs Inference: {format_memory(abs(peak_load_vram - peak_infer_vram))}")

# Clean up CUDA memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()