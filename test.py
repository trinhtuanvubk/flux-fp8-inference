import io
from flux_pipeline import FluxPipeline

import time 

start_load = time.time()
pipe = FluxPipeline.load_pipeline_from_config_path(
    "configs/config-schnell-cuda0.json"  # or whatever your config is
)

print(f"loading time: {time.time() - start_load}")
start = time.time()
output_jpeg_bytes: io.BytesIO = pipe.generate(
    # Required args:
    prompt="a bird",
    # Optional args:
    width=1024,
    height=1024,
    num_steps=4,
    guidance=0.0,
    seed=2203,
    strength=0.8,
)
print(f"infer time: {time.time() - start}")
with open("output.jpg", "wb") as f:
    f.write(output_jpeg_bytes.getvalue())
