# GPU memory and speed of training steps, to choose batch sizes for new architectures:
#   python src/benchmark.py convnext_base:448:16,24 eva_x_base:448:16,32
#
# Each argument is model:image_size:batch sizes. Runs eager (no torch.compile, which usually
# lowers memory) with AMP and channels_last, as in training; reports peak memory and steps/s,
# or OOM. Run with the GPU otherwise idle.

from __future__ import annotations

import argparse
import time

import torch

from models.factory import build_model

BF16 = {"swin_v2_b", "maxvit_t", "vit_b_16", "eva_x_base"}  # transformers train in bfloat16 here


def benchmark(model_name: str, image_size: int, batch_size: int, steps: int = 12) -> str:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model({"model_name": model_name, "pretrained": False, "image_size": image_size, "drop_path": 0.2}, 27)
    model = model.cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dtype = torch.bfloat16 if model_name in BF16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    images = torch.randn(batch_size, 3, image_size, image_size, device="cuda").contiguous(memory_format=torch.channels_last)
    targets = torch.rand(batch_size, 27, device="cuda")
    try:
        for step in range(steps):
            if step == 2:
                torch.cuda.synchronize()
                start = time.time()
            with torch.autocast("cuda", dtype=dtype):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images).float(), targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        rate = (steps - 2) / (time.time() - start)
        return f"{rate:5.2f} steps/s ({rate * batch_size:6.1f} img/s)  peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GiB"
    except torch.OutOfMemoryError:
        return "OOM"
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-step memory and speed per architecture.")
    parser.add_argument("specs", nargs="+", help="model:image_size:batch[,batch...]")
    args = parser.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{torch.cuda.get_device_name(0)}, {total:.1f} GiB")
    for spec in args.specs:
        model_name, image_size, batches = spec.split(":")
        for batch_size in (int(b) for b in batches.split(",")):
            print(f"  {model_name:18} {image_size:>4}px  batch {batch_size:3d}: {benchmark(model_name, int(image_size), batch_size)}")


if __name__ == "__main__":
    main()
