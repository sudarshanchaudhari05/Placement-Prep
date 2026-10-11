"""
DataLoader Multiprocessing & Throughput Benchmark Script
=======================================================
Benchmarks:
- Config 1: workers=0, pin_memory=False, persistent=False, prefetch=None (Baseline)
- Config 2: workers=2, pin_memory=True, persistent=True, prefetch=2
- Config 3: workers=4, pin_memory=True, persistent=True, prefetch=2

Measures:
- Warmup time (spawn & first batch)
- Steady-state batch throughput (samples/sec and ms/batch)
- RAM usage (RSS before and during execution)
- Tensor & label equality across configurations
"""

import os
import sys
import time
import json
from pathlib import Path
import psutil
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from src.data.dataset import RowAnchorDataset, build_dataloader


def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def benchmark():
    print("=" * 75)
    print("PHASE 4C-G: DATALOADER PERFORMANCE & MULTIPROCESSING BENCHMARK")
    print("=" * 75)

    cfg = load_config()
    train_ds = RowAnchorDataset("data/synthetic/train/annotations.json", config=cfg)
    total_samples = len(train_ds)
    batch_size = 32
    num_eval_batches = 10
    warmup_batches = 2

    print(f"Dataset: {total_samples} samples | Batch size: {batch_size}")
    base_ram = get_ram_mb()
    print(f"Initial process RAM: {base_ram:.1f} MB\n")

    configs = [
        {"name": "workers=0 (Baseline)", "num_workers": 0, "pin_memory": False, "persistent": False, "prefetch": None},
        {"name": "workers=2 + pinned + persistent", "num_workers": 2, "pin_memory": True, "persistent": True, "prefetch": 2},
        {"name": "workers=4 + pinned + persistent", "num_workers": 4, "pin_memory": True, "persistent": True, "prefetch": 2},
    ]

    results = []
    first_batch_tensors = {}

    for c in configs:
        print(f"Testing configuration: {c['name']} ...")
        t_start = time.time()
        ram_before = get_ram_mb()

        # Fixed generator seed to guarantee identical sample ordering
        g = torch.Generator()
        g.manual_seed(42)

        loader = build_dataloader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,  # deterministic order for comparison
            num_workers=c["num_workers"],
            pin_memory=c["pin_memory"],
            persistent_workers=c["persistent"],
            prefetch_factor=c["prefetch"],
            generator=g,
        )

        # Warmup
        t_warm_start = time.time()
        loader_iter = iter(loader)
        first_batch = None
        for i in range(warmup_batches):
            b = next(loader_iter)
            if i == 0:
                first_batch = b
        warmup_time = time.time() - t_warm_start

        # Steady-state timing
        t_bench_start = time.time()
        for _ in range(num_eval_batches):
            _ = next(loader_iter)
        steady_time = time.time() - t_bench_start

        peak_ram = get_ram_mb()
        total_time = time.time() - t_start

        ms_per_batch = (steady_time / num_eval_batches) * 1000.0
        samples_per_sec = (num_eval_batches * batch_size) / steady_time
        est_epoch_time_sec = (total_samples / batch_size) * (steady_time / num_eval_batches)

        first_batch_tensors[c["name"]] = {
            "image": first_batch["image"].clone(),
            "targets": first_batch["targets"].clone(),
        }

        res = {
            "name": c["name"],
            "num_workers": c["num_workers"],
            "pin_memory": c["pin_memory"],
            "persistent_workers": c["persistent"],
            "prefetch_factor": c["prefetch"],
            "warmup_sec": warmup_time,
            "steady_batch_ms": ms_per_batch,
            "samples_per_sec": samples_per_sec,
            "estimated_epoch_sec": est_epoch_time_sec,
            "ram_increase_mb": peak_ram - ram_before,
            "peak_ram_mb": peak_ram,
        }
        results.append(res)
        print(f"  Warmup (startup + {warmup_batches} batches): {warmup_time:.2f} s")
        print(f"  Steady-state throughput: {samples_per_sec:.1f} samples/s ({ms_per_batch:.1f} ms/batch)")
        print(f"  Estimated 4,200-sample epoch I/O duration: {est_epoch_time_sec:.1f} s ({est_epoch_time_sec/60:.2f} min)")
        print(f"  Process RAM: {peak_ram:.1f} MB (Delta: +{peak_ram - ram_before:.1f} MB)\n")

    # Verify tensor equality across all configurations
    ref_name = configs[0]["name"]
    ref_img = first_batch_tensors[ref_name]["image"]
    ref_tgt = first_batch_tensors[ref_name]["targets"]

    for c in configs[1:]:
        c_name = c["name"]
        test_img = first_batch_tensors[c_name]["image"]
        test_tgt = first_batch_tensors[c_name]["targets"]

        img_match = torch.equal(ref_img, test_img)
        tgt_match = torch.equal(ref_tgt, test_tgt)
        assert img_match, f"Image tensor mismatch between {ref_name} and {c_name}!"
        assert tgt_match, f"Target tensor mismatch between {ref_name} and {c_name}!"
        print(f"Verified: Tensors and labels for {c_name} match {ref_name} 100% bitwise.")

    # Save results to experiments/
    out_path = PROJECT_ROOT / "experiments" / "dataloader_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to: {out_path}")


if __name__ == "__main__":
    benchmark()
