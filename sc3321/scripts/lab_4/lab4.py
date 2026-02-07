import torch
import time
from chop.models import get_model
from chop.dataset import get_dataset_info

import csv
from pathlib import Path



OUT_DIR = Path("mase/sc3321/outputs/lab_4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "lab4_torch_compile_results.csv"

with OUT_CSV.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "device",
        "n_iterations",
        "baseline_time_s",
        "optimized_time_s",
    ])


def timed_gpu(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def timed_cpu(fn):
    start = time.time()
    result = fn()
    return result, time.time() - start


def get_data():
    return torch.randn(length, 3, 224, 224)


def time_model(fn, n=1000, device="cpu"):
    times = []
    data = get_data().to(device)
    for _ in range(n):
        if device == "cpu":
            _, t = timed_cpu(lambda: fn(data.cpu()))
        else:
            _, t = timed_gpu(lambda: fn(data))
        times.append(t)
    avg_time = sum(times) / len(times)
    return avg_time


cifar10_info = get_dataset_info("imagenet")
model = get_model("resnet18", pretrained=True, dataset_info=cifar10_info)
image = torch.randn(64, 3, 224, 224)

opt_model = torch.compile(model)


devices = ["cuda", "cpu"]
num_its = [1, 10, 25]

for device in devices:
    if device == "cuda" and not torch.cuda.is_available():
        continue    
    for n in num_its:

        model.to(device)
        opt_model.to(device)
        avg_t = time_model(model, n=n, device=device)
        opt_avg_t = time_model(opt_model, n=n, device=device)
        print(f"[{device}] n={n} baseline={avg_t:.4f}s optimized={opt_avg_t:.4f}s")

        with OUT_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                device,
                n,
                avg_t,
                opt_avg_t,
            ]) 



### Task 2 SDPA

import math
import torch.nn.functional as F


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, query, key, value):
        scale_factor = 1 / math.sqrt(query.size(-1))
        score = query @ key.transpose(-2, -1) * scale_factor
        attn = F.softmax(score, -1)
        context = attn @ value
        return context


class ScaledDotProductAttentionFused(torch.nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value)


sequence_lengths = [64, 128, 256, 512]
for device in devices:
    for length in sequence_lengths:

		query = torch.ones(32, 8, length, 64, dtype=torch.float16, device=device)
		key = torch.ones(32, 8, length, 64, dtype=torch.float16, device=device)
		value = torch.ones(32, 8, length, 64, dtype=torch.float16, device=device)

		y1 = ScaledDotProductAttention()(query, key, value)
		y2 = ScaledDotProductAttentionFused()(query, key, value)
		print(y1[0, 0, 0, 0], y2[0, 0, 0, 0])







