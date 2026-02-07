import torch
import time
from chop.models import get_model
from chop.dataset import get_dataset_info

'''
Returns result of the function and time in ?
'''
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
    return torch.randn(128, 3, 224, 224)

'''
Modified to return an array of individual execution times
'''
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
    return avg_time, times