"""Host NVML gate and ownership telemetry; no GPU contexts are created."""
import json
import os
from pathlib import Path
import time
import pynvml as nv

def sample():
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    mem = nv.nvmlDeviceGetMemoryInfo(h,version=nv.nvmlMemory_v2)
    return {'timestamp':time.time(),'uuid':nv.nvmlDeviceGetUUID(h),
            'utilization_percent':nv.nvmlDeviceGetUtilizationRates(h).gpu,
            'memory_used_bytes':mem.used,'driver_reserved_bytes':mem.reserved,
            'compute_pids':[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h)],
            'graphics_pids':[p.pid for p in nv.nvmlDeviceGetGraphicsRunningProcesses(h)]}

def gate(path):
    # Established host idle baseline: NVML v2 1,114,112 bytes, nvidia-smi rounds
    # to 2 MiB; no compute/graphics processes, unchanged across idle observations.
    baseline = 1114112
    start = time.monotonic()
    attempts = []
    while True:
        rows = []
        for i in range(3):
            rows.append(sample())
            if i < 2: time.sleep(1)
        attempts.append(rows)
        passed = all(not r['compute_pids'] and not r['graphics_pids'] and r['utilization_percent']==0 and r['memory_used_bytes']==baseline for r in rows)
        result={'status':'passed' if passed else 'waiting','idle_baseline_bytes':baseline,'baseline_evidence':'Host observation before any benchmark CUDA context, no compute or graphics processes; exact stable NVML v2 baseline, driver reservation recorded separately','attempts':attempts}
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(result,indent=2)+'\n')
        if passed:return result
        print('GPU gate waiting: '+json.dumps(rows[-1]),flush=True)
        if time.monotonic()-start>=600:raise RuntimeError('GPU gate unresolved after ten minutes')
        time.sleep(30)

def owned(allowed=()):
    row=sample()
    unexpected=(set(row['compute_pids'])|set(row['graphics_pids']))-set(allowed)-{os.getpid()}
    row['unexpected_pids']=sorted(unexpected)
    if unexpected:raise RuntimeError('Competing GPU processes: '+str(sorted(unexpected)))
    return row
