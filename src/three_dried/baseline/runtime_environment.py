"""Fresh process/hardware provenance for each benchmark invocation."""
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

import numpy as np
import torch


def capture_runtime_environment(device, nvcc='/usr/local/cuda/bin/nvcc'):
    def command(args):
        try:
            result=subprocess.run(args,capture_output=True,text=True,check=False)
            return {'command':args,'returncode':result.returncode,
                    'stdout':result.stdout,'stderr':result.stderr}
        except OSError as error:
            return {'command':args,'error':str(error)}
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')),'unknown')
    selected=torch.device(device)
    cuda=selected.type=='cuda' and torch.cuda.is_available()
    return dict(capture_id=str(uuid.uuid4()),captured_at_utc=datetime.now(timezone.utc).isoformat(),
        environment_source='live_process_and_hardware',pid=os.getpid(),hostname=platform.node(),
        python_executable=sys.executable,python=sys.version,numpy=np.__version__,torch=torch.__version__,
        platform=platform.platform(),cpu=cpu,device=str(selected),
        cuda_runtime=torch.version.cuda,cuda_device_count=torch.cuda.device_count(),
        gpu=torch.cuda.get_device_name(selected) if cuda else None,
        gpu_capability=list(torch.cuda.get_device_capability(selected)) if cuda else None,
        nvidia_smi=command(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used',
                            '--format=csv,noheader']),nvcc=command([nvcc,'--version']))
