
import subprocess
from io import StringIO
import pandas as pd

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode()), names=['memory.free'], skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = gpu_df['memory.free'].idxmax()
    return idx

if __name__ == "__main__":
    free_gpu_id = get_free_gpu()
    print(free_gpu_id)