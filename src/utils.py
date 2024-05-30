import torch, pandas as pd, os,  subprocess
from timeit import default_timer as timer
import subprocess
import hashlib
from IPython.display import display
import torch
from pynvml import (nvmlInit,
                    nvmlDeviceGetCount, 
                    nvmlDeviceGetHandleByIndex, 
                    nvmlDeviceGetMemoryInfo, 
                    nvmlDeviceGetName)

class timecode:
    """This class is used for timing code"""
    def __enter__(self):
        self.t0 = timer()
        return self

    def __exit__(self, type, value, traceback):
        self.t = timer() - self.t0


def print_device_info():
    """
    Prints some statistics around versions and the GPU's available for
    the host machine
    """
    import torch
    import sys
    print("######## Diagnostics and version information ######## ")
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION', )
    from subprocess import call
    # call(["nvcc", "--version"]) does not work
    #! nvcc --version
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print ('Available devices ', torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name())
    print ('Current cuda device ', torch.cuda.current_device())
    print("#################################################################")

def round_t(t, dp=2):
    """Return rounded tensors for easy viewing. t is a tensor, dp=decimal places"""
    if t.device.type == "cuda": t=t.cpu()
    return t.detach().numpy().round(dp)

def merge_dicts(d1, d2):
    """Merge the two dicts and return the result. Check first that there is no key overlap."""
    assert set(d1.keys()).isdisjoint(d2.keys())
    return {**d1, **d2}

def display_all(df):
    with pd.option_context("display.max_rows", 3000):
        with pd.option_context("display.max_columns", 1000):
            with pd.option_context("max_colwidth", 480):
                display(df)

def unpack_nested_lists_in_df(df, scalar_cols=[]):
    """Take a df where we have lists stored in the cells and convert it to many rows.
    Put all columns without lists stored in the cells into `scalar_cols`."""
    return df.set_index(scalar_cols).apply(pd.Series.explode).reset_index()

def is_0d_tensor(x): 
    if isinstance(x, torch.Tensor): 
        if x.dim() == 0: return True
    return False


def profile(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        from line_profiler import LineProfiler
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            with open('profile_results.txt', 'w') as f:
                prof.print_stats(stream=f)
            # prof.print_stats()
    return wrapper


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector. 
    Useful when running into an out of memory error on the GPU. """
    import gc
    def pretty_size(size):
        """Pretty prints a torch.Size object"""
        assert(isinstance(size, torch.Size))
        return " × ".join(map(str, size))
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__, 
                                            " GPU" if obj.is_cuda else "",
                                            " pinned" if obj.is_pinned else "",
                                            pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                                                    type(obj.data).__name__, 
                                                    " GPU" if obj.is_cuda else "",
                                                    " pinned" if obj.data.is_pinned else "",
                                                    " grad" if obj.requires_grad else "", 
                                                    " volatile" if obj.volatile else "",
                                                    pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass        
    print("Total size:", total_size)


def print_gpu_tensors():
    #  prints currently alive Tensors and Variables
    import torch
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if str(obj.device) != 'cpu':
                    mem = obj.element_size() * obj.nelement()
                    print(mem, type(obj), obj.size(),  obj.device)
        except:
            pass

def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result):
        return int(result.strip().split('\n')[0])
    
    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')    


def file_is_unchanged(fname): 
    with open(fname, "rb") as f: 
        current_checksum = hashlib.md5(f.read()).hexdigest()    
    checksum_fname = f"./cache/{fname.split('/')[-1]}.checksum"
    if not os.path.exists(checksum_fname):
    # write current_checksum to file and return False
        if not os.path.exists("./cache/"): os.makedirs("./cache/")
        with open(checksum_fname, "x") as f: f.write(current_checksum)
        return False  # file doesn't exist already, easier to do this 
    with open(checksum_fname, "r") as f:  
        old_checksum = f.read()
    if current_checksum == old_checksum: 
        return True
    else: 
        with open(checksum_fname, "w") as f:f.write(current_checksum)
        return False 


def is_t5(x):  return "t5" in x or "T5" in x

def get_least_occupied_gpu():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    most_memory = None
    most_free_gpu = None

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        print(f"Device: cuda:{i} ({nvmlDeviceGetName(handle)})")
        print(f"Total memory: {meminfo.total / 1024**2} MB")
        print(f"Free memory: {meminfo.free / 1024**2} MB")
        print(f"Used memory: {meminfo.used / 1024**2} MB")
        print("-----------------------------")

        if most_memory is None or meminfo.free > most_memory:
            most_memory = meminfo.free
            most_free_gpu = i
    return most_free_gpu