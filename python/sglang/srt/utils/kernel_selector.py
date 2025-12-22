import os
import importlib
import warnings
import pkgutil
import fcntl
from typing import List, Optional
from sglang.srt.utils import get_bool_env_var, get_int_env_var

import torch

def generate_level1_key(device_type, op_type, graph_mode, data_type):
    return f"{device_type}_{data_type}_{op_type}_graphmode_{graph_mode}"

def append_string_if_not_exists(filename, target_str):
    if os.name == 'posix':
        import fcntl
    elif os.name == 'nt':
        import msvcrt

    with open(filename, 'a+', encoding='utf-8') as f:
        if os.name == 'posix':
            fcntl.flock(f, fcntl.LOCK_EX)
        elif os.name == 'nt':
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        
        f.seek(0)
        exists = False
        for line in f:
            if line.rstrip('\n') == target_str:
                exists = True
                break
        
        if not exists:
            # print("target_str {}".format(target_str))
            f.write(target_str + '\n')

        if os.name == 'posix':
            fcntl.flock(f, fcntl.LOCK_UN)
        elif os.name == 'nt':
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def append_string_if_not_exists(filename, target_str):
    if os.name == 'posix':
        import fcntl
    elif os.name == 'nt':
        import msvcrt

    with open(filename, 'a+', encoding='utf-8') as f:
        if os.name == 'posix':
            fcntl.flock(f, fcntl.LOCK_EX)
        elif os.name == 'nt':
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        
        f.seek(0)
        exists = False
        for line in f:
            if line.rstrip('\n') == target_str:
                exists = True
                break
        
        if not exists:
            # print("target_str {}".format(target_str))
            f.write(target_str + '\n')

        if os.name == 'posix':
            fcntl.flock(f, fcntl.LOCK_UN)
        elif os.name == 'nt':
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


class KernelSelector:
           
    def __init__(self):
        self.all_kernel_data = {}
        self.file_map = {}  # NEW: Maps level1_key -> file_path
        self.package_dir = None # NEW: Stores the directory of the package
        self.import_success = False
        
        self.launch_data = {}
        self.hit_times = {}
        self.launch_times = {}
        self.unhit_shape = {}
        
        self.acc_map = {}
        self.report_step_num = 1000

        
        # 1. Try to import the main module
        try:
            module = importlib.import_module("kernel_select_data")
            self.import_success = True
            
            # Save the package directory (used for creating NEW files later)
            if hasattr(module, "__path__"):
                self.package_dir = list(module.__path__)[0]

            # 2. Iterate over submodules
            if hasattr(module, "__path__"):
                for loader, module_name, is_pkg in pkgutil.iter_modules(module.__path__):
                    if not is_pkg:
                        full_module_name = f"kernel_select_data.{module_name}"
                        submod = importlib.import_module(full_module_name)
                        
                        if hasattr(submod, "kernel_data"):
                            self.all_kernel_data[module_name] = getattr(submod, "kernel_data")
                            
                            # NEW: Store the file path for this module
                            if hasattr(submod, "__file__") and submod.__file__:
                                # Ensure we point to .py, not .pyc
                                file_path = submod.__file__
                                if file_path.endswith('.pyc'):
                                    file_path = file_path[:-1]
                                self.file_map[module_name] = file_path
            
            print("import kernel_data @ {}".format(self.package_dir))
            print("file_map : {}".format(self.file_map))
            # Debug print
            # print(f"Collected data: {self.all_kernel_data}")
            # print(f"File mapping: {self.file_map}")

        except (ImportError, ModuleNotFoundError) as e:
            warnings.warn(f"Failed to import 'kernel_select_data': {e}. Kernel selection will use defaults.")
            self.import_success = False

    def summary_status(self):
        print("========summary kernel selector status========")
        for op_type in self.launch_times:
            print("---> summary op {} status".format(op_type))
            print("\t\t launch_times {}".format(self.launch_times[op_type]))
            print("\t\t hit times {}".format(self.hit_times[op_type]))
            print("\t\t unhit_shape")
            for s in self.unhit_shape[op_type]:
                graph_mode = False
                if len(s.split("_")) == 4:
                    s.replace("_graph", "")
                    graph_mode = True
                print("\t\t\t {}".format(s))
                dump_file_path =  os.getenv("unhit_shape_dump_file")
                        
                if dump_file_path:
                    if graph_mode:
                        if "False" in dump_file_path:
                            dump_file_path.replace("False", "True")
                    else:
                        if "True" in dump_file_path:
                            dump_file_path.replace("True, False")
                    append_string_if_not_exists(dump_file_path, s)
                    print("dumped to {}".format(dump_file_path))
            print("\t\t kernel_launch_times")
            for kernel in self.launch_data[op_type]:
                print("\t\t\t {} : {} ".format(kernel, self.launch_data[op_type][kernel]))
        
        # for shape_str in self.acc_map:
        #     print("\n=====check shape {} speedup======".format(shape_str))
        #     print("\tabs {} rel {} hit_times {}".format(self.acc_map[shape_str]["abs_acc"], self.acc_map[shape_str]["rel_acc"], self.acc_map[shape_str]["hit_times"]))
                
        print("======summary end======")

    def update_kernel_data(self, key, op_type):
        if not self.import_success:
            return
        
        if key not in self.launch_data[op_type]:
            self.launch_data[op_type][key] = 1
        else:
            self.launch_data[op_type][key]+=1
    
    def query_kernel_data(self, device_type: str, shape: List[int], data_type: str, op_type: str, run_in_graph: bool, **kwargs) -> str:
        if not self.import_success:
            return "default"
        if op_type not in self.launch_times:
            self.launch_times[op_type] = 0
        self.launch_times[op_type]+=1
        
        if op_type not in self.launch_data:
            self.launch_data[op_type] = {}
        
        # level1 : show hit status/ unhit shape / and performance advantage versus default
        if get_int_env_var("SHOW_SELECTOR_STATUS") == 1:
            if self.launch_times[op_type] % self.report_step_num == 0 and self.launch_times[op_type] > 0:
                self.summary_status()
        
       
        if op_type not in self.hit_times:
            self.hit_times[op_type] = 0
        
        if op_type not in self.unhit_shape:
            self.unhit_shape[op_type] = set()
        
        shape_str = "_".join(map(str, shape))
        level1_key = generate_level1_key(device_type, op_type, run_in_graph, data_type)
        
        if level1_key in self.all_kernel_data:
            specific_dict = self.all_kernel_data[level1_key]
            if shape_str in specific_dict:
                self.hit_times[op_type]+=1
                ret =  specific_dict[shape_str]
                kernel_type = ret[-1]["kernel_type"]
                if get_int_env_var("SHOW_SELECTOR_STATUS") == 1 and kernel_type !="default":
                    if shape_str not in self.acc_map:
                        self.acc_map[shape_str] = {}                                            
                        default_letency = 0
                        for data_dic in ret:
                            if data_dic["kernel_type"] == "default":
                                default_letency = data_dic["latency"]
                                break
                        
                        selected_latency = ret[-1]["latency"]
                        abs_accerlate = default_letency - selected_latency
                        relative_accerlate = abs_accerlate / default_letency
                        self.acc_map[shape_str]["abs_acc"] =  abs_accerlate
                        self.acc_map[shape_str]["rel_acc"] = relative_accerlate
                        self.acc_map[shape_str]["hit_times"]= 1
                    else:
                        self.acc_map[shape_str]["hit_times"]+=1
                return ret
            else :
                if run_in_graph:
                    shape_str+="_graph"
                self.unhit_shape[op_type].add(shape_str)
                
                return "default"

    def refresh_kernel_data(self, level1_key: str):
            """
            Updates kernel data with Process Safety:
            1. Acquires an exclusive file lock.
            2. Refreshes in-memory data by reading the latest file content from disk.
            3. Merges the new updates.
            4. Writes back to disk and releases lock.
            """
            if not self.import_success or not self.package_dir:
                print("Cannot update: kernel_select_data module was not loaded successfully.")
                return

            # 1. Determine File Path
            if level1_key in self.file_map:
                file_path = self.file_map[level1_key]
            else:
                file_path = os.path.join(self.package_dir, f"{level1_key}.py")
                self.file_map[level1_key] = file_path

            # 2. Enter Critical Section (File Locking)
            # We open with 'a+' to create the file if it doesn't exist, but allow reading.
            # However, to read efficiently from start, 'r+' is better if file exists.
            mode = 'r+' if os.path.exists(file_path) else 'w+'
            
            with open(file_path, mode) as f:
                # save to read with lock , other process will write with this lock
                fcntl.flock(f, fcntl.LOCK_EX)
                
                try:
                    # 3. REFRESH: Read the LATEST content from disk (ignore import cache)
                    # We cannot rely on 'self.all_kernel_data' or 'importlib' here because 
                    # another process might have just updated the file 1ms ago.
                    f.seek(0)
                    content = f.read()
                    
                    disk_data = {}
                    if content.strip():
                        try:
                            # We use exec() to parse the file content as a dictionary 
                            # without triggering Python's import caching mechanism.
                            local_scope = {}
                            exec(content, {}, local_scope)
                            if "kernel_data" in local_scope:
                                disk_data = local_scope["kernel_data"]
                        except Exception as e:
                            print(f"Warning: Failed to parse existing file {file_path}: {e}")
                            # If parse fails, we proceed with empty disk_data (overwrite risk, but safer than crashing)

                    # 4. MERGE: Disk Data + Current Memory + New Updates
                    if level1_key not in self.all_kernel_data:
                        self.all_kernel_data[level1_key] = {}
                    
                    # Step A: Update our memory with what was actually on disk 
                    # (Syncs us with other processes)
                    self.all_kernel_data[level1_key].update(disk_data)
                    
                finally:
                    # [UNLOCK] Release the lock
                    fcntl.flock(f, fcntl.LOCK_UN)
    # should update background
    def update_all_kernel_datas(self):
        for level_1_key in self.all_kernel_data:
            self.refresh_kernel_data(level_1_key)
# Example Usage:
# if __name__ == "__main__":
#     selector = get_kernel_selector()
#     # result = selector.query_kernel_data("A100", [128, 128, 128], "FP16", "GEMM")
#     # print(result)

# Global variable to hold the singleton instance
_kernel_selector_instance: Optional['KernelSelector'] = None

def get_kernel_selector() -> 'KernelSelector':
    """
    Returns the singleton instance of KernelSelector. 
    Creates it if it doesn't already exist.
    """
    global _kernel_selector_instance
    if _kernel_selector_instance is None:
        _kernel_selector_instance = KernelSelector()
    return _kernel_selector_instance
