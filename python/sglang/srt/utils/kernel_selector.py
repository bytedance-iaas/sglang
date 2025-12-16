import os
import importlib
import warnings
import pkgutil
from typing import List, Optional


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
            print("target_str {}".format(target_str))
            f.write(target_str + '\n')

        if os.name == 'posix':
            fcntl.flock(f, fcntl.LOCK_UN)
        elif os.name == 'nt':
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


class KernelSelector:
    def __init__(self):
        self.all_kernel_data = {}
        self.import_success = False
        
        # 1. Load environment variable
        # data_path = os.getenv("kernel_select_data_path")
        
        # 2. Try to import the module
        try:
            # We assume the module 'kernel_select_data' is in the python path
            # (e.g., the directory containing it is in sys.path)
            module = importlib.import_module("kernel_select_data")
            self.import_success = True
            
            # 3. Iterate over submodules in the package
            # pkgutil.iter_modules looks at the __path__ of the imported module
            if hasattr(module, "__path__"):
                for loader, module_name, is_pkg in pkgutil.iter_modules(module.__path__):
                    # We only care about .py files (not sub-packages)
                    if not is_pkg:
                        full_module_name = f"kernel_select_data.{module_name}"
                        submod = importlib.import_module(full_module_name)
                        
                        # Get the 'kernel_data' dict from the submodule
                        if hasattr(submod, "kernel_data"):
                            self.all_kernel_data[module_name] = getattr(submod, "kernel_data")
            
            print("collect all data is {}".format(self.all_kernel_data))
        except (ImportError, ModuleNotFoundError) as e:
            warnings.warn(f"Failed to import 'kernel_select_data': {e}. Kernel selection will use defaults.")
            self.import_success = False

    def query_kernel_data(self, device_type: str, shape: List[int], data_type: str, op_type: str, run_in_graph: bool, **kwargs) -> str:
        """
        Queries the loaded kernel data based on device, shape, and operation type.
        """
        if not self.import_success:
            return "default"
        
        # Create shape string: "128_128_128"
        shape_str = "_".join(map(str, shape))
        
        # Create first-level key: "A100_FP32_MATMUL"
        # Join: device_type, data_type, op_type
        level1_key = f"{device_type}_{data_type}_{op_type}_graphmode_{run_in_graph}"
        
        # Search in loaded data
        # Each 'module_name' in all_kernel_data acts as the level1_key 
        # based on the file-naming convention.
        if level1_key not in self.all_kernel_data:
            print("can not find lv1 key {}".format(level1_key))
        
        if level1_key in self.all_kernel_data:
            specific_dict = self.all_kernel_data[level1_key]
            # return specific_dict.get(shape_str, "default")
            if not (shape_str in specific_dict):
                print("can not find shape {}".format(shape_str))
            else:
                return specific_dict[shape_str]
        
        return "default"

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