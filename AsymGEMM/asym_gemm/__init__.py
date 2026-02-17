import os
import subprocess
import torch
from torch.version import cuda as cuda_version
from packaging import version

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Configs
from . import _C
from ._C import (
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
)

if version.parse(cuda_version) >= version.parse('12.1'):
    def _maybe_import_from_C(names):
        for name in names:
            if hasattr(_C, name):
                globals()[name] = getattr(_C, name)

    def _missing_kernel(kernel_name):
        def _raise_missing(*args, **kwargs):
            raise RuntimeError(
                f"`{kernel_name}` is not available in this build of asym_gemm. "
                "Rebuild with matching CUDA/architecture flags to enable this kernel."
            )
        return _raise_missing

    def _export_kernel_alias(alias_name, target_name):
        globals()[alias_name] = globals().get(target_name, _missing_kernel(target_name))

    # DeepGEMM Kernels (may vary by build flags / arch)
    _maybe_import_from_C([
        # FP8 GEMMs
        "m_grouped_fp8_asym_gemm_nt_masked",
        "m_grouped_fp8_asym_gemm_nt_contiguous",
        # BF16 GEMMs
        "m_grouped_bf16_asym_gemm_nt_contiguous",
        "m_grouped_bf16_gemm_nt_contiguous",
        "m_grouped_bf16_asym_gemm_nt_masked",
        # Einsum kernels
        "einsum",
        "fp8_einsum",
        # Attention kernels
        "fp8_mqa_logits",
        "get_paged_mqa_logits_metadata",
        "fp8_paged_mqa_logits",
        # Layout kernels
        "transform_sf_into_required_layout",
        "get_mk_alignment_for_contiguous_layout",
    ])

    # Some alias for legacy supports
    # TODO: remove these later
    _export_kernel_alias("fp8_m_grouped_asym_gemm_nt_masked", "m_grouped_fp8_asym_gemm_nt_masked")
    _export_kernel_alias("fp8_m_grouped_gemm_nt_masked", "m_grouped_fp8_asym_gemm_nt_masked")
    _export_kernel_alias("bf16_m_grouped_asym_gemm_nt_masked", "m_grouped_bf16_asym_gemm_nt_masked")
    _export_kernel_alias("bf16_m_grouped_gemm_nt_masked", "m_grouped_bf16_asym_gemm_nt_masked")

# Some utils
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels for A100
from . import legacy

# Initialize CPP modules
def _find_cuda_home() -> str:
    # TODO: reuse PyTorch API later
    # For some PyTorch versions, the original `_find_cuda_home` will initialize CUDA, which is incompatible with process forks
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # noinspection PyBroadException
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


_C.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    _find_cuda_home()                           # CUDA home
)

__version__ = '2.2.0'
