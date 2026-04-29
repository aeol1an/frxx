import enum

class Backend(enum.Enum):
    TORCH_CUDA = "torch_cuda"
    TORCH_MPS = "torch_mps"
    NUMBA = "numba"

def detect_backend(prefer: str | None = None):
    """
    Determine the best available backend.
    
    prefer: None, "numba"
        - None: use torch+GPU if available, else numba
        - "numba": force numba regardless of GPU availability
    """
    if prefer == "numba":
        return Backend.NUMBA

    torch_available = False
    try:
        import torch
        torch_available = True
    except ImportError:
        pass

    if not torch_available:
        return Backend.NUMBA

    if prefer is None or prefer == "torch":
        import torch

        if torch.cuda.is_available():
            return Backend.TORCH_CUDA
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return Backend.TORCH_MPS
        else:
            # auto: no GPU, just use numba since it's faster on CPU for loopy code
            return Backend.NUMBA

    return Backend.NUMBA