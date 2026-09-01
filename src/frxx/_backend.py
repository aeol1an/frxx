import importlib

from . import Backend

def detect_backend(prefer: str | None = None):
    if prefer is None or prefer.lower() != "torch":
        return Backend.CPU

    print("frxx: importing PyTorch to detect an available accelerator...", flush=True)

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        print("frxx: PyTorch was not found; using the CPU backend.", flush=True)
        return Backend.CPU

    print("frxx: PyTorch imported successfully.", flush=True)

    if torch.cuda.is_available():
        backend = Backend.TORCH_CUDA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backend = Backend.TORCH_MPS
    else:
        backend = Backend.CPU

    print(f"frxx: resolved backend to {backend.value}.", flush=True)
    return backend
