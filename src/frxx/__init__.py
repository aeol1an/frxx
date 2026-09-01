import enum
import os

class Backend(enum.Enum):
    TORCH_CUDA = "torch_cuda"
    TORCH_MPS = "torch_mps"
    CPU = "cpu"

from ._backend import detect_backend

BACKEND = Backend.CPU

from . import core, io, proc, viz, utils

if os.getenv("FRXX_BACKEND", "").lower() == "torch":
    BACKEND = detect_backend(prefer="torch")

def get_backend():
    return BACKEND
def set_backend(backend: str):
    global BACKEND
    backend = backend.lower()
    if backend not in ["torch", "cpu"]:
        raise ValueError('Valid backends are "torch" and "cpu".')
    if backend == "cpu":
        BACKEND = Backend.CPU
    else:
        BACKEND = detect_backend(prefer="torch")
