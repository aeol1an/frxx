import threading
import importlib
import importlib.util

from . import Backend

class _LazyModule:
    def __init__(self, name):
        self._name = name
        self._mod = None
        self._found = importlib.util.find_spec(name) is not None
        self._event = threading.Event()
        if self._found:
            threading.Thread(target=self._bg_load, daemon=True).start()
        else:
            self._event.set()

    def _bg_load(self):
        try:
            self._mod = importlib.import_module(self._name)
        except ImportError:
            self._found = False
        self._event.set()

    @property
    def available(self):
        """Check if the module is available (blocks until load attempt finishes)."""
        self._event.wait()
        return self._mod is not None

    def __getattr__(self, attr):
        self._event.wait()
        if self._mod is None:
            raise ImportError(f"No module named '{self._name}'")
        return getattr(self._mod, attr)
    
_torch = _LazyModule("torch")

def detect_backend(prefer: str | None = None):
    if prefer is not None:
        prefer = prefer.lower()

    if prefer == "cpu":
        return Backend.CPU

    if not _torch.available:
        return Backend.CPU

    if _torch.cuda.is_available():
        return Backend.TORCH_CUDA
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        return Backend.TORCH_MPS
    else:
        return Backend.CPU
