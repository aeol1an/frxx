from ._backend import detect_backend

import os

# Detect once at import time
BACKEND = detect_backend(prefer=os.getenv("FRXX_BACKEND"))

def get_backend():
    return BACKEND

from . import core, io, proc, viz, utils