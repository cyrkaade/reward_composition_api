import os
import sys
from pathlib import Path

# On this Windows/conda box, torch and numpy-MKL each link their own copy of
# libiomp5md.dll, and the second one to initialise aborts the process:
#     OMP: Error #15: Initializing libiomp5md.dll, but found it already
#     initialized.  ->  Fatal Python error: Aborted
# It is not specific to this project -- `import torch; np.ones((4,4)) @ ...`
# reproduces it on its own -- but the STARC tests are the first in the suite to
# run a numpy matmul after torch has loaded, so they are where it surfaces.
#
# MKL reads this only at load time, so the setdefault below wins the race when
# conftest is imported before numpy (running a single test file) and loses it
# when a pytest plugin got there first.  For the whole suite, export it in the
# shell:  MKL_THREADING_LAYER=SEQUENTIAL python -m pytest tests/ -q
# Linux (Triton) is unaffected.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))
