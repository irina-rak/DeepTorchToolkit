from __future__ import annotations

import sys
from pathlib import Path

# Ensure 'src' is on sys.path for imports without installing the package
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
