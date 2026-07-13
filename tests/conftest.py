"""Import the inner package without executing the repo-root __init__.py
(a ComfyUI entry point). Run: pytest tests -q --import-mode=importlib"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
