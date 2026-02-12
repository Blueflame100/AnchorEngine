"""Pytest configuration: force mock LLM before any app imports."""

import os
import sys
from pathlib import Path

# Must run before any imports that touch GrokClient or the registry
os.environ["USE_MOCK_LLM"] = "true"

# Ensure repo root is on path for "from src.main import app"
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
