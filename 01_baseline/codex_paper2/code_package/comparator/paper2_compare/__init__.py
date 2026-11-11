"""Prospectively specified finite-frame comparisons; execution needs approval."""
from pathlib import Path
import sys

# Preserve original numerical source byte-for-byte with its absolute module
# imports. All modules resolve within the self-contained vendored directory.
VENDOR = Path(__file__).resolve().parent / 'vendor'
sys.path.insert(0, str(VENDOR))
__version__ = '1.0.0'
