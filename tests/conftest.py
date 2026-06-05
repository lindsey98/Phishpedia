import os
import sys

# Make the repository root and the WEBtool package importable from tests.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "WEBtool"))
