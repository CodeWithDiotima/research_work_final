"""
metrics.py (Root Launcher)
==========================
Wrapper script allowing direct execution of the metrics suite
from the repository root directory.
"""
import os
import sys

# Ensure dss_system is in python path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
dss_dir = os.path.join(workspace_dir, "dss_system")
if dss_dir not in sys.path:
    sys.path.insert(0, dss_dir)

from metrics import main

if __name__ == "__main__":
    main()
