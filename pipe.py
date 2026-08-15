#!/usr/bin/env python3
"""Compatibility shim so `./pipe.py ...` keeps working after the move to a
package. The real entry point is `photo-pipe`, or `python -m photopipe.cli`."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from photopipe.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
