"""Offline ML lifecycle: dataset audit, simulator, training, evaluation, promotion (M3)."""

import os

# joblib/loky probes physical cores via a subprocess that fails on some Windows hosts.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
