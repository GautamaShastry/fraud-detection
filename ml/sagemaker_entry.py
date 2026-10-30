"""SageMaker container entry point: trains into /opt/ml/model (uploaded to S3 by SageMaker).

Reads hyperparameters from /opt/ml/input/config/hyperparameters.json. The produced artifact is
PENDING approval; promotion (ml/promote.py) is a separate manual step.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.train import train

CONFIG = Path("/opt/ml/input/config/hyperparameters.json")
MODEL_DIR = Path("/opt/ml/model")


def main() -> None:
    params = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    directory = train(int(params.get("seed", 7)), MODEL_DIR, int(params.get("days", 90)))
    print(f"Artifact written to {directory}")


if __name__ == "__main__":
    main()
