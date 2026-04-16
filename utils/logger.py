"""
utils/logger.py — TensorBoard + CSV training logger
"""

import os
import csv
from typing import Dict, Any

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    print("[Logger] TensorBoard not available — CSV-only logging.")


class TrainingLogger:

    def __init__(self, log_dir: str, run_name: str):
        import time
        run_dir = os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"[logger] Logging to {run_dir}")

        self.tb = SummaryWriter(log_dir=run_dir) if _TB_AVAILABLE else None

        self.csv_path    = os.path.join(run_dir, "metrics.csv")
        self._csv_file   = None
        self._csv_writer = None
        self._headers    = None

    def log(self, metrics: Dict[str, Any], step: int):
        if self.tb is not None:
            for k, v in metrics.items():
                self.tb.add_scalar(k, float(v), global_step=step)

        row = {"step": step, **metrics}
        if self._csv_writer is None:
            self._headers    = list(row.keys())
            self._csv_file   = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._headers)
            self._csv_writer.writeheader()

        self._csv_writer.writerow({k: row.get(k, "") for k in self._headers})
        self._csv_file.flush()

    def close(self):
        if self.tb is not None:
            self.tb.close()
        if self._csv_file is not None:
            self._csv_file.close()
