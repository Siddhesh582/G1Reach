"""
utils/logger.py
---------------
TensorBoard + CSV training logger.

Usage:
    logger = TrainingLogger(log_dir="logs", cfg=G1Config())
    logger.log(iteration=0, mean_reward=-1.2, success_rate=0.0, ...)
    logger.close()

Then visualise with:
    tensorboard --logdir logs/
"""

from __future__ import annotations

import os
import csv
import time
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import G1Config

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


# Metrics logged every iteration
_LOG_KEYS = [
    "mean_reward",
    "mean_value",
    "explained_variance",
    "success_rate",
    "action_std",
    "lr",
    "fps",
    "policy_loss",
    "value_loss",
    "entropy",
    "kl_approx",
    "clip_frac",
]


class TrainingLogger:
    """
    Writes training metrics to:
      - TensorBoard (if torch.utils.tensorboard is available)
      - A CSV file  (always — useful for plotting without TensorBoard)
    """

    def __init__(self, log_dir: str, cfg: G1Config):
        # Create a timestamped run subdirectory so runs don't overwrite each other
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # TensorBoard
        if TB_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.run_dir)
        else:
            self.writer = None
            print("[logger] TensorBoard not available — CSV only.")

        # CSV
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["iteration", "wall_time"] + _LOG_KEYS,
        )
        self._csv_writer.writeheader()

        self._start_time = time.time()
        print(f"[logger] Logging to {self.run_dir}")

    def log(self, iteration: int, **kwargs):
        """
        Log a dict of scalar metrics at a given training iteration.

        Any keys not in _LOG_KEYS are still written to TensorBoard
        under the "extra/" prefix but omitted from CSV.
        """
        wall_time = time.time() - self._start_time

        # TensorBoard
        if self.writer is not None:
            for key, val in kwargs.items():
                if val is None:
                    continue
                tag = key if key in _LOG_KEYS else f"extra/{key}"
                self.writer.add_scalar(tag, val, global_step=iteration)

        # CSV — only standard keys
        row = {"iteration": iteration, "wall_time": f"{wall_time:.1f}"}
        for key in _LOG_KEYS:
            row[key] = f"{kwargs.get(key, ''):.6f}" \
                       if isinstance(kwargs.get(key), float) else kwargs.get(key, "")
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self._csv_file.close()