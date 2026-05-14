"""
Logging Utilities
=================
Author: Hardik Tankaria
Date: May 2026
Description:
    Structured logging for experiments with optional Weights & Biases
    integration. Provides a unified logger that works with and without
    wandb installed.
"""

import os
import sys
import logging
from typing import Dict, Optional

logger = logging.getLogger("nysadam")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure structured logging for experiments.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path for file logging.

    Returns:
        Configured logger.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.setLevel(level)
    return logger


class WandbLogger:
    """
    Optional Weights & Biases logger wrapper.
    Gracefully degrades if wandb is not installed.
    """

    def __init__(
        self,
        project: str = "nysadam",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._run = None

        if enabled:
            try:
                import wandb
                self._run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    reinit=True,
                )
            except (ImportError, Exception) as e:
                logger.warning(f"wandb not available: {e}. Logging locally only.")
                self.enabled = False

    def log(self, metrics: Dict, step: Optional[int] = None) -> None:
        """Log metrics dict."""
        if self.enabled and self._run:
            import wandb
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Close wandb run."""
        if self.enabled and self._run:
            import wandb
            wandb.finish()
