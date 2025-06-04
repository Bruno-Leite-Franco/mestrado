"""Entry point for example model."""

from .train import main as train_main
from .test import main as test_main

__all__ = ["train_main", "test_main"]
