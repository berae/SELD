"""Shared official metric bridge; installed as SELD_evaluation_metrics_aligned.py."""
import sys
from pathlib import Path

_shared = Path(__file__).resolve().parents[7] / 'evaluation'
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))
from einv2_adapter import METRIC_PROTOCOL, SELDMetrics, early_stopping_metric
