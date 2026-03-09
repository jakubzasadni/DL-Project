"""Klasyczne baseline'y dla projektu malware detection."""

from src.baselines.ewoa_knn import EWOAKNNConfig, run_ewoa_knn_pipeline

__all__ = ["EWOAKNNConfig", "run_ewoa_knn_pipeline"]