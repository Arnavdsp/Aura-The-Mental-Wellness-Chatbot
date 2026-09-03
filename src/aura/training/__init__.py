"""Dataset construction, fine-tuning and behavioural evaluation."""

from aura.training.data import PreferencePair, build_preference_dataset
from aura.training.evaluate import EvaluationReport, evaluate
from aura.training.train import TrainingConfig, run

__all__ = [
    "EvaluationReport",
    "PreferencePair",
    "TrainingConfig",
    "build_preference_dataset",
    "evaluate",
    "run",
]
