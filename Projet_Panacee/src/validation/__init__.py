"""
Package de validation académique Panacée.

Ré-exporte l'API publique pour permettre `from src.validation import X`
(sinon ImportError, car les classes vivent dans les sous-modules).
"""
from .reproducibility_utils import (
    SeedManager, EnvironmentManager, EnvironmentSnapshot,
    HyperparameterConfig, ModelVersion, ModelVersionManager, ExperimentLogger,
)
from .validation_framework import (
    MetricResult, ValidationResult, CrossValidator,
    AblationStudy, BaselineComparison, SignificanceTest,
)
from .calibration_metrics import (
    CalibrationAnalyzer, UncertaintyDecomposition,
    ConfidenceThreshold, SelectivePrediction,
)
from .scientific_reporting import (
    LaTeXReportGenerator, MarkdownReportGenerator,
    ResultSummarizer, ComparisonTableGenerator,
)
from .profiling_utils import (
    PerformanceProfiler, MemoryProfiler, GPUMonitor,
    ScalabilityAnalyzer, ProfilingReport,
)
from .clinical_metrics import (
    summarize, per_task_metrics, evaluate_checkpoint, TOX21_TASKS,
)

__all__ = [
    "SeedManager", "EnvironmentManager", "EnvironmentSnapshot",
    "HyperparameterConfig", "ModelVersion", "ModelVersionManager", "ExperimentLogger",
    "MetricResult", "ValidationResult", "CrossValidator",
    "AblationStudy", "BaselineComparison", "SignificanceTest",
    "CalibrationAnalyzer", "UncertaintyDecomposition",
    "ConfidenceThreshold", "SelectivePrediction",
    "LaTeXReportGenerator", "MarkdownReportGenerator",
    "ResultSummarizer", "ComparisonTableGenerator",
    "PerformanceProfiler", "MemoryProfiler", "GPUMonitor",
    "ScalabilityAnalyzer", "ProfilingReport",
    "summarize", "per_task_metrics", "evaluate_checkpoint", "TOX21_TASKS",
]
