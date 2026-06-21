"""
Package de validation académique Panacée.

Ré-exporte l'API publique pour permettre `from src.validation import X`
(sinon ImportError, car les classes vivent dans les sous-modules).
"""
from .calibration_metrics import (
    CalibrationAnalyzer,
    ConfidenceThreshold,
    SelectivePrediction,
    UncertaintyDecomposition,
)
from .clinical_metrics import (
    TOX21_TASKS,
    evaluate_checkpoint,
    per_task_metrics,
    summarize,
)
from .profiling_utils import (
    GPUMonitor,
    MemoryProfiler,
    PerformanceProfiler,
    ProfilingReport,
    ScalabilityAnalyzer,
)
from .reproducibility_utils import (
    EnvironmentManager,
    EnvironmentSnapshot,
    ExperimentLogger,
    HyperparameterConfig,
    ModelVersion,
    ModelVersionManager,
    SeedManager,
)
from .scientific_reporting import (
    ComparisonTableGenerator,
    LaTeXReportGenerator,
    MarkdownReportGenerator,
    ResultSummarizer,
)
from .validation_framework import (
    AblationStudy,
    BaselineComparison,
    CrossValidator,
    MetricResult,
    SignificanceTest,
    ValidationResult,
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
