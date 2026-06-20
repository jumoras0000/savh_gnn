"""
Profiling et benchmarking pour optimisation de performance.

Mesure :
  - Timing de chaque étape
  - Memory profiling
  - GPU/CPU utilization
  - Throughput et latency
  - Scalability analysis
"""
import logging
import time
import psutil
import numpy as np
import torch
from typing import Callable, Dict, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger("panacee.profiling")


# ═══════════════════════════════════════════════════════════════
#  Timing & Performance
# ═══════════════════════════════════════════════════════════════

@dataclass
class TimingResult:
    """Résultat de mesure de temps."""
    name: str
    elapsed_time: float  # en secondes
    n_calls: int = 1
    mean_time: float = 0.0
    std_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0

    def __post_init__(self):
        if self.n_calls > 0:
            self.mean_time = self.elapsed_time / self.n_calls

    def __str__(self):
        if self.n_calls == 1:
            return f"{self.name}: {self.elapsed_time:.4f}s"
        else:
            return (f"{self.name}: {self.mean_time:.6f}s/call "
                   f"(total: {self.elapsed_time:.4f}s, n={self.n_calls})")


class PerformanceProfiler:
    """
    Profile les performances : temps, mémoire, GPU.
    """

    def __init__(self):
        """Initialise le profiler."""
        self.timings: Dict[str, list] = {}
        self.memory_stats: Dict[str, Dict] = {}

    @contextmanager
    def timer(self, name: str):
        """
        Context manager pour mesurer le temps.

        Usage:
            with profiler.timer("my_operation"):
                do_something()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
            logger.debug(f"{name}: {elapsed:.4f}s")

    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """
        Statistiques de timing pour une opération.

        Returns:
            {mean, std, min, max, count}
        """
        if name not in self.timings or not self.timings[name]:
            return {}

        times = np.array(self.timings[name])
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "count": len(times),
            "total": float(np.sum(times)),
        }

    def profile_function(
        self,
        func: Callable,
        n_runs: int = 10,
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Profile une fonction avec plusieurs exécutions.

        Args:
            func: fonction à profiler
            n_runs: nombre d'exécutions
            args, kwargs: arguments de la fonction

        Returns:
            Statistiques de timing
        """
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - start)

        times = np.array(times)
        return {
            "function": func.__name__,
            "n_runs": n_runs,
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "total": float(np.sum(times)),
        }

    def report_timing(self) -> str:
        """Rapport textuel des timings."""
        report = "\n=== TIMING REPORT ===\n"

        for name, times in sorted(self.timings.items()):
            times_array = np.array(times)
            report += (
                f"{name}: "
                f"mean={np.mean(times_array):.4f}s, "
                f"std={np.std(times_array):.4f}s, "
                f"total={np.sum(times_array):.4f}s, "
                f"count={len(times)}\n"
            )

        return report


# ═══════════════════════════════════════════════════════════════
#  Memory Profiling
# ═══════════════════════════════════════════════════════════════

@dataclass
class MemoryStats:
    """Statistiques mémoire."""
    timestamp: str
    rss_mb: float  # Resident Set Size (mémoire physique)
    vms_mb: float  # Virtual Memory Size
    percent: float  # % de mémoire système utilisée
    gpu_memory_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None


class MemoryProfiler:
    """
    Profile l'utilisation mémoire (CPU et GPU).
    """

    def __init__(self):
        """Initialise le profiler mémoire."""
        self.snapshots: list = []
        self.process = psutil.Process()

    def take_snapshot(self, label: str = "") -> MemoryStats:
        """
        Prend un snapshot de la mémoire.

        Args:
            label: description du snapshot

        Returns:
            MemoryStats
        """
        # CPU memory
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / (1024 ** 2)
        vms_mb = mem_info.vms / (1024 ** 2)
        percent = self.process.memory_percent()

        # GPU memory
        gpu_mem_mb = None
        gpu_mem_percent = None

        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            gpu_mem_percent = (gpu_mem_mb / gpu_mem_total) * 100

        stats = MemoryStats(
            timestamp=datetime.now().isoformat(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            percent=percent,
            gpu_memory_mb=gpu_mem_mb,
            gpu_memory_percent=gpu_mem_percent,
        )

        self.snapshots.append((label, stats))
        # gpu_mem_mb peut être None (CPU) : ne pas le formater avec :.1f
        gpu_str = f"{gpu_mem_mb:.1f}MB" if gpu_mem_mb is not None else "N/A"
        logger.debug(f"Memory snapshot ({label}): CPU={rss_mb:.1f}MB, GPU={gpu_str}")

        return stats

    def memory_delta(self, label1: str, label2: str) -> Dict[str, float]:
        """
        Calcule la différence mémoire entre deux snapshots.

        Returns:
            {cpu_delta_mb, gpu_delta_mb}
        """
        stats1 = next((s for l, s in self.snapshots if l == label1), None)
        stats2 = next((s for l, s in self.snapshots if l == label2), None)

        if not stats1 or not stats2:
            return {}

        return {
            "cpu_delta_mb": stats2.rss_mb - stats1.rss_mb,
            "gpu_delta_mb": (stats2.gpu_memory_mb - stats1.gpu_memory_mb)
                           if stats1.gpu_memory_mb and stats2.gpu_memory_mb
                           else None,
        }

    def report_memory(self) -> str:
        """Rapport sur utilisation mémoire."""
        report = "\n=== MEMORY USAGE REPORT ===\n"

        for label, stats in self.snapshots:
            report += f"\n{label}:\n"
            report += f"  CPU (RSS): {stats.rss_mb:.1f} MB ({stats.percent:.1f}%)\n"
            if stats.gpu_memory_mb is not None:
                report += f"  GPU: {stats.gpu_memory_mb:.1f} MB ({stats.gpu_memory_percent:.1f}%)\n"

        return report


# ═══════════════════════════════════════════════════════════════
#  GPU Monitoring
# ═══════════════════════════════════════════════════════════════

class GPUMonitor:
    """
    Monitor les statistiques GPU en temps réel.
    """

    @staticmethod
    def get_gpu_stats() -> Dict[str, Any]:
        """
        Récupère les stats GPU actuelles.

        Returns:
            {allocated, reserved, free, total, efficiency}
        """
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free = total - allocated

        return {
            "allocated_gb": float(allocated),
            "reserved_gb": float(reserved),
            "free_gb": float(free),
            "total_gb": float(total),
            "utilization_percent": float((allocated / total) * 100),
            "fragmentation_percent": float(((reserved - allocated) / reserved) * 100)
            if reserved > 0 else 0.0,
        }

    @staticmethod
    def clear_cache():
        """Nettoie le cache GPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    @staticmethod
    def reset_peak_memory():
        """Réinitialise le pic mémoire."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.info("GPU peak memory stats reset")


# ═══════════════════════════════════════════════════════════════
#  Throughput & Scalability
# ═══════════════════════════════════════════════════════════════

class ScalabilityAnalyzer:
    """
    Analyse la scalabilité : comment les performances changent avec la taille.
    """

    @staticmethod
    def throughput_test(
        func: Callable,
        input_sizes: list,
        n_runs: int = 3,
    ) -> Dict[int, float]:
        """
        Mesure le throughput pour différentes tailles d'entrée.

        Args:
            func: fonction qui prend une taille en entrée
            input_sizes: tailles à tester [100, 1000, 10000, ...]
            n_runs: répétitions par taille

        Returns:
            {input_size: throughput_ops_per_sec}
        """
        results = {}

        for size in input_sizes:
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                func(size)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            mean_time = np.mean(times)
            throughput = size / mean_time  # samples/sec

            results[size] = throughput
            logger.info(f"Size {size}: {throughput:.0f} ops/sec")

        return results

    @staticmethod
    def strong_scaling(
        func: Callable,
        n_workers_list: list,
        problem_size: int,
        n_runs: int = 3,
    ) -> Dict[int, Dict]:
        """
        Analyse strong scaling (même problème avec plus de travailleurs).

        Returns:
            {n_workers: {time, speedup, efficiency}}
        """
        reference_time = None
        results = {}

        for n_workers in n_workers_list:
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                func(problem_size, n_workers)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            mean_time = np.mean(times)

            if reference_time is None:
                reference_time = mean_time

            speedup = reference_time / mean_time
            efficiency = speedup / n_workers

            results[n_workers] = {
                "time": float(mean_time),
                "speedup": float(speedup),
                "efficiency": float(efficiency),
            }

            logger.info(f"Workers {n_workers}: speedup={speedup:.2f}x, eff={efficiency:.2%}")

        return results


# ═══════════════════════════════════════════════════════════════
#  Profiling Report
# ═══════════════════════════════════════════════════════════════

class ProfilingReport:
    """
    Génère un rapport complet de profiling.
    """

    def __init__(
        self,
        perf_profiler: PerformanceProfiler,
        mem_profiler: MemoryProfiler,
    ):
        """
        Args:
            perf_profiler: PerformanceProfiler
            mem_profiler: MemoryProfiler
        """
        self.perf = perf_profiler
        self.mem = mem_profiler

    def generate_report(self, output_path: Path) -> Path:
        """
        Génère un rapport complet (JSON + texte).

        Returns:
            Chemin du fichier rapport
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "timings": {},
            "memory": [],
            "gpu": {},
        }

        # Timings
        for name in self.perf.timings:
            report_data["timings"][name] = self.perf.get_timing_stats(name)

        # Memory snapshots
        for label, stats in self.mem.snapshots:
            report_data["memory"].append({
                "label": label,
                "rss_mb": stats.rss_mb,
                "gpu_memory_mb": stats.gpu_memory_mb,
            })

        # GPU stats
        report_data["gpu"] = GPUMonitor.get_gpu_stats()

        # Sauvegarder JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Profiling report saved: {output_path}")

        return output_path
