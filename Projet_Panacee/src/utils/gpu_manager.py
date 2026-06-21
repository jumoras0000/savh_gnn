"""
Gestionnaire GPU et ressources matérielles.

Fonctionnalités :
  - Détection automatique du meilleur GPU
  - Monitoring mémoire GPU en temps réel
  - Ajustement automatique du batch_size selon la VRAM
  - Mixed precision (FP16) si supporté
  - Gestion mémoire avec garbage collection
"""
import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger("panacee.gpu")


class GPUManager:
    """Gestionnaire centralisé des ressources GPU."""

    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu
        self._device = None
        self._gpu_info = None
        self._supports_fp16 = False

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = self._detect_best_device()
        return self._device

    def _detect_best_device(self) -> torch.device:
        """Détecte le meilleur device disponible."""
        if self.force_cpu:
            logger.info("Mode CPU forcé")
            return torch.device("cpu")

        if not torch.cuda.is_available():
            logger.warning("CUDA non disponible, utilisation du CPU")
            return torch.device("cpu")

        # Sélectionner le GPU avec le plus de mémoire libre
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            return torch.device("cpu")

        best_gpu = 0
        best_free = 0
        for i in range(n_gpus):
            try:
                free, total = torch.cuda.mem_get_info(i)
                name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {name} | "
                           f"{free / 1e9:.1f} GB libre / {total / 1e9:.1f} GB total")
                if free > best_free:
                    best_free = free
                    best_gpu = i
            except Exception as e:
                logger.warning(f"  GPU {i}: erreur détection - {e}")

        device = torch.device(f"cuda:{best_gpu}")

        # Vérifier support FP16
        cap = torch.cuda.get_device_capability(best_gpu)
        self._supports_fp16 = cap[0] >= 7  # Volta+ (compute capability 7.0+)

        self._gpu_info = {
            "name": torch.cuda.get_device_name(best_gpu),
            "index": best_gpu,
            "total_memory_gb": torch.cuda.get_device_properties(best_gpu).total_memory / 1e9,
            "compute_capability": f"{cap[0]}.{cap[1]}",
            "supports_fp16": self._supports_fp16,
        }

        return device

    def get_gpu_info(self) -> dict:
        """Retourne les informations du GPU sélectionné."""
        _ = self.device  # Force la détection
        if self._gpu_info is None:
            return {"device": "cpu", "name": "CPU"}
        return self._gpu_info

    def get_memory_stats(self) -> dict:
        """Retourne les statistiques mémoire GPU actuelles."""
        if not torch.cuda.is_available() or self.device.type == "cpu":
            return {"device": "cpu"}

        idx = self.device.index or 0
        try:
            allocated = torch.cuda.memory_allocated(idx) / 1e9
            reserved = torch.cuda.memory_reserved(idx) / 1e9
            free, total = torch.cuda.mem_get_info(idx)
            return {
                "allocated_gb": round(allocated, 3),
                "reserved_gb": round(reserved, 3),
                "free_gb": round(free / 1e9, 3),
                "total_gb": round(total / 1e9, 3),
                "utilization_pct": round((1 - free / total) * 100, 1),
            }
        except Exception:
            return {"error": "impossible de lire la mémoire GPU"}

    def optimize_batch_size(self, base_batch_size: int, model_size_mb: float = 100) -> int:
        """
        Ajuste le batch_size selon la VRAM disponible.

        Args:
            base_batch_size: taille de batch souhaitée
            model_size_mb: taille estimée du modèle en MB

        Returns:
            batch_size ajusté
        """
        if self.device.type == "cpu":
            return max(base_batch_size // 4, 1)

        try:
            free, _total = torch.cuda.mem_get_info(self.device.index or 0)
            free_mb = free / 1e6

            # Réserver 500 MB pour le système et overhead
            usable_mb = max(free_mb - 500, 100)

            # Estimer la mémoire par sample (heuristique)
            mem_per_sample_mb = model_size_mb * 0.05
            max_batch = int(usable_mb / max(mem_per_sample_mb, 1))

            adjusted = min(base_batch_size, max(max_batch, 1))
            if adjusted < base_batch_size:
                logger.warning(
                    f"Batch size ajusté: {base_batch_size} → {adjusted} "
                    f"(VRAM libre: {free_mb:.0f} MB)"
                )
            return adjusted
        except Exception:
            return base_batch_size

    def clear_memory(self):
        """Libère la mémoire GPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Retourne un GradScaler pour mixed precision si supporté."""
        if self._supports_fp16 and self.device.type == "cuda":
            return torch.amp.GradScaler("cuda")
        return None

    @property
    def supports_amp(self) -> bool:
        """Vérifie si Automatic Mixed Precision est supporté."""
        return self._supports_fp16 and self.device.type == "cuda"

    def print_summary(self):
        """Affiche un résumé des ressources."""
        info = self.get_gpu_info()
        mem = self.get_memory_stats()

        print(f"  Device      : {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU         : {info.get('name', 'N/A')}")
            print(f"  VRAM        : {mem.get('total_gb', 'N/A')} GB total, "
                  f"{mem.get('free_gb', 'N/A')} GB libre")
            print(f"  Compute Cap : {info.get('compute_capability', 'N/A')}")
            print(f"  FP16/AMP    : {'Oui' if self._supports_fp16 else 'Non'}")
        else:
            print("  Mode CPU (pas de GPU)")


# Singleton global
_gpu_manager = None

def get_gpu_manager(force_cpu: bool = False) -> GPUManager:
    """Retourne le gestionnaire GPU global."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager(force_cpu=force_cpu)
    return _gpu_manager
