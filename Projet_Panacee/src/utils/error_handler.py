"""
Gestionnaire centralisé des erreurs et logging.

Fonctionnalités :
  - Logging hiérarchique (console + fichier)
  - Décorateur de gestion d'erreurs avec retry
  - Alertes de santé du système
  - Sauvegarde automatique en cas de crash
"""
import os
import sys
import time
import logging
import functools
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    name: str = "panacee",
) -> logging.Logger:
    """
    Configure le logging pour tout le projet.

    Args:
        log_dir: dossier pour les fichiers de log
        level: niveau de logging
        name: nom du logger racine
    """
    if log_dir is None:
        log_dir = str(PROJECT_ROOT / "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Éviter les doublons
    if logger.handlers:
        return logger

    # Format
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        os.path.join(log_dir, f"panacee_{timestamp}.log"),
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class PanaceeError(Exception):
    """Classe de base pour les erreurs Panacée."""
    pass


class DataError(PanaceeError):
    """Erreur liée aux données (fichiers manquants, format invalide)."""
    pass


class ModelError(PanaceeError):
    """Erreur liée au modèle (architecture incompatible, poids corrompus)."""
    pass


class GPUError(PanaceeError):
    """Erreur liée au GPU (CUDA OOM, device indisponible)."""
    pass


class WebSearchError(PanaceeError):
    """Erreur liée à la recherche web (timeout, API indisponible)."""
    pass


def safe_execution(
    retries: int = 1,
    delay: float = 1.0,
    fallback: Any = None,
    catch: tuple = (Exception,),
    logger_name: str = "panacee",
):
    """
    Décorateur pour exécution sûre avec retry et fallback.

    Args:
        retries: nombre de tentatives
        delay: délai entre tentatives (secondes)
        fallback: valeur de retour en cas d'échec total
        catch: types d'exceptions à attraper
        logger_name: nom du logger
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logging.getLogger(logger_name)
            last_exc = None

            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except catch as e:
                    last_exc = e
                    if attempt < retries:
                        log.warning(
                            f"{func.__name__} échec (tentative {attempt}/{retries}): {e}. "
                            f"Retry dans {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        log.error(
                            f"{func.__name__} échec définitif après {retries} tentatives: {e}"
                        )

            if fallback is not None:
                log.info(f"{func.__name__}: utilisation de la valeur de fallback")
                return fallback

            raise last_exc

        return wrapper
    return decorator


def emergency_save(model, optimizer, epoch: int, path: str):
    """
    Sauvegarde d'urgence du modèle en cas de crash.

    Args:
        model: le modèle PyTorch
        optimizer: l'optimiseur
        epoch: époque courante
        path: chemin de sauvegarde
    """
    import torch

    logger = logging.getLogger("panacee.emergency")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": datetime.now().isoformat(),
            "reason": "emergency_save",
        }
        torch.save(checkpoint, path)
        logger.info(f"Sauvegarde d'urgence réussie: {path}")
    except Exception as e:
        logger.critical(f"Échec de la sauvegarde d'urgence: {e}")


class HealthMonitor:
    """Moniteur de santé du système pendant l'entraînement."""

    def __init__(self, check_interval: int = 50):
        """
        Args:
            check_interval: vérifier tous les N batches
        """
        self.check_interval = check_interval
        self.logger = logging.getLogger("panacee.health")
        self._step = 0
        self._loss_history = []
        self._nan_count = 0

    def step(self, loss_value: float) -> bool:
        """
        Enregistre un pas d'entraînement.

        Returns:
            True si le système est sain, False sinon
        """
        import math
        self._step += 1

        # Détection NaN/Inf
        if math.isnan(loss_value) or math.isinf(loss_value):
            self._nan_count += 1
            self.logger.warning(
                f"Step {self._step}: loss = {loss_value} "
                f"(NaN/Inf #{self._nan_count})"
            )
            if self._nan_count >= 5:
                self.logger.error("Trop de NaN/Inf consécutifs - arrêt recommandé")
                return False
        else:
            self._nan_count = 0
            self._loss_history.append(loss_value)

        # Vérifications périodiques
        if self._step % self.check_interval == 0:
            self._check_gpu_health()
            self._check_loss_trend()

        return True

    def _check_gpu_health(self):
        """Vérifie l'état du GPU."""
        import torch
        if not torch.cuda.is_available():
            return

        try:
            free, total = torch.cuda.mem_get_info()
            usage_pct = (1 - free / total) * 100
            if usage_pct > 95:
                self.logger.warning(
                    f"VRAM critique: {usage_pct:.1f}% utilisée "
                    f"({free / 1e6:.0f} MB libre)"
                )
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _check_loss_trend(self):
        """Vérifie la tendance de la loss."""
        if len(self._loss_history) < 20:
            return

        recent = self._loss_history[-10:]
        older = self._loss_history[-20:-10]
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)

        if avg_recent > avg_older * 1.5:
            self.logger.warning(
                f"Loss en augmentation: {avg_older:.4f} → {avg_recent:.4f}"
            )
