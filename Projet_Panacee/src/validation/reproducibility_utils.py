"""
Système de reproducibilité complète pour recherche académique.

Garantit que les expériences peuvent être exactement reproduites :
  - Seeds fixes pour tous les RNGs
  - Snapshots de l'environnement Python
  - Versioning des modèles avec métadonnées
  - Tracking des hyperparamètres
  - Logging structuré
"""
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger("panacee.reproducibility")


# ═══════════════════════════════════════════════════════════════
#  Seed Management
# ═══════════════════════════════════════════════════════════════

class SeedManager:
    """Gestion centralisée des seeds pour reproducibilité."""

    _seed_value = None
    _is_set = False

    @classmethod
    def set_seed(cls, seed: int = 42):
        """
        Fixe tous les seeds pour reproducibilité complète.

        Affecte :
          - Python random
          - NumPy
          - PyTorch (CPU et CUDA)
          - CuDNN
        """
        cls._seed_value = seed
        cls._is_set = True

        logger.info(f"Setting reproducibility seed: {seed}")

        # Python
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch CPU
        torch.manual_seed(seed)

        # PyTorch CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Multi-GPU

        # CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info("All seeds set successfully")

    @classmethod
    def get_seed(cls) -> Optional[int]:
        """Retourne le seed actuellement fixé."""
        return cls._seed_value

    @classmethod
    def is_reproducible(cls) -> bool:
        """Vérifie si la reproducibilité est activée."""
        return cls._is_set


# ═══════════════════════════════════════════════════════════════
#  Environment Snapshot
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnvironmentSnapshot:
    """Snapshot complète de l'environnement."""
    timestamp: str
    python_version: str
    python_executable: str
    platform_info: Dict[str, str]
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    gpu_info: Optional[Dict[str, Any]]
    installed_packages: Dict[str, str]
    seed_value: Optional[int]
    working_directory: str


class EnvironmentManager:
    """Capture et sauvegarde l'environnement complet."""

    @staticmethod
    def capture_environment() -> EnvironmentSnapshot:
        """
        Capture un snapshot complet de l'environnement d'exécution.

        Returns:
            EnvironmentSnapshot avec tous les détails
        """
        # Platform info
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        # CUDA info
        cuda_version = None
        cudnn_version = None
        gpu_info = None

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()

            # GPU details
            gpu_info = {
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": torch.cuda.get_device_capability(0),
            }

            if hasattr(torch.cuda, "get_device_properties"):
                props = torch.cuda.get_device_properties(0)
                gpu_info["total_memory_gb"] = props.total_memory / (1024**3)
                gpu_info["compute_capability"] = f"{props.major}.{props.minor}"

        # Packages installed
        packages = {}
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                pip_list = json.loads(result.stdout)
                packages = {pkg["name"]: pkg["version"] for pkg in pip_list}
        except Exception as e:
            logger.warning(f"Failed to get pip list: {e}")

        snapshot = EnvironmentSnapshot(
            timestamp=datetime.now().isoformat(),
            python_version=platform.python_version(),
            python_executable=sys.executable,
            platform_info=platform_info,
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            gpu_info=gpu_info,
            installed_packages=packages,
            seed_value=SeedManager.get_seed(),
            working_directory=os.getcwd(),
        )

        return snapshot

    @staticmethod
    def save_environment(
        snapshot: EnvironmentSnapshot,
        filepath: Path,
    ):
        """Sauvegarde le snapshot en JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for field, value in asdict(snapshot).items():
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                data[field] = value
            else:
                data[field] = str(value)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Environment snapshot saved to {filepath}")


# ═══════════════════════════════════════════════════════════════
#  Hyperparameter Management
# ═══════════════════════════════════════════════════════════════

@dataclass
class HyperparameterConfig:
    """Configuration des hyperparamètres avec métadonnées."""
    name: str
    description: str
    parameters: Dict[str, Any]
    created_at: str = ""
    modified_at: str = ""
    experiment_id: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.modified_at:
            self.modified_at = datetime.now().isoformat()
        if not self.experiment_id:
            self.experiment_id = self._generate_id()

    def _generate_id(self) -> str:
        """Génère un ID unique basé sur le hash des paramètres."""
        param_str = json.dumps(self.parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:12]

    def save(self, filepath: Path):
        """Sauvegarde la configuration."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)

        logger.info(f"Hyperparameters saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> "HyperparameterConfig":
        """Charge une configuration."""
        with open(filepath) as f:
            data = json.load(f)

        return HyperparameterConfig(**data)


# ═══════════════════════════════════════════════════════════════
#  Model Versioning
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelVersion:
    """Métadonnées d'une version de modèle."""
    version_id: str
    model_name: str
    created_at: str
    parent_version: Optional[str] = None
    description: str = ""
    performance_metrics: Dict[str, float] = None
    training_config: Dict[str, Any] = None
    data_sha256: Optional[str] = None
    model_sha256: Optional[str] = None
    architecture_notes: str = ""

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.training_config is None:
            self.training_config = {}


class ModelVersionManager:
    """Gère le versioning des modèles entraînés."""

    def __init__(self, base_dir: Path):
        """
        Args:
            base_dir: répertoire pour stocker les versions
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.base_dir / "versions.json"
        self.versions: Dict[str, ModelVersion] = self._load_versions()

    def _load_versions(self) -> Dict[str, ModelVersion]:
        """Charge l'historique des versions (tolérant aux fichiers corrompus)."""
        if not self.versions_file.exists():
            return {}

        try:
            with open(self.versions_file, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("registre attendu sous forme d'objet JSON")
            return {
                vid: ModelVersion(**v_data)
                for vid, v_data in data.items()
            }
        except (json.JSONDecodeError, ValueError, TypeError, UnicodeDecodeError) as e:
            # Registre illisible / corrompu : on repart d'un registre vide plutôt
            # que de planter. L'ancien fichier est mis de côté pour inspection.
            logger.warning(f"versions.json illisible ({e}); registre réinitialisé.")
            try:
                self.versions_file.replace(
                    self.versions_file.with_suffix(".json.corrupt")
                )
            except OSError:
                pass
            return {}

    def _save_versions(self):
        """Sauvegarde l'historique des versions."""
        data = {}
        for vid, version in self.versions.items():
            version_dict = asdict(version)
            data[vid] = version_dict

        with open(self.versions_file, "w") as f:
            json.dump(data, f, indent=2)

    def save_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        performance_metrics: Dict[str, float],
        training_config: Dict[str, Any],
        description: str = "",
        parent_version: Optional[str] = None,
    ) -> ModelVersion:
        """
        Sauvegarde un modèle avec métadonnées complètes.

        Returns:
            ModelVersion créée
        """
        # Générer ID unique
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.base_dir / f"{model_name}_{version_id}.pt"

        # Sauvegarder le modèle
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # Calculer hash
        model_sha = self._compute_file_sha256(model_path)

        # Créer version metadata
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            created_at=datetime.now().isoformat(),
            parent_version=parent_version,
            description=description,
            performance_metrics=performance_metrics,
            training_config=training_config,
            model_sha256=model_sha,
        )

        self.versions[version_id] = version
        self._save_versions()

        logger.info(f"Model version {version_id} created")

        return version

    def load_model(
        self,
        model: torch.nn.Module,
        version_id: str,
    ):
        """Charge un modèle d'une version spécifique."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        version = self.versions[version_id]
        model_path = self.base_dir / f"{version.model_name}_{version_id}.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # weights_only=True : ne désérialise que des tenseurs (pas de pickle arbitraire)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        logger.info(f"Model version {version_id} loaded")

        return model

    def get_version_info(self, version_id: str) -> ModelVersion:
        """Récupère les informations sur une version."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        return self.versions[version_id]

    def list_versions(self, model_name: Optional[str] = None):
        """Liste toutes les versions ou pour un modèle spécifique."""
        versions = self.versions

        if model_name:
            versions = {
                vid: v for vid, v in versions.items()
                if v.model_name == model_name
            }

        # Trier par date
        sorted_versions = sorted(
            versions.items(),
            key=lambda x: x[1].created_at,
            reverse=True,
        )

        return sorted_versions

    def rollback(self, version_id: str) -> ModelVersion:
        """Revient à une version précédente."""
        version = self.get_version_info(version_id)
        logger.info(f"Rolling back to version {version_id}")
        return version

    @staticmethod
    def _compute_file_sha256(filepath: Path) -> str:
        """Calcule le SHA256 d'un fichier."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# ═══════════════════════════════════════════════════════════════
#  Experiment Logger
# ═══════════════════════════════════════════════════════════════

class ExperimentLogger:
    """Logging structuré pour les expériences."""

    def __init__(self, log_dir: Path, experiment_name: str):
        """
        Args:
            log_dir: répertoire pour logs
            experiment_name: nom de l'expérience
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.config_file = self.log_dir / "config.json"
        self.log_file = self.log_dir / "experiment.log"

        # Logger
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)

    def log_metric(self, epoch: int, metrics: Dict[str, float]):
        """
        Enregistre les métriques d'une époque.

        Format : JSON Lines (une ligne JSON par époque)
        """
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_config(self, config: Dict[str, Any]):
        """Sauvegarde la configuration de l'expérience."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

    def get_metrics(self) -> list:
        """Charge tous les métriques enregistrées."""
        if not self.metrics_file.exists():
            return []

        metrics = []
        with open(self.metrics_file) as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))

        return metrics
