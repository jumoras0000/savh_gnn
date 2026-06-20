# -*- coding: utf-8 -*-
"""
Gestionnaire d'entraînement — lance/arrête les phases depuis l'interface web.

Un seul entraînement actif à la fois. Le sous-processus (`python run_phaseX.py …`)
écrit son `live_metrics.jsonl` que le flux SSE existant diffuse en temps réel ;
ici on ne gère que le cycle de vie du process + la capture des logs.

Sécurité : pas de `shell=True`, arguments passés en liste, phase ∈ {1,2,3},
valeurs numériques validées → pas d'injection. Le serveur étant lié à 127.0.0.1
par défaut, le contrôle reste local.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# phase -> (script, dossier de sortie relatif, libellé)
_PHASES = {
    1: ("run_phase1.py", "checkpoints/phase1", "Phase 1 · Pré-entraînement MGM"),
    2: ("run_phase2.py", "checkpoints/phase2", "Phase 2 · Fine-tuning toxicité"),
    3: ("run_phase3.py", "checkpoints/phase3", "Phase 3 · Multi-propriétés + IA"),
}


def _as_int(v, lo, hi, default):
    try:
        x = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


class TrainingManager:
    """Cycle de vie d'un unique entraînement (thread-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._info: dict = {}

    # ── état ──────────────────────────────────────────────────────────
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        with self._lock:
            info = dict(self._info)
            if self._proc is not None:
                rc = self._proc.poll()
                if rc is None:
                    info["state"] = "running"
                else:
                    info["state"] = "finished" if rc == 0 else "failed"
                    info["returncode"] = rc
            else:
                info.setdefault("state", "idle")
        info["log_tail"] = self._read_log_tail(info.get("log_file"), 40)
        return info

    # ── démarrage ─────────────────────────────────────────────────────
    def start(self, phase: int, params: dict) -> dict:
        if phase not in _PHASES:
            return {"ok": False, "error": f"phase invalide: {phase}"}
        if self.is_running():
            return {"ok": False, "error": "un entraînement est déjà en cours"}

        script, out_dir, label = _PHASES[phase]
        epochs = _as_int(params.get("epochs"), 1, 1000, 20)
        max_mol = params.get("max_molecules")
        max_mol = _as_int(max_mol, 1, 10_000_000, 0) if max_mol not in (None, "", 0, "0") else 0

        argv = [sys.executable, script, "--epochs", str(epochs)]
        if params.get("download", True):
            argv.append("--download")
        if max_mol:
            argv += ["--max_molecules", str(max_mol)]
        if phase == 2:
            argv += ["--cv_folds", str(_as_int(params.get("cv_folds"), 0, 10, 0))]
            argv += ["--ema", "1" if params.get("ema", True) else "0"]

        log_dir = PROJECT_ROOT / "logs" / "webapp"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_phase{phase}_{ts}.log"

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        fh = open(log_file, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                argv, cwd=str(PROJECT_ROOT), env=env,
                stdout=fh, stderr=subprocess.STDOUT,
            )
        except Exception as e:
            fh.close()
            return {"ok": False, "error": f"lancement impossible: {e}"}

        with self._lock:
            self._proc = proc
            self._log_fh = fh
            self._info = {
                "phase": phase, "label": label, "pid": proc.pid,
                "cmd": " ".join(argv[1:]), "started_at": time.time(),
                "log_file": str(log_file),
                "run_id": Path(out_dir).name,  # ex: "phase2" → id de run pour le SSE
                "state": "running",
            }
        return {"ok": True, **self._info}

    # ── arrêt ─────────────────────────────────────────────────────────
    def stop(self) -> dict:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return {"ok": False, "error": "aucun entraînement en cours"}
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        with self._lock:
            if getattr(self, "_log_fh", None):
                with contextlib.suppress(Exception):
                    self._log_fh.close()
            self._info["state"] = "stopped"
        return {"ok": True, "state": "stopped"}

    # ── logs ──────────────────────────────────────────────────────────
    @staticmethod
    def _read_log_tail(path, n=40) -> list[str]:
        if not path or not Path(path).exists():
            return []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return [ln.rstrip("\n") for ln in lines[-n:]]
        except OSError:
            return []


# Singleton partagé par les routes
MANAGER = TrainingManager()
