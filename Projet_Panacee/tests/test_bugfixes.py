# -*- coding: utf-8 -*-
"""
Tests unitaires des corrections de bugs (audit approfondi).

Couvre :
  - medical_rules : inversion des métriques de RISQUE dans admet_global
  - calibration_metrics : AUC couverture/accuracy non négative + ECE bornée
  - reasoner : SynergyAnalyzer respecte le mask, forward + clamp positions
  - reproducibility_utils : snapshot environnement (total_memory non-tuple)

Lancer :  python -m tests.test_bugfixes    (depuis Projet_Panacee/)
"""
import sys
import os
import tempfile
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_ok = True


def check(name, cond):
    global _ok
    print(("  OK   " if cond else "  FAIL ") + name)
    _ok = _ok and bool(cond)


# ──────────────────────────────────────────────────────────────────────
def test_admet_risk_inversion():
    from src.knowledge.medical_rules import MolecularProfile, ADMETEvaluator

    # Molécule "à risque" : LogP/MW/aromatiques élevés -> cyp_inhibition_risk élevé
    bad = MolecularProfile(smiles="x", mw=600, logp=6.0, hba=8, hbd=2,
                           tpsa=80, rotatable_bonds=12, aromatic_rings=4)
    scores = ADMETEvaluator.evaluate_all(bad)

    check("cyp_inhibition_risk élevé pour molécule à risque",
          scores["cyp_inhibition_risk"] > 0.5)

    # admet_global doit inverser cyp_inhibition_risk (risque haut -> contribue peu)
    contributions = [
        (1.0 - v) if k == "cyp_inhibition_risk" else v
        for k, v in scores.items() if k != "admet_global"
    ]
    expected = sum(contributions) / len(contributions)
    check("admet_global inverse bien le risque CYP",
          abs(expected - scores["admet_global"]) < 1e-9)

    # Sanity : une molécule propre score plus haut qu'une molécule à risque
    good = MolecularProfile(smiles="y", mw=300, logp=2.0, hba=4, hbd=1,
                            tpsa=60, rotatable_bonds=3, aromatic_rings=1)
    good_scores = ADMETEvaluator.evaluate_all(good)
    check("admet_global(propre) > admet_global(à risque)",
          good_scores["admet_global"] > scores["admet_global"])


# ──────────────────────────────────────────────────────────────────────
def test_calibration():
    from src.validation.calibration_metrics import CalibrationAnalyzer, ConfidenceThreshold
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 300)
    y_proba = rng.rand(300)

    ca = CalibrationAnalyzer(n_bins=10)
    ece = ca.expected_calibration_error(y_true, y_proba)
    check("ECE bornée dans [0,1]", 0.0 <= ece <= 1.0)

    curve = ConfidenceThreshold.coverage_accuracy_curve(y_true, y_proba)
    check("AUC couverture/accuracy non négative (bug trapz corrigé)",
          curve["area_under_curve"] >= 0.0)

    # parfaitement calibré -> ECE proche de 0
    y2 = np.array([0, 0, 1, 1] * 50)
    p2 = np.array([0.0, 0.0, 1.0, 1.0] * 50)
    check("ECE ~0 si parfaitement calibré", ca.expected_calibration_error(y2, p2) < 0.05)


# ──────────────────────────────────────────────────────────────────────
def test_reasoner_mask_and_forward():
    import torch
    from src.models.reasoner import SynergyAnalyzer, MolecularReasoner
    torch.manual_seed(0)

    # SynergyAnalyzer : le mask doit annuler la synergie des molécules de padding
    sa = SynergyAnalyzer(d_model=32)
    emb = torch.randn(1, 3, 32)
    mask = torch.tensor([[False, False, True]])  # molécule 2 = padding
    sm = sa(emb, mask)
    check("synergie nulle pour paires avec padding",
          sm[0, 0, 2].item() == 0.0 and sm[0, 1, 2].item() == 0.0 and sm[0, 2, 2].item() == 0.0)
    check("synergie non nulle pour paire réelle (0,1)", sm[0, 0, 1].item() > 0.0)

    # MolecularReasoner : shapes correctes
    r = MolecularReasoner(mol_dim=16, hidden_dim=32, num_heads=4, num_layers=2,
                          max_molecules=5, num_dose_levels=7)
    out = r(torch.randn(2, 3, 16))
    check("synergy_matrix [B,N,N]", tuple(out["synergy_matrix"].shape) == (2, 3, 3))
    check("dose_distributions [B,N,doses]", tuple(out["dose_distributions"].shape) == (2, 3, 7))
    check("success_score [B,1]", tuple(out["success_score"].shape) == (2, 1))

    # N == max_molecules : positions valides (pas d'IndexError)
    out2 = r(torch.randn(1, 5, 16))
    check("forward OK quand N == max_molecules", tuple(out2["synergy_matrix"].shape) == (1, 5, 5))

    # mask propagé jusqu'à la sortie
    out3 = r(torch.randn(1, 3, 16), torch.tensor([[False, False, True]]))
    check("synergie sortie nulle pour padding", out3["synergy_matrix"][0, 0, 2].item() == 0.0)


# ──────────────────────────────────────────────────────────────────────
def test_env_snapshot():
    from src.validation.reproducibility_utils import EnvironmentManager, SeedManager

    SeedManager.set_seed(123)
    check("SeedManager mémorise le seed", SeedManager.get_seed() == 123)

    snap = EnvironmentManager.capture_environment()
    check("snapshot torch_version présent", bool(snap.torch_version))

    # Le bug corrigé : total_memory_gb ne doit PAS être un tuple
    if snap.gpu_info:
        tm = snap.gpu_info.get("total_memory_gb")
        check("gpu_info.total_memory_gb est un scalaire (pas un tuple)",
              tm is None or isinstance(tm, (int, float)))
    else:
        check("gpu_info None sur CPU (pas de crash)", snap.gpu_info is None)

    # save round-trip JSON
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "env.json"
        EnvironmentManager.save_environment(snap, p)
        check("snapshot sauvé en JSON valide", p.exists() and isinstance(json.load(open(p)), dict))


if __name__ == "__main__":
    print("== ADMET risk inversion =="); test_admet_risk_inversion()
    print("== calibration =="); test_calibration()
    print("== reasoner mask/forward =="); test_reasoner_mask_and_forward()
    print("== env snapshot =="); test_env_snapshot()
    print("\n" + ("==> BUGFIX TESTS OK" if _ok else "==> ECHEC"))
    sys.exit(0 if _ok else 1)
