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


# ──────────────────────────────────────────────────────────────────────
def test_latex_report():
    from src.validation.scientific_reporting import LaTeXReportGenerator

    esc = LaTeXReportGenerator._escape_latex
    # Échappement simple, SANS double-échappement des backslashes introduits
    check("échappe '&' en '\\&'", esc("a & b") == r"a \& b")
    check("échappe '_' et '%' proprement (pas de textbackslash parasite)",
          esc("a_b 100%") == r"a\_b 100\%" and "textbackslash" not in esc("a_b 100%"))
    check("backslash réel -> textbackslash",
          esc("x\\y") == r"x\textbackslash{}y")

    with tempfile.TemporaryDirectory() as d:
        gen = LaTeXReportGenerator(Path(d))
        header = gen._create_document_header("Titre", ["Auteur"])
        check("inputenc utf8 (pas utf-8 invalide)",
              "[utf8]" in header and "[utf-8]" not in header)


def test_profiling_cpu():
    from src.validation.profiling_utils import MemoryProfiler
    prof = MemoryProfiler()
    snap = prof.take_snapshot("test")  # ne doit PAS crasher sur CPU
    check("take_snapshot OK sur CPU (pas de crash f-string None)", snap.rss_mb > 0)
    check("gpu_memory_mb None sur CPU", snap.gpu_memory_mb is None)


def test_known_interactions():
    from src.knowledge.medical_rules import check_known_interactions
    found = check_known_interactions(["warfarin", "aspirin"])
    check("interaction warfarin+aspirin détectée", len(found) == 1)
    check("une seule molécule -> aucune interaction",
          len(check_known_interactions(["warfarin"])) == 0)


def test_gradual_unfreeze():
    from src.models.encoder import MolecularEncoder
    from src.models.toxicity_classifier import ToxicityClassifier
    enc = MolecularEncoder(num_layers=3)
    model = ToxicityClassifier(encoder=enc, num_tasks=12, freeze_encoder=True)

    check("atom_embedding gelé au départ",
          all(not p.requires_grad for p in model.encoder.atom_embedding.parameters()))

    # Dégel complet (epoch >> freeze + n_layers)
    model.gradual_unfreeze(epoch=100, total_freeze_epochs=2)
    check("atom_embedding dégelé au dégel complet (bug corrigé)",
          all(p.requires_grad for p in model.encoder.atom_embedding.parameters()))
    check("pool_gate dégelé au dégel complet (bug corrigé)",
          all(p.requires_grad for p in model.encoder.pool_gate.parameters()))
    check("toutes les convs dégelées",
          all(p.requires_grad for p in model.encoder.convs.parameters()))


def test_graph_features_clamped():
    from src.preprocessing.graph_builder import smiles_to_graph
    # Molécule variée (cycles, hétéroatomes) -> features doivent rester dans [0,1]
    for smi in ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "O=S(=O)(O)c1ccccc1", "[O-][N+](=O)c1ccccc1"]:
        g = smiles_to_graph(smi)
        check(f"features ∈ [0,1] pour {smi[:18]}",
              g is not None and float(g.x.min()) >= 0.0 and float(g.x.max()) <= 1.0)


if __name__ == "__main__":
    print("== ADMET risk inversion =="); test_admet_risk_inversion()
    print("== calibration =="); test_calibration()
    print("== reasoner mask/forward =="); test_reasoner_mask_and_forward()
    print("== env snapshot =="); test_env_snapshot()
    print("== latex report =="); test_latex_report()
    print("== profiling CPU =="); test_profiling_cpu()
    print("== known interactions =="); test_known_interactions()
    print("== gradual unfreeze =="); test_gradual_unfreeze()
    print("== graph features clamp =="); test_graph_features_clamped()
    print("\n" + ("==> BUGFIX TESTS OK" if _ok else "==> ECHEC"))
    sys.exit(0 if _ok else 1)
