# -*- coding: utf-8 -*-
"""
Tests du module de metriques cliniques + logging temps reel + dashboard.

Lancer :  python -m tests.test_clinical_metrics    (depuis Projet_Panacee/)
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_ok = True


def check(name, cond):
    global _ok
    print(("  OK   " if cond else "  FAIL ") + name)
    _ok = _ok and bool(cond)


def test_clinical():
    from src.validation.clinical_metrics import per_task_metrics, summarize

    N = 40
    probs = np.zeros((N, 2))
    targets = np.zeros((N, 2))

    # Tache 0 : parfaite (toxiques bien detectes)
    targets[:20, 0] = 1.0
    probs[:20, 0] = 0.9
    probs[20:, 0] = 0.1

    # Tache 1 : DANGEREUSE (modele predit tout "sur" alors que 20 sont toxiques)
    targets[:20, 1] = 1.0
    probs[:, 1] = 0.1  # tout sous le seuil -> faux negatifs massifs

    tasks = per_task_metrics(probs, targets, task_names=["GOOD", "BAD"])
    g, b = tasks[0], tasks[1]

    check("tache parfaite : sensibilite=1", abs(g["sensitivity"] - 1.0) < 1e-9)
    check("tache parfaite : FNR=0", g["fnr"] == 0.0)
    check("tache parfaite : danger=OK", g["danger"] == "OK")

    check("tache dangereuse : sensibilite=0", b["sensitivity"] == 0.0)
    check("tache dangereuse : FNR=1", abs(b["fnr"] - 1.0) < 1e-9)
    check("tache dangereuse : danger=DANGER", b["danger"] == "DANGER")

    res = summarize(probs, targets, task_names=["GOOD", "BAD"])
    check("agregat : 1 endpoint DANGER", res["aggregate"]["n_danger"] == 1)
    check("agregat : FNR macro = 0.5", abs(res["aggregate"]["macro_fnr"] - 0.5) < 1e-9)
    check("alertes : BAD signalee", any(a["task"] == "BAD" and a["level"] == "DANGER"
                                        for a in res["alerts"]))

    # NaN ignores
    t2 = targets.copy()
    t2[:5, 0] = np.nan
    tasks2 = per_task_metrics(probs, t2, task_names=["GOOD", "BAD"])
    check("NaN exclus du support", tasks2[0]["support"] == N - 5)


def test_live_logger():
    from src.utils.live_logger import LiveLogger, read_live

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "live_metrics.jsonl")
        lg = LiveLogger(path, meta={"phase": "phase2", "epochs_total": 3})
        lg.log({"epoch": 1, "val_auc": 0.7, "macro_fnr": 0.4, "n_danger": 2})
        lg.log({"epoch": 2, "val_auc": 0.8, "macro_fnr": 0.2, "n_danger": 0})
        meta, epochs = read_live(path)
        check("meta lue", meta.get("phase") == "phase2")
        check("2 epochs lus", len(epochs) == 2)
        check("derniere valeur correcte", epochs[-1]["val_auc"] == 0.8)


def test_dashboard_io_and_app():
    # data_io
    from dashboard.data_io import compare_to_expected, epochs_series
    rows = compare_to_expected({"val_auc": 0.9, "macro_sensitivity": 0.8, "macro_fnr": 0.1})
    check("compare_to_expected : 3 lignes", len(rows) == 3)
    check("compare_to_expected : AUC 0.9 OK", rows[0]["ok"] is True)
    xs, ys = epochs_series([{"epoch": 1, "val_auc": 0.7}, {"epoch": 2, "val_auc": 0.8}], "val_auc")
    check("epochs_series extrait la serie", xs == [1, 2] and ys == [0.7, 0.8])

    # App Streamlit headless via AppTest (si dispo)
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:
        print("  SKIP  AppTest indisponible (streamlit non installe ?)")
        return
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "dashboard", "app.py")
    at = AppTest.from_file(app_path, default_timeout=30).run()
    check("dashboard s'execute sans exception", not at.exception)
    check("dashboard a un titre", len(at.title) >= 1)


if __name__ == "__main__":
    print("== metriques cliniques =="); test_clinical()
    print("== live logger =="); test_live_logger()
    print("== dashboard io + app =="); test_dashboard_io_and_app()
    print("\n" + ("==> CLINICAL/DASHBOARD TESTS OK" if _ok else "==> ECHEC"))
    sys.exit(0 if _ok else 1)
