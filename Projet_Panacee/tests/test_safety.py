# -*- coding: utf-8 -*-
"""Tests des alertes structurelles (Ames/DILI/hERG) et du domaine d'applicabilité."""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")


def _eps(alerts):
    return {e["key"] for e in alerts.get("endpoints", [])}


def test_clean_molecule_has_no_alerts():
    from webapp import cheminfo
    a = cheminfo.structural_alerts("CC(=O)Oc1ccccc1C(=O)O")  # aspirine
    assert a["valid"] and a["level"] == "OK"
    assert not a["endpoints"]


def test_nitroaromatic_flags_ames():
    from webapp import cheminfo
    a = cheminfo.structural_alerts("c1ccccc1[N+](=O)[O-]")  # nitrobenzène
    assert "ames_mutagenicity" in _eps(a)
    assert a["level"] == "DANGER"


def test_haloperidol_flags_herg():
    from webapp import cheminfo
    halo = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
    a = cheminfo.structural_alerts(halo)
    assert a["herg"]["risk"] is True
    assert "herg_cardiotox" in _eps(a)


def test_paracetamol_flags_dili():
    from webapp import cheminfo
    a = cheminfo.structural_alerts("CC(=O)Nc1ccc(O)cc1")
    assert "dili_hepatotox" in _eps(a)


def test_invalid_smiles():
    from webapp import cheminfo
    assert cheminfo.structural_alerts("pas_un_smiles")["valid"] is False


def test_applicability_in_domain_for_known_drug():
    from webapp import cheminfo
    # AZT figure dans le catalogue → similarité maximale
    d = cheminfo.applicability_domain("CC1=CN(C2CC(N=[N+]=[N-])C(CO)O2)C(=O)NC1=O")
    assert d["valid"] and d["in_domain"] is True
    assert d["max_similarity"] >= 0.9


def test_applicability_out_of_domain_for_exotic():
    from webapp import cheminfo
    d = cheminfo.applicability_domain("C1CC2CCC3CCC4CCCCC4C3C2C1")  # squelette inhabituel
    assert d["in_domain"] is False
    assert d["level"] == "DANGER"


def test_safety_endpoint(client):
    r = client.post("/api/safety", json={"smiles": "c1ccccc1[N+](=O)[O-]"})
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 1
    assert res[0]["alerts"]["level"] == "DANGER"
    assert "applicability" in res[0]


def test_safety_endpoint_requires_smiles(client):
    assert client.post("/api/safety", json={}).status_code == 400
