"""
ChEMBL HIV target loader — récupère les inhibiteurs validés des 5 cibles anti-VIH clés.

Cibles ChEMBL (vérifiées via ebi.ac.uk/chembl 2026-10) :
  - CHEMBL247  : HIV-1 Reverse Transcriptase (RT)
  - CHEMBL243  : HIV-1 Protease (PR)
  - CHEMBL3471 : HIV-1 Integrase (IN)
  - CHEMBL5658 : HIV-1 gp120
  - CHEMBL274  : CCR5 (co-récepteur, cible maraviroc)

Pour chaque cible : récupère activities IC50/Ki/Kd en nM, filtre pChEMBL >= 5
(= IC50 <= 10 µM, convention standard virtual screening), extrait SMILES canonique.

API ChEMBL REST : https://www.ebi.ac.uk/chembl/api/data/activity.json
Pas de clé requise, rate-limit implicite (soyons polis : 3 req/s).

Sortie : CSV par cible + CSV merged avec colonnes :
  smiles, chembl_id, target, pchembl_value, standard_type, standard_units
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXTERNAL_DIR

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Cibles HIV avec UniProt + description
HIV_TARGETS = {
    "CHEMBL247": {
        "name": "HIV-1 Reverse Transcriptase",
        "uniprot": "P03366",
        "role": "RT",
    },
    "CHEMBL243": {
        "name": "HIV-1 Protease",
        "uniprot": "P03367",
        "role": "PR",
    },
    "CHEMBL3471": {
        "name": "HIV-1 Integrase",
        "uniprot": "P12497",
        "role": "IN",
    },
    "CHEMBL5658": {
        "name": "HIV-1 gp120",
        "uniprot": "P04578",
        "role": "gp120",
    },
    "CHEMBL274": {
        "name": "C-C chemokine receptor type 5",
        "uniprot": "P51681",
        "role": "CCR5",
    },
}

_MIN_PCHEMBL = 5.0  # Convention: pChEMBL >= 5 → IC50 <= 10 µM = actif
_RATE_LIMIT_SEC = 0.35  # ~3 req/s


def _polite_sleep(last_t: float) -> float:
    """Respect rate limit — retourne nouveau timestamp."""
    now = time.monotonic()
    dt = now - last_t
    if dt < _RATE_LIMIT_SEC:
        time.sleep(_RATE_LIMIT_SEC - dt)
    return time.monotonic()


def _fetch_json(url: str, timeout: int = 30) -> dict:
    """GET JSON avec User-Agent explicite (politesse ChEMBL)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Panacee-GNN/1.0 (research; makon052026@gmail.com)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_target_activities(
    target_chembl_id: str,
    min_pchembl: float = _MIN_PCHEMBL,
    max_records: Optional[int] = None,
) -> list[dict]:
    """
    Récupère toutes les activities bioactives pour une cible ChEMBL.

    Filtres :
      - pchembl_value >= min_pchembl (actifs confirmés)
      - standard_type ∈ {IC50, Ki, Kd, EC50}
      - standard_relation = '=' (valeur exacte, pas '<' ou '>')
      - canonical_smiles non-null

    Pagination : ChEMBL retourne 20 records/page par défaut, max 1000.
    """
    rows = []
    offset = 0
    limit = 1000
    last_t = 0.0

    while True:
        params = urllib.parse.urlencode({
            "target_chembl_id": target_chembl_id,
            "pchembl_value__gte": min_pchembl,
            "standard_relation": "=",
            "standard_type__in": "IC50,Ki,Kd,EC50",
            "limit": limit,
            "offset": offset,
            "format": "json",
        })
        url = f"{CHEMBL_BASE}/activity.json?{params}"

        last_t = _polite_sleep(last_t)
        try:
            data = _fetch_json(url)
        except Exception as e:
            print(f"  ✗ {target_chembl_id} page offset={offset}: {e}")
            break

        activities = data.get("activities", [])
        if not activities:
            break

        for a in activities:
            smi = a.get("canonical_smiles")
            if not smi:
                continue
            rows.append({
                "smiles": smi,
                "chembl_id": a.get("molecule_chembl_id", ""),
                "target": target_chembl_id,
                "pchembl_value": a.get("pchembl_value"),
                "standard_type": a.get("standard_type"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
            })

        page_meta = data.get("page_meta", {})
        total = page_meta.get("total_count", len(rows))
        print(f"    {target_chembl_id}: {len(rows)}/{total} récupérés")

        if max_records and len(rows) >= max_records:
            break
        if len(activities) < limit:
            break
        offset += limit

    return rows


def _canonicalize_and_dedupe(rows: list[dict]) -> list[dict]:
    """Dédupe par SMILES canonique isomerique, garde pChEMBL le plus élevé."""
    try:
        from rdkit import Chem
    except ImportError:
        print("  ⚠ RDKit absent — pas de canonicalisation, dédupe str simple")
        seen = set()
        uniq = []
        for r in rows:
            if r["smiles"] not in seen:
                seen.add(r["smiles"])
                uniq.append(r)
        return uniq

    best: dict[str, dict] = {}
    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        r["smiles"] = canon
        pv = r.get("pchembl_value")
        try:
            pv_f = float(pv) if pv is not None else 0.0
        except (TypeError, ValueError):
            pv_f = 0.0
        if canon not in best or float(best[canon].get("pchembl_value") or 0) < pv_f:
            best[canon] = r
    return list(best.values())


def download_chembl_hiv(
    save_dir: Optional[str] = None,
    targets: Optional[list[str]] = None,
    min_pchembl: float = _MIN_PCHEMBL,
    max_per_target: Optional[int] = None,
) -> dict[str, str]:
    """
    Télécharge les actifs ChEMBL pour chaque cible HIV, sauvegarde CSV.

    Returns:
        dict {role: csv_path} (RT, PR, IN, gp120, CCR5, merged)
    """
    import pandas as pd

    if save_dir is None:
        save_dir = str(EXTERNAL_DIR / "chembl_hiv")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if targets is None:
        targets = list(HIV_TARGETS.keys())

    results = {}
    all_rows = []

    for target_id in targets:
        meta = HIV_TARGETS.get(target_id, {})
        role = meta.get("role", target_id)
        print(f"\n  → {target_id} ({meta.get('name', '?')})")

        csv_path = save_path / f"chembl_hiv_{role}.csv"
        if csv_path.exists() and csv_path.stat().st_size > 100:
            print(f"    déjà présent → {csv_path}")
            df = pd.read_csv(csv_path)
            all_rows.extend(df.to_dict(orient="records"))
            results[role] = str(csv_path)
            continue

        rows = fetch_target_activities(
            target_id, min_pchembl=min_pchembl, max_records=max_per_target,
        )
        if not rows:
            print(f"    ⚠ aucune activity pour {target_id}")
            continue

        rows = _canonicalize_and_dedupe(rows)
        print(f"    {len(rows)} molécules uniques canoniques")

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        results[role] = str(csv_path)
        all_rows.extend(rows)

    # Merged CSV avec label binaire (tous actifs pChEMBL >= min)
    if all_rows:
        merged = _canonicalize_and_dedupe(all_rows)
        import pandas as pd
        df_m = pd.DataFrame(merged)
        df_m["hiv_active_chembl"] = 1  # tous sont actifs par filtre
        merged_path = save_path / "chembl_hiv_merged.csv"
        df_m.to_csv(merged_path, index=False)
        results["merged"] = str(merged_path)
        print(f"\n  ✓ Merged: {len(df_m)} actifs HIV uniques → {merged_path}")

    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download ChEMBL HIV target actives")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--targets", nargs="+", default=None,
                   help="Target ChEMBL IDs (default: all 5 HIV targets)")
    p.add_argument("--min_pchembl", type=float, default=_MIN_PCHEMBL)
    p.add_argument("--max_per_target", type=int, default=None)
    args = p.parse_args()

    results = download_chembl_hiv(
        save_dir=args.save_dir,
        targets=args.targets,
        min_pchembl=args.min_pchembl,
        max_per_target=args.max_per_target,
    )
    print("\n=== Résultats ===")
    for role, path in results.items():
        print(f"  {role}: {path}")
