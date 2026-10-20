"""
Multi-virus antiviral loader — pretrain intermédiaire pour transfer learning HIV.

Hypothèse testée : les molécules actives contre d'autres virus (HCV, HBV,
Influenza, SARS-CoV-2) partagent des features chimico-structurales avec les
antirétroviraux (chaînes azotées, hétérocycles, nucléosides modifiés, bases
peptidomimétiques pour PI). Un pretrain sur ce corpus multi-virus devrait
fournir un biais inductif anti-viral utile avant fine-tune HIV ciblé.

Cibles virales ChEMBL (vérifiées 2026-10) :
  - HCV NS5A      : CHEMBL5375
  - HCV NS5B      : CHEMBL5071
  - HCV NS3-4A    : CHEMBL4523
  - HBV capsid    : CHEMBL1795183
  - HBV polymerase: CHEMBL1795186
  - Influenza NA  : CHEMBL4036
  - Influenza PA  : CHEMBL1293281
  - SARS-CoV-2 Mpro (3CLpro) : CHEMBL4523582
  - SARS-CoV-2 RdRp          : CHEMBL4523587
  - SARS-CoV-2 Spike/ACE2    : CHEMBL4523967

Référence transfer learning anti-viral : Jin et al. 2021 (Nature Mach. Intell.),
DeepCoVDR (Chen 2023), invariant learning Krueger et al. 2021.
"""
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXTERNAL_DIR
from src.preprocessing.chembl_hiv_loader import (
    _canonicalize_and_dedupe,
    fetch_target_activities,
)

ANTIVIRAL_TARGETS = {
    # HCV (hepatitis C — virus guéri, modèle à transférer)
    "CHEMBL5375":     {"virus": "HCV",       "target": "NS5A",       "role": "polymerase_inhibitor"},
    "CHEMBL5071":     {"virus": "HCV",       "target": "NS5B",       "role": "polymerase_inhibitor"},
    "CHEMBL4523":     {"virus": "HCV",       "target": "NS3-4A",     "role": "protease_inhibitor"},
    # HBV (hepatitis B — capsid inhibitor, parallèle lenacapavir HIV)
    "CHEMBL1795183":  {"virus": "HBV",       "target": "capsid",     "role": "capsid_inhibitor"},
    "CHEMBL1795186":  {"virus": "HBV",       "target": "polymerase", "role": "polymerase_inhibitor"},
    # Influenza
    "CHEMBL4036":     {"virus": "Influenza", "target": "neuraminidase",       "role": "neuraminidase_inhibitor"},
    "CHEMBL1293281":  {"virus": "Influenza", "target": "endonuclease",        "role": "endonuclease_inhibitor"},
    # SARS-CoV-2
    "CHEMBL4523582":  {"virus": "SARS-CoV-2","target": "3CLpro",     "role": "protease_inhibitor"},
    "CHEMBL4523587":  {"virus": "SARS-CoV-2","target": "RdRp",       "role": "polymerase_inhibitor"},
    "CHEMBL4523967":  {"virus": "SARS-CoV-2","target": "Spike",      "role": "entry_inhibitor"},
}


def download_multivirus_antivirals(
    save_dir: Optional[str] = None,
    min_pchembl: float = 5.0,
    max_per_target: Optional[int] = 2000,
    include_hiv: bool = False,
) -> str:
    """
    Télécharge les actifs des cibles anti-virales non-HIV depuis ChEMBL.

    Args:
        save_dir: dossier de sortie
        min_pchembl: seuil activité (5.0 = IC50 <= 10 µM)
        max_per_target: borne pour contrôler le temps de download
        include_hiv: inclure aussi les cibles HIV (via chembl_hiv_loader)

    Returns:
        chemin du CSV mergé consolidé pour pretrain multi-virus
    """
    import pandas as pd

    if save_dir is None:
        save_dir = str(EXTERNAL_DIR / "multivirus_antivirals")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for target_id, meta in ANTIVIRAL_TARGETS.items():
        virus = meta["virus"]
        target_name = meta["target"]
        print(f"\n  → {target_id} ({virus} {target_name})")

        csv_path = save_path / f"{virus}_{target_name}.csv"
        if csv_path.exists() and csv_path.stat().st_size > 100:
            print(f"    déjà présent → {csv_path}")
            df = pd.read_csv(csv_path)
            all_rows.extend(df.to_dict(orient="records"))
            continue

        rows = fetch_target_activities(
            target_id, min_pchembl=min_pchembl, max_records=max_per_target,
        )
        if not rows:
            print(f"    ⚠ aucune activity pour {target_id}")
            continue

        # Enrichit avec métadonnées virus/target
        for r in rows:
            r["virus"] = virus
            r["viral_target"] = target_name
            r["mechanistic_class"] = meta["role"]

        rows = _canonicalize_and_dedupe(rows)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"    {len(rows)} molécules uniques → {csv_path}")
        all_rows.extend(rows)

    if include_hiv:
        from src.preprocessing.chembl_hiv_loader import download_chembl_hiv
        print("\n  → Ajout des cibles HIV")
        hiv_results = download_chembl_hiv(
            save_dir=str(save_path / "hiv_subset"),
            min_pchembl=min_pchembl,
            max_per_target=max_per_target,
        )
        if "merged" in hiv_results:
            hiv_df = pd.read_csv(hiv_results["merged"])
            hiv_df["virus"] = "HIV"
            hiv_df["viral_target"] = hiv_df.get("target", "unknown")
            hiv_df["mechanistic_class"] = "hiv_inhibitor"
            all_rows.extend(hiv_df.to_dict(orient="records"))

    if not all_rows:
        raise RuntimeError("Aucun antiviral téléchargé — vérifier connexion ChEMBL")

    # Dédupe global (même SMILES contre plusieurs virus → on garde toutes
    # les étiquettes virus en liste pour multi-label antiviral training)
    merged = _canonicalize_and_dedupe(all_rows)
    merged_path = save_path / "multivirus_antivirals_merged.csv"
    df_m = pd.DataFrame(merged)
    df_m["antiviral_active"] = 1  # tous actifs par filtre pChEMBL >= 5
    df_m.to_csv(merged_path, index=False)
    print(f"\n  ✓ {len(df_m)} antiviraux multi-virus uniques → {merged_path}")
    return str(merged_path)


def build_antiviral_pretrain_dataset(
    output_csv: Optional[str] = None,
    include_zinc_decoys: bool = True,
    n_decoys_ratio: float = 10.0,
) -> str:
    """
    Construit le dataset pretrain antiviral avec actifs + decoys ZINC.

    Pour un pretrain intermédiaire efficace :
      - Positifs : antiviraux multi-virus (ChEMBL pChEMBL >= 5)
      - Négatifs : échantillon aléatoire ZINC (hypothèse decoy inactif)
      - Ratio neg:pos ≈ 10:1 (évite overfitting sur positifs rares)

    Returns:
        chemin du CSV binaire antiviral_active ∈ {0, 1}
    """
    import pandas as pd

    actives_csv = download_multivirus_antivirals()
    actives = pd.read_csv(actives_csv)

    if not include_zinc_decoys:
        if output_csv is None:
            output_csv = str(EXTERNAL_DIR / "multivirus_antivirals" / "pretrain_actives_only.csv")
        actives.to_csv(output_csv, index=False)
        return output_csv

    # Decoys : échantillon ZINC si disponible
    zinc_path = EXTERNAL_DIR.parent / "raw" / "250k_rndm_zinc_drugs_clean_3.csv"
    if not zinc_path.exists():
        print(f"  ⚠ ZINC non trouvé à {zinc_path} → export actifs seuls")
        if output_csv is None:
            output_csv = str(EXTERNAL_DIR / "multivirus_antivirals" / "pretrain_actives_only.csv")
        actives.to_csv(output_csv, index=False)
        return output_csv

    n_decoys = int(len(actives) * n_decoys_ratio)
    zinc = pd.read_csv(zinc_path).sample(n=min(n_decoys, 200000), random_state=42)
    zinc["smiles"] = zinc["smiles"].astype(str).str.strip()
    zinc["antiviral_active"] = 0
    zinc = zinc[["smiles", "antiviral_active"]]

    actives_min = actives[["smiles"]].copy()
    actives_min["antiviral_active"] = 1

    combined = pd.concat([actives_min, zinc], ignore_index=True)
    combined = combined.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    if output_csv is None:
        output_csv = str(EXTERNAL_DIR / "multivirus_antivirals" / "pretrain_multivirus_binary.csv")
    combined.to_csv(output_csv, index=False)
    print(f"\n  ✓ Pretrain antiviral: {int(combined['antiviral_active'].sum())} actifs "
          f"+ {int((combined['antiviral_active']==0).sum())} decoys → {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download multivirus antivirals for pretrain")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--min_pchembl", type=float, default=5.0)
    p.add_argument("--max_per_target", type=int, default=2000)
    p.add_argument("--include_hiv", action="store_true",
                   help="Inclure HIV (sinon pretrain pur non-HIV pour transfer HIV)")
    p.add_argument("--build_pretrain", action="store_true",
                   help="Construit le dataset binaire pretrain (actifs + decoys ZINC)")
    args = p.parse_args()

    if args.build_pretrain:
        build_antiviral_pretrain_dataset()
    else:
        download_multivirus_antivirals(
            save_dir=args.save_dir,
            min_pchembl=args.min_pchembl,
            max_per_target=args.max_per_target,
            include_hiv=args.include_hiv,
        )
