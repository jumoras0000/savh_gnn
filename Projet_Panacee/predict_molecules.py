"""
Script de prédiction et analyse – Panacée Phase 3.

Utilise le modèle Phase 3 entraîné pour :
  - Prédire les propriétés d'une ou plusieurs molécules
  - Analyser les combinaisons et synergies
  - Recommander les doses optimales
  - Générer un rapport complet avec scores de réussite

Usage :
    # Prédire les propriétés d'une molécule
    python predict_molecules.py --smiles "Cc1cn(C2CC(N=[N+]=[N-])C(CO)O2)c(=O)[nH]c1=O"

    # Analyser plusieurs molécules (combinaisons)
    python predict_molecules.py --smiles "CCO,CC(=O)O,CC(=O)NC1=CC=C(O)C=C1"

    # Depuis un fichier CSV
    python predict_molecules.py --smiles_file candidates.csv

    # Rapport complet avec fichier de sortie
    python predict_molecules.py --smiles_file candidates.csv --report rapport.json

    # Prédiction seule (sans analyse combinatoire)
    python predict_molecules.py --smiles "CCO" --predict_only

    # Combinaisons de taille 3
    python predict_molecules.py --smiles_file candidates.csv --combo_size 3 --top_k 10
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Panacée – Prédiction et Analyse de Molécules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python predict_molecules.py --smiles "CCO"
  python predict_molecules.py --smiles "CCO,CC(=O)O" --combo_size 2
  python predict_molecules.py --smiles_file molecules.csv --report output.json
  python predict_molecules.py --smiles_file molecules.csv --predict_only
        """,
    )
    p.add_argument("--smiles", type=str, default=None,
                   help="SMILES séparés par des virgules")
    p.add_argument("--smiles_file", type=str, default=None,
                   help="Fichier CSV avec colonne 'smiles'")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Chemin du checkpoint Phase 3")
    p.add_argument("--combo_size", type=int, default=2,
                   help="Taille des combinaisons à tester (défaut: 2)")
    p.add_argument("--top_k", type=int, default=5,
                   help="Nombre de meilleures combinaisons (défaut: 5)")
    p.add_argument("--report", type=str, default=None,
                   help="Fichier de sortie JSON pour le rapport")
    p.add_argument("--predict_only", action="store_true",
                   help="Prédire les propriétés sans analyse combinatoire")
    p.add_argument("--device", type=str, default=None,
                   help="Device (cuda ou cpu)")
    args = p.parse_args()

    # ── Bannière ──
    print("=" * 80)
    print("🧪 PANACÉE – ANALYSE DE MOLÉCULES")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ── Récupérer les SMILES ──
    import pandas as pd
    smiles_list = []

    if args.smiles:
        smiles_list = [s.strip() for s in args.smiles.split(",") if s.strip()]
        print(f"\n📥 {len(smiles_list)} molécule(s) fournies en ligne de commande")

    elif args.smiles_file:
        if not os.path.exists(args.smiles_file):
            print(f"❌ Fichier introuvable : {args.smiles_file}")
            sys.exit(1)

        df = pd.read_csv(args.smiles_file)
        smiles_col = None
        for candidate in ["smiles", "SMILES", "canonical_smiles", "ids", "mol", "molecule"]:
            if candidate in df.columns:
                smiles_col = candidate
                break

        if smiles_col is None:
            print(f"❌ Pas de colonne SMILES trouvée dans {args.smiles_file}")
            print(f"   Colonnes disponibles : {list(df.columns)}")
            sys.exit(1)

        smiles_list = df[smiles_col].dropna().astype(str).tolist()
        print(f"\n📥 {len(smiles_list)} molécules chargées depuis {args.smiles_file}")

    else:
        print("❌ Spécifiez --smiles ou --smiles_file")
        print("   Exemple : python predict_molecules.py --smiles \"CCO,CC(=O)O\"")
        sys.exit(1)

    if not smiles_list:
        print("❌ Aucune molécule valide fournie")
        sys.exit(1)

    # ── Lister les molécules ──
    print("\n📋 Molécules à analyser :")
    for i, smi in enumerate(smiles_list, 1):
        display = smi if len(smi) <= 60 else smi[:57] + "..."
        print(f"   {i}. {display}")

    # ── Charger le modèle ──
    print("\n🔄 Chargement du modèle...")
    try:
        from src.analysis.combinatorial_engine import PanaceeAnalyzer
        analyzer = PanaceeAnalyzer(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("   Lancez d'abord l'entraînement Phase 3 :")
        print("   python run_phase3.py --download")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement du modèle : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Mode prédiction uniquement ──
    if args.predict_only:
        print("\n" + "=" * 80)
        print("📊 PRÉDICTIONS INDIVIDUELLES")
        print("=" * 80)

        all_results = []
        for smi in smiles_list:
            result = analyzer.predict_properties(smi)
            if result:
                analyzer._print_molecule_summary(result)
                all_results.append(result)
            else:
                print(f"\n  ⚠ Impossible d'analyser : {smi}")

        # Sauvegarder si demandé
        if args.report:
            report_data = {
                "date": datetime.now().isoformat(),
                "mode": "predict_only",
                "results": all_results,
            }
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📄 Résultats sauvegardés : {args.report}")

        print(f"\n✅ {len(all_results)} molécules analysées")
        return

    # ── Mode rapport complet avec analyse combinatoire ──
    output_file = args.report or f"rapport_panacee_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        analyzer.generate_report(
            smiles_list,
            combo_size=args.combo_size,
            top_k=args.top_k,
            output_file=output_file,
        )
    except Exception as e:
        print(f"\n❌ Erreur analyse : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ Analyse terminée")
    print(f"📄 Rapport : {output_file}")


if __name__ == "__main__":
    main()
