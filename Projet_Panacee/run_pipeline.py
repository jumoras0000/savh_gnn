"""
Pipeline Orchestrator – Panacée.

Orchestre les 3 phases d'entraînement et l'analyse avancée.

Usage :
  python run_pipeline.py --all               # Tout exécuter (Phase 1 + 2 + 3)
  python run_pipeline.py --phase1            # Phase 1 uniquement
  python run_pipeline.py --phase2            # Phase 2 uniquement
  python run_pipeline.py --phase3            # Phase 3 uniquement
  python run_pipeline.py --analyze           # Analyse avancée seulement
  python run_pipeline.py --status            # Afficher l'état du pipeline
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_checkpoint(path: str) -> bool:
    """Vérifie si un checkpoint existe."""
    return os.path.exists(path)


def get_python() -> str:
    """Retourne le chemin de l'interpréteur Python."""
    return sys.executable


def run_phase(script: str, args: list, phase_name: str) -> bool:
    """
    Lance une phase dans un sous-processus.

    Args:
        script: nom du script (ex: "run_phase1.py")
        args: arguments supplémentaires
        phase_name: nom pour l'affichage

    Returns:
        True si succès
    """
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        print(f"  ERREUR: Script introuvable : {script_path}")
        return False

    print(f"\n{'='*80}")
    print(f"  {phase_name}")
    print(f"  Script: {script_path}")
    print(f"  Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    t0 = time.time()
    cmd = [get_python(), str(script_path)] + args

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"\n  {phase_name} terminée en {timedelta(seconds=int(elapsed))}")
            return True
        else:
            print(f"\n  ERREUR: {phase_name} échouée (code={result.returncode})")
            return False

    except Exception as e:
        print(f"\n  ERREUR: {e}")
        return False


def print_status():
    """Affiche l'état complet du pipeline."""
    from src.config import CHECKPOINT_DIR

    print("\n" + "=" * 70)
    print("  ÉTAT DU PIPELINE PANACÉE")
    print("=" * 70)

    phases = [
        ("Phase 1 – Pré-entraînement GNN",
         CHECKPOINT_DIR / "phase1" / "sovereign_encoder_v1.pth"),
        ("Phase 2 – Fine-tuning Toxicité",
         CHECKPOINT_DIR / "phase2" / "best_toxicity_model.pth"),
        ("Phase 3 – Multi-propriétés + Raisonnement",
         CHECKPOINT_DIR / "phase3" / "panacee_phase3_complete.pth"),
    ]

    all_done = True
    for name, path in phases:
        exists = check_checkpoint(str(path))
        status = "COMPLETED" if exists else "A FAIRE"
        icon = "[OK]" if exists else "[  ]"
        print(f"  {icon} {name}")
        if exists:
            stat = os.stat(str(path))
            size_mb = stat.st_size / 1e6
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"        {path}")
            print(f"        {size_mb:.1f} MB | {mod_time}")
        else:
            all_done = False
            print(f"        Checkpoint manquant : {path}")

    # Modules disponibles
    print(f"\n  MODULES AVANCÉS:")
    modules = [
        ("GPU Manager", "src/utils/gpu_manager.py"),
        ("Error Handler", "src/utils/error_handler.py"),
        ("Connaissances médicales", "src/knowledge/medical_rules.py"),
        ("Recherche web", "src/knowledge/web_search.py"),
        ("Raisonneur avancé", "src/models/advanced_reasoner.py"),
    ]
    for name, path in modules:
        exists = (PROJECT_ROOT / path).exists()
        icon = "[OK]" if exists else "[  ]"
        print(f"  {icon} {name} ({path})")

    # GPU
    print(f"\n  MATÉRIEL:")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            print(f"  [OK] GPU: {name} ({total/1e9:.1f} GB VRAM, {free/1e9:.1f} GB libre)")
        else:
            print(f"  [  ] GPU: CUDA non disponible, mode CPU")
    except Exception:
        print(f"  [  ] GPU: information indisponible")

    print(f"\n  PRÊT POUR L'ANALYSE: {'OUI' if all_done else 'NON'}")
    if not all_done:
        print("  Lancez: python run_pipeline.py --all")
    else:
        print("  Lancez: python run_pipeline.py --analyze --smiles \"CCO,CC(=O)O\"")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Panacée – Orchestration des phases"
    )

    # Phases
    parser.add_argument("--all", action="store_true",
                        help="Exécuter les 3 phases séquentiellement")
    parser.add_argument("--phase1", action="store_true",
                        help="Lancer Phase 1 (pré-entraînement)")
    parser.add_argument("--phase2", action="store_true",
                        help="Lancer Phase 2 (toxicité)")
    parser.add_argument("--phase3", action="store_true",
                        help="Lancer Phase 3 (multi-propriétés)")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyse avancée")
    parser.add_argument("--status", action="store_true",
                        help="Afficher l'état du pipeline")

    # Options communes
    parser.add_argument("--download", action="store_true",
                        help="Télécharger les données nécessaires")
    parser.add_argument("--force", action="store_true",
                        help="Relancer même si le checkpoint existe")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    # Options d'analyse
    parser.add_argument("--smiles", type=str, default=None,
                        help="SMILES séparés par virgules")
    parser.add_argument("--smiles_file", type=str, default=None,
                        help="Fichier CSV avec colonne smiles")
    parser.add_argument("--combo_size", type=int, default=2,
                        help="Taille des combinaisons")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--use_mcts", action="store_true")
    parser.add_argument("--use_web", action="store_true")
    parser.add_argument("--indication", type=str, default="",
                        help="Indication thérapeutique visée")
    parser.add_argument("--output", type=str, default=None,
                        help="Fichier de sortie JSON")

    args = parser.parse_args()

    # ── Status ──
    if args.status or (not args.all and not args.phase1 and not args.phase2
                       and not args.phase3 and not args.analyze):
        print_status()
        return

    from src.config import CHECKPOINT_DIR

    # ── Phase 1 ──
    if args.all or args.phase1:
        ckpt1 = str(CHECKPOINT_DIR / "phase1" / "sovereign_encoder_v1.pth")
        if check_checkpoint(ckpt1) and not args.force:
            print(f"  Phase 1 déjà complétée ({ckpt1})")
            print("  Utilisez --force pour relancer")
        else:
            phase1_args = ["--download"] if args.download else []
            if args.device:
                phase1_args += ["--device", args.device]

            ok = run_phase("run_phase1.py", phase1_args, "PHASE 1 – PRÉ-ENTRAÎNEMENT GNN")
            if not ok and not args.force:
                print("  Phase 1 échouée, arrêt du pipeline.")
                sys.exit(1)

    # ── Phase 2 ──
    if args.all or args.phase2:
        ckpt2 = str(CHECKPOINT_DIR / "phase2" / "best_toxicity_model.pth")
        if check_checkpoint(ckpt2) and not args.force:
            print(f"  Phase 2 déjà complétée ({ckpt2})")
        else:
            phase2_args = ["--download"] if args.download else []
            if args.device:
                phase2_args += ["--device", args.device]

            ok = run_phase("run_phase2.py", phase2_args, "PHASE 2 – FINE-TUNING TOXICITÉ")
            if not ok and not args.force:
                print("  Phase 2 échouée, arrêt du pipeline.")
                sys.exit(1)

    # ── Phase 3 ──
    if args.all or args.phase3:
        ckpt3 = str(CHECKPOINT_DIR / "phase3" / "panacee_phase3_complete.pth")
        if check_checkpoint(ckpt3) and not args.force:
            print(f"  Phase 3 déjà complétée ({ckpt3})")
        else:
            phase3_args = ["--download"] if args.download else []
            if args.device:
                phase3_args += ["--device", args.device]

            ok = run_phase("run_phase3.py", phase3_args, "PHASE 3 – MULTI-PROPRIÉTÉS + IA")
            if not ok:
                print("  Phase 3 échouée.")
                sys.exit(1)

    # ── Analyse avancée ──
    if args.analyze:
        ckpt3 = str(CHECKPOINT_DIR / "phase3" / "panacee_phase3_complete.pth")
        if not check_checkpoint(ckpt3):
            print(f"  ERREUR: Phase 3 non complétée. Lancez --phase3 d'abord.")
            sys.exit(1)

        # Charger les SMILES
        smiles_list = []
        if args.smiles:
            smiles_list = [s.strip() for s in args.smiles.split(",")]
        elif args.smiles_file:
            import pandas as pd
            df = pd.read_csv(args.smiles_file)
            for col in ["smiles", "SMILES", "canonical_smiles", "mol"]:
                if col in df.columns:
                    smiles_list = df[col].dropna().tolist()
                    break

        if not smiles_list:
            print("  ERREUR: Spécifiez --smiles ou --smiles_file")
            sys.exit(1)

        print(f"\n  Analyse de {len(smiles_list)} molécules...")
        from src.analysis.combinatorial_engine import PanaceeAnalyzer
        analyzer = PanaceeAnalyzer(checkpoint_path=ckpt3)
        report = analyzer.advanced_analysis(
            smiles_list=smiles_list,
            combo_size=args.combo_size,
            top_k=args.top_k,
            use_mcts=args.use_mcts,
            use_pareto=True,
            use_knowledge=True,
            use_web=args.use_web,
            indication=args.indication,
            output_file=args.output,
        )

        print(f"\n  Analyse terminée. Score final: {report.get('advanced_reasoning', {}).get('final_score', 'N/A')}")


if __name__ == "__main__":
    main()
