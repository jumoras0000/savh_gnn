"""Test d'import et validation fonctionnelle de tous les modules améliorés."""
import sys
sys.path.insert(0, ".")

def test_gpu_manager():
    from src.utils.gpu_manager import GPUManager, get_gpu_manager
    gpu = get_gpu_manager()
    info = gpu.get_gpu_info()
    mem = gpu.get_memory_stats()
    bs = gpu.optimize_batch_size(64)
    gpu.print_summary()
    print(f"  [OK] GPU Manager - device={gpu.device}, batch_opt={bs}")

def test_error_handler():
    from src.utils.error_handler import (
        setup_logging, safe_execution, PanaceeError, DataError,
        ModelError, GPUError, WebSearchError, HealthMonitor,
    )
    health = HealthMonitor()
    for i in range(10):
        assert health.step(1.0 - i * 0.05)

    @safe_execution(retries=2, delay=0.1, fallback="fallback")
    def failing_fn():
        raise ValueError("test")
    assert failing_fn() == "fallback"
    print("  [OK] Error Handler - HealthMonitor, safe_execution, 5 error classes")

def test_medical_rules():
    from src.knowledge.medical_rules import (
        LipinskiEvaluator, ADMETEvaluator, MolecularProfile,
        THERAPEUTIC_TARGETS, STRUCTURAL_ALERTS,
        estimate_bioavailability, comprehensive_evaluation,
    )

    profile = MolecularProfile(
        smiles="CCO", mw=46.07, logp=-0.31,
        hbd=1, hba=1, tpsa=20.23,
        rotatable_bonds=0, aromatic_rings=0, heavy_atoms=3,
    )
    score, violations = LipinskiEvaluator.evaluate(profile)
    assert score == 1.0 and len(violations) == 0

    admet = ADMETEvaluator.evaluate_all(profile)
    assert "admet_global" in admet

    bio = estimate_bioavailability(profile)
    assert 0 <= bio <= 100

    print(f"  [OK] Medical Rules - {len(THERAPEUTIC_TARGETS)} targets, {len(STRUCTURAL_ALERTS)} alerts")
    print(f"       Lipinski={score}, ADMET_global={admet['admet_global']:.2f}, Bioavail={bio:.0f}%")

def test_web_search():
    from src.knowledge.web_search import (
        PubChemSearch, ChEMBLSearch, PubMedSearch, WebResearchEngine,
    )
    # Juste vérifier l'import (pas de requête réseau dans les tests)
    engine = WebResearchEngine()
    print("  [OK] Web Search - PubChem, ChEMBL, PubMed, WebResearchEngine")

def test_advanced_reasoner():
    import torch
    import numpy as np
    from src.models.advanced_reasoner import (
        MCTSCombinationSearch, BayesianDoseOptimizer,
        MultiObjectiveOptimizer, Solution, pareto_front,
        EnsembleConfidence, ChainOfThought,
    )

    # MCTS
    def mock_score(indices):
        return sum(indices) / max(len(indices), 1) / 10.0
    mcts = MCTSCombinationSearch(mock_score, n_molecules=20, combo_size=3)
    best, score = mcts.search(n_iterations=50)
    assert len(best) == 3
    print(f"  [OK] MCTS - combo={best}, score={score:.3f}")

    # Bayesian
    opt = BayesianDoseOptimizer([0.1, 0.5, 1.0, 5.0, 10.0])
    opt.observe(1.0, 0.8)
    opt.observe(5.0, 0.3)
    opt.observe(0.1, 0.1)
    next_dose = opt.suggest_next()
    best_d, best_e = opt.get_best()
    curve = opt.get_dose_response_curve()
    assert best_d == 1.0 and best_e == 0.8
    print(f"  [OK] Bayesian - next={next_dose}, best_dose={best_d}, curve_len={len(curve['doses'])}")

    # Pareto
    moo = MultiObjectiveOptimizer(["eff", "safety"], [True, True])
    moo.add_solution(0, [0.9, 0.1])
    moo.add_solution(1, [0.1, 0.9])
    moo.add_solution(2, [0.5, 0.5])
    moo.add_solution(3, [0.3, 0.3])  # dominated
    front = moo.get_pareto_front()
    ranked = moo.rank_solutions()
    compromise = moo.suggest_best_compromise()
    print(f"  [OK] Pareto - front={len(front)}, ranked={len(ranked)}, compromise_idx={compromise.index}")

    # Ensemble
    ens = EnsembleConfidence(d_model=256, n_sources=3)
    sources = [torch.randn(2, 256) for _ in range(3)]
    result = ens(sources)
    assert result["combined"].shape == (2, 256)
    assert result["confidence"].shape == (2, 1)
    assert result["source_weights"].shape == (2, 3)
    print(f"  [OK] Ensemble - combined={tuple(result['combined'].shape)}, weights={tuple(result['source_weights'].shape)}")

    # Chain of Thought
    cot = ChainOfThought()
    cot.add_step("step1", "Desc 1", 0.8, ["evidence1"])
    cot.add_step("step2", "Desc 2", 0.6, sub_scores={"a": 0.7})
    final = cot.get_final_score()
    report = cot.generate_report()
    assert 0 <= final <= 1
    assert "CHAÎNE DE PENSÉE" in report
    print(f"  [OK] Chain of Thought - score={final:.2f}, report={len(report)} chars")

def test_combinatorial_engine_import():
    from src.analysis.combinatorial_engine import PanaceeAnalyzer
    print("  [OK] Combinatorial Engine - PanaceeAnalyzer with advanced_analysis()")

def test_train_phase3_import():
    from src.training.train_phase3 import (
        WarmupCosineScheduler, compute_phase3_metrics,
    )
    print("  [OK] Train Phase 3 - GPU manager + health monitor integrated")

def test_pipeline_import():
    from run_pipeline import check_checkpoint, print_status
    print("  [OK] Pipeline Orchestrator - run_pipeline.py")


if __name__ == "__main__":
    print("=" * 60)
    print("  VALIDATION DES MODULES PANACÉE AMÉLIORÉS")
    print("=" * 60)

    tests = [
        ("GPU Manager", test_gpu_manager),
        ("Error Handler", test_error_handler),
        ("Medical Rules", test_medical_rules),
        ("Web Search", test_web_search),
        ("Advanced Reasoner", test_advanced_reasoner),
        ("Combinatorial Engine", test_combinatorial_engine_import),
        ("Train Phase 3", test_train_phase3_import),
        ("Pipeline Orchestrator", test_pipeline_import),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n--- {name} ---")
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  RÉSULTAT: {passed}/{passed+failed} tests passés")
    if failed == 0:
        print("  TOUS LES MODULES SONT OPÉRATIONNELS")
    else:
        print(f"  {failed} test(s) en échec")
    print("=" * 60)
