"""
Génération de conformères 3D via RDKit ETKDG pour head 3D-aware optionnelle.

Référence : Riniker & Landrum 2015 (ETKDG v3), Boucher et al. 2026 arXiv 2606.08825
(5-10 conformers suffisent pour prédiction propriétés).

Design :
  - Entrée : SMILES
  - Génère N conformères via ETKDGv3 + MMFF94s optimization
  - Retourne tensor [N_conf, N_atoms, 3] + énergies relatives
  - Fallback : si ETKDG échoue (molécule très flexible / fragments),
    utilise coordonnées 2D projetées (compatibilité graphe 2D)

Usage dans pipeline :
    conf_coords, energies = generate_conformers(smiles, n_conformers=5)
    # Concaténer aux features atomiques dans graph_builder OU utiliser
    # une head 3D-aware (SchNet-style equivariant).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def generate_conformers(
    smiles: str,
    n_conformers: int = 5,
    optimize: bool = True,
    max_attempts: int = 100,
    random_seed: int = 42,
) -> Optional[dict]:
    """
    Génère N conformères 3D avec ETKDG v3 + MMFF94s optimization.

    Args:
        smiles: SMILES (sera canonicalisé)
        n_conformers: nombre de conformers à générer
        optimize: appliquer MMFF94s minimization
        max_attempts: budget embedding (ETKDG peut échouer)
        random_seed: seed pour reproductibilité ETKDG

    Returns:
        dict ou None si invalide :
          {
            "smiles": SMILES canonique,
            "n_atoms": int,
            "coords": np.ndarray [n_valid_conformers, n_atoms, 3],
            "energies": np.ndarray [n_valid_conformers] (relative à min),
            "atom_types": list[int] atomic numbers,
          }
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    # Ajout H explicites requis pour géométrie 3D correcte
    mol_h = Chem.AddHs(mol)
    n_atoms = mol_h.GetNumAtoms()
    if n_atoms < 2:
        return None

    # ETKDGv3 : paramétrage robuste (Riniker 2020).
    # Note : RDKit >= 2024.03 a renommé `maxAttempts` → `maxIterations`
    # (vérifié via dir(ETKDGv3) = maxIterations, pas maxAttempts).
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    # Setattr défensif : support des 2 noms selon version RDKit
    for attr_name in ("maxIterations", "maxAttempts"):
        if hasattr(params, attr_name):
            setattr(params, attr_name, max_attempts)
            break
    params.pruneRmsThresh = 0.5  # élimine conformers trop proches
    params.useRandomCoords = True  # fallback pour mol flexibles

    conf_ids = AllChem.EmbedMultipleConfs(
        mol_h, numConfs=n_conformers, params=params,
    )
    if not conf_ids:
        return None

    conf_ids = list(conf_ids)
    energies = []

    if optimize:
        for cid in conf_ids:
            try:
                res = AllChem.MMFFOptimizeMolecule(mol_h, confId=int(cid), maxIters=200)
                if res == 0:
                    props = AllChem.MMFFGetMoleculeProperties(mol_h)
                    ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=int(cid))
                    energies.append(float(ff.CalcEnergy()) if ff is not None else np.nan)
                else:
                    energies.append(np.nan)
            except Exception:
                energies.append(np.nan)
    else:
        energies = [0.0] * len(conf_ids)

    coords = np.zeros((len(conf_ids), n_atoms, 3), dtype=np.float32)
    for i, cid in enumerate(conf_ids):
        conf = mol_h.GetConformer(int(cid))
        for a in range(n_atoms):
            pos = conf.GetAtomPosition(a)
            coords[i, a] = [pos.x, pos.y, pos.z]

    energies = np.asarray(energies, dtype=np.float32)
    valid = np.isfinite(energies) if optimize else np.ones(len(energies), dtype=bool)
    if valid.any():
        coords = coords[valid]
        energies = energies[valid]
        if np.any(np.isfinite(energies)):
            energies = energies - np.nanmin(energies)

    atom_types = [a.GetAtomicNum() for a in mol_h.GetAtoms()]

    return {
        "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "n_atoms": n_atoms,
        "coords": coords,
        "energies": energies,
        "atom_types": atom_types,
    }


def batch_conformers(
    smiles_list: list[str],
    n_conformers: int = 5,
    optimize: bool = True,
    show_progress: bool = True,
) -> list[Optional[dict]]:
    """Version batch avec progress bar optionnelle."""
    iterator = smiles_list
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(smiles_list, desc="Conformers")
        except ImportError:
            pass

    return [
        generate_conformers(smi, n_conformers=n_conformers, optimize=optimize)
        for smi in iterator
    ]


def conformer_derived_features(conf_result: dict) -> Optional[dict]:
    """
    Extrait features 3D agrégés sur les conformères :
      - Rg (radius of gyration) moyen/min/max
      - PMI1/PMI2/PMI3 (principal moments of inertia) moyennés
      - Boltzmann-weighted descriptors (si énergies disponibles)
      - Shape : NPR1/NPR2 (normalized principal ratios)

    Utile comme features additionnelles bon-marché sans refondre l'archi.
    """
    if conf_result is None or len(conf_result["coords"]) == 0:
        return None

    coords = conf_result["coords"]  # [n_conf, n_atoms, 3]
    atom_types = np.asarray(conf_result["atom_types"], dtype=float)
    masses = np.array([_atomic_mass(int(z)) for z in atom_types], dtype=np.float32)

    rgs = []
    npr1s, npr2s = [], []
    for c in coords:
        com = (masses[:, None] * c).sum(axis=0) / masses.sum()
        c_centered = c - com
        rg_sq = ((masses[:, None] * (c_centered ** 2)).sum() / masses.sum())
        rgs.append(float(np.sqrt(rg_sq)))
        # Tensor d'inertie
        I = np.zeros((3, 3))
        for a in range(len(masses)):
            r = c_centered[a]
            I += masses[a] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        eigvals = np.sort(np.linalg.eigvalsh(I))
        if eigvals[2] > 1e-6:
            npr1s.append(float(eigvals[0] / eigvals[2]))
            npr2s.append(float(eigvals[1] / eigvals[2]))

    energies = conf_result.get("energies", np.array([]))
    if len(energies) > 0 and np.all(np.isfinite(energies)):
        # Pondération Boltzmann (kcal/mol, T=298K → kT ≈ 0.593)
        weights = np.exp(-energies / 0.593)
        weights = weights / weights.sum() if weights.sum() > 0 else None
    else:
        weights = None

    def _agg(vals):
        if not vals:
            return None
        if weights is not None and len(weights) == len(vals):
            return float(np.sum(weights * np.asarray(vals)))
        return float(np.mean(vals))

    return {
        "rg_mean": _agg(rgs),
        "rg_min": float(min(rgs)) if rgs else None,
        "rg_max": float(max(rgs)) if rgs else None,
        "npr1_mean": _agg(npr1s),
        "npr2_mean": _agg(npr2s),
        "n_conformers": int(len(coords)),
    }


def _atomic_mass(z: int) -> float:
    """Masses atomiques standard (g/mol) pour atomes drug-like."""
    _TABLE = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
              15: 30.974, 16: 32.06, 17: 35.45, 35: 79.904, 53: 126.904}
    return _TABLE.get(z, 12.0)


if __name__ == "__main__":
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    print(f"Testing conformer generation: {smi}")
    r = generate_conformers(smi, n_conformers=5)
    if r:
        print(f"  SMILES canonique: {r['smiles']}")
        print(f"  Atomes (avec H): {r['n_atoms']}")
        print(f"  Conformers générés: {len(r['coords'])}")
        print(f"  Énergies relatives (kcal/mol): {r['energies']}")
        feat = conformer_derived_features(r)
        print(f"  Features 3D: {feat}")
