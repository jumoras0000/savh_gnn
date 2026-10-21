"""
Fingerprints Morgan centralisés — API moderne RDKit avec fallback version ancienne.

RDKit >= 2023.03 déprécie `AllChem.GetMorganFingerprintAsBitVect` en faveur
de `rdFingerprintGenerator.GetMorganGenerator().GetFingerprint()`.

Ce module encapsule la nouvelle API avec fallback transparent → aucune
dépendance de version dans le code appelant.

Références :
  - https://greglandrum.github.io/rdkit-blog/posts/2023-01-13-fingerprint-generator-example.html
  - https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html
"""
from __future__ import annotations

from typing import Any

try:
    from rdkit.Chem import rdFingerprintGenerator
    _HAS_MODERN_API = True
except ImportError:
    _HAS_MODERN_API = False

# Cache les générateurs par (radius, n_bits) pour éviter réinstanciation coûteuse
_GENERATOR_CACHE: dict[tuple[int, int], Any] = {}


def _get_generator(radius: int, n_bits: int):
    """Retourne un générateur Morgan réutilisable (thread-safe pour lectures)."""
    key = (radius, n_bits)
    if key not in _GENERATOR_CACHE:
        _GENERATOR_CACHE[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits,
        )
    return _GENERATOR_CACHE[key]


def morgan_fp_bitvect(mol, radius: int = 2, n_bits: int = 2048):
    """
    Morgan fingerprint ECFPn → BitVect RDKit (compatible BulkTanimotoSimilarity).

    API unifiée : utilise rdFingerprintGenerator (RDKit >= 2023.03) ou fallback.
    Mol invalide → ValueError (l'appelant doit vérifier avec MolFromSmiles d'abord).
    """
    if mol is None:
        raise ValueError("mol est None — vérifier MolFromSmiles en amont")

    if _HAS_MODERN_API:
        gen = _get_generator(radius, n_bits)
        return gen.GetFingerprint(mol)

    # Fallback RDKit ancien (devrait rarement s'exécuter en production)
    from rdkit.Chem import AllChem
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def morgan_fp_count(mol, radius: int = 2, n_bits: int = 2048):
    """
    Count fingerprint Morgan (uint32) — alternative au bit vector.
    Utile pour similarity count-based (Tanimoto Dice sur comptes).
    """
    if mol is None:
        raise ValueError("mol est None — vérifier MolFromSmiles en amont")

    if _HAS_MODERN_API:
        gen = _get_generator(radius, n_bits)
        return gen.GetCountFingerprint(mol)

    from rdkit.Chem import AllChem
    return AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)


def smiles_to_morgan_array(smiles_list, radius: int = 2, n_bits: int = 2048):
    """
    Convertit batch de SMILES → np.ndarray [N, n_bits] uint8 + masque validité.

    Utility haute-performance pour baselines (RF, LogReg) et Tanimoto AD.
    """
    import numpy as np
    from rdkit import Chem

    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    valid = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = morgan_fp_bitvect(mol, radius, n_bits)
        X[i] = np.array(fp)
        valid[i] = True
    return X, valid
