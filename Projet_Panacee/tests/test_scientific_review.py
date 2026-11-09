"""Contrôle de lisibilité et de contenu de la revue scientifique publiée."""
from pathlib import Path

import pytest


def test_scientific_review_has_searchable_text_and_figures():
    pypdf = pytest.importorskip("pypdf")
    file = Path(__file__).resolve().parents[1] / "reports" / (
        "revue_scientifique_panacee_2026-11.pdf")
    reader = pypdf.PdfReader(file)
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert 5 <= len(reader.pages) <= 12
    assert sum(len(page.images) for page in reader.pages) >= 8
    assert all(phrase in text for phrase in (
        "Phase 1", "Phase 2", "Phase 3", "Phase 4", "32/34",
        "aucune", "Traçabilité", "Références scientifiques"))
    assert "ChatGPT" not in text
    assert "généré par une IA" not in text
