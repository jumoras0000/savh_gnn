"""Prépare le dossier aveugle à transmettre à un laboratoire partenaire.

Le ZIP ne contient ni clé de décodage, ni prédictions, ni données personnelles.
Il décrit des résultats attendus, pas un protocole d'essai biologique.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.translational import package, sha256

OUT = ROOT / "data" / "experiment_package"


def _font() -> str:
    for path in ("C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            pdfmetrics.registerFont(TTFont("BriefSans", path))
            return "BriefSans"
    raise FileNotFoundError("Police Unicode TrueType introuvable")


def build() -> Path:
    manifest, _, _ = package(ROOT)
    font = _font()
    pdf = OUT / "demande_essais_aveugles.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=2.2*cm,
                            rightMargin=2.2*cm, topMargin=2*cm,
                            bottomMargin=1.8*cm,
                            title="Panacée — demande de mesures aveugles",
                            author="Projet Panacée")
    title = ParagraphStyle("title", fontName=font, fontSize=16, leading=21,
                           textColor=colors.HexColor("#18333b"), spaceAfter=15)
    section = ParagraphStyle("section", fontName=font, fontSize=10.5, leading=14,
                             textColor=colors.HexColor("#116a6e"), spaceBefore=11,
                             spaceAfter=5)
    body = ParagraphStyle("body", fontName=font, fontSize=9, leading=13,
                          spaceAfter=8)
    small = ParagraphStyle("small", fontName=font, fontSize=7.8, leading=11)
    story = [Paragraph("Demande de mesures indépendantes et aveugles", title),
             Paragraph("Projet Panacée · dossier de transmission du 10 novembre 2026", body)]

    def P(value: str, style=body):
        story.append(Paragraph(value, style))

    P("Objet", section)
    P("Évaluer un ensemble gelé de 26 structures proposé par un modèle de classement VIH. Les identifiants PZ-001 à PZ-026 ne révèlent ni la catégorie de sélection ni le score informatique. Les résultats doivent être retournés avant tout décodage. Aucun composé du lot n'est présenté comme antiviral confirmé.")
    P("Fichiers transmis", section)
    P("<b>submission_blinded.csv</b> : identifiant et SMILES. <b>assay_results_template.csv</b> : schéma de retour. Le laboratoire peut utiliser son propre système d'information si les champs, unités, limites de mesure et identifiants sont conservés. La clé de décodage et les prédictions ne figurent pas dans ce dossier.")
    P("Résultats à retourner", section)
    rows = [
        ("Identification", "blind_id, assay_id, laboratoire, date de mesure, réplicat"),
        ("Activité", "EC50 en µM, ou borne >x si non quantifiable ; résultat brut et intervalle/erreur si disponible"),
        ("Cytotoxicité", "CC50 en µM, ou borne >x ; contexte cellulaire et limites de mesure"),
        ("Qualité", "Contrôles positifs/négatifs validés par le laboratoire, critères d'acceptation, échecs techniques, répétitions"),
        ("Traçabilité", "Méthode d'identification chimique, pureté si déterminée, version de l'essai, fichier source daté"),
    ]
    table = Table([[Paragraph(a, small), Paragraph(b, small)] for a,b in rows],
                  colWidths=[3.7*cm,12.8*cm])
    table.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LINEBELOW",(0,0),(-1,-1),.25,colors.HexColor("#d4dddf")),
        ("LEFTPADDING",(0,0),(-1,-1),5),
        ("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
    ]))
    story.append(table)
    P("Lecture et conservation", section)
    P("Deux réplicats par structure sont demandés pour le contrôle informatique initial. Le laboratoire définit et atteste ses propres critères de qualité ; le logiciel ne juge pas la validité d'un essai biologique. Les valeurs censurées restent des bornes, non des concentrations exactes. Les interprétations de mécanisme, résistance et sélectivité nécessitent des études distinctes.")
    P("Traçabilité du lot", section)
    P(f"Modèle de sélection : {manifest['model_id']} ({manifest['model_created']}). "
      f"Nombre de structures : {manifest['count']}. Empreinte SHA-256 du CSV aveugle : "
      f"{manifest['submission_sha256']}.", small)
    P("Ce document est une demande de données pour une étude exploratoire. Il n'atteste ni une efficacité clinique ni une autorisation de développement.", small)
    doc.build(story)

    public = {"model_id": manifest["model_id"],
              "model_created": manifest["model_created"],
              "n_structures": manifest["count"],
              "submission_sha256": manifest["submission_sha256"],
              "request_pdf_sha256": sha256(pdf),
              "contents": ["submission_blinded.csv", "assay_results_template.csv",
                           "demande_essais_aveugles.pdf"]}
    public_file = OUT / "manifest_public.json"
    public_file.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    bundle = OUT / "dossier_laboratoire_aveugle.zip"
    names = public["contents"] + ["manifest_public.json"]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(OUT / name, arcname=name)
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == set(names)
        assert "key_local.json" not in archive.namelist()
        assert "predictions_local.json" not in archive.namelist()
    (OUT / "dossier_laboratoire_aveugle.sha256").write_text(
        sha256(bundle) + "\n", encoding="ascii")
    return bundle


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    file = build()
    print(json.dumps({"file": str(file), "size": file.stat().st_size,
                      "sha256": hashlib.sha256(file.read_bytes()).hexdigest()},
                     ensure_ascii=False))
