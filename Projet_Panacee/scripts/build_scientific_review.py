"""Construit une revue scientifique traçable des quatre phases Panacée.

Toutes les figures chiffrées sont recalculées depuis les artefacts du dépôt.
Usage : python scripts/build_scientific_review.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import Draw
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.studio_support import verify_registry

OUT = ROOT / "reports"
FIG = OUT / "figures"
PDF = OUT / "revue_scientifique_panacee_2026-11.pdf"
INK = "#18333b"
TEAL = "#116a6e"
ORANGE = "#b4672d"
GRAY = "#64737a"
LIGHT = "#e8eef0"


def source(path: str):
    return ROOT / path


def read_json(path: str):
    return json.loads(source(path).read_text(encoding="utf-8"))


def epochs(phase: str) -> list[dict]:
    rows = [json.loads(line) for line in source(
        f"checkpoints/{phase}/live_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [row for row in rows if row.get("_type") == "epoch"]


def set_plot_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": GRAY, "ytick.color": GRAY,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})


def save_fig(fig, name: str) -> Path:
    path = FIG / name
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def make_figures(p1, p2, p3, augmented, zinc, rep, zinc_summary, rep_raw_count):
    set_plot_style()
    paths = {}
    fig, ax = plt.subplots(figsize=(10.2, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
    boxes = [
        (0.08, "1 · Représentation", "ZINC · MGM", "encodeur"),
        (2.55, "2 · Toxicité", "Tox21 · 12 tâches", "transfert"),
        (5.02, "3 · VIH", "MoleculeNet + ChEMBL", "classement A''"),
        (7.49, "4 · Priorisation", "ZINC + antiviraux", "26 structures gelées"),
    ]
    for x, title, subtitle, outcome in boxes:
        box = FancyBboxPatch((x, .35), 2.2, 1.25, boxstyle="round,pad=0.05",
                             edgecolor=TEAL, facecolor="#f4f8f8", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x+.12, 1.30, title, weight="bold", fontsize=10)
        ax.text(x+.12, .98, subtitle, fontsize=8.5)
        ax.text(x+.12, .60, outcome, fontsize=8.5, color=TEAL)
    for x in (2.30, 4.77, 7.24):
        ax.annotate("", xy=(x+.20,.98), xytext=(x,.98),
                    arrowprops={"arrowstyle":"->", "color":ORANGE, "lw":1.6})
    paths["overview"] = save_fig(fig, "fig01_parcours.png")

    x = np.array([row["epoch"] for row in p1])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.1))
    axs[0].plot(x, [r["train_loss"] for r in p1], color=GRAY, label="entraînement")
    axs[0].plot(x, [r["val_loss"] for r in p1], color=TEAL, label="validation")
    axs[0].axvline(98, color=ORANGE, ls="--", lw=1, label="checkpoint 98")
    axs[0].set(xlabel="Époque", ylabel="Perte MGM", title="Apprentissage de représentation")
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].plot(x, [r["train_acc"] for r in p1], color=GRAY, label="entraînement")
    axs[1].plot(x, [r["val_acc"] for r in p1], color=TEAL, label="validation")
    axs[1].axvline(98, color=ORANGE, ls="--", lw=1)
    axs[1].set(xlabel="Époque", ylabel="Exactitude des atomes masqués", ylim=(.85, 1),
               title="Objectif prétexte, non activité VIH")
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths["p1"] = save_fig(fig, "fig02_phase1_mgm.png")

    x = np.array([row["epoch"] for row in p2])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    axs[0].plot(x, [r["train_auc"] for r in p2], color=GRAY, label="entraînement")
    axs[0].plot(x, [r["val_auc"] for r in p2], color=TEAL, label="validation")
    axs[0].axvline(41, color=ORANGE, ls="--", lw=1, label="checkpoint 41")
    axs[0].set(xlabel="Époque", ylabel="ROC-AUC macro", title="Tox21 : discrimination")
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].plot(x, [r["clinical_score"] for r in p2], color=TEAL,
                label="score de sélection")
    axs[1].plot(x, [r["macro_sensitivity"] for r in p2], color=ORANGE,
                label="sensibilité macro")
    axs[1].axvline(41, color=GRAY, ls="--", lw=1)
    axs[1].set(xlabel="Époque", ylabel="Valeur", title="Sélection et rappel")
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths["p2"] = save_fig(fig, "fig03_phase2_tox21.png")

    selected = next(r for r in p2 if r["epoch"] == 41)
    auc = selected["per_task_auc"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    names = list(auc)
    ax.bar(range(len(names)), [auc[n] for n in names], color=TEAL, width=.7)
    ax.axhline(.5, color=ORANGE, ls="--", lw=1)
    ax.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=7.5)
    ax.set(ylim=(0, 1), ylabel="ROC-AUC de validation",
           title="Tox21 : hétérogénéité entre les 12 cibles, époque 41")
    fig.tight_layout()
    paths["p2_tasks"] = save_fig(fig, "fig04_phase2_taches.png")

    x = np.array([row["epoch"] for row in p3])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    axs[0].plot(x, [r["train_loss"] for r in p3], color=GRAY, label="entraînement")
    axs[0].plot(x, [r["val_loss"] for r in p3], color=ORANGE, label="validation")
    axs[0].axvline(13, color=TEAL, ls="--", lw=1, label="checkpoint 13")
    axs[0].set(xlabel="Époque", ylabel="Perte multitâche", yscale="log",
               title="Divergence de la perte de validation")
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].plot(x, [r["eff_auc"] for r in p3], color=TEAL, label="VIH, val. ROC-AUC")
    axs[1].plot(x, [r["composite_score"] for r in p3], color=ORANGE,
                label="score multitâche")
    axs[1].axvline(13, color=GRAY, ls="--", lw=1)
    axs[1].set(xlabel="Époque", ylabel="Valeur", title="Sélection de la phase 3")
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths["p3"] = save_fig(fig, "fig05_phase3_multitache.png")

    a, ap = augmented["model_A"], augmented["model_A_prime"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.1))
    metrics = [("ROC-AUC", "roc_auc"), ("PRC-AUC", "prc_auc"),
               ("Hit top 100", "hit_rate_top_100"), ("BEDROC", "bedroc_alpha20")]
    xx = np.arange(len(metrics))
    axs[0].bar(xx-.18, [a[k] for _,k in metrics], .36, label="A (rapport comparatif)",
               color=GRAY)
    axs[0].bar(xx+.18, [ap[k] for _,k in metrics], .36, label="A'' retenu", color=TEAL)
    axs[0].set_xticks(xx, [n for n,_ in metrics], fontsize=8)
    axs[0].set(ylim=(0,.75), ylabel="Valeur", title="Test HIV : compromis de classement")
    axs[0].legend(frameon=False, fontsize=7.5)
    axs[1].bar([0,1], [a["drugbank"], ap["drugbank"]], color=[GRAY,TEAL], width=.52)
    axs[1].axhline(25, color=ORANGE, ls="--", label="seuil interne 25/34")
    axs[1].set_xticks([0,1], ["A", "A''"])
    axs[1].set(ylim=(0,34), ylabel="ARV dans top 10 % (sur 34)",
               title="Gate DrugBank : hausse du rappel")
    axs[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    paths["models"] = save_fig(fig, "fig06_comparaison_A_Aprime.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.1))
    for ax, values, title in ((axs[0], [zinc_summary["n_scored"],
                                       zinc_summary["n_annotated"],len(zinc)], "ZINC"),
                              (axs[1], [rep_raw_count,len(rep)], "Repurposing")):
        names = ["Scorées", "Annotées", "Hits propres"] if title == "ZINC" else [
            "Lignes conservées", "Hits propres"]
        ax.bar(range(len(values)), values, color=[GRAY]* (len(values)-1)+[TEAL])
        ax.set_yscale("log")
        ax.set_xticks(range(len(values)), names, fontsize=8)
        ax.set(ylabel="Nombre de structures (log)", title=title)
        for i, v in enumerate(values):
            ax.text(i, v*1.12, f"{v:,}".replace(",", " "), ha="center", fontsize=8)
    fig.tight_layout()
    paths["p4"] = save_fig(fig, "fig07_phase4_entonnoir.png")

    # Dessins chimiques calculés à partir des SMILES des CSV de Phase 4.
    chosen = [("ZINC 1", zinc[0]["smiles"].strip()),
              ("ZINC 2", zinc[1]["smiles"].strip()),
              (rep[0]["chembl_id"], rep[0]["smiles"].strip())]
    molecules = [Chem.MolFromSmiles(s) for _, s in chosen]
    if any(mol is None for mol in molecules):
        raise ValueError("SMILES de figure chimique non valide")
    grid = Draw.MolsToGridImage(molecules, molsPerRow=3, subImgSize=(330,260),
                                legends=[name for name,_ in chosen],
                                returnPNG=False)
    paths["molecules"] = FIG / "fig08_structures_2D.png"
    grid.save(paths["molecules"])
    return paths


def fonts():
    candidates = [
        ("C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/georgiab.ttf",
         "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for files in candidates:
        if all(Path(p).is_file() for p in files):
            for name, file in zip(("Body", "BodyBold", "Head", "HeadBold"), files):
                pdfmetrics.registerFont(TTFont(name, file))
            pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold")
            pdfmetrics.registerFontFamily("Head", normal="Head", bold="HeadBold")
            return
    raise FileNotFoundError("Polices TrueType nécessaires pour un PDF Unicode")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleResearch", parent=base["Title"], fontName="HeadBold",
                                fontSize=21, leading=27, textColor=colors.HexColor(INK),
                                spaceAfter=12),
        "subtitle": ParagraphStyle("SubResearch", parent=base["Normal"],
                                   fontName="Head", fontSize=10, leading=14,
                                   textColor=colors.HexColor(GRAY), alignment=TA_CENTER,
                                   spaceAfter=18),
        "h1": ParagraphStyle("H1Research", parent=base["Heading1"], fontName="HeadBold",
                             fontSize=13.5, leading=17, textColor=colors.HexColor(INK),
                             spaceBefore=14, spaceAfter=8),
        "h2": ParagraphStyle("H2Research", parent=base["Heading2"], fontName="HeadBold",
                             fontSize=10.5, leading=14, textColor=colors.HexColor(TEAL),
                             spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("BodyResearch", parent=base["Normal"], fontName="Body",
                               fontSize=9.15, leading=13.35, alignment=TA_JUSTIFY,
                               spaceAfter=8),
        "small": ParagraphStyle("SmallResearch", parent=base["Normal"], fontName="Body",
                                fontSize=8.1, leading=11.2, spaceAfter=5),
        "caption": ParagraphStyle("CaptionResearch", parent=base["Normal"],
                                  fontName="Body", fontSize=7.7, leading=10.5,
                                  textColor=colors.HexColor(GRAY), spaceAfter=13),
        "table": ParagraphStyle("TableResearch", parent=base["Normal"],
                                fontName="Head", fontSize=7.7, leading=10),
    }


def build_pdf(data, paths):
    fonts()
    st = styles()
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2.05*cm,
                            rightMargin=2.05*cm, topMargin=1.85*cm,
                            bottomMargin=1.7*cm, title="Revue scientifique Panacée",
                            author="Projet Panacée",
                            subject="Analyse critique des quatre phases du pipeline VIH")
    story = []

    def P(text, kind="body"):
        story.append(Paragraph(text.replace("`", ""), st[kind]))

    def H(text, level=1):
        P(text, "h1" if level == 1 else "h2")

    def F(name, caption, width=16.3*cm):
        from PIL import Image as PILImage

        with PILImage.open(paths[name]) as img:
            ratio = img.height / img.width
        story.append(Image(str(paths[name]), width=width, height=width*ratio))
        P(caption, "caption")

    def T(headers, rows, widths=None):
        cells = [[Paragraph(str(v), st["table"]) for v in headers]]
        cells += [[Paragraph(str(v), st["table"]) for v in row] for row in rows]
        table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(LIGHT)),
            ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor(TEAL)),
            ("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#d5dfe1")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),6),
            ("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ]))
        story.append(table)
        story.append(Spacer(1, 9))

    p1,p2,p3 = data["logs"]
    aug = data["augmented"]
    audit = data["member_audit"]
    registry = data["registry"]

    story.append(Spacer(1, 26))
    P("Panacée : revue critique des quatre phases du criblage moléculaire anti-VIH", "title")
    P("Synthèse des artefacts Jupyter et des résultats numériques archivés · 9 novembre 2026 · Projet Panacée", "subtitle")
    H("Résumé")
    P("Ce rapport reconstitue la chaîne de calcul allant du pré-entraînement sur ZINC au criblage virtuel de Phase 4. Il distingue les observations tirées des journaux et checkpoints, les comparaisons entre versions du modèle et les hypothèses biologiques non testées. Le modèle VIH retenu, A'' (GNN propre et quatre forêts aléatoires augmentées), reproduit une ROC-AUC de 0,6201, une PRC-AUC de 0,1541 et le classement de 32 antirétroviraux sur 34 dans le décile supérieur d'un pool défini. Ce dernier résultat est un test de reconnaissance de médicaments connus, non une efficacité cellulaire mesurée des candidats. La Phase 4 fournit 124 structures ZINC et six entrées de repurposing après filtrage informatique ; aucune n'a encore été confirmée dans ce projet par un essai indépendant.")
    P("La progression apparente du rappel DrugBank s'accompagne d'une baisse de plusieurs métriques sur le test HIV par rapport au comparateur A du même rapport. Des fichiers antérieurs donnent des valeurs différentes pour ce comparateur ; ces versions ne sont pas fusionnées. La sélection d'un composé prometteur exige désormais des mesures antivirales et cytotoxiques indépendantes, puis une confirmation de la résistance, de la pharmacologie et de la sécurité.")
    F("overview", "Figure 1. Parcours documentaire des quatre phases. Les flèches indiquent un transfert de représentation ou de résultats, non une validation biologique.")
    H("Périmètre, sources et méthode")
    P("L'analyse repose sur les journaux <i>live_metrics.jsonl</i> des Phases 1 à 3, les métadonnées des checkpoints retenus, les rapports JSON de comparaison, les CSV de hits et le registre SHA-256 du modèle A''. Les courbes ont été retracées à partir des enregistrements par époque ; les tableaux n'assemblent des modèles que lorsque leurs valeurs proviennent du même rapport comparatif. Les splits HIV Jupyter sont présents localement et leurs empreintes concordent avec le registre. Le test HIV compte 3 778 molécules, dont 84 positives. Les proportions et scores de classement sont conditionnels à cette population et à ses définitions de label [1].")

    H("1. Pré-entraînement moléculaire : Phase 1")
    P("La Phase 1 apprend une représentation moléculaire par <i>masked graph modeling</i> sur ZINC. Le checkpoint `sovereign_encoder_v1.pth` possède six couches d'attention et une dimension cachée de 256. Le journal contient 113 époques observées, bien que 3 000 aient été configurées ; ce nombre configuré ne doit pas être présenté comme le nombre exécuté. L'époque 98, retenue dans le checkpoint, porte une perte de validation de 0,1141. L'exactitude de validation atteint son maximum observé à l'époque 104 (0,9561).")
    F("p1", "Figure 2. Évolution de la perte MGM et de l'exactitude des atomes masqués, 113 époques enregistrées. La ligne verticale situe le checkpoint de l'époque 98. Source : checkpoints/phase1/live_metrics.jsonl.")
    P("L'amélioration de l'objectif prétexte témoigne d'un apprentissage de la tâche de reconstruction, pas d'une capacité à prédire l'inhibition du VIH. L'absence d'évaluation externe spécifique de cette représentation empêche d'attribuer le gain ultérieur à la seule Phase 1. Le champ `best_val_loss` du checkpoint (0,1161) diffère de sa perte de validation enregistrée (0,1141) ; le présent rapport retient la valeur mesurée à l'époque 98 et signale l'écart de sérialisation.")

    H("2. Transfert toxicologique : Phase 2")
    P("La Phase 2 adapte l'encodeur aux 12 tâches Tox21. Le journal conserve 61 époques, contre 800 programmées. Le checkpoint sélectionné correspond à l'époque 41 : ROC-AUC macro de validation 0,7533, F1 0,3036, rappel 0,6544 et score de sélection 0,6489. La ROC-AUC du journal augmente encore jusqu'à 0,7642 à l'époque 61, mais la sélection n'était pas fondée sur cette seule métrique ; le score composite de sélection culmine à l'époque 41. La sensibilité macro, en revanche, passe de 0,7214 à l'époque 8 à 0,5979 à l'époque 61.")
    F("p2", "Figure 3. Trajectoires d'apprentissage et de sélection Tox21. Le checkpoint de l'époque 41 n'est pas l'époque de ROC-AUC maximale observée. Source : checkpoints/phase2/live_metrics.jsonl.")
    F("p2_tasks", "Figure 4. ROC-AUC de validation par tâche au checkpoint retenu. Les 12 endpoints ne présentent pas une performance uniforme ; les scores ne constituent pas une évaluation toxicologique du candidat VIH.")
    P("Les tâches Tox21 sont des proxys expérimentaux distincts. Ni la moyenne de ROC-AUC ni le checkpoint Phase 2 ne renseignent, à eux seuls, sur la fenêtre thérapeutique, la toxicité chronique, les interactions ou la sécurité clinique d'une structure priorisée [2].")

    H("3. Modélisation multipropriété et ensemble VIH : Phase 3")
    H("3.1 Modèle multipropriété initial", 2)
    P("Le checkpoint `panacee_phase3_complete.pth` a été retenu à l'époque 13 parmi 38 époques enregistrées, sur un score composite multitâche de 0,6144. Sa ROC-AUC VIH est 0,7925 sur validation, mais 0,6568 sur le test ; la PRC-AUC VIH du test est 0,1665. Le rapport d'évaluation indépendant de ce checkpoint donne une ROC-AUC de classement de 0,6582, un EF@1 % de 16,57, une PRC-AUC de 0,1661 et seulement 15/34 ARV au gate DrugBank, sous le seuil interne de 25/34. La perte de validation croît fortement au fil des époques malgré une perte d'entraînement décroissante : le résultat final ne saurait être déduit de l'époque 38.")
    F("p3", "Figure 5. Phase 3 multipropriété : perte entraînement/validation en échelle logarithmique et critères de sélection. La ligne pointillée indique le checkpoint retenu (époque 13). Source : checkpoints/phase3/live_metrics.jsonl.")
    H("3.2 Exploration des représentations et échecs utiles", 2)
    P("Les rapports de stacking comparent GNN, forêts sur ECFP4/ECFP6/MACCS/RDKit, fusion interne, métamodèle logistique et agrégation par rang. Le GNN seul de cette génération atteint une ROC-AUC test de 0,7194, mais une PRC-AUC de 0,1131 et 17/34 au gate. La forêt ECFP4 atteint 0,6524 de ROC-AUC, 0,1567 de PRC-AUC et 20/34. La fusion interne n'apporte pas le meilleur compromis de reconnaissance précoce ; l'agrégation de rang a été poursuivie. Ces chiffres concernent la génération antérieure, pas le GNN du paquet A'' actuel. Le Model A de cette génération échoue au gate avec 24/34 dans `push_ensemble_report.json`. Le plafond a conduit à augmenter l'entraînement, plutôt qu'à annoncer un succès prématuré.")
    H("3.3 Augmentation ChEMBL et modèle A'' retenu", 2)
    P("Quinze mille cinq cent quatre-vingts structures actives ChEMBL ont été ajoutées à l'entraînement des forêts ; le GNN utilisé dans le manifeste A'' reste le checkpoint propre déjà enregistré. Le filtre documenté exclut trois identités ARV, 146 composés partageant un scaffold avec validation/test, 36 analogues au-dessus du seuil Tanimoto 0,80 vis-à-vis des ARV et 77 doublons de train. Il demeure 189 ajouts partageant un scaffold avec un ARV. Deux des 34 ARV sont également des positifs connus du train MoleculeNet. Ces éléments imposent de distinguer un gate de reconnaissance de médicaments connus d'une validation externe totalement indépendante.")
    P("Le registre A'' contrôle 12 artefacts. Le recalcul local à partir des poids et des 3 778 structures de test reproduit les huit métriques numériques du rapport `augmented_report.json` à la tolérance déclarée. Le tableau suivant provient exclusivement de ce rapport comparatif, afin de ne pas mêler des baselines de versions différentes.")
    a, ap = aug["model_A"], aug["model_A_prime"]
    T(["Métrique, test HIV ou gate", "A, même rapport", "A'' retenu"], [
        ("ROC-AUC", f"{a['roc_auc']:.3f}", f"{ap['roc_auc']:.3f}"),
        ("PRC-AUC", f"{a['prc_auc']:.3f}", f"{ap['prc_auc']:.3f}"),
        ("EF@1 %", f"{a['ef_1pct']:.2f}", f"{ap['ef_1pct']:.2f}"),
        ("EF@5 %", f"{a['ef_5pct']:.2f}", f"{ap['ef_5pct']:.2f}"),
        ("BEDROC α=20", f"{a['bedroc_alpha20']:.3f}", f"{ap['bedroc_alpha20']:.3f}"),
        ("Hit rate top 100", f"{a['hit_rate_top_100']:.2f}", f"{ap['hit_rate_top_100']:.2f}"),
        ("Rappel top 10 %", f"{a['recall_top_10pct']:.3f}", f"{ap['recall_top_10pct']:.3f}"),
        ("Gate DrugBank", f"{int(a['drugbank'])}/34", f"{int(ap['drugbank'])}/34"),
    ], [7.0*cm,4.55*cm,4.55*cm])
    P("Tableau 1. Comparaison interne issue d'augmented_report.json ; les métriques du test et le gate DrugBank portent sur des populations et objectifs distincts.", "caption")
    F("models", "Figure 6. Comparaison interne A versus A'' d'après un seul rapport (`augmented_report.json`). Le gain de rappel des ARV coexiste avec une baisse de ROC-AUC, PRC-AUC, BEDROC et hit rate top 100 sur le test HIV.")
    P("Le gain du gate est de 26 à 32/34 dans cette comparaison. Le coût sur le test HIV est mesurable : BEDROC passe de 0,325 à 0,291 et le hit rate top 100 de 0,23 à 0,20. Le bootstrap du rapport note un intervalle à 95 % de la différence BEDROC excluant zéro (−0,0686 à −0,0021) ; les intervalles du ROC-AUC et de la PRC-AUC recouvrent zéro. La hausse du gate ne suffit donc pas à conclure à une supériorité générale.")
    P("Un audit des cinq membres du paquet A'' actuel retrouve, sur le même test, une PRC-AUC de 0,0776 pour le GNN, 0,1073 pour RF ECFP4 et 0,1541 pour l'ensemble. Le GNN de cette version diffère du GNN ancien à ROC-AUC 0,7194 ; les chiffres ne sont pas interchangeables. Les scores bruts ont des Brier scores de 0,039 à 0,140 contre 0,0217 pour la constante empirique de prévalence : le rang composite et les scores membres ne sont pas des probabilités calibrées. Le domaine d'applicabilité est calculé sur 49 150 empreintes train + augmentation ; la calibration conformal GNN couvre 93,09 % du test interne avec 67,44 % de sorties ambiguës. Cela ne démontre pas une couverture prospective.")
    audit_rows = [(label, f"{values['roc_auc']:.3f}",
                   f"{values['prc_auc']:.3f}",
                   f"{values['precision_at_100']:.2f}")
                  for label, values in [
                      ("GNN", audit["members"]["gnn"]),
                      ("RF ECFP4", audit["members"]["rf_ecfp4"]),
                      ("RF ECFP6", audit["members"]["rf_ecfp6"]),
                      ("RF MACCS", audit["members"]["rf_maccs"]),
                      ("RF RDKit", audit["members"]["rf_rdkit"]),
                      ("A'' ensemble", audit["ensemble_rank_average"]),
                  ]]
    T(["Membre actuel", "ROC-AUC", "PRC-AUC", "Précision top 100"], audit_rows,
      [5.3*cm,3.0*cm,3.0*cm,4.8*cm])
    P("Tableau 2. Audit des membres réellement chargés par A''. Il ne doit pas servir à réajuster les poids sur ce test déjà examiné.", "caption")

    H("4. Criblage et sélection de structures : Phase 4")
    P("La bibliothèque ZINC compte 249 455 structures scorées selon son résumé de traitement. Deux mille lignes ont été annotées et 124 satisfont les critères de sortie du CSV propre. Le fichier repurposing conserve 500 lignes scorées et six hits propres. Ces nombres décrivent un filtrage informatique ; ils ne constituent pas des confirmations antivirales. Les 124 hits ZINC sont tous étiquetés `in_domain_moderate` par le pipeline historique ; 118 ont une sortie conformal `ambiguous` et six `active`. Les six hits repurposing portent tous `uncertain`. Les étiquettes ne sont ni des mesures de puissance ni des garanties de sécurité.")
    F("p4", "Figure 7. Entonnoir de Phase 4. Pour ZINC, les comptes proviennent du résumé de criblage et du CSV propre ; pour repurposing, « 500 lignes conservées » désigne le fichier archivé, non la taille certaine de la bibliothèque initiale. Échelle verticale logarithmique.")
    F("molecules", "Figure 8. Représentations 2D calculées avec RDKit pour les deux premières lignes ZINC et la première ligne repurposing du CSV propre. Le dessin indique la connectivité déclarée par SMILES, non un mécanisme d'action ni une activité mesurée.")
    P("La médiane QED des 124 hits ZINC est 0,687, celle des six hits repurposing 0,555. QED et Lipinski sont des descripteurs, tandis que les colonnes de toxicité et d'ADMET du CSV sont des sorties de modèles. Leur présence ne transforme pas ces structures en candidats précliniques. Les cartes d'attention stockées pour 20 hits montrent des poids du modèle ; elles ne démontrent ni liaison à une cible du VIH ni pharmacophore causal.")
    P("Un paquet prospectif distinct a été gelé localement : 12 ZINC, six repurposing, quatre ARV de référence et quatre comparateurs négatifs du criblage MoleculeNet. Sa soumission aveugle, sa clé locale et ses prédictions ont des empreintes SHA-256. Aucun résultat expérimental n'a été reçu ; les quatre comparateurs ne sont pas des contrôles négatifs certifiés pour un essai biologique.")

    H("Discussion : interprétation et points non résolus")
    P("Trois résultats structurent la décision. D'abord, le pré-entraînement et le transfert toxicologique ont produit des checkpoints documentés, mais leurs métriques ne valident pas l'usage antiviral ou la sécurité d'un nouveau composé. Ensuite, la Phase 3 a franchi le seuil interne de reconnaissance des ARV avec A'', au prix d'une baisse de certaines métriques de classement HIV. Enfin, la Phase 4 a réduit des bibliothèques en listes testables, sans preuve cellulaire. Le choix d'A'' est cohérent avec un objectif de rappel d'ARV connus, mais ne prouve pas qu'il maximise la découverte de nouvelles séries chimiques.")
    P("Plusieurs discordances documentaires interdisent une méta-analyse naïve : le Model A vaut 24/34 dans un rapport et 26/34 dans le rapport comparatif A/A'' ; un autre audit d'honnêteté donne 31/34 pour A' alors que la reproduction actuelle donne 32/34 pour A''. Le fichier `phase3_eval.json` contient BEDROC=1,0, incompatible avec `reports/eval.json` (0,2979) pour les autres métriques identiques ; la valeur 1,0 n'est pas retenue. Le checkpoint GNN propre contient un `test_report` avec zéro positif et une ROC-AUC non définie sur un jeu de leurres : ce rapport ne peut pas être cité comme performance HIV. La validation croisée à cinq folds (ROC-AUC 0,8338 ± 0,0276) évalue une configuration de forêts différente ; elle ne constitue pas une validation indépendante du modèle A'' complet.")
    P("La publication du rapport suit une logique de traçabilité inspirée des principes de transparence des modèles prédictifs [3] et des critères OCDE de domaine d'applicabilité [4], sans revendiquer une conformité réglementaire. Pour un antirétroviral, les données précliniques pertinentes incluent activité cellulaire, cytotoxicité, mécanisme, résistance et interactions [2]. La puissance, la tolérance et la barrière à la résistance ont une portée clinique que le classement moléculaire ne mesure pas [5].")

    H("Conclusion")
    P("Le projet a atteint une étape de priorisation moléculaire traçable, pas celle de découverte d'un médicament VIH confirmé. La prochaine décision scientifique dépend d'un essai indépendant et aveugle des structures gelées, avec mesures d'activité et de cytotoxicité, contrôle qualité du laboratoire et confirmation des hits. Les données quantitatives ainsi obtenues permettront de définir un modèle vNext orienté vers la puissance et la sélectivité ; le test déjà utilisé pour sélectionner A'' ne doit pas servir de nouveau terrain d'optimisation.")

    H("Traçabilité et reproductibilité")
    P("Le registre du modèle du 6 novembre associe les poids, rapports et jeux Jupyter à des empreintes SHA-256. Le présent PDF a été produit par `scripts/build_scientific_review.py`, qui refuse sa génération si le registre, les splits HIV ou les huit métriques A'' reproduites divergent. Les figures proviennent exclusivement des journaux et des CSV indiqués dans leurs légendes.")
    T(["Artefact", "SHA-256 (préfixe)", "Fonction"], [
        ("GNN propre", registry["artifacts"]["phase3/panacee_phase3_hiv_chembl_clean.pth"][:16], "Membre A''"),
        ("Manifest A''", registry["artifacts"]["model_A_prime_augmented/manifest.json"][:16], "Version du paquet"),
        ("Split HIV test", registry["datasets"]["hiv_test.csv"][:16], "Évaluation 3 778"),
        ("Reproduction A''", registry["artifacts"]["model_A_prime_augmented/validation_reproduction.json"][:16], "Huit métriques"),
    ], [5.2*cm,4.1*cm,6.8*cm])

    H("Références scientifiques et réglementaires")
    refs = [
        "[1] Wu Z et al. MoleculeNet: a benchmark for molecular machine learning. Chem Sci. 2018;9:513–530. DOI 10.1039/C7SC02664A. https://doi.org/10.1039/C7SC02664A",
        "[2] US FDA. Division of Antivirals: information for pre-IND submissions. https://www.fda.gov/about-fda/cder-offices-and-divisions/division-antivirals-dav-information-pre-ind-submissions",
        "[3] Collins GS et al. TRIPOD+AI statement. BMJ. 2024;385:e078378. DOI 10.1136/bmj-2023-078378. https://www.bmj.com/content/385/bmj-2023-078378",
        "[4] OCDE. Guidance Document on the Validation of (Q)SAR Models, principe du domaine d'applicabilité. https://www.oecd.org/content/dam/oecd/en/publications/reports/2014/09/guidance-document-on-the-validation-of-quantitative-structure-activity-relationship-q-sar-models_g1ghcc68/9789264085442-en.pdf",
        "[5] NIH. Principles of Antiretroviral Therapy, mise à jour 2026. https://clinicalinfo.hiv.gov/en/guidelines/hiv-clinical-guidelines-adult-and-adolescent-arv/principles-antiretroviral-therapy",
    ]
    for ref in refs:
        P(ref, "small")
    H("Sources internes principales")
    P("checkpoints/phase{1,2,3}/live_metrics.jsonl ; checkpoints/phase{1,2,3}/*.pth ; reports/eval.json ; checkpoints/phase3_stacking/{push_ensemble_report,augmented_report,bootstrap_ci_report,kfold_cv_report,arv_leakage_report}.json ; checkpoints/model_A_prime_augmented/{registry,validation_reproduction}.json ; checkpoints/phase4/{library_hits.summary,library_hits_clean,repurposing_hits_clean}.*. Les valeurs A'' et les empreintes ont été revérifiées lors de la génération. Les données brutes d'entraînement et le paquet prospectif restent locaux et hors du PDF.", "small")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5d7"))
        canvas.line(2.05*cm, 1.25*cm, A4[0]-2.05*cm, 1.25*cm)
        canvas.setFont("Head", 7.3)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(2.05*cm, .88*cm, "PANACÉE  |  revue scientifique des artefacts  |  09.11.2026")
        canvas.drawRightString(A4[0]-2.05*cm, .88*cm, str(doc.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    check = verify_registry(ROOT / "checkpoints", datasets=True)
    if check["status"] != "verified":
        raise ValueError("Registre A'' ou splits Jupyter non concordants")
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    logs = [epochs(n) for n in ("phase1", "phase2", "phase3")]
    ckpts = [torch.load(source(p), map_location="cpu", weights_only=False) for p in (
        "checkpoints/phase1/sovereign_encoder_v1.pth",
        "checkpoints/phase2/best_toxicity_model.pth",
        "checkpoints/phase3/panacee_phase3_complete.pth")]
    if [ck["epoch"] for ck in ckpts] != [98, 41, 13]:
        raise ValueError("Époques des checkpoints modifiées")
    augmented = read_json("checkpoints/phase3_stacking/augmented_report.json")
    reproduction = read_json("checkpoints/model_A_prime_augmented/validation_reproduction.json")
    if not all(reproduction["within_tolerance"].values()):
        raise ValueError("Métriques A'' non reproduites")
    if any(abs(reproduction["metrics"][k] - augmented["model_A_prime"][k]) > 1e-6
           for k in reproduction["metrics"]):
        raise ValueError("Écart entre rapport et reproduction A''")
    with source("checkpoints/phase4/library_hits_clean.csv").open(
        encoding="utf-8-sig", newline="") as handle:
        zinc = list(csv.DictReader(handle))
    with source("checkpoints/phase4/repurposing_hits_clean.csv").open(
        encoding="utf-8-sig", newline="") as handle:
        rep = list(csv.DictReader(handle))
    if len(zinc) != 124 or len(rep) != 6:
        raise ValueError("Comptes Phase 4 modifiés")
    zinc_summary = read_json("checkpoints/phase4/library_hits.summary.json")
    with source("checkpoints/phase4/repurposing_hits.csv").open(
        encoding="utf-8-sig", newline="") as handle:
        rep_raw_count = sum(1 for _ in csv.DictReader(handle))
    audit = read_json("reports/model_member_audit_2026-11.json")
    if audit["n_test"] != reproduction["n_test"] or abs(
        audit["ensemble_rank_average"]["roc_auc"] -
        reproduction["metrics"]["roc_auc"]) > 1e-6:
        raise ValueError("Audit des membres incohérent")
    paths = make_figures(*logs, augmented, zinc, rep, zinc_summary, rep_raw_count)
    phase3_eval = read_json("reports/eval.json")
    if phase3_eval["drugbank_validation"]["recall"] != 15:
        raise ValueError("Gate Phase 3 initial modifié")
    build_pdf({"logs":logs, "augmented":augmented, "member_audit":audit,
               "registry":check["registry"]}, paths)
    print(f"PDF créé : {PDF} ({PDF.stat().st_size:,} octets, {len(paths)} figures)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
