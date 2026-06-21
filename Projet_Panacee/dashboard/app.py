# -*- coding: utf-8 -*-
"""
🧬 Panacée — Tableau de bord d'entraînement (Streamlit).

Lancement :
    cd Projet_Panacee
    streamlit run dashboard/app.py

Onglets :
  📈 Évolution        — courbes loss / AUC / sécurité en temps réel (tail du .jsonl)
  🏥 Métriques cliniques — sensibilité, spécificité, FNR, calibration par endpoint
  🚨 Sécurité          — alertes DANGER/WARN (faux négatifs toxicologiques)
  🔬 Comparaison       — attendu vs obtenu, comparaison de runs

Le temps réel repose sur la lecture du fichier checkpoints/.../live_metrics.jsonl
écrit par l'entraînement (un point par epoch). Pas de couplage réseau.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_io import (
    EXPECTED,
    compare_to_expected,
    find_live_files,
    load_run,
)

# Auto-refresh optionnel (sans dépendance dure)
try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False

_DANGER_COLORS = {"DANGER": "#d62728", "WARN": "#ff7f0e", "OK": "#2ca02c", "NA": "#888888"}

st.set_page_config(page_title="Panacée — Dashboard", page_icon="🧬", layout="wide")


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("🧬 Panacée Dashboard")

ckpt_root = st.sidebar.text_input("Dossier des runs", value="checkpoints")
live_files = find_live_files(ckpt_root)

if not live_files:
    st.sidebar.warning("Aucun live_metrics.jsonl trouvé. Lance un entraînement Phase 2.")
run_path = None
if live_files:
    run_path = st.sidebar.selectbox("Run (live_metrics.jsonl)",
                                    options=live_files, format_func=lambda p: str(Path(p).parent))

realtime = st.sidebar.checkbox("⏱️ Temps réel (auto-refresh)", value=True)
interval = st.sidebar.slider("Intervalle (s)", 2, 30, 5)
if realtime:
    if _HAS_AUTOREFRESH:
        st_autorefresh(interval=interval * 1000, key="refresh")
    else:
        st.sidebar.caption("`pip install streamlit-autorefresh` pour le rafraîchissement auto. "
                           "Sinon, utilise le bouton ci-dessous.")
        if st.sidebar.button("🔄 Rafraîchir"):
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Sécurité : on surveille le **FNR** (faux négatifs = toxiques manqués).")

# Charger le run sélectionné
meta, epochs = ({}, [])
if run_path:
    meta, epochs = load_run(run_path)
latest = epochs[-1] if epochs else {}


# ──────────────────────────────────────────────────────────────────────
# En-tête / KPIs
# ──────────────────────────────────────────────────────────────────────
st.title("🧬 Panacée — Suivi d'entraînement & sécurité médicale")
if meta:
    st.caption(f"Phase: {meta.get('phase','?')} | conv: {meta.get('conv_type','?')} | "
               f"EMA: {meta.get('ema','?')} | epochs prévus: {meta.get('epochs_total','?')} | "
               f"points reçus: {len(epochs)}")

if latest:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Epoch", latest.get("epoch", "—"))
    c2.metric("ROC-AUC (val)", f"{latest.get('val_auc', 0):.3f}")
    c3.metric("Sensibilité", f"{latest.get('macro_sensitivity', 0):.3f}")
    fnr = latest.get("macro_fnr", 0)
    c4.metric("FNR (faux négatifs)", f"{fnr:.3f}",
              delta="⚠️" if fnr > EXPECTED["macro_fnr_max"] else "ok",
              delta_color="inverse")
    nd = latest.get("n_danger", 0)
    c5.metric("Endpoints en DANGER", nd, delta=None)
    if nd and nd > 0:
        st.error(f"🚨 {nd} endpoint(s) toxicologique(s) en DANGER — voir l'onglet Sécurité.")

tab_evo, tab_clin, tab_sec, tab_cmp = st.tabs(
    ["📈 Évolution", "🏥 Métriques cliniques", "🚨 Sécurité", "🔬 Comparaison"])


# ──────────────────────────────────────────────────────────────────────
# Onglet Évolution
# ──────────────────────────────────────────────────────────────────────
with tab_evo:
    if not epochs:
        st.info("En attente de données d'entraînement…")
    else:
        df = pd.DataFrame(epochs)
        st.subheader("Loss (train / val)")
        loss_cols = [c for c in ["train_loss", "val_loss"] if c in df]
        if loss_cols:
            st.line_chart(df.set_index("epoch")[loss_cols])

        st.subheader("ROC-AUC (train / val)")
        auc_cols = [c for c in ["train_auc", "val_auc"] if c in df]
        if auc_cols:
            st.line_chart(df.set_index("epoch")[auc_cols])

        st.subheader("Sécurité dans le temps : sensibilité & FNR")
        sec_cols = [c for c in ["macro_sensitivity", "macro_fnr"] if c in df]
        if sec_cols:
            st.line_chart(df.set_index("epoch")[sec_cols])

        # Détection de surapprentissage
        if {"train_auc", "val_auc"} <= set(df.columns):
            gap = float(df["train_auc"].iloc[-1] - df["val_auc"].iloc[-1])
            if gap > 0.15:
                st.warning(f"Surapprentissage probable : écart train-val AUC = {gap:.2f}")


# ──────────────────────────────────────────────────────────────────────
# Onglet Métriques cliniques
# ──────────────────────────────────────────────────────────────────────
with tab_clin:
    st.subheader("Métriques par endpoint toxicologique")
    st.caption("Évaluation approfondie depuis un checkpoint + un CSV de validation.")

    default_ckpt = str(Path(run_path).parent / "best_toxicity_model.pth") if run_path else ""
    ckpt_in = st.text_input("Checkpoint Phase 2 (.pth)", value=default_ckpt)
    val_csv_in = st.text_input("CSV de validation", value="data/external/tox21/tox21_val.csv")

    if st.button("🔬 Évaluer le checkpoint"):
        if not (Path(ckpt_in).exists() and Path(val_csv_in).exists()):
            st.error("Checkpoint ou CSV introuvable.")
        else:
            with st.spinner("Inférence + métriques cliniques…"):
                from src.validation.clinical_metrics import evaluate_checkpoint
                res = evaluate_checkpoint(ckpt_in, val_csv_in)
            st.session_state["clinical"] = res

    res = st.session_state.get("clinical")
    if res:
        agg = res["aggregate"]
        a, b, c, d = st.columns(4)
        a.metric("AUC macro", f"{agg['macro_roc_auc']:.3f}")
        b.metric("Sensibilité macro", f"{agg['macro_sensitivity']:.3f}")
        c.metric("FNR macro", f"{agg['macro_fnr']:.3f}")
        d.metric("ECE (calibration)", f"{agg['mean_ece']:.3f}")

        tdf = pd.DataFrame(res["tasks"])
        show_cols = ["task", "danger", "support", "prevalence", "sensitivity",
                     "specificity", "fnr", "precision", "f1", "roc_auc", "pr_auc", "ece"]
        show_cols = [c for c in show_cols if c in tdf.columns]

        def _hl(row):
            color = _DANGER_COLORS.get(row["danger"], "")
            return [f"background-color: {color}33"] * len(row)

        st.dataframe(tdf[show_cols].style.apply(_hl, axis=1), use_container_width=True)
    else:
        st.info("Clique sur « Évaluer le checkpoint » pour calculer les métriques par endpoint.")


# ──────────────────────────────────────────────────────────────────────
# Onglet Sécurité
# ──────────────────────────────────────────────────────────────────────
with tab_sec:
    st.subheader("🚨 Alertes de sécurité (faux négatifs toxicologiques)")
    st.caption("Un faux négatif = molécule TOXIQUE prédite « sûre » → risque clinique majeur.")

    res = st.session_state.get("clinical")
    if res and res.get("alerts"):
        for al in res["alerts"]:
            msg = al["message"]
            if al["level"] == "DANGER":
                st.error(f"🔴 DANGER — {msg}")
            else:
                st.warning(f"🟠 ATTENTION — {msg}")
    elif res:
        st.success("✅ Aucun endpoint en danger sur ce checkpoint.")
    else:
        # Fallback temps réel depuis le live log
        nd = latest.get("n_danger", 0) if latest else 0
        nw = latest.get("n_warn", 0) if latest else 0
        if nd:
            st.error(f"🔴 {nd} endpoint(s) en DANGER (live). Évalue le checkpoint pour le détail.")
        if nw:
            st.warning(f"🟠 {nw} endpoint(s) en attention (live).")
        if not nd and not nw:
            st.info("Pas d'alerte pour l'instant (ou évaluation cliniques non lancée).")

    st.markdown(
        "**Barème** : DANGER si FNR ≥ 50 % ou sensibilité < 50 % ou AUC < 0.60 — "
        "WARN si FNR ≥ 30 % ou AUC < 0.70.")


# ──────────────────────────────────────────────────────────────────────
# Onglet Comparaison
# ──────────────────────────────────────────────────────────────────────
with tab_cmp:
    st.subheader("Attendu vs obtenu")
    rows = compare_to_expected(latest)
    if rows:
        cdf = pd.DataFrame(rows)
        cdf["obtenu"] = cdf["obtenu"].apply(lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else v)
        cdf["statut"] = cdf["ok"].apply(lambda ok: "✅" if ok else "❌")
        st.table(cdf[["metric", "obtenu", "attendu", "statut"]])
    else:
        st.info("Pas encore de métriques à comparer.")

    st.subheader("Comparer les runs (AUC val finale)")
    comp = []
    for lf in live_files:
        m, eps = load_run(lf)
        if eps:
            comp.append({"run": str(Path(lf).parent),
                         "epochs": len(eps),
                         "val_auc_final": eps[-1].get("val_auc"),
                         "fnr_final": eps[-1].get("macro_fnr"),
                         "n_danger": eps[-1].get("n_danger")})
    if comp:
        st.dataframe(pd.DataFrame(comp), use_container_width=True)
