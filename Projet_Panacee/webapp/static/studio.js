"use strict";

const $ = id => document.getElementById(id);
const app = { overview: null, ranking: null, source: "zinc", hits: [], hitVisible: 20 };

async function api(path, options = {}) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Erreur HTTP ${response.status}`);
  return data;
}

function cell(row, value, className = "") {
  const td = document.createElement("td");
  td.textContent = value === null || value === undefined || value === "" ? "—" : String(value);
  if (className) td.className = className;
  row.append(td);
}

function fmt(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits).replace(/\.?0+$/, "") : "—";
}

function setStatus(message, error = false) {
  $("screenStatus").textContent = message;
  $("screenStatus").dataset.error = error ? "true" : "false";
}

function parseMolecules(raw) {
  return raw.split(/\r?\n/).map(line => line.trim()).filter(Boolean).map((line, index) => {
    const comma = line.indexOf(",");
    return comma < 0
      ? {name: `Molécule ${index + 1}`, smiles: line}
      : {name: line.slice(0, comma).trim() || `Molécule ${index + 1}`,
         smiles: line.slice(comma + 1).trim()};
  });
}

function renderRanking(data) {
  app.ranking = data;
  const body = $("screenRows"); body.replaceChildren();
  for (const molecule of data.ranked) {
    const row = document.createElement("tr");
    cell(row, `#${molecule.position}`, "rank-cell");
    const name = document.createElement("td");
    const strong = document.createElement("strong"); strong.textContent = molecule.name;
    const smi = document.createElement("small"); smi.textContent = molecule.smiles;
    name.append(strong, smi); row.append(name);
    cell(row, fmt(molecule.rank_average, 3), "score-cell");
    cell(row, fmt(molecule.reference_percentile, 1), "score-cell");
    cell(row, `${molecule.ad_class || "—"} (${fmt(molecule.ad_max_sim, 2)})`);
    cell(row, molecule.gnn_conformal);
    cell(row, fmt(molecule.qed, 3));
    cell(row, fmt(molecule.mw, 1));
    cell(row, fmt(molecule.logp, 2));
    const flags = molecule.structural_alerts || [];
    cell(row, flags.length ? flags.join(", ") :
         (molecule.lipinski_pass ? "Aucune alerte détectée" : "Lipinski à examiner"),
         flags.length ? "flag-cell" : "");
    body.append(row);
  }
  $("rankingNote").textContent = `${data.ranking_scope}. ${data.interpretation} Référence fixe : test HIV sans labels (3 778 structures). Domaine : train + actifs ajoutés. Conformal : GNN seul.`;
  $("invalidNote").textContent = data.invalid.length
    ? `${data.invalid.length} entrée(s) invalide(s) ou non scorée(s) : ${data.invalid.map(x => x.name).join(", ")}` : "";
  $("screenResults").hidden = false;
  $("screenResults").scrollIntoView({behavior: "smooth", block: "start"});
}

function exportRanking() {
  if (!app.ranking?.ranked?.length) return;
  const columns = ["position", "name", "smiles", "rank_average", "reference_percentile", "ad_class", "ad_max_sim", "gnn_conformal", "qed", "mw", "logp", "lipinski_pass", "structural_alerts"];
  const quote = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const csv = [columns.join(","), ...app.ranking.ranked.map(row =>
    columns.map(key => quote(Array.isArray(row[key]) ? row[key].join(";") : row[key])).join(","))].join("\r\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["\uFEFF", csv], {type: "text/csv;charset=utf-8"}));
  a.download = "panacee_classement_A_prime_prime.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function renderHits() {
  const query = $("hitSearch").value.trim().toLowerCase();
  const filtered = app.hits.filter(row => !query || Object.values(row).some(value =>
    String(value).toLowerCase().includes(query)));
  $("hitCount").textContent = `${Math.min(filtered.length, app.hitVisible)} affichés · ${filtered.length} trouvés`;
  const body = $("hitRows"); body.replaceChildren();
  filtered.slice(0, app.hitVisible).forEach((hit, index) => {
    const row = document.createElement("tr");
    cell(row, index + 1, "rank-cell");
    const mol = document.createElement("td");
    const main = document.createElement("strong"); main.textContent = hit.chembl_id || `Hit ${index + 1}`;
    const smi = document.createElement("small"); smi.textContent = hit.smiles;
    mol.append(main, smi); row.append(mol);
    cell(row, fmt(hit.hiv_rank_score || hit.score_ensemble, 2), "score-cell");
    cell(row, fmt(hit.qed, 3));
    cell(row, hit.ad_class || hit.ad_label);
    cell(row, hit.conformal);
    cell(row, hit.source_virus || "ZINC");
    cell(row, hit.mechanism);
    const action = document.createElement("td");
    const add = document.createElement("button");
    add.type = "button"; add.className = "text-button"; add.textContent = "Ajouter au lot";
    add.setAttribute("aria-label", `Ajouter ${hit.chembl_id || `hit ${index + 1}`} au lot`);
    add.addEventListener("click", () => {
      const name = hit.chembl_id || `Hit ${index + 1}`;
      const current = $("molecules").value.trim();
      $("molecules").value = `${current}${current ? "\n" : ""}${name},${hit.smiles}`;
      $("molecules").focus();
      $("screen").scrollIntoView({behavior: "smooth"});
    });
    action.append(add); row.append(action);
    body.append(row);
  });
  $("moreHits").hidden = filtered.length <= app.hitVisible;
}

async function loadHits(source) {
  app.source = source;
  app.hitVisible = 20;
  document.querySelectorAll("[data-source]").forEach(button =>
    button.classList.toggle("active", button.dataset.source === source));
  try {
    const data = await api(`api/studio/hits?source=${encodeURIComponent(source)}`);
    app.hits = data.rows || [];
    renderHits();
  } catch (error) {
    app.hits = []; renderHits(); $("hitCount").textContent = error.message;
  }
}

function renderOverview(data) {
  app.overview = data;
  const model = data.model;
  $("modelState").textContent = model ? `A'' · ${model.created} · poids vérifiés` : "Modèle validé indisponible";
  $("modelState").dataset.ready = model ? "true" : "false";
  $("heroModel").textContent = model ? `A'' propre · ${model.created}` : "Indisponible";
  const metrics = data.validation.metrics || {};
  $("heroDrugbank").textContent = metrics.drugbank !== undefined
    ? `${metrics.drugbank}/${data.validation.evaluation.n_arv}` : "—";
  $("heroHits").textContent = `${data.hits.zinc} ZINC · ${data.hits.repurposing} repurposing`;
  $("screenBtn").disabled = !model;
  const selected = [
    ["ROC-AUC", metrics.roc_auc], ["PRC-AUC", metrics.prc_auc],
    ["EF@1 %", metrics.ef_1pct], ["EF@5 %", metrics.ef_5pct],
    ["BEDROC α=20", metrics.bedroc_alpha20],
    ["Hit top 100", metrics.hit_rate_top_100],
    ["Recall top 10 %", metrics.recall_top_10pct],
    ["DrugBank", metrics.drugbank !== undefined
      ? `${metrics.drugbank}/${data.validation.evaluation.n_arv}` : undefined],
  ];
  const grid = $("metricGrid"); grid.replaceChildren();
  selected.forEach(([label, value]) => {
    const item = document.createElement("div");
    const name = document.createElement("span"); name.textContent = label;
    const number = document.createElement("strong");
    number.textContent = typeof value === "number" ? fmt(value, 4) : (value || "—");
    item.append(name, number); grid.append(item);
  });
  $("metricSource").textContent = data.validation.source
    ? `Reproduction exacte sur ${data.validation.evaluation.n_test} molécules · ${data.validation.source} · SHA GNN ${data.registry.gnn_sha256?.slice(0, 12) || "—"} · couverture conformal GNN sur test : ${fmt(100 * data.validation.conformal_test.empirical_coverage, 1)} % (cible 90 %).`
    : "Rapport de validation non disponible.";
  const external = data.external_validation;
  $("externalStatus").textContent = external?.status === "evaluated"
    ? `${external.reports.length} évaluation(s) externe(s) enregistrée(s), après audit de recouvrement.`
    : "Évaluation externe indépendante : aucune donnée admissible fournie. Les performances affichées restent celles du test interne.";
  const packageInfo = data.candidate_package;
  $("packageStatus").textContent = packageInfo
    ? `Liste aveugle prête : ${packageInfo.count} structures (${packageInfo.zinc} ZINC, ${packageInfo.repurposing} repurposing, ${packageInfo.arv_reference} ARV et ${packageInfo.screen_negative_comparator} comparateurs du criblage). Résultats suivis dans la section Validation.`
    : "Liste de validation expérimentale non préparée.";
  $("packageDownload").hidden = !packageInfo;
  const phases = $("phaseRoles"); phases.replaceChildren();
  for (const phase of data.phases) {
    const li = document.createElement("li");
    const num = document.createElement("b"); num.textContent = String(phase.number).padStart(2, "0");
    const content = document.createElement("span");
    const title = document.createElement("strong"); title.textContent = phase.role;
    const note = document.createElement("small"); note.textContent = phase.use;
    content.append(title, note); li.append(num, content); phases.append(li);
  }
}

function renderDevelopment(data) {
  const names = {
    frozen: "Gelé", unavailable: "Indisponible", awaiting_lab_results: "En attente",
    partial: "Partiel", data_inconsistent: "Incohérent", data_complete: "Données complètes",
    evidence_recorded: "Pièce archivée, non validée",
    requires_cellular_data: "Données cellulaires manquantes",
    source_audit_unavailable: "Audit source indisponible",
    not_performed: "Non réalisé", not_started: "Non commencé"
  };
  const list = $("developmentGates"); list.replaceChildren();
  for (const gate of data.gates) {
    const item = document.createElement("li");
    const mark = document.createElement("b"); mark.textContent = gate.status === "data_complete" || gate.status === "frozen" ? "✓" : "○";
    const content = document.createElement("span");
    const title = document.createElement("strong"); title.textContent = gate.name;
    const state = document.createElement("small"); state.textContent = names[gate.status] || gate.status;
    content.append(title, state); item.append(mark, content); list.append(item);
  }
  const assay = data.assay;
  $("assayState").textContent = assay.status === "awaiting_lab_results"
    ? "Aucun résultat expérimental reçu. Le paquet aveugle est prêt pour un laboratoire partenaire."
    : `${names[assay.status] || assay.status} · ${assay.n_measured}/${assay.n_expected} structures mesurées.${assay.issues?.length ? ` ${assay.issues.join(" ; ")}` : ""}`;
    if (assay.prospective_evaluation?.top_budget?.length) {
      const budgets = assay.prospective_evaluation.top_budget.map(item =>
        `top ${item.budget} : ${item.hit} signal(s), ${item.indeterminate} indéterminé(s)`).join(" ; ");
      $("assayState").textContent += ` Bilan descriptif du lot présélectionné : ${budgets}.`;
    }
  const readiness = data.model_readiness;
  $("modelReadiness").textContent = readiness.status === "requires_cellular_data"
    ? `La source ChEMBL disponible contient ${readiness.n_chembl.toLocaleString("fr-FR")} mesures positives, dont ${readiness.assay_types.EC50 || 0} EC50, sans couples EC50/CC50 cellulaires exploitables. Le prochain modèle doit apprendre une activité et une sélectivité mesurées, sur des campagnes indépendantes ; A'' reste gelé en attendant ces données.`
    : "Audit des données quantitatives indisponible.";
}

async function loadDevelopment() {
  try { renderDevelopment(await api("api/studio/development")); }
  catch (error) { $("assayState").textContent = error.message; }
}

document.addEventListener("DOMContentLoaded", () => {
  $("exampleBtn").addEventListener("click", () => {
    $("molecules").value = "Éthanol,CCO\nÉthylamine,CCN\nBenzène,c1ccccc1";
    $("molecules").focus();
  });
  $("screenForm").addEventListener("submit", async event => {
    event.preventDefault();
    const molecules = parseMolecules($("molecules").value);
    if (!molecules.length || molecules.length > 100) return setStatus("Fournir 1 à 100 molécules.", true);
    $("screenBtn").disabled = true; setStatus("Classement en cours sur le modèle local…");
    try {
      const data = await api("api/studio/screen", {method: "POST",
        headers: {"Content-Type": "application/json"}, body: JSON.stringify({molecules})});
      renderRanking(data);
      setStatus(`${data.ranked.length} molécule(s) classée(s).`);
    } catch (error) { setStatus(error.message, true); }
    finally { $("screenBtn").disabled = false; }
  });
  $("exportBtn").addEventListener("click", exportRanking);
  $("hitSearch").addEventListener("input", () => { app.hitVisible = 20; renderHits(); });
  $("moreHits").addEventListener("click", () => { app.hitVisible += 20; renderHits(); });
  document.querySelectorAll("[data-source]").forEach(button => button.addEventListener("click", () =>
    loadHits(button.dataset.source)));
  $("assayFile").addEventListener("change", async event => {
    const file = event.target.files?.[0]; if (!file) return;
    $("assayMessage").textContent = "Contrôle du fichier en cours…";
    try {
      const result = await api("api/studio/assays", {method: "POST",
        headers: {"Content-Type": "text/csv;charset=utf-8"}, body: file});
      $("assayMessage").textContent = `${result.status} · ${result.n_measured}/${result.n_expected} structures. ${result.issues.join(" ; ")}`;
      await loadDevelopment();
    } catch (error) { $("assayMessage").textContent = error.message; }
    event.target.value = "";
  });
  api("api/studio/overview").then(renderOverview).catch(error => {
    $("modelState").textContent = error.message;
    $("screenBtn").disabled = true;
  });
  loadHits("zinc");
  loadDevelopment();
});
