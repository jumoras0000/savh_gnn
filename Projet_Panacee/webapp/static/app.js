/* ====================================================================
   PANACÉE — Console de signes vitaux (frontend autonome, sans dépendance)
   Graphiques canvas faits main + flux SSE temps réel.
   ==================================================================== */
"use strict";

const CSS = getComputedStyle(document.documentElement);
const C = (n) => CSS.getPropertyValue(n).trim();
const COL = {
  vital: C("--vital") || "#28e0bf", blue: C("--blue") || "#4d9bff",
  ok: C("--ok") || "#36d399", warn: C("--warn") || "#ffb02e",
  danger: C("--danger") || "#ff3d68", line: C("--line") || "#243044",
  muted: C("--muted") || "#8294ae", faint: C("--faint") || "#56657f",
  text: C("--text") || "#e8eef7",
};
const MONO = '11px "JetBrains Mono", monospace';

/* ---------- état global ---------- */
const state = {
  runId: null, meta: {}, epochs: [], latest: {},
  expected: {}, thresholds: {}, status: "idle",
  verdict: null, compare: [], perTask: {}, es: null,
};

/* ====================================================================
   Utilitaires canvas haute densité
   ==================================================================== */
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(r.width)), h = Math.max(1, Math.floor(r.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

const nf = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const pct = (v) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (v * 100).toFixed(0) + "%";

/* ====================================================================
   Graphique en courbes (multi-séries + lignes de référence + zones)
   series: [{name, color, data:[{x,y}], dashed?}]
   opts:   {yMin,yMax, refs:[{y,color,label}], bands:[{from,to,color}]}
   ==================================================================== */
function lineChart(canvas, series, opts = {}) {
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const padL = 44, padR = 14, padT = 14, padB = 26;
  const plotW = w - padL - padR, plotH = h - padT - padB;

  const allY = [], allX = [];
  series.forEach(s => s.data.forEach(p => { if (p.y != null) allY.push(p.y); allX.push(p.x); }));
  (opts.refs || []).forEach(r => allY.push(r.y));
  if (allX.length === 0) { emptyChart(ctx, w, h); return; }

  let yMin = opts.yMin != null ? opts.yMin : Math.min(...allY);
  let yMax = opts.yMax != null ? opts.yMax : Math.max(...allY);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const X = (x) => padL + (xMax === xMin ? plotW / 2 : (x - xMin) / (xMax - xMin) * plotW);
  const Y = (y) => padT + plotH - (y - yMin) / (yMax - yMin) * plotH;

  // bandes (zones de danger)
  (opts.bands || []).forEach(b => {
    const y1 = Y(Math.min(b.to, yMax)), y2 = Y(Math.max(b.from, yMin));
    ctx.fillStyle = b.color; ctx.fillRect(padL, y1, plotW, y2 - y1);
  });

  // grille + axe Y
  ctx.strokeStyle = COL.line; ctx.fillStyle = COL.faint;
  ctx.font = MONO; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const val = yMin + (yMax - yMin) * i / ticks, yy = Y(val);
    ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
    ctx.globalAlpha = 1; ctx.fillText(val.toFixed(2), padL - 6, yy);
  }

  // axe X (epochs)
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xticks = Math.min(6, Math.max(1, xMax - xMin));
  for (let i = 0; i <= xticks; i++) {
    const xv = Math.round(xMin + (xMax - xMin) * i / xticks);
    ctx.fillText(String(xv), X(xv), h - padB + 6);
  }

  // lignes de référence
  (opts.refs || []).forEach(r => {
    const yy = Y(r.y);
    ctx.strokeStyle = r.color; ctx.globalAlpha = 0.9; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  });

  // séries
  series.forEach(s => {
    const pts = s.data.filter(p => p.y != null);
    if (!pts.length) return;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = "round";
    if (s.dashed) ctx.setLineDash([4, 3]);
    ctx.beginPath();
    pts.forEach((p, i) => { const px = X(p.x), py = Y(p.y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke(); ctx.setLineDash([]);
    // dernier point en évidence
    const last = pts[pts.length - 1];
    ctx.fillStyle = s.color; ctx.beginPath();
    ctx.arc(X(last.x), Y(last.y), 3, 0, Math.PI * 2); ctx.fill();
  });
}

function emptyChart(ctx, w, h) {
  ctx.fillStyle = COL.faint; ctx.font = MONO; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("en attente de données…", w / 2, h / 2);
}

/* ---------- barres (per-task AUC vs cible) ---------- */
function barChart(canvas, items, opts = {}) {
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const padL = 36, padR = 12, padT = 12, padB = 64;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  if (!items.length) { emptyChart(ctx, w, h); return; }
  const yMin = opts.yMin != null ? opts.yMin : 0.4, yMax = 1.0;
  const Y = (y) => padT + plotH - (y - yMin) / (yMax - yMin) * plotH;

  ctx.strokeStyle = COL.line; ctx.fillStyle = COL.faint; ctx.font = MONO;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const val = yMin + (yMax - yMin) * i / 4, yy = Y(val);
    ctx.globalAlpha = .35; ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
    ctx.globalAlpha = 1; ctx.fillText(val.toFixed(2), padL - 6, yy);
  }

  const n = items.length, gap = 6, bw = Math.max(4, (plotW - gap * (n - 1)) / n);
  items.forEach((it, i) => {
    const x = padL + i * (bw + gap), v = it.value == null ? yMin : it.value;
    const col = v < (opts.danger || 0.6) ? COL.danger : v < (opts.target || 0.85) ? COL.warn : COL.ok;
    const yy = Y(Math.max(v, yMin));
    ctx.fillStyle = col; ctx.fillRect(x, yy, bw, padT + plotH - yy);
    ctx.save(); ctx.translate(x + bw / 2, h - padB + 6); ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = COL.muted; ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(it.label, 0, 0); ctx.restore();
  });

  // lignes cible + danger
  [[opts.target || 0.85, COL.warn], [opts.danger || 0.6, COL.danger]].forEach(([yv, c]) => {
    const yy = Y(yv); ctx.strokeStyle = c; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke(); ctx.setLineDash([]);
  });
}

/* ---------- sparkline KPI ---------- */
function sparkline(canvas, data, color) {
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const pts = data.filter(v => v != null);
  if (pts.length < 2) return;
  const mn = Math.min(...pts), mx = Math.max(...pts), pad = 3;
  const X = (i) => pad + i / (pts.length - 1) * (w - 2 * pad);
  const Y = (v) => mx === mn ? h / 2 : pad + (1 - (v - mn) / (mx - mn)) * (h - 2 * pad);
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.lineJoin = "round"; ctx.beginPath();
  pts.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)));
  ctx.stroke();
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(X(pts.length - 1), Y(pts[pts.length - 1]), 2.2, 0, 7); ctx.fill();
}

/* ====================================================================
   ECG — battement animé (signature visuelle)
   Vitesse/amplitude pilotées par l'état (running = vif, done = calme)
   ==================================================================== */
const ecg = { canvas: null, phase: 0, beats: [], active: false };
function ecgInit() {
  ecg.canvas = document.getElementById("ecg");
  requestAnimationFrame(ecgLoop);
}
function ecgWave(t) {
  // forme P-QRS-T simplifiée sur t∈[0,1)
  const x = t % 1;
  if (x < 0.10) return 0.12 * Math.sin(x / 0.10 * Math.PI);       // P
  if (x < 0.16) return -0.10;                                      // Q
  if (x < 0.20) return 1.0;                                        // R (pic)
  if (x < 0.26) return -0.28;                                      // S
  if (x < 0.45) return 0.22 * Math.sin((x - 0.26) / 0.19 * Math.PI); // T
  return 0;
}
function ecgLoop() {
  const cv = ecg.canvas; if (!cv) return;
  const { ctx, w, h } = fitCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const running = state.status === "running";
  const amp = running ? 1.0 : 0.32;
  const col = state.verdict && state.verdict.level === "DANGER" ? COL.danger
            : state.verdict && state.verdict.level === "WARN" ? COL.warn : COL.vital;
  const speed = running ? 0.020 : 0.006;
  ecg.phase = (ecg.phase + speed) % 1;
  const cycles = 2.2, mid = h * 0.55, scale = h * 0.34 * amp;
  ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.lineJoin = "round";
  ctx.shadowColor = col; ctx.shadowBlur = running ? 8 : 2;
  ctx.beginPath();
  for (let px = 0; px <= w; px++) {
    const t = (px / w) * cycles + ecg.phase * cycles;
    const y = mid - ecgWave(t) * scale;
    px ? ctx.lineTo(px, y) : ctx.moveTo(px, y);
  }
  ctx.stroke(); ctx.shadowBlur = 0;
  requestAnimationFrame(ecgLoop);
}

/* ====================================================================
   Rendu : KPIs, verdict, charts, tables
   ==================================================================== */
function series(key) { return state.epochs.map(e => ({ x: e.epoch, y: e[key] ?? null })); }
function col(key) { return state.epochs.map(e => e[key] ?? null); }

function renderKPIs() {
  const L = state.latest;
  // Epoch + progression (on garde le <span class="unit"> en réécrivant le contenu)
  const total = state.meta.epochs_total;
  const epEl = document.querySelector("#kpiEpoch .value");
  epEl.innerHTML = (L.epoch != null ? String(L.epoch) : "—") +
    `<span class="unit" id="epochTotal">${total ? " / " + total : ""}</span>`;
  const frac = (total && L.epoch) ? Math.min(1, L.epoch / total) : 0;
  document.getElementById("epochBar").style.width = (frac * 100) + "%";

  // AUC
  setKpi("kpiAuc", L.val_auc, nf(L.val_auc), col("val_auc"), COL.vital,
    flag(L.val_auc, state.expected.val_auc, "min"));
  // Sensibilité
  setKpi("kpiSens", L.macro_sensitivity, pct(L.macro_sensitivity), col("macro_sensitivity"), COL.ok,
    flag(L.macro_sensitivity, state.expected.macro_sensitivity, "min"));
  // FNR
  setKpi("kpiFnr", L.macro_fnr, pct(L.macro_fnr), col("macro_fnr"), COL.danger,
    flag(L.macro_fnr, state.expected.macro_fnr_max, "max"));
  // Danger
  const nd = L.n_danger;
  const dEl = document.getElementById("kpiDanger");
  dEl.querySelector(".value").textContent = nd != null ? String(nd) : "—";
  dEl.dataset.flag = (nd > 0) ? "bad" : (nd === 0 ? "good" : "");

  // ETA
  const eta = estimateEta();
  document.getElementById("etaTag").textContent = eta || " ";
}

function setKpi(id, raw, text, sparkData, color, flagVal) {
  const el = document.getElementById(id);
  el.querySelector(".value").textContent = text;
  el.dataset.flag = flagVal;
  const spark = el.querySelector(".spark");
  if (spark) sparkline(spark, sparkData, color);
}

function flag(v, target, sense) {
  if (v == null || target == null) return "";
  if (sense === "min") return v >= target ? "good" : (v >= target * 0.9 ? "warn" : "bad");
  return v <= target ? "good" : (v <= target * 1.3 ? "warn" : "bad");
}

function estimateEta() {
  const ep = state.epochs;
  if (state.status !== "running" || ep.length < 2 || !state.meta.epochs_total) return "";
  const dts = [];
  for (let i = 1; i < ep.length; i++) if (ep[i].time && ep[i - 1].time) dts.push(ep[i].time - ep[i - 1].time);
  if (!dts.length) return "";
  const avg = dts.reduce((a, b) => a + b, 0) / dts.length;
  const remain = state.meta.epochs_total - ep[ep.length - 1].epoch;
  if (remain <= 0) return "terminé";
  const s = Math.round(avg * remain);
  return "ETA ~" + (s >= 60 ? Math.floor(s / 60) + "m" + (s % 60) + "s" : s + "s");
}

function renderVerdict() {
  const v = state.verdict || { level: "NA", title: "En attente de données", reasons: [] };
  const el = document.getElementById("verdict");
  el.dataset.level = v.level;
  const icons = { OK: "🟢", WARN: "🟠", DANGER: "🔴", NA: "🩺" };
  document.getElementById("verdictIcon").textContent = icons[v.level] || "🩺";
  document.getElementById("verdictTitle").textContent = v.title;
  document.getElementById("verdictReasons").innerHTML =
    (v.reasons || []).map(r => `<li>${esc(r)}</li>`).join("");
}

function renderEvolution() {
  const has = state.epochs.length > 0;
  document.getElementById("evoEmpty").style.display = has ? "none" : "block";
  document.getElementById("evoCharts").style.display = has ? "block" : "none";
  if (!has) return;

  lineChart(document.getElementById("chartLoss"),
    [{ name: "train", color: COL.blue, data: series("train_loss") },
     { name: "val", color: COL.vital, data: series("val_loss") }],
    { yMin: 0 });

  lineChart(document.getElementById("chartAuc"),
    [{ name: "train", color: COL.blue, data: series("train_auc") },
     { name: "val", color: COL.vital, data: series("val_auc") }],
    { yMin: 0.4, yMax: 1.0, refs: [{ y: state.expected.val_auc || 0.85, color: COL.warn }] });

  const fnrDanger = state.thresholds.fnr_danger || 0.5;
  lineChart(document.getElementById("chartSafety"),
    [{ name: "sens", color: COL.ok, data: series("macro_sensitivity") },
     { name: "fnr", color: COL.danger, data: series("macro_fnr") }],
    { yMin: 0, yMax: 1.0,
      bands: [{ from: fnrDanger, to: 1.0, color: "rgba(255,61,104,.06)" }],
      refs: [{ y: state.expected.macro_fnr_max || 0.3, color: COL.warn },
             { y: fnrDanger, color: COL.danger }] });
}

/* ---------- métriques cliniques ---------- */
const CLIN_COLS = [
  ["task", "Endpoint"], ["danger", "Niveau"], ["support", "N"], ["prevalence", "Prév."],
  ["sensitivity", "Sensib."], ["specificity", "Spécif."], ["fnr", "FNR"],
  ["precision", "Précis."], ["f1", "F1"], ["roc_auc", "ROC-AUC"], ["pr_auc", "PR-AUC"], ["ece", "ECE"],
];

function renderClinicalFromLive() {
  // À défaut d'évaluation, on affiche le per-task AUC du dernier point
  const pt = state.perTask || {};
  const tbody = document.querySelector("#clinTable tbody");
  const thead = document.querySelector("#clinTable thead");
  thead.innerHTML = "<tr><th>Endpoint</th><th>ROC-AUC (live)</th><th>Niveau</th></tr>";
  const dT = state.thresholds.auc_danger || 0.6, wT = state.thresholds.auc_warn || 0.7;
  const rows = Object.entries(pt).map(([k, v]) => {
    const lvl = v == null ? "NA" : v < dT ? "DANGER" : v < wT ? "WARN" : "OK";
    return `<tr data-level="${lvl}"><td>${esc(k)}</td><td>${nf(v)}</td><td><span class="badge ${lvl}">${lvl}</span></td></tr>`;
  });
  tbody.innerHTML = rows.join("") || `<tr><td colspan="3" style="color:var(--faint)">Pas encore de données per-endpoint.</td></tr>`;
}

function renderClinicalFromEval(res) {
  const agg = res.aggregate || {};
  document.getElementById("evalAgg").innerHTML = `
    <div class="kpis" style="margin:0">
      ${aggCard("AUC macro", nf(agg.macro_roc_auc))}
      ${aggCard("Sensibilité macro", pct(agg.macro_sensitivity))}
      ${aggCard("FNR macro", pct(agg.macro_fnr))}
      ${aggCard("ECE (calibration)", nf(agg.mean_ece))}
      ${aggCard("Endpoints DANGER", agg.n_danger ?? "—")}
    </div>`;
  const thead = document.querySelector("#clinTable thead");
  const tbody = document.querySelector("#clinTable tbody");
  thead.innerHTML = "<tr>" + CLIN_COLS.map(([, h]) => `<th>${h}</th>`).join("") + "</tr>";
  tbody.innerHTML = (res.tasks || []).map(t => {
    const cells = CLIN_COLS.map(([k]) => {
      if (k === "task") return `<td>${esc(t.task)}</td>`;
      if (k === "danger") return `<td><span class="badge ${t.danger}">${t.danger}</span></td>`;
      if (k === "support") return `<td>${t.support}</td>`;
      if (["prevalence", "sensitivity", "specificity", "fnr", "precision", "f1"].includes(k))
        return `<td>${pct(t[k])}</td>`;
      return `<td>${nf(t[k])}</td>`;
    });
    return `<tr data-level="${t.danger}">${cells.join("")}</tr>`;
  }).join("");
  document.getElementById("clinSource").textContent =
    `Source : évaluation du checkpoint (${res.meta ? res.meta.n_molecules : "?"} molécules).`;
  // alimente aussi l'onglet sécurité
  renderAlerts(res.alerts);
}

function aggCard(label, val) {
  return `<div class="kpi"><div class="label">${label}</div><div class="value" style="font-size:24px">${val}</div></div>`;
}

/* ---------- sécurité ---------- */
function renderAlerts(alerts) {
  const box = document.getElementById("alerts");
  if (alerts && alerts.length) {
    box.innerHTML = alerts.map(a => {
      const ico = a.level === "DANGER" ? "🔴" : "🟠";
      return `<div class="alert ${a.level}"><div class="a-ico">${ico}</div>
        <div><div class="a-title">${a.level === "DANGER" ? "DANGER" : "ATTENTION"} — ${esc(a.task)}</div>
        <div class="a-msg">${esc(a.message)}</div></div></div>`;
    }).join("");
    return;
  }
  // fallback temps réel
  const L = state.latest, nd = L.n_danger || 0, nw = L.n_warn || 0;
  if (nd) box.innerHTML = `<div class="alert DANGER"><div class="a-ico">🔴</div><div><div class="a-title">${nd} endpoint(s) en DANGER (live)</div><div class="a-msg">Lance une évaluation de checkpoint pour le détail par endpoint.</div></div></div>`;
  else if (nw) box.innerHTML = `<div class="alert WARN"><div class="a-ico">🟠</div><div><div class="a-title">${nw} endpoint(s) à surveiller (live)</div><div class="a-msg">Surveille le FNR par endpoint.</div></div></div>`;
  else if (state.epochs.length) box.innerHTML = `<div class="alert OK"><div class="a-ico">🟢</div><div><div class="a-title">Aucun endpoint en danger</div><div class="a-msg">Sensibilité et FNR dans les cibles sur ce point.</div></div></div>`;
  else box.innerHTML = `<div class="empty">Pas d'alerte — en attente de données.</div>`;
}

function renderBarème() {
  const t = state.thresholds;
  document.getElementById("barème").innerHTML = `
    <b>DANGER</b> si FNR ≥ ${pct(t.fnr_danger)} &nbsp;ou&nbsp; sensibilité &lt; ${pct(t.sens_danger)} &nbsp;ou&nbsp; ROC-AUC &lt; ${nf(t.auc_danger, 2)}<br>
    <b>ATTENTION</b> si FNR ≥ ${pct(t.fnr_warn)} &nbsp;ou&nbsp; ROC-AUC &lt; ${nf(t.auc_warn, 2)}<br>
    <b>Rappel clinique</b> : un faux négatif (toxique prédit « sûr ») est l'erreur la plus grave →
    on privilégie la <b>sensibilité</b> et on plafonne le <b>FNR</b>.`;
}

/* ---------- comparaison ---------- */
function renderCompare() {
  // attendu vs obtenu
  const cmp = state.compare || [];
  const thead = document.querySelector("#cmpTable thead");
  const tbody = document.querySelector("#cmpTable tbody");
  thead.innerHTML = "<tr><th>Métrique</th><th>Obtenu</th><th>Attendu</th><th>Statut</th></tr>";
  tbody.innerHTML = cmp.map(r => {
    const op = r.sens === "max" ? "≤" : "≥";
    const got = (r.key === "val_auc") ? nf(r.obtenu) : pct(r.obtenu);
    const exp = (r.key === "val_auc") ? nf(r.attendu, 2) : pct(r.attendu);
    return `<tr><td>${esc(r.metric)}</td><td>${got}</td><td>${op} ${exp}</td>
      <td><span class="badge ${r.ok ? "OK" : "DANGER"}">${r.ok ? "OK" : "RATÉ"}</span></td></tr>`;
  }).join("") || `<tr><td colspan="4" style="color:var(--faint)">Pas encore de métriques.</td></tr>`;

  // barres per-task
  const items = Object.entries(state.perTask || {}).map(([k, v]) => ({ label: k, value: v }));
  barChart(document.getElementById("chartPerTask"), items,
    { target: state.expected.val_auc || 0.85, danger: state.thresholds.auc_danger || 0.6 });
}

async function renderRunsTable() {
  const data = await fetchJSON("/api/compare");
  const thead = document.querySelector("#runsTable thead");
  const tbody = document.querySelector("#runsTable tbody");
  thead.innerHTML = "<tr><th>Run</th><th>Phase</th><th>Epochs</th><th>ROC-AUC</th><th>Sensib.</th><th>FNR</th><th>Danger</th><th>Verdict</th></tr>";
  tbody.innerHTML = (data.runs || []).map(r => `
    <tr data-level="${r.verdict}">
      <td>${esc(r.id)}</td><td>${esc(r.phase)}</td><td>${r.epochs}</td>
      <td>${nf(r.val_auc)}</td><td>${pct(r.macro_sensitivity)}</td><td>${pct(r.macro_fnr)}</td>
      <td>${r.n_danger ?? "—"}</td><td><span class="badge ${r.verdict}">${r.verdict}</span></td>
    </tr>`).join("") || `<tr><td colspan="8" style="color:var(--faint)">Aucun run détecté.</td></tr>`;
}

/* ---------- statut header ---------- */
function renderStatus() {
  const pill = document.getElementById("statusPill");
  pill.dataset.state = state.status;
  const txt = { running: "EN COURS", done: "TERMINÉ", idle: "EN ATTENTE" }[state.status] || "—";
  document.getElementById("statusText").textContent = txt;
}

function renderAll() {
  renderStatus(); renderVerdict(); renderKPIs(); renderEvolution();
  renderClinicalFromLive(); renderAlerts(); renderBarème(); renderCompare();
}

/* ====================================================================
   Données : REST + SSE
   ==================================================================== */
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).error || m; } catch (e) {} throw new Error(m); }
  return r.json();
}

async function loadConfig() {
  try {
    const cfg = await fetchJSON("/api/config");
    state.expected = cfg.expected; state.thresholds = cfg.thresholds;
  } catch (e) { /* défauts gérés côté getters */ }
}

async function loadRuns() {
  const data = await fetchJSON("/api/runs");
  const sel = document.getElementById("runSelect");
  const runs = data.runs || [];
  sel.innerHTML = runs.map(r =>
    `<option value="${esc(r.id)}">${esc(r.id)} · ${r.status} · ${r.points} pts</option>`).join("");
  if (!runs.length) { sel.innerHTML = `<option value="">aucun run</option>`; renderAll(); return; }
  // garde le run courant si présent, sinon prend le plus récent
  const ids = runs.map(r => r.id);
  if (!state.runId || !ids.includes(state.runId)) state.runId = runs[0].id;
  sel.value = state.runId;
  connectStream(state.runId);
}

function applySnapshot(d) {
  state.meta = d.meta || {}; state.epochs = d.epochs || []; state.latest = d.latest || {};
  state.status = d.status; state.verdict = d.verdict; state.compare = d.compare || [];
  state.perTask = d.per_task_auc || {}; state.expected = d.expected || state.expected;
  state.thresholds = d.thresholds || state.thresholds;
  renderAll();
}

function connectStream(runId) {
  if (state.es) { state.es.close(); state.es = null; }
  const conn = document.getElementById("connText");
  conn.textContent = "connexion…";
  const es = new EventSource(`/api/stream?id=${encodeURIComponent(runId)}`);
  state.es = es;

  es.addEventListener("snapshot", (ev) => { conn.innerHTML = "<b>actif</b>"; applySnapshot(JSON.parse(ev.data)); });
  es.addEventListener("waiting", () => { conn.textContent = "en attente"; });
  es.addEventListener("epoch", (ev) => {
    const e = JSON.parse(ev.data);
    state.epochs.push(e); state.latest = e;
    if (e.per_task_auc) state.perTask = e.per_task_auc;
    renderKPIs(); renderEvolution(); renderClinicalFromLive(); renderAlerts(); renderCompare();
    flashToast(`Epoch ${e.epoch} · AUC ${nf(e.val_auc)} · FNR ${pct(e.macro_fnr)}`);
  });
  es.addEventListener("status", (ev) => {
    const s = JSON.parse(ev.data);
    state.status = s.status; state.verdict = s.verdict; state.compare = s.compare || state.compare;
    if (s.epochs_total) state.meta.epochs_total = s.epochs_total;
    renderStatus(); renderVerdict(); renderKPIs();
  });
  es.addEventListener("ping", () => { conn.innerHTML = "<b>actif</b>"; });
  es.onerror = () => { conn.textContent = "reconnexion…"; };
}

/* ====================================================================
   Interactions
   ==================================================================== */
function setupTabs() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".panel-view").forEach(v => v.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("view-" + btn.dataset.tab).classList.add("active");
      // redessine les charts du panneau affiché (canvas a maintenant une taille)
      requestAnimationFrame(() => { renderEvolution(); renderCompare(); });
    });
  });
}

function setupRunSelect() {
  document.getElementById("runSelect").addEventListener("change", (e) => {
    state.runId = e.target.value;
    if (state.runId) connectStream(state.runId);
  });
}

function setupEval() {
  const btn = document.getElementById("evalBtn");
  btn.addEventListener("click", async () => {
    const checkpoint = document.getElementById("evalCkpt").value.trim();
    const val_csv = document.getElementById("evalCsv").value.trim();
    if (!checkpoint || !val_csv) { flashToast("Renseigne le checkpoint et le CSV.", true); return; }
    const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Évaluation…';
    try {
      const res = await fetchJSON("/api/evaluate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint, val_csv }),
      });
      renderClinicalFromEval(res);
      flashToast("Évaluation terminée.");
    } catch (e) { flashToast("Échec : " + e.message, true); }
    finally { btn.disabled = false; btn.innerHTML = old; }
  });
}

let toastTimer = null;
function flashToast(msg, err = false) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.className = "toast show" + (err ? " err" : "");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), 3200);
}

const esc = (s) => String(s).replace(/[&<>"']/g, c => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* redessine au redimensionnement */
let rz = null;
window.addEventListener("resize", () => {
  clearTimeout(rz); rz = setTimeout(() => { renderEvolution(); renderCompare(); renderKPIs(); }, 120);
});

/* ====================================================================
   Démarrage
   ==================================================================== */
async function main() {
  ecgInit(); setupTabs(); setupRunSelect(); setupEval();
  await loadConfig();
  renderBarème();
  await loadRuns();
  await renderRunsTable();
  // rafraîchit la liste des runs périodiquement (nouveaux runs)
  setInterval(() => loadRuns().catch(() => {}), 15000);
  setInterval(() => renderRunsTable().catch(() => {}), 15000);
}
document.addEventListener("DOMContentLoaded", main);
