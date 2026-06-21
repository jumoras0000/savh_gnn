/* ====================================================================
   PANACÉE — Console de signes vitaux (frontend autonome, sans dépendance)
   Graphiques canvas faits main + flux SSE temps réel.
   ==================================================================== */
"use strict";

const C = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = {};
function readColors() {
  Object.assign(COL, {
    vital: C("--vital") || "#28e0bf", blue: C("--blue") || "#4d9bff",
    ok: C("--ok") || "#36d399", warn: C("--warn") || "#ffb02e",
    danger: C("--danger") || "#ff3d68", line: C("--line") || "#243044",
    muted: C("--muted") || "#8294ae", faint: C("--faint") || "#56657f",
    text: C("--text") || "#e8eef7",
  });
}
readColors();
const MONO = '11px "JetBrains Mono", monospace';

/* ---------- état global ---------- */
const state = {
  runId: null, meta: {}, epochs: [], latest: {},
  expected: {}, thresholds: {}, status: "idle",
  verdict: null, compare: [], perTask: {}, es: null,
  observations: [], files: { checkpoints: [], csvs: [] },
  trainTimer: null, chatHistory: [], currentConv: null, _loadConversations: null,
  runs: [],
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

/* ---------- observation & risque (live, miroir du backend) ---------- */
function computeObservations() {
  const L = state.latest, e = state.expected, t = state.thresholds, out = [];
  if (!L || L.epoch == null) {
    return [{ level: "INFO", metric: "—", text: "Aucune métrique reçue. Lance un entraînement." }];
  }
  const auc = L.val_auc;
  if (auc != null) {
    if (auc < (t.auc_danger ?? 0.6)) out.push({ level: "DANGER", metric: "ROC-AUC", text: `AUC ${auc.toFixed(2)} ≈ aléatoire : le modèle ne discrimine pas.` });
    else if (auc < (e.val_auc ?? 0.85)) out.push({ level: "WARN", metric: "ROC-AUC", text: `AUC ${auc.toFixed(2)} sous la cible ${e.val_auc ?? 0.85} : marge de progrès.` });
    else out.push({ level: "OK", metric: "ROC-AUC", text: `AUC ${auc.toFixed(2)} ≥ cible : bon pouvoir discriminant.` });
  }
  const fnr = L.macro_fnr;
  if (fnr != null) {
    if (fnr >= (t.fnr_danger ?? 0.5)) out.push({ level: "DANGER", metric: "FNR", text: `FNR ${(fnr * 100).toFixed(0)}% : trop de composés TOXIQUES manqués (risque clinique).` });
    else if (fnr > (e.macro_fnr_max ?? 0.3)) out.push({ level: "WARN", metric: "FNR", text: `FNR ${(fnr * 100).toFixed(0)}% au-dessus du seuil toléré.` });
    else out.push({ level: "OK", metric: "FNR", text: `FNR ${(fnr * 100).toFixed(0)}% sous le seuil : peu de toxiques manqués.` });
  }
  const sens = L.macro_sensitivity;
  if (sens != null && sens < (t.sens_danger ?? 0.5)) out.push({ level: "DANGER", metric: "Sensibilité", text: `Sensibilité ${(sens * 100).toFixed(0)}% : détection insuffisante.` });
  if (L.train_auc != null && L.val_auc != null) {
    const gap = L.train_auc - L.val_auc;
    if (gap > 0.15) out.push({ level: "WARN", metric: "Surapprentissage", text: `Écart train-val AUC = ${gap.toFixed(2)} : surapprentissage probable.` });
  }
  if ((L.n_danger || 0) > 0) out.push({ level: "DANGER", metric: "Endpoints", text: `${L.n_danger} endpoint(s) en DANGER — voir Sécurité.` });
  // Phase 1/3 : pas de toxicité → s'appuyer sur la perte
  if (auc == null && L.val_loss != null) out.push({ level: "INFO", metric: "Perte", text: `val_loss = ${nf(L.val_loss, 4)} (phase sans métrique de toxicité).` });
  return out.length ? out : [{ level: "OK", metric: "—", text: "Indicateurs nominaux." }];
}

function renderObservations() {
  const box = document.getElementById("observations");
  if (!box) return;
  const obs = computeObservations();
  const ico = { OK: "🟢", WARN: "🟠", DANGER: "🔴", INFO: "•" };
  box.innerHTML = obs.map(o =>
    `<div class="obs ${o.level}"><span class="o-ico">${ico[o.level] || "•"}</span>
      <span class="o-metric">${esc(o.metric)}</span><span class="o-text">${esc(o.text)}</span></div>`).join("");
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
  renderObservations(); renderClinicalFromLive(); renderAlerts(); renderBarème(); renderCompare();
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
  state.runs = runs;
  sel.innerHTML = runs.map(r => {
    const tag = r.source === "remote" ? "🛰️ Kaggle · " : "";
    return `<option value="${esc(r.id)}">${tag}${esc(r.id)} · ${r.status} · ${r.points} pts</option>`;
  }).join("");
  updateKaggleBanner();
  if (!runs.length) { sel.innerHTML = `<option value="">aucun run</option>`; renderAll(); return; }
  // garde le run courant si présent, sinon prend le plus récent
  const ids = runs.map(r => r.id);
  if (!state.runId || !ids.includes(state.runId)) state.runId = runs[0].id;
  sel.value = state.runId;
  connectStream(state.runId);
}

// Bannière « Entraînement Kaggle en cours » : visible dès qu'un run distant tourne.
function updateKaggleBanner() {
  const banner = document.getElementById("kaggleBanner");
  if (!banner) return;
  const remote = (state.runs || []).filter(r => r.source === "remote" && r.status === "running");
  if (!remote.length) { banner.style.display = "none"; return; }
  const r = remote[0];
  const auc = (r.val_auc != null) ? nf(r.val_auc) : "—";
  const ep = r.epochs_total ? `${r.last_epoch || 0}/${r.epochs_total}` : `${r.last_epoch || 0}`;
  banner.style.display = "flex";
  banner.innerHTML =
    `<span class="kb-pulse"></span>` +
    `<div><b>🛰️ Entraînement Kaggle en cours</b> — run <code>${esc(r.id)}</code> · ` +
    `epoch ${ep} · AUC ${auc} · ${r.points} points` +
    `<div class="kb-meta">Les courbes et le verdict clinique se mettent à jour en temps réel. ` +
    `<a href="#" id="kbGoEvo">Voir l'évolution →</a></div></div>`;
  const go = document.getElementById("kbGoEvo");
  if (go) go.addEventListener("click", (e) => {
    e.preventDefault();
    if (r.id !== state.runId) { state.runId = r.id; connectStream(r.id);
      const sel = document.getElementById("runSelect"); if (sel) sel.value = r.id; }
    document.querySelector('.tab[data-tab="evo"]').click();
  });
}

function applySnapshot(d) {
  state.meta = d.meta || {}; state.epochs = d.epochs || []; state.latest = d.latest || {};
  state.status = d.status; state.verdict = d.verdict; state.compare = d.compare || [];
  state.perTask = d.per_task_auc || {}; state.expected = d.expected || state.expected;
  state.thresholds = d.thresholds || state.thresholds;
  state.observations = d.observations || [];
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
    renderKPIs(); renderEvolution(); renderObservations(); renderClinicalFromLive(); renderAlerts(); renderCompare();
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
      const tab = btn.dataset.tab;
      document.getElementById("view-" + tab).classList.add("active");
      // redessine les charts du panneau affiché (canvas a maintenant une taille)
      requestAnimationFrame(() => { renderEvolution(); renderCompare(); });
      if (tab === "train") pollTrain();
      if (tab === "clin" || tab === "research") loadFiles();
      if (tab === "screen") { loadFiles(); loadLibraries(); }
      if (tab === "info") loadCapabilities();
      if (tab === "chat") refreshChatMode();
    });
  });
}

/* ====================================================================
   Aide contextuelle « En savoir plus » — une fiche par page
   ==================================================================== */
const PAGE_HELP = {
  evo: {
    title: "📈 Évolution — suivi temps réel de l'entraînement",
    role: "Visualise, <b>en direct</b>, l'apprentissage du modèle GNN epoch après epoch : courbes de perte, ROC-AUC et signes vitaux de sécurité. Les données arrivent par flux SSE depuis l'entraînement local <i>ou</i> Kaggle.",
    entries: [
      "<b>KPI (cartes du haut)</b> — epoch courant + ETA, ROC-AUC validation, sensibilité, FNR (faux négatifs), endpoints en danger. Chaque carte a une <i>sparkline</i> de tendance.",
      "<b>Courbes loss / AUC</b> — comparaison train vs validation. Un écart croissant = surapprentissage.",
      "<b>Signes vitaux de sécurité</b> — sensibilité &amp; FNR avec lignes de cible et zone de danger.",
      "<b>ECG animé</b> — bat plus vite quand l'entraînement tourne, vire au rouge en cas de DANGER.",
      "<b>Sélecteur de run</b> (barre du haut) — bascule entre les entraînements détectés.",
    ],
    howto: [
      "Lance un entraînement (onglet Entraînement) ou pousse-le depuis Kaggle.",
      "Sélectionne le run dans la liste déroulante en haut.",
      "Observe les courbes se remplir en temps réel ; surveille le verdict clinique en haut de page.",
    ],
    objective: "Détecter tôt si un modèle dérape (FNR qui monte, AUC qui stagne) pour arrêter et réajuster sans attendre la fin.",
  },
  clin: {
    title: "🏥 Métriques cliniques — évaluation par endpoint",
    role: "Calcule le détail des performances <b>endpoint toxicologique par endpoint</b> (les 12 cibles Tox21) à partir d'un checkpoint et d'un CSV de validation.",
    entries: [
      "<b>Sélecteur de checkpoint (.pth)</b> — le modèle entraîné à évaluer (détecté automatiquement).",
      "<b>Sélecteur de CSV de validation</b> — le jeu de données étiqueté pour la mesure.",
      "<b>Importer</b> — charge un .pth ou .csv qui n'est pas déjà dans le projet.",
      "<b>Tableau</b> — par endpoint : sensibilité, spécificité, FNR, précision, F1, ROC-AUC, PR-AUC, ECE (calibration).",
    ],
    howto: [
      "Choisis un checkpoint et un CSV dans les sélecteurs.",
      "Clique « Évaluer ».",
      "Lis le tableau : repère les endpoints à FNR élevé (toxiques manqués) — les plus risqués.",
    ],
    objective: "Savoir précisément OÙ le modèle se trompe pour juger s'il est déployable et sur quels dangers il faut se méfier.",
  },
  sec: {
    title: "🚨 Sécurité — alertes triées par gravité",
    role: "Concentre tous les <b>signaux de danger</b> du run courant : endpoints où le modèle manque trop de composés toxiques, sensibilité/AUC sous les seuils.",
    entries: [
      "<b>Liste d'alertes</b> — chaque ligne = un risque, trié DANGER puis WARN.",
      "<b>Barème</b> — rappel des seuils (FNR &lt; 30 %, sensibilité &amp; AUC cibles).",
    ],
    howto: [
      "Ouvre l'onglet pendant ou après un entraînement.",
      "Traite d'abord les alertes 🔴 DANGER, puis 🟠 WARN.",
    ],
    objective: "Transformer des chiffres en décisions de sécurité : ce modèle est-il assez sûr pour passer à l'étape suivante ?",
  },
  cmp: {
    title: "🔬 Comparaison — attendu vs obtenu, et runs entre eux",
    role: "Compare les résultats <b>obtenus</b> aux <b>cibles attendues</b>, et tous les runs entre eux (ROC-AUC par endpoint, métriques globales).",
    entries: [
      "<b>Attendu vs obtenu</b> — l'écart à la cible pour chaque métrique clé.",
      "<b>ROC-AUC par endpoint</b> — barres comparées à la cible.",
      "<b>Tableau multi-runs</b> — toutes les expériences côte à côte.",
    ],
    howto: [
      "Lance plusieurs entraînements (phases / hyperparamètres différents).",
      "Reviens ici pour identifier la meilleure configuration.",
    ],
    objective: "Choisir objectivement le meilleur modèle et mesurer le chemin restant jusqu'aux objectifs cliniques.",
  },
  train: {
    title: "🎛️ Entraînement — lancer / arrêter une phase",
    role: "Pilote l'entraînement directement depuis le navigateur, avec console de logs et statut en direct.",
    entries: [
      "<b>Phase</b> — 1 (pré-entraînement MGM), 2 (toxicité), 3 (multi-propriétés + anti-VIH).",
      "<b>Epochs</b> — nombre de passes sur les données.",
      "<b>Max molécules</b> — limite la taille du jeu (utile pour tester vite).",
      "<b>Démarrer / Arrêter</b> — contrôle le process ; le suivi temps réel bascule automatiquement dessus.",
      "<b>Console</b> — logs en direct du process d'entraînement.",
    ],
    howto: [
      "Choisis la phase et les paramètres.",
      "Clique « Démarrer » — va dans Évolution pour voir les courbes.",
      "« Arrêter » stoppe proprement à tout moment.",
    ],
    objective: "Entraîner un modèle de bout en bout sans ligne de commande, et tout suivre depuis l'interface.",
  },
  research: {
    title: "🧪 Recherche — analyser de vraies molécules",
    role: "Analyse une ou plusieurs <b>molécules réelles (SMILES)</b> : structure 2D, descripteurs RDKit, puis toxicité / efficacité / ADME / risque selon le modèle disponible.",
    entries: [
      "<b>Champ SMILES</b> — colle une ou plusieurs molécules (ex. <code>CC(=O)Nc1ccc(O)cc1</code>).",
      "<b>Structure 2D</b> — dessin de la molécule, généré automatiquement.",
      "<b>Descripteurs</b> — MW, LogP, TPSA, HBD/HBA, QED, Lipinski (toujours disponibles, sans modèle).",
      "<b>Risque &amp; propriétés</b> — toxicité/efficacité si un modèle Phase 2/3 est entraîné.",
      "<b>Combinaison</b> — ≥2 molécules → synergie, doses, score (MolecularReasoner, Phase 3).",
      "<b>Export JSON</b> — sauvegarde l'analyse.",
    ],
    howto: [
      "Colle un SMILES et clique « Analyser ».",
      "Mode adaptatif : Phase 3 (complet) → Phase 2 (toxicité) → descripteurs seuls.",
      "Pour une combinaison, entre plusieurs SMILES séparés par une virgule.",
    ],
    objective: "Évaluer in-silico un candidat médicament en quelques secondes, comme une fiche d'identité chimique + risque.",
  },
  screen: {
    title: "🧬 Criblage VIH — virtual screening (HTS in-silico)",
    role: "Classe toute une <b>bibliothèque de molécules</b> par objectif : efficacité anti-VIH, sécurité, ou drug-likeness.",
    entries: [
      "<b>Bibliothèque</b> — intégrée (antirétroviraux de référence, médicaments courants), collée ou importée.",
      "<b>Objectif</b> — Efficacité anti-VIH (Phase 3), Sécurité (Phase 2-3) ou Drug-likeness/QED (sans modèle).",
      "<b>Classement</b> — molécules triées par score, avec QED.",
      "<b>Export CSV</b> — récupère le palmarès.",
    ],
    howto: [
      "Choisis une bibliothèque (ou colle tes SMILES).",
      "Sélectionne l'objectif et clique « Cribler ».",
      "Exporte le top candidats en CSV.",
    ],
    objective: "Reproduire un criblage à haut débit (HTS) sans paillasse : trier des centaines de molécules pour ne garder que les meilleures.",
  },
  chat: {
    title: "💬 Assistant — copilote synchronisé au modèle",
    role: "Un chatbot qui <b>orchestre les outils du modèle GNN</b> (toxicité, efficacité VIH, descripteurs, criblage, synergie). Avec une clé Claude il raisonne et analyse les images ; sans clé, un assistant local appelle quand même les outils.",
    entries: [
      "<b>Clé API Anthropic</b> — colle <code>sk-ant-…</code> puis « Activer Claude ». « Déconnecter » revient au mode local.",
      "<b>Nouvelle conversation / liste</b> — historique multi-chats à gauche (créer, ouvrir, supprimer).",
      "<b>Recherche</b> — filtre les conversations (Échap pour vider).",
      "<b>🖼️ Image</b> — joins une image pour analyse vision (nécessite Claude).",
      "<b>Structures 2D</b> — générées automatiquement à partir des SMILES détectés.",
      "<b>Exporter</b> — télécharge la conversation courante en JSON.",
    ],
    howto: [
      "(Optionnel) Active Claude avec ta clé pour des analyses avancées.",
      "Pose une question avec un SMILES, ex. « toxicité et risque de CC(=O)Nc1ccc(O)cc1 ».",
      "Réponses en streaming ; tout est sauvegardé automatiquement.",
    ],
    objective: "Parler en langage naturel à ton modèle et obtenir des analyses complètes, même hors-ligne.",
  },
  info: {
    title: "ℹ️ Guide — métriques, risque et flux de travail",
    role: "Documentation de référence : signification de chaque métrique, lecture du risque, équivalence laboratoire et flux de travail recommandé.",
    entries: [
      "<b>Tout ce que vous pouvez faire</b> — catalogue des capacités.",
      "<b>Équivalence laboratoire</b> — ce que chaque analyse représente « à la paillasse ».",
      "<b>Pourquoi ces métriques</b> — ROC-AUC, sensibilité, FNR, etc.",
      "<b>Lecture du risque</b> — OK / WARN / DANGER.",
    ],
    howto: [
      "Consulte cette page en cas de doute sur un terme ou une métrique.",
      "Suis le flux de travail recommandé en bas de page.",
    ],
    objective: "Comprendre ce que les chiffres veulent dire pour prendre de bonnes décisions.",
  },
};

function helpHtml(h) {
  const li = (arr) => "<ul>" + arr.map(x => "<li>" + x + "</li>").join("") + "</ul>";
  return (
    `<h4>À quoi sert cette page</h4><p>${h.role}</p>` +
    `<h4>Ce que fait chaque élément</h4>${li(h.entries)}` +
    `<h4>Comment l'utiliser</h4>${li(h.howto)}` +
    `<div class="obj">🎯 <b>Objectif</b> — ${h.objective}</div>`
  );
}

function openHelp(tab) {
  const h = PAGE_HELP[tab];
  if (!h) return;
  document.getElementById("helpTitle").innerHTML = h.title;
  document.getElementById("helpBody").innerHTML = helpHtml(h);
  document.getElementById("helpModal").style.display = "flex";
}

function closeHelp() { document.getElementById("helpModal").style.display = "none"; }

function setupHelp() {
  // Injecte un bouton « En savoir plus » en tête de chaque page
  Object.keys(PAGE_HELP).forEach(tab => {
    const view = document.getElementById("view-" + tab);
    if (!view) return;
    const btn = document.createElement("button");
    btn.className = "page-help";
    btn.type = "button";
    btn.innerHTML = "ℹ️ En savoir plus";
    btn.addEventListener("click", () => openHelp(tab));
    view.insertBefore(btn, view.firstChild);
  });
  document.getElementById("helpClose").addEventListener("click", closeHelp);
  document.getElementById("helpModal").addEventListener("click", (e) => {
    if (e.target.id === "helpModal") closeHelp();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && document.getElementById("helpModal").style.display === "flex") closeHelp();
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
   Sélecteurs de fichiers + import
   ==================================================================== */
async function loadFiles() {
  try {
    state.files = await fetchJSON("/api/files");
  } catch (e) { return; }
  const cks = state.files.checkpoints || [];
  const csvs = state.files.csvs || [];
  fillPicker("evalCkpt", cks, "rel");
  fillPicker("evalCsv", csvs, "rel");
  // Recherche / criblage : privilégier les checkpoints Phase 3 puis Phase 2
  const phase3 = cks.filter(c => /phase3/i.test(c.rel));
  const preferred = phase3.length ? phase3 : cks;
  fillPicker("rsCkpt", preferred, "rel");
  fillPicker("scCkpt", preferred, "rel");
}

function fillPicker(id, items, key) {
  const sel = document.getElementById(id);
  if (!sel) return;
  const prev = sel.value;
  if (!items.length) { sel.innerHTML = `<option value="">(aucun fichier détecté)</option>`; return; }
  sel.innerHTML = items.map(it =>
    `<option value="${esc(it.path)}">${esc(it[key])} · ${it.size_mb} Mo</option>`).join("");
  if (prev && items.some(it => it.path === prev)) sel.value = prev;
}

async function uploadFile(file) {
  const url = `/api/upload?name=${encodeURIComponent(file.name)}`;
  const r = await fetch(url, { method: "POST", body: file });
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).error || m; } catch (e) {} throw new Error(m); }
  return r.json();
}

function setupFiles() {
  document.getElementById("refreshFiles")?.addEventListener("click", () => {
    loadFiles(); flashToast("Listes de fichiers rafraîchies.");
  });
  const wire = (inputId, msg) => {
    const inp = document.getElementById(inputId);
    inp?.addEventListener("change", async () => {
      if (!inp.files || !inp.files[0]) return;
      const f = inp.files[0];
      const el = document.getElementById("uploadMsg");
      el.textContent = `Import de ${f.name}…`;
      try {
        const res = await uploadFile(f);
        el.textContent = `✓ ${res.name} importé (${res.size_mb} Mo)`;
        await loadFiles();
        flashToast(`${msg} importé : ${res.name}`);
      } catch (e) { el.textContent = "✗ " + e.message; flashToast("Import échoué : " + e.message, true); }
      inp.value = "";
    });
  };
  wire("upCkpt", "Checkpoint");
  wire("upCsv", "CSV");
}

/* ====================================================================
   Contrôle de l'entraînement
   ==================================================================== */
function setupTraining() {
  const phaseSel = document.getElementById("trPhase");
  const cvWrap = document.getElementById("trCvWrap");
  const syncCv = () => { cvWrap.style.display = phaseSel.value === "2" ? "" : "none"; };
  phaseSel.addEventListener("change", syncCv); syncCv();

  document.getElementById("trStart").addEventListener("click", async () => {
    const body = {
      phase: parseInt(phaseSel.value, 10),
      epochs: parseInt(document.getElementById("trEpochs").value, 10) || 20,
      max_molecules: parseInt(document.getElementById("trMaxMol").value, 10) || 0,
      cv_folds: parseInt(document.getElementById("trCv").value, 10) || 0,
      download: document.getElementById("trDownload").checked,
    };
    try {
      const res = await fetchJSON("/api/train/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      flashToast(`Entraînement lancé : ${res.label} (pid ${res.pid})`);
      pollTrain(true);
      // bascule le suivi live sur le run de cette phase
      if (res.run_id) { setTimeout(() => { loadRuns(); state.runId = res.run_id; }, 1500); }
    } catch (e) { flashToast("Échec du lancement : " + e.message, true); }
  });

  document.getElementById("trStop").addEventListener("click", async () => {
    try { await fetchJSON("/api/train/stop", { method: "POST" }); flashToast("Entraînement arrêté."); }
    catch (e) { flashToast("Arrêt : " + e.message, true); }
    pollTrain(true);
  });
}

async function pollTrain(once = false) {
  try {
    const s = await fetchJSON("/api/train/status");
    const badge = document.getElementById("trStateBadge");
    const map = { running: "WARN", finished: "OK", failed: "DANGER", stopped: "NA", idle: "NA" };
    badge.className = "badge " + (map[s.state] || "NA");
    badge.textContent = s.state || "idle";
    const remote = (state.runs || []).filter(r => r.source === "remote" && r.status === "running");
    const meta = s.pid
      ? `${s.label || ""} · pid ${s.pid} · ${s.cmd || ""}`
      : (remote.length
          ? `🛰️ Aucun process local — un entraînement Kaggle (${remote[0].id}) pousse ses métriques en temps réel.`
          : "Aucun entraînement lancé depuis l'interface.");
    document.getElementById("trMeta").textContent = meta;
    document.getElementById("trLog").textContent = (s.log_tail && s.log_tail.length)
      ? s.log_tail.join("\n") : "—";
  } catch (e) { /* ignore */ }
}

/* ====================================================================
   Recherche — analyse de molécules réelles
   ==================================================================== */
function setupResearch() {
  document.querySelectorAll("#view-research .lnk").forEach(btn => {
    btn.addEventListener("click", () => {
      const ta = document.getElementById("rsSmiles");
      ta.value = (ta.value.trim() ? ta.value.trim() + "\n" : "") + btn.dataset.smiles;
    });
  });

  document.getElementById("rsPredict").addEventListener("click", async () => {
    const smiles = document.getElementById("rsSmiles").value.trim();
    if (!smiles) { flashToast("Saisis au moins un SMILES.", true); return; }
    const checkpoint = document.getElementById("rsCkpt").value || null;
    const box = document.getElementById("rsResults");
    box.innerHTML = `<div class="card"><span class="spinner"></span> Analyse en cours…</div>`;
    document.getElementById("rsComboResult").innerHTML = "";
    try {
      const res = await fetchJSON("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles, checkpoint }),
      });
      renderPredictions(res);
    } catch (e) { box.innerHTML = errCard(e.message); }
  });

  document.getElementById("rsExport").addEventListener("click", () => {
    if (!state.lastPredict) { flashToast("Lance d'abord une analyse.", true); return; }
    downloadFile("analyse_molecules.json", JSON.stringify(state.lastPredict, null, 2), "application/json");
  });

  document.getElementById("rsCombo").addEventListener("click", async () => {
    const smiles = document.getElementById("rsSmiles").value.trim();
    const checkpoint = document.getElementById("rsCkpt").value || null;
    const box = document.getElementById("rsComboResult");
    box.innerHTML = `<div class="card"><span class="spinner"></span> Analyse de la combinaison…</div>`;
    try {
      const res = await fetchJSON("/api/combo", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles, checkpoint }),
      });
      renderCombo(res);
    } catch (e) { box.innerHTML = errCard(e.message); }
  });
}

function errCard(msg) {
  return `<div class="card"><div class="alert DANGER"><div class="a-ico">⚠️</div>
    <div><div class="a-title">Analyse impossible</div><div class="a-msg">${esc(msg)}</div></div></div></div>`;
}

function renderPredictions(res) {
  state.lastPredict = res;
  document.getElementById("rsMode").textContent = res.note ? `Mode : ${res.note}` : "";
  const box = document.getElementById("rsResults");
  const cards = (res.results || []).map(r => moleculeCard(r)).join("");
  const inv = (res.invalid && res.invalid.length)
    ? `<div class="card"><div class="a-msg">SMILES invalides ignorés : ${res.invalid.map(esc).join(", ")}</div></div>` : "";
  box.innerHTML = (cards || `<div class="empty">Aucune molécule valide.</div>`) + inv;
}

function moleculeCard(r) {
  const risk = r.risk || { level: "OK", observations: [] };
  const tox = r.toxicity || {};
  const d = r.descriptors || {};
  const hasModel = r.safety_score != null || Object.keys(tox).length > 0;
  const toxic = Object.entries(tox).filter(([, x]) => x.toxique).map(([k]) => k);

  const modelRows = [
    ["Sécurité", r.safety_score != null ? r.safety_score + "%" : "—"],
    ["Efficacité (anti-VIH)", r.efficacy?.probabilite_activite != null ? r.efficacy.probabilite_activite + "%" : "—"],
    ["Solubilité (LogS)", r.solubility ? `${r.solubility.log_s} · ${r.solubility.interpretation}` : "—"],
    ["Lipophilie (LogP préd.)", r.lipophilicity ? `${r.lipophilicity.log_p} · ${r.lipophilicity.interpretation}` : "—"],
    ["Biodisponibilité", r.bioavailability?.probabilite != null ? r.bioavailability.probabilite + "%" : "—"],
    ["Stabilité métab.", r.metabolic_stability?.probabilite != null ? r.metabolic_stability.probabilite + "%" : "—"],
    ["Drug-likeness (modèle)", r.drug_likeness?.score_global ?? "—"],
  ];
  const descRows = [
    ["Formule", d.formula ?? "—"],
    ["Masse molaire", d.mw != null ? d.mw + " g/mol" : "—"],
    ["LogP (calc.)", d.logp ?? "—"],
    ["TPSA", d.tpsa != null ? d.tpsa + " Å²" : "—"],
    ["Donneurs/Accepteurs H", `${d.hbd ?? "—"} / ${d.hba ?? "—"}`],
    ["Liaisons rotatives", d.rotatable_bonds ?? "—"],
    ["Cycles (arom.)", `${d.rings ?? "—"} (${d.aromatic_rings ?? "—"})`],
    ["QED", d.qed ?? "—"],
    ["Lipinski", d.lipinski_pass != null ? (d.lipinski_pass ? `✅ OK (${d.lipinski_violations} viol.)` : `❌ ${d.lipinski_violations} violations`) : "—"],
  ];
  const obs = (risk.observations || []).map(o =>
    `<div class="obs ${o.level}"><span class="o-ico">${o.level === "DANGER" ? "🔴" : o.level === "WARN" ? "🟠" : "🟢"}</span><span class="o-text">${esc(o.text)}</span></div>`).join("");

  const modelBlock = hasModel ? `
      <div class="o-head">Prédictions du modèle</div>
      <table class="mol-tbl">${modelRows.map(([k, v]) => `<tr><td>${k}</td><td>${esc(String(v))}</td></tr>`).join("")}</table>
      <div class="o-head" style="margin-top:10px">Toxicité Tox21</div>
      ${toxic.length ? `<div class="tox-alert">⚠ ${toxic.map(esc).join(", ")}</div>` : `<div class="tox-ok">✅ aucun endpoint toxique</div>`}` : "";

  return `<div class="card mol">
    <h3><span class="badge ${risk.level}">${risk.level}</span>
      <span class="smi">${esc(d.canonical || r.smiles)}</span></h3>
    <div class="mol-grid3">
      <div class="mol-struct"><img loading="lazy" alt="structure" src="/api/depict?smiles=${encodeURIComponent(r.smiles)}" /></div>
      <div>
        <div class="o-head">Descripteurs (RDKit)</div>
        <table class="mol-tbl">${descRows.map(([k, v]) => `<tr><td>${k}</td><td>${esc(String(v))}</td></tr>`).join("")}</table>
      </div>
      <div class="mol-side">
        ${modelBlock}
        <div class="o-head" style="margin-top:10px">Observation &amp; risque</div>
        ${obs}
      </div>
    </div></div>`;
}

function renderCombo(res) {
  const box = document.getElementById("rsComboResult");
  if (res.error) { box.innerHTML = errCard(res.error); return; }
  const pairs = (res.synergy_pairs || []).map(p =>
    `<tr><td>${esc(p.molecule_1.slice(0, 28))}</td><td>${esc(p.molecule_2.slice(0, 28))}</td>
      <td>${p.synergie}%</td><td><span class="badge ${p.type === "Synergique" ? "OK" : p.type === "Antagoniste" ? "DANGER" : "WARN"}">${p.type}</span></td></tr>`).join("");
  const doses = (res.dose_recommendations || []).map(d =>
    `<tr><td>${esc(d.smiles.slice(0, 36))}</td><td>${d.dose_optimale_mg_kg} mg/kg</td></tr>`).join("");
  box.innerHTML = `<div class="card">
    <h3>🔗 Combinaison — score de réussite ${res.success_score}% · confiance ${res.confidence}% · sécurité ${res.combined_safety}%</h3>
    <div class="grid-2">
      <div>
        <div class="o-head">Synergie par paire</div>
        <table><thead><tr><th>Molécule 1</th><th>Molécule 2</th><th>Synergie</th><th>Type</th></tr></thead><tbody>${pairs}</tbody></table>
      </div>
      <div>
        <div class="o-head">Doses optimales</div>
        <table><thead><tr><th>Molécule</th><th>Dose</th></tr></thead><tbody>${doses}</tbody></table>
      </div>
    </div>
    <div class="o-head" style="margin-top:12px">Interprétation IA</div>
    <pre class="console">${esc(res.interpretation || "")}</pre>
  </div>`;
}

/* ---------- export ---------- */
function downloadFile(name, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function toCSV(rows, cols) {
  const head = cols.join(",");
  const body = rows.map(r => cols.map(c => {
    const v = r[c] == null ? "" : String(r[c]).replace(/"/g, '""');
    return /[",\n]/.test(v) ? `"${v}"` : v;
  }).join(",")).join("\n");
  return head + "\n" + body;
}

/* ====================================================================
   Criblage virtuel (VIH)
   ==================================================================== */
async function loadLibraries() {
  let libs;
  try { libs = await fetchJSON("/api/libraries"); } catch (e) { return; }
  state.libraries = libs;
  const sel = document.getElementById("scLib");
  sel.innerHTML = Object.entries(libs).map(([k, v]) =>
    `<option value="${esc(k)}">${esc(v.label)} (${v.count})</option>`).join("");
}

function setupScreening() {
  document.getElementById("scRun").addEventListener("click", async () => {
    const objective = document.getElementById("scObjective").value;
    const checkpoint = document.getElementById("scCkpt").value || null;
    const smiles = document.getElementById("scSmiles").value.trim();
    const library = document.getElementById("scLib").value;
    const body = smiles ? { objective, checkpoint, smiles } : { objective, checkpoint, library };
    const meta = document.getElementById("scMeta");
    meta.innerHTML = '<span class="spinner"></span> Criblage…';
    try {
      const res = await fetchJSON("/api/screen", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      renderScreen(res); meta.textContent = "";
    } catch (e) { meta.textContent = ""; document.getElementById("scResultCard").style.display = "block";
      document.getElementById("scHint").innerHTML = `<span style="color:var(--danger)">${esc(e.message)}</span>`;
      document.querySelector("#scTable tbody").innerHTML = ""; document.querySelector("#scTable thead").innerHTML = ""; }
  });
  document.getElementById("scExport").addEventListener("click", () => {
    const r = state.lastScreen;
    if (!r || !r.ranked || !r.ranked.length) { flashToast("Lance d'abord un criblage.", true); return; }
    const cols = ["rank", "name", "smiles", "score", "mw", "logp", "qed", "lipinski_pass", "safety_score", "risk"];
    downloadFile(`criblage_${r.objective}.csv`, toCSV(r.ranked, cols), "text/csv");
  });
}

function renderScreen(res) {
  state.lastScreen = res;
  document.getElementById("scResultCard").style.display = "block";
  document.getElementById("scHint").textContent =
    `${res.objective_label} · mode ${res.mode} · ${res.n_valid}/${res.n_input} molécules valides`;
  const thead = document.querySelector("#scTable thead");
  const tbody = document.querySelector("#scTable tbody");
  thead.innerHTML = "<tr><th>#</th><th>Nom</th><th>SMILES</th><th>Score</th><th>MW</th><th>LogP</th><th>QED</th><th>Lipinski</th><th>Risque</th></tr>";
  tbody.innerHTML = (res.ranked || []).map(r => `<tr data-level="${r.risk || ""}">
    <td>${r.rank}</td><td>${esc(r.name || "—")}</td>
    <td class="smi-cell">${esc(r.smiles)}</td>
    <td><b>${r.score}</b></td><td>${r.mw}</td><td>${r.logp}</td><td>${r.qed ?? "—"}</td>
    <td>${r.lipinski_pass ? "✅" : "❌"}</td>
    <td>${r.risk ? `<span class="badge ${r.risk}">${r.risk}</span>` : "—"}</td></tr>`).join("")
    || `<tr><td colspan="9" style="color:var(--faint)">Aucun résultat.</td></tr>`;
}

/* ====================================================================
   Guide : capacités + équivalence laboratoire
   ==================================================================== */
async function loadCapabilities() {
  let data;
  try { data = await fetchJSON("/api/capabilities"); } catch (e) { return; }
  document.getElementById("capabilities").innerHTML = (data.capabilities || []).map(g =>
    `<div class="cap-group"><div class="cap-title">${esc(g.group)}</div>
      <ul>${g.items.map(i => `<li>${esc(i)}</li>`).join("")}</ul></div>`).join("");
  const thead = document.querySelector("#labTable thead");
  const tbody = document.querySelector("#labTable tbody");
  thead.innerHTML = "<tr><th>Analyse</th><th>In silico (cette app)</th><th>Équivalent laboratoire</th></tr>";
  tbody.innerHTML = (data.lab_equivalence || []).map(r =>
    `<tr><td>${esc(r.analyse)}</td><td>${esc(r.in_silico)}</td><td>${esc(r.labo)}</td></tr>`).join("");
}

/* ====================================================================
   Thème clair / sombre
   ==================================================================== */
function applyTheme(theme) {
  if (theme === "light") document.documentElement.setAttribute("data-theme", "light");
  else document.documentElement.removeAttribute("data-theme");
  const btn = document.getElementById("themeToggle");
  if (btn) btn.textContent = theme === "light" ? "☀️" : "🌙";
  readColors();
  requestAnimationFrame(() => { renderEvolution(); renderCompare(); renderKPIs(); });
}
function setupTheme() {
  let theme = "dark";
  try { theme = localStorage.getItem("panacee-theme") || "dark"; } catch (e) {}
  applyTheme(theme);
  document.getElementById("themeToggle")?.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    const next = cur === "light" ? "dark" : "light";
    try { localStorage.setItem("panacee-theme", next); } catch (e) {}
    applyTheme(next);
  });
}

/* ====================================================================
   Assistant (chatbot)
   ==================================================================== */
function setupChat() {
  const log = document.getElementById("chatLog");
  const input = document.getElementById("chatInput");
  let attachedImage = null; // {media_type, data}

  const addBubble = (cls, text) => {
    const div = document.createElement("div");
    div.className = "bubble " + cls;
    const txt = document.createElement("span");
    txt.className = "txt"; txt.textContent = text || "";
    div.appendChild(txt);
    log.appendChild(div); log.scrollTop = log.scrollHeight;
    return div;
  };
  const setToolNote = (bubble, tools) => {
    if (!tools || !tools.length) return;
    let n = bubble.querySelector(".toolnote");
    if (!n) { n = document.createElement("span"); n.className = "toolnote"; bubble.appendChild(n); }
    n.textContent = "🔧 " + tools.join(", ");
  };
  const addImage = (bubble, src, cls) => {
    const img = document.createElement("img");
    img.className = "chat-img " + (cls || ""); img.src = src; img.loading = "lazy";
    const note = bubble.querySelector(".toolnote");
    bubble.insertBefore(img, note || null);
    log.scrollTop = log.scrollHeight;
  };

  // ---- conversations ----
  // La barre de recherche est la source unique de vérité : la liste reflète
  // toujours son contenu (évite la désynchro liste ↔ champ de recherche).
  async function loadConversations() {
    const sb = document.getElementById("chatSearch");
    const q = sb ? sb.value.trim() : "";
    try {
      const data = q ? await fetchJSON("/api/conversations/search?q=" + encodeURIComponent(q))
                     : await fetchJSON("/api/conversations");
      renderConvList(q ? data.results : data.conversations);
    } catch (e) { /* ignore */ }
  }
  function renderConvList(items) {
    const box = document.getElementById("convList");
    box.innerHTML = (items || []).map(c => `
      <div class="conv ${c.id === state.currentConv ? "active" : ""}" data-id="${c.id}">
        <div class="conv-title">${esc(c.title || "Conversation")}</div>
        <div class="conv-snip">${esc(c.last_snippet || c.snippet || "")}</div>
        <button class="conv-del" data-id="${c.id}" title="Supprimer">✕</button>
      </div>`).join("") || `<div class="empty" style="padding:14px">Aucune conversation.</div>`;
    box.querySelectorAll(".conv").forEach(el => el.addEventListener("click", (e) => {
      if (e.target.classList.contains("conv-del")) return;
      openConversation(el.dataset.id);
    }));
    box.querySelectorAll(".conv-del").forEach(b => b.addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch("/api/conversations/" + b.dataset.id, { method: "DELETE" });
      if (state.currentConv === b.dataset.id) { state.currentConv = null; log.innerHTML = ""; }
      loadConversations();
    }));
  }
  function resetSearch() {
    const sb = document.getElementById("chatSearch");
    if (sb) sb.value = "";
  }
  async function openConversation(cid) {
    state.currentConv = cid;
    log.innerHTML = "";
    resetSearch();
    try {
      const data = await fetchJSON("/api/conversations/" + cid);
      for (const m of data.messages) {
        const b = addBubble(m.role === "user" ? "user" : "bot", m.content);
        if (m.image) addImage(b, "/api/chat/image?name=" + encodeURIComponent(m.image), "user-img");
        if (m.tools && m.tools.length) setToolNote(b, m.tools.map(t => t.tool || t));
      }
    } catch (e) {}
    log.scrollTop = log.scrollHeight;
    loadConversations();
  }
  async function newConversation() {
    resetSearch();
    try {
      const c = await fetchJSON("/api/conversations", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: "{}" });
      state.currentConv = c.id; log.innerHTML = ""; await loadConversations();
    } catch (e) {}
  }

  // ---- image attach ----
  function clearImage() {
    attachedImage = null;
    const p = document.getElementById("imgPreview");
    p.style.display = "none"; p.innerHTML = "";
    document.getElementById("chatImage").value = "";
  }
  document.getElementById("chatImage").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = () => {
      attachedImage = { media_type: f.type, data: r.result.split(",")[1] };
      const p = document.getElementById("imgPreview");
      p.style.display = "flex";
      p.innerHTML = `<img src="${r.result}" alt="aperçu"/><button class="btn ghost" id="imgClear">✕ retirer</button>`;
      document.getElementById("imgClear").addEventListener("click", clearImage);
    };
    r.readAsDataURL(f);
  });

  // ---- send (streaming + persistance) ----
  const send = async () => {
    const text = input.value.trim();
    if (!text && !attachedImage) return;
    input.value = "";
    const ub = addBubble("user", text);
    const payloadImage = attachedImage;
    if (payloadImage) addImage(ub, "data:" + payloadImage.media_type + ";base64," + payloadImage.data, "user-img");
    clearImage();
    const bubble = addBubble("bot", "");
    const txt = bubble.querySelector(".txt");
    txt.innerHTML = '<span class="typing">…</span>';
    let full = "", tools = [], started = false;

    const onEvent = (ev, data) => {
      if (ev === "meta") { if (data.conversation_id) state.currentConv = data.conversation_id; }
      else if (ev === "image") { addImage(bubble, data.url, "mol-img"); }
      else if (ev === "tool") { tools.push(data.tool); setToolNote(bubble, tools); }
      else if (ev === "delta") {
        if (!started) { txt.textContent = ""; started = true; }
        full += data.text || ""; txt.textContent = full; log.scrollTop = log.scrollHeight;
      } else if (ev === "done") { setToolNote(bubble, (data.tools || []).map(t => t.tool || t)); }
      else if (ev === "error") { txt.textContent = "Erreur : " + (data.error || "?"); }
    };

    try {
      const resp = await fetch("/api/chat/stream", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: state.currentConv, content: text, image: payloadImage }),
      });
      if (!resp.ok || !resp.body) throw new Error("HTTP " + resp.status);
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, i); buf = buf.slice(i + 2);
          let ev = "msg", dat = {};
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) ev = line.slice(6).trim();
            else if (line.startsWith("data:")) { try { dat = JSON.parse(line.slice(5).trim()); } catch (e) {} }
          }
          onEvent(ev, dat);
        }
      }
      if (!started && !full) txt.textContent = "(pas de réponse)";
      loadConversations();
    } catch (e) { txt.textContent = "Erreur : " + e.message; }
  };

  // ---- wiring ----
  document.getElementById("chatSend").addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  document.querySelectorAll("#view-chat .chip").forEach(c =>
    c.addEventListener("click", () => { input.value = c.dataset.msg; send(); }));
  document.getElementById("chatNew").addEventListener("click", newConversation);
  let stq = null;
  const searchBox = document.getElementById("chatSearch");
  searchBox.addEventListener("input", () => {
    clearTimeout(stq);
    stq = setTimeout(() => loadConversations(), 250);
  });
  // Échap vide la recherche et restaure la liste complète
  searchBox.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { searchBox.value = ""; loadConversations(); }
  });
  document.getElementById("chatExport").addEventListener("click", () => {
    if (!state.currentConv) { flashToast("Ouvre une conversation d'abord.", true); return; }
    window.open("/api/conversations/" + state.currentConv + "/export", "_blank");
  });
  document.getElementById("apiKeySave").addEventListener("click", async () => {
    const k = document.getElementById("apiKeyInput").value.trim();
    if (!k) { flashToast("Colle ta clé API.", true); return; }
    try {
      const result = await fetchJSON("/api/settings/apikey", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: k }) });
      document.getElementById("apiKeyInput").value = "";
      flashToast(result.claude ? "Clé enregistrée — Claude activé." : "Clé enregistrée.");
      refreshChatMode();
    } catch (e) {
      flashToast("Échec : " + e.message, true);
    }
  });
  const discoBtn = document.getElementById("apiKeyDisconnect");
  if (discoBtn) discoBtn.addEventListener("click", async () => {
    if (!confirm("Déconnecter la clé Claude ? L'assistant repassera en mode local.")) return;
    try {
      const r = await fetchJSON("/api/settings/apikey", { method: "DELETE" });
      flashToast(r.env_locked
        ? "Clé locale effacée (une clé ANTHROPIC_API_KEY reste active côté serveur)."
        : "Claude déconnecté — mode local.");
      refreshChatMode();
    } catch (e) { flashToast("Échec : " + e.message, true); }
  });

  state._loadConversations = loadConversations;
}

async function refreshChatMode() {
  const badge = document.getElementById("chatMode");
  if (!badge) return;
  try {
    const s = await fetchJSON("/api/chat/status");
    badge.textContent = s.claude ? "Claude " + s.model : "local";
    badge.className = "badge " + (s.claude ? "OK" : "NA");
    const row = document.getElementById("apiKeyRow");
    if (row) row.style.display = s.claude ? "none" : "flex";
    // Bouton déconnexion : visible si Claude actif ET clé stockée localement (pas env)
    const disco = document.getElementById("apiKeyDisconnect");
    if (disco) disco.style.display = (s.claude && s.key_source === "store") ? "inline-flex" : "none";
  } catch (e) { badge.textContent = "?"; }
  if (state._loadConversations) state._loadConversations();
}

/* ====================================================================
   Démarrage
   ==================================================================== */
async function main() {
  setupTheme();
  ecgInit(); setupTabs(); setupHelp(); setupRunSelect(); setupEval();
  setupFiles(); setupTraining(); setupResearch(); setupScreening(); setupChat();
  await loadConfig();
  renderBarème();
  await loadFiles();
  await loadRuns();
  await renderRunsTable();
  // rafraîchit la liste des runs périodiquement (nouveaux runs)
  setInterval(() => loadRuns().catch(() => {}), 15000);
  setInterval(() => renderRunsTable().catch(() => {}), 15000);
  // statut d'entraînement (poll régulier tant que l'app est ouverte)
  setInterval(() => { pollTrain(); }, 4000);
}
document.addEventListener("DOMContentLoaded", main);
