/* Flowcast dashboard. Plain ES modules, no build step; data comes from dashboard/server.py.
   Layout and interaction patterns follow Torque Stripe (MIT, (c) 2026 Aljabri Alam, github.com/aljabrialam/torque-stripe). */
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = { get: k => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };
async function api(path, q = {}) {
  const r = await fetch(`/api/${path}?${new URLSearchParams(q)}`), j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText || `HTTP ${r.status}`);
  return j;
}

// ---------- icons (paths from Torque Stripe's Icons.tsx, plus outlook and wind) ----------
const svg = d => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
const I = {
  outlook: svg('<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.2"/><path d="M12 3.5V12l5.5 3.2"/>'),
  flights: svg('<path d="M2.5 13.5 21 8l-1.2 3.2-6.3 2.1-3 6.2H8.6l1.3-5.2-4.6 1.4-1.6 2H2.2l.9-3.3Z"/>'),
  history: svg('<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'),
  eval: svg('<path d="M12 3 4.5 6v5.5c0 4.6 3.1 8.2 7.5 9.5 4.4-1.3 7.5-4.9 7.5-9.5V6Z"/><path d="m8.8 12 2.2 2.2 4.4-4.6"/>'),
  help: svg('<circle cx="12" cy="12" r="9"/><path d="M9.6 9.3a2.5 2.5 0 0 1 4.8.9c0 1.7-2.4 2.1-2.4 3.6M12 17.2v.1"/>'),
  checks: svg('<rect x="3.5" y="4" width="17" height="16" rx="3"/><path d="M7.5 9h4M7.5 13h6M7.5 17h3M16 9l1 1 2-2"/>'),
  doc: svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>'),
  cube: svg('<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9Z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9"/>'),
  wind: svg('<path d="M3 8h10.5a2.5 2.5 0 1 0-2.5-2.5M3 12h15.5a2.5 2.5 0 1 1-2.5 2.5M3 16h7"/>'),
  storm: svg('<path d="M7 16.5a4.5 4.5 0 1 1 1.2-8.8A5.5 5.5 0 0 1 18.6 9 3.8 3.8 0 0 1 18 16.5"/><path d="m12.5 12-2.5 4.5h3.5L11 21"/>'),
};
const LOGO = `<svg width="30" height="30" viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="26" fill="#00205B"/><rect x="28" y="12" width="8" height="40" rx="1.5" fill="#fff"/><path d="M32 16v5M32 27v5M32 38v5" stroke="#00205B" stroke-width="1.8"/><path d="M32 6a26 26 0 0 1 26 26" fill="none" stroke="#0085CA" stroke-width="6" stroke-linecap="round"/></svg>`;

// ---------- state ----------
const VIEWS = [["outlook", "Outlook", I.outlook], ["flights", "Flights", I.flights], ["history", "History", I.history], ["eval", "Evaluation", I.eval],
               ["storms", "Storms", I.storm]];
const SEV = { clear: "Clear", moderate: "Reduced", severe: "Severe" };
const S = { view: "outlook", meta: null, icao: "KATL", t0: "", sens: "balanced", thr: 0.25, reveal: true, out: null, hist: null, ev: null,
            st: null, expl: null, lead: 3, busy: false, sel: null, cam: "3d", render: "solid" };
let stage = null;

const hhmm = s => s ? `${s.slice(11, 16)}Z` : "—";
const day = s => new Date(s).toLocaleDateString("en-GB", { day: "2-digit", month: "short", timeZone: "UTC" });
const pct = p => p == null ? "—" : Math.round(p * 100);
const n0 = v => v == null ? "—" : Math.round(v).toLocaleString("en-US");
const airport = () => S.meta.airports.find(a => a.icao === S.icao);
const dirWord = d => d === "arr" ? "from" : "to";
const riskTone = p => p >= S.thr ? "stop" : p >= 0.2 ? "warn" : "mute";

function readUrl() {
  const q = new URLSearchParams(location.search), h = location.hash.replace(/^#\/?/, "");
  if (VIEWS.some(v => v[0] === h)) S.view = h;
  if (q.get("airport") && S.meta.airports.some(a => a.icao === q.get("airport"))) S.icao = q.get("airport");
  S.t0 = q.get("t0") ? `${q.get("t0").slice(0, 13)}:00Z` : S.meta.default.t0;
}
function writeUrl() { history.replaceState(null, "", `?airport=${S.icao}&t0=${S.t0.slice(0, 13)}#/${S.view}`); }

// ---------- data ----------
async function forecast() {
  S.busy = true; paintAskButton();
  try {
    S.out = await api("outlook", { airport: S.icao, t0: S.t0, sens: S.sens, thr: S.thr });
    S.t0 = S.out.t0; S.hist = null; S.ev = null; S.st = null; S.expl = null; S.sel = null; writeUrl(); syncStage(true);
  } catch (e) { toast(e.message); }
  S.busy = false; render();
}
function syncStage(scan) {
  if (!stage || !S.out) return;
  const a = airport(); stage.setAirport(a.iata, a.runway_heading); stage.setWeather(S.out.now);
  stage.setRing(S.out.hours.map(h => h.sev)); if (scan) stage.scan();
}
async function ensure(view) {
  try {
    if (view === "history" && !S.hist) { S.hist = await api("history", { airport: S.icao, t0: S.t0 }); render(); }
    if (view === "eval" && (!S.ev || S.ev.lead !== S.lead || S.ev.icao !== S.icao)) { S.ev = { ...await api("eval", { airport: S.icao, lead: S.lead }), icao: S.icao }; render(); }
    if (view === "storms" && (!S.st || S.st.lead !== S.lead || S.st.icao !== S.icao)) { S.st = await api("storm", { airport: S.icao, lead: S.lead }); render(); }
  } catch (e) { toast(e.message); }
}
function toast(msg) {
  const t = document.createElement("div"); t.className = "toast glass strong"; t.textContent = msg; document.body.append(t); setTimeout(() => t.remove(), 4200);
}

// ---------- shell ----------
function go(v) { S.view = v; writeUrl(); render(); ensure(v); }
function renderShell() {
  $("#rail").innerHTML = `<a class="rail-logo" href="#/outlook" data-go="outlook" aria-label="Flowcast, outlook">${LOGO}</a>`
    + VIEWS.map(([id, l, ic]) => `<button class="rail-btn" data-go="${id}" data-tab="${id}"${S.view === id ? ' aria-current="page"' : ""}>${ic}<span>${l}</span></button>`).join("")
    + `<span class="rail-gap"></span><button class="rail-btn" data-tour-open aria-label="Show the guided tour">${I.help}<span>Tour</span></button>`;
  const m = S.meta, a = airport();
  $("#topbar").innerHTML = `<span class="tb-item brand"><b>Flowcast</b></span>
    <span class="tb-item hide-sm"><span class="dot ok"></span>Data · METAR · TAF · BTS · ${n0(m.flights)} test flights</span>
    <span class="tb-item hide-sm"><span class="dot"></span>Mode · replay of held-out 2025</span><span class="tb-sp"></span>
    <span class="tb-item hide-sm"><span class="dot"></span>Airport · <span class="mono">${a.iata}</span></span>
    <span class="tb-item hide-sm"><span class="dot"></span>Issued · <span class="mono">${S.t0.slice(0, 10)} ${hhmm(S.t0)}</span></span>`;
  $("#stage").classList.toggle("dim", S.view !== "outlook"); stage?.setActive(S.view === "outlook");
}
function render() {
  renderShell();
  const v = $("#view"), keep = v.firstElementChild?.dataset.view === S.view ? v.firstElementChild.scrollTop : 0;
  v.innerHTML = { outlook: outlookView, flights: flightsView, history: historyView, eval: evalView, storms: stormView }[S.view]();
  if (v.firstElementChild) v.firstElementChild.scrollTop = keep;
}
const loading = (view, what) => `<div class="page scroll" data-view="${view}"><p class="empty-note">Loading ${what}…</p></div>`;

// ---------- outlook: the console over the stage, then the advisory ----------
function outlookView() {
  const m = S.meta, o = S.out, a = airport();
  const date = S.t0.slice(0, 10), hour = +S.t0.slice(11, 13);
  const askbar = `<form class="askbar glass strong" data-tour="ask">
      <label class="ab-model"><span class="label">Airport</span>
        <select class="mono" name="airport" aria-label="Airport">${m.airports.map(x => `<option value="${x.icao}"${x.icao === S.icao ? " selected" : ""}>${x.iata} · ${x.icao}</option>`).join("")}</select></label>
      <div class="ab-q"><span class="label">Forecast issue time, UTC · replay of the held-out 2025 period</span>
        <span class="row"><input type="date" name="date" class="mono" aria-label="Issue date (UTC)" min="${m.start.slice(0, 10)}" max="${m.end.slice(0, 10)}" value="${date}" required />
        <select name="hour" class="mono" aria-label="Issue hour (UTC)">${Array.from({ length: 24 }, (_, h) => `<option value="${h}"${h === hour ? " selected" : ""}>${String(h).padStart(2, "0")}:00</option>`).join("")}</select>
        ${o ? `<span class="label hide-sm">local ${esc(o.local)}</span>` : ""}</span></div>
      <button class="glass-btn pri" type="submit"${S.busy ? " disabled" : ""}>${S.busy ? "Forecasting…" : "Forecast"}</button>
    </form>`;
  const chips = `<div class="chips" data-tour="chips">${m.events.map((e, i) => `<button class="chip glass" data-ev="${i}" title="Issue the forecast 6 hours before the dip"><span class="mono k">${e.iata}</span>${day(e.at)} ${hhmm(e.at)} · AAR fell to ${n0(e.aar)}/h</button>`).join("")}</div>`;
  if (!o) return `<div class="outlook" data-view="outlook"><section class="console">${askbar}${chips}</section></div>`;

  const low = o.hours.reduce((x, h) => (h.pred < x.pred ? h : x), o.hours[0]), lowPct = Math.round(100 * low.pred / o.aar_vfr);
  const capacity = `<aside class="hud left glass" aria-label="Capacity, hour by hour" data-tour="capacity">
      <div class="ph">${I.checks}Capacity · next 12 h<span class="meta"><span class="seg" role="group" aria-label="Alert sensitivity">${["balanced", "cautious"].map(s => `<button type="button" data-sens="${s}" aria-pressed="${S.sens === s}">${s === "balanced" ? "Balanced" : "Cautious"}</button>`).join("")}</span></span></div>
      <div class="readiness" title="Lowest forecast AAR as a share of the best-case rate"><div class="segs">${o.hours.map(h => `<i class="${h.sev}"></i>`).join("")}</div><span class="cond pct">${lowPct}%</span></div>
      <ul class="checks scroll">${o.hours.map((h, i) => `<li data-hour="${i}"><span class="dot ${h.sev}"></span><span class="ct"><b>${hhmm(h.hour)} · +${h.lead} h</b><span>AAR ${n0(h.pred)}/h · range ${n0(h.lo)}–${n0(h.hi)} · TAF ${n0(h.taf)}${S.reveal ? ` · actual ${n0(h.actual)}` : ""}</span></span><span class="cs t-${h.sev}">${SEV[h.sev]}</span></li>`).join("")}</ul>
    </aside>`;
  const w = o.now;
  const weather = `<aside class="hud right glass" aria-label="Weather now" data-tour="weather">
      <div class="ph">${I.wind}Weather now<span class="meta mono">${w ? `METAR ${hhmm(w.hour)}` : "no report"}</span></div>
      <div class="pad">${w ? weatherTiles(w, o) : '<p class="label">No published METAR in the hour before the issue time.</p>'}</div>
    </aside>`;
  const atRisk = o.n_flights ? Math.round(100 * o.n_high / o.n_flights) : 0;
  const tag = `<div class="hud bl" data-tour="stage">
      <div class="model-tag"><div class="readout" style="font-size:46px">${a.iata}</div><div class="label">${esc(a.name)}</div>
        <div class="model-sum">Issued ${day(o.t0)} ${hhmm(o.t0)} · local ${esc(o.local)}${w ? ` · ${flow(w, a)}` : ""}</div>
        <div class="label">Airfield © OpenStreetMap contributors</div></div>
      <div class="readouts"><div><div class="readout">${n0(low.pred)}<small>/h</small></div><div class="label">Lowest forecast AAR</div></div>
        <div><div class="readout">${atRisk}<small>%</small></div><div class="label">Flights at risk ≥ ${pct(S.thr)}%</div></div></div>
    </div>`;
  const verdict = `<div class="hud bc glass"><span class="heat ${o.verdict.kind}" aria-hidden="true"></span>
      <div class="sl-t"><b>${esc(o.verdict.title)}</b><span class="label">${w ? windText(w) : ""}${w?.wx ? ` · ${esc(w.wx)}` : ""}${w?.ceiling_ft != null && w.ceiling_ft < 12000 ? ` · ceiling ${n0(w.ceiling_ft)} ft` : ""}</span></div>
      <div class="views-col">
        <div class="views" role="group" aria-label="Camera">${I.cube}${[["3d", "3D"], ["top", "Top"], ["side", "Side"]].map(([k, l]) => `<button type="button" class="glass-btn" data-cam="${k}" aria-pressed="${S.cam === k}">${l}</button>`).join("")}</div>
        <div class="views" role="group" aria-label="Render mode">${[["solid", "Render"], ["wire", "Wire"]].map(([k, l]) => `<button type="button" class="glass-btn" data-render="${k}" aria-pressed="${S.render === k}">${l}</button>`).join("")}</div>
      </div></div>`;
  return `<div class="outlook" data-view="outlook"><section class="console">${askbar}${chips}${capacity}${weather}${tag}${verdict}${radar(o)}</section>${advisory(o, a)}</div>`;
}

function windText(w) {
  if (w.wind_kt == null) return "Wind not reported";
  if (w.wind_kt < 1) return "Calm";
  return `${w.wind_dir == null ? "VRB" : String(w.wind_dir).padStart(3, "0") + "°"} ${w.wind_kt}${w.gust_kt ? `G${w.gust_kt}` : ""} kt`;
}
function flow(w, a) {
  const land = ((w.headwind_kt ?? 1) >= 0 ? a.runway_heading : a.runway_heading + 180) % 360;
  const rwy = String(Math.round(land / 10) || 36).padStart(2, "0");
  return `landing runway ${rwy}, ${["north", "east", "south", "west"][Math.round(land / 90) % 4]} flow`;
}
function weatherTiles(w, o) {
  const tile = (l, v, tone = "") => `<div class="tile"><div class="label">${l}</div><div class="tv t-${tone}">${v}</div></div>`;
  const cat = w.category || "n/a", x = Math.abs(w.crosswind_kt ?? 0), c = w.ceiling_ft, v = w.vis_mi;
  const frac = w.aar == null ? 0 : Math.max(0, Math.min(1, w.aar / o.aar_vfr)), C = 2 * Math.PI * 34;
  const tone = frac <= 0.6 ? "var(--stop)" : frac <= 0.8 ? "var(--ring-moderate)" : "var(--sky)";
  return `<div class="tiles">
      ${tile("Flight category", cat, cat === "VFR" ? "ok" : cat === "MVFR" ? "warn" : cat === "n/a" ? "mute" : "stop")}
      ${tile("Wind", windText(w))}
      ${tile("Crosswind", w.crosswind_kt == null ? "—" : `${x}<small>kt</small>`, x >= 15 ? "stop" : x >= 10 ? "warn" : "")}
      ${tile("Ceiling", c == null || c >= 12000 ? "None" : `${n0(c)}<small>ft</small>`, c != null && c < 1000 ? "stop" : c != null && c < 3000 ? "warn" : "")}
      ${tile("Visibility", v == null ? "—" : `${v >= 10 ? "10+" : v}<small>SM</small>`, v != null && v < 1 ? "stop" : v != null && v < 3 ? "warn" : "")}
      ${tile("Weather", w.wx ? esc(w.wx) : "None", w.thunder ? "stop" : w.wx ? "warn" : "mute")}
    </div>
    <div class="dial">
      <svg viewBox="0 0 84 84" aria-hidden="true"><circle cx="42" cy="42" r="34" fill="none" stroke="rgba(0,32,91,.08)" stroke-width="6"/>
        <circle cx="42" cy="42" r="34" fill="none" stroke="${tone}" stroke-width="6" stroke-linecap="round" stroke-dasharray="${(C * frac).toFixed(1)} ${C.toFixed(1)}" transform="rotate(-90 42 42)"/></svg>
      <div><div class="label">AAR now</div><div class="readout" style="font-size:34px">${n0(w.aar)}<small>/h</small></div>
        <div class="label">of ${o.aar_vfr} best case · from this METAR</div></div>
    </div>`;
}

function radar(o) {
  const R = 62, H = 75, blips = o.flights.map(f => {
    const r = 8 + (R - 8) * Math.max(0, Math.min(1, (f.h - 2.5) / 10)), b = (f.bearing ?? (f.id * 47) % 360) * Math.PI / 180;
    const x = H + r * Math.sin(b), y = H - r * Math.cos(b), t = riskTone(f.p);
    const fillc = t === "stop" ? "var(--stop)" : t === "warn" ? "var(--ring-moderate)" : "var(--sky)";
    return `<g class="blip" data-flight="${f.id}"><title>${esc(`${f.carrier} ${f.dir === "arr" ? "arrival from" : "departure to"} ${f.other} · ${hhmm(f.t)} · risk ${pct(f.p)}%`)}</title>
      <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="8" fill="transparent"/><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(2.4 + 5 * f.p).toFixed(1)}" fill="${fillc}" opacity="${t === "mute" ? 0.55 : 1}"/></g>`;
  }).join("");
  const top = o.flights[0];
  return `<div class="hud br glass" data-tour="radar">
      <div class="ph" style="padding:12px 14px 4px">Delay radar<span class="meta"><select class="seg mono" data-thr aria-label="Risk threshold" style="height:26px;padding:0 6px">${[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5].map(t => `<option value="${t}"${t === S.thr ? " selected" : ""}>≥ ${pct(t)}%</option>`).join("")}</select></span></div>
      <div class="radar">
        <svg viewBox="0 0 150 150" role="img" aria-label="${o.n_high} of ${o.n_flights} sampled flights at or above the risk threshold">
          <defs><radialGradient id="rg"><stop offset="0" stop-color="rgba(0,133,202,.12)"/><stop offset="1" stop-color="rgba(0,133,202,0)"/></radialGradient>
            <linearGradient id="sw" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="rgba(0,133,202,0)"/><stop offset="1" stop-color="rgba(0,133,202,.35)"/></linearGradient></defs>
          <circle cx="${H}" cy="${H}" r="${R}" fill="url(#rg)"/>
          ${[1, 2, 3, 4].map(i => `<circle cx="${H}" cy="${H}" r="${R * i / 4}" fill="none" stroke="rgba(0,32,91,.14)"/>`).join("")}
          <path d="M${H} ${H - R}V${H + R}M${H - R} ${H}H${H + R}" stroke="rgba(0,32,91,.1)"/>
          <text x="${H}" y="${H - R - 4}" text-anchor="middle" font-size="8" fill="var(--ink-3)" class="mono">N</text>
          <g class="sweep"><path d="M${H} ${H}L${H + R} ${H}A${R} ${R} 0 0 0 ${(H + R * Math.cos(-0.7)).toFixed(1)} ${(H + R * Math.sin(-0.7)).toFixed(1)}Z" fill="url(#sw)"/></g>
          ${blips}
        </svg>
        <div class="radar-l">
          <div class="readout" style="font-size:40px">${o.n_high}</div>
          <div class="label">at risk ≥ ${pct(S.thr)}% of ${o.n_flights} sampled flights · centre = soonest, angle = route bearing</div>
          ${top ? `<button class="glass-btn" data-flight="${top.id}"><span class="mono">${esc(top.carrier)} ${dirWord(top.dir)} ${esc(top.other)} · ${pct(top.p)}%</span></button>` : ""}
        </div>
      </div></div>`;
}

function advisory(o, a) {
  const k = o.verdict.kind, worst = o.windows.find(w => w.severity === "severe") || o.windows[0];
  const kv = [["Lowest forecast AAR", worst ? `${worst.min_aar}/h (${worst.lo}–${worst.hi})` : `${n0(Math.min(...o.hours.map(h => h.pred)))}/h`],
              ["Window", o.windows.length ? o.windows.map(w => `${hhmm(w.start)}–${hhmm(w.end)}`).join(", ") : "None"],
              ["Confidence", worst ? worst.confidence : "—"], ["Drivers", worst?.hazards.length ? worst.hazards.join(", ") : "—"],
              ["Best-case AAR", `${o.aar_vfr}/h`], ["Flights at risk", `${o.n_high} of ${o.n_flights} sampled`]];
  const advice = o.windows.length
    ? o.windows.flatMap((w, wi) => w.advice.map(s => `<p class="sent"><span class="cite mono">W${wi + 1}</span><span>${esc(s)}</span></p>`)).join("")
    : '<p class="sent"><span class="cite mono">OK</span><span>No capacity-reduction window is forecast. Routine monitoring: re-run at the next METAR or TAF update.</span></p>';
  const shown = o.flights.slice(0, 10);
  const flights = shown.length ? shown.map(f => `<button class="src${S.sel === f.id ? " cur" : ""}" data-flight="${f.id}" id="f${f.id}">
      <div class="src-h"><b class="readout t-${riskTone(f.p)}">${pct(f.p)}<small>%</small></b><span class="mono">${esc(f.carrier)} · ${f.dir === "arr" ? "arrival from" : "departure to"} ${esc(f.other)}</span><span class="label">${hhmm(f.t)} · in ${f.h} h</span></div>
      <p>${f.drivers.length ? f.drivers.map(([n, v]) => `${esc(n)} <span class="mono">+${v.toFixed(2)}</span>`).join(" · ") : "No strong driver; close to the average flight."}</p>
      ${S.reveal ? `<div class="label">Actual delay ${n0(f.delay_min)} min${f.severe ? " · <b class=\"t-stop\">severe</b>" : ""}</div>` : ""}
    </button>`).join("") : '<p class="empty-note">No sampled flights are scheduled 3 to 12 hours after this issue time.</p>';
  return `<section class="results" id="answer"><div class="answer-grid">
    <article class="glass strong card" data-tour="advice">
      <div class="pillrow"><span class="pill ${k}">${k === "clear" ? "No reduction" : k === "moderate" ? "Reduced capacity" : "Severe reduction"}</span>
        <span class="pill">${S.sens === "balanced" ? "Balanced alerts" : "Cautious alerts"}</span><span class="pill mono">${a.iata} · ${day(o.t0)} ${hhmm(o.t0)}</span>
        <span style="flex:1"></span><button type="button" class="glass-btn" data-reveal aria-pressed="${S.reveal}">${S.reveal ? "Hide what happened" : "Reveal what happened"}</button></div>
      <h2 class="headline">${esc(o.verdict.title)}</h2><p class="lede">${esc(o.verdict.sub)}</p>
      ${aarChart(o)}
      <dl class="kv">${kv.map(([t, d]) => `<div><dt>${t}</dt><dd>${esc(d)}</dd></div>`).join("")}</dl>
      <div class="ph" style="padding:0 0 10px">${I.doc}Recommendations<span class="meta">rule-based</span></div>
      ${advice}
      <p class="label mode">The advice comes from fixed, readable rules (src/advisory.py); the models only supply the forecasts. A qualified person decides.</p>
      ${S.meta.explain ? `<div class="explain">${S.expl ? `<p class="sent"><span class="cite mono">AI</span><span>${esc(S.expl)}</span></p>
        <p class="label">Plain-English summary by Claude, written only from the numbers above. It explains; it does not forecast or decide.</p>`
        : `<button type="button" class="glass-btn" data-explain${S.busy ? " disabled" : ""}>Explain this outlook in plain English</button>`}</div>` : ""}
      <details class="trace"><summary class="label">How this outlook was made · 5 steps</summary><ol class="mono">
        <li>Weather: latest METAR published before ${hhmm(o.t0)}, used only if under 2 h old; TAF only if issued before ${hhmm(o.t0)} and valid for the target hour.</li>
        <li>Capacity: LightGBM forecasts AAR as a share of the ${o.aar_vfr}/h best case for +3 to +12 h; quantile models give the 80% range.</li>
        <li>Alerts: an hour is reduced at ≤ 80% of best case and severe at ≤ 60%, on the ${S.sens === "balanced" ? "central forecast" : "10% lower bound"}.</li>
        <li>Delay risk: calibrated logistic regression per lead time; drivers are each feature's log-odds push against the average flight.</li>
        <li>Models were trained on data before 2025 only; this is a replay of the held-out period.</li></ol></details>
    </article>
    <aside class="glass strong card sources" aria-label="Flights at risk" data-tour="flights">
      <div class="ph" style="padding:0 0 10px">${I.flights}Flights in the next 12 h<span class="meta">${o.n_flights} sampled</span></div>
      ${flights}
      ${o.flights.length > shown.length ? `<button class="glass-btn" data-go="flights" style="margin-top:12px">All ${o.flights.length} flights</button>` : ""}
      <p class="label" style="margin-top:12px">The flight file is a sample of real traffic, so compare risk rates, not raw counts, between airports.</p>
    </aside>
  </div></section>`;
}

function aarChart(o) {
  const W = 760, Hh = 270, P = { l: 38, r: 12, t: 12, b: 24 };
  const t0 = Date.parse(o.t0), x0 = t0 - 24 * 3600e3, x1 = t0 + 13 * 3600e3;
  const X = t => P.l + (Date.parse(t) - x0) / (x1 - x0) * (W - P.l - P.r);
  const vals = [...o.hist.map(h => h.aar), ...o.hours.flatMap(h => [h.lo, h.hi, h.taf, h.persist, S.reveal ? h.actual : null])].filter(v => v != null);
  const yMax = Math.ceil(Math.max(o.aar_vfr * 1.06, ...vals) / 20) * 20, yMin = Math.max(0, Math.floor(Math.min(o.aar_vfr * 0.5, ...vals) / 20) * 20);
  const Y = v => P.t + (1 - (v - yMin) / (yMax - yMin)) * (Hh - P.t - P.b);
  const path = pts => pts.filter(p => p[1] != null).map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)} ${Y(p[1]).toFixed(1)}`).join("");
  const band = o.hours.map(h => [h.hour, h.hi]).concat(o.hours.slice().reverse().map(h => [h.hour, h.lo]));
  const grid = []; for (let v = yMin; v <= yMax; v += 20) grid.push(`<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
  const ticks = []; for (let k = -24; k <= 12; k += 6) { const t = new Date(t0 + k * 3600e3).toISOString(); ticks.push(`<text x="${X(t)}" y="${Hh - 6}" text-anchor="middle">${k === 0 ? "issued" : `${t.slice(11, 13)}Z`}</text>`); }
  const thr = [[0.8, "var(--ring-moderate)", "−20%"], [0.6, "var(--stop)", "−40%"]].map(([f, c, l]) => `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(f * o.aar_vfr)}" y2="${Y(f * o.aar_vfr)}" stroke="${c}" stroke-dasharray="4 4" opacity=".8"/><text x="${W - P.r - 2}" y="${Y(f * o.aar_vfr) - 4}" text-anchor="end" style="fill:${c}">${l}</text>`);
  const dots = o.hours.map(h => `<circle cx="${X(h.hour)}" cy="${Y(h.pred)}" r="3.4" fill="#fff" stroke="var(--sky)" stroke-width="2"><title>${hhmm(h.hour)} · forecast ${h.pred}/h (${h.lo}–${h.hi})${S.reveal ? ` · actual ${h.actual}` : ""}</title></circle>`).join("");
  const legend = [["var(--ink-3)", "Observed (weather-derived)"], ["var(--sky)", "Model forecast"], ["rgba(0,133,202,.25)", "80% range", true], ["var(--ok)", "TAF-implied"], ["#a9b6c8", "Persistence"]]
    .concat(S.reveal ? [["var(--stop)", "What actually happened"]] : []).map(([c, l, box]) => `<span><i class="${box ? "box" : ""}" style="background:${c}"></i>${l}</span>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Arrival acceptance rate: last 24 hours observed and the next 12 hours forecast">
      ${grid.join("")}${thr.join("")}
      <line x1="${X(o.t0)}" x2="${X(o.t0)}" y1="${P.t}" y2="${Hh - P.b}" stroke="var(--navy)" stroke-dasharray="2 3" opacity=".5"/>
      <path d="${path(band)}Z" fill="rgba(0,133,202,.14)"/>
      <path d="${path(o.hist.map(h => [h.hour, h.aar]))}" fill="none" stroke="var(--ink-3)" stroke-width="2"/>
      <path d="${path(o.hours.map(h => [h.hour, h.persist]))}" fill="none" stroke="#a9b6c8" stroke-width="1.6" stroke-dasharray="3 4"/>
      <path d="${path(o.hours.map(h => [h.hour, h.taf]))}" fill="none" stroke="var(--ok)" stroke-width="1.8" stroke-dasharray="5 4"/>
      ${S.reveal ? `<path d="${path(o.hours.map(h => [h.hour, h.actual]))}" fill="none" stroke="var(--stop)" stroke-width="2"/>` : ""}
      <path d="${path(o.hours.map(h => [h.hour, h.pred]))}" fill="none" stroke="var(--sky)" stroke-width="2.6"/>
      ${dots}${ticks.join("")}
    </svg><div class="legend label">${legend}</div>`;
}

// ---------- flights ----------
function flightsView() {
  const o = S.out; if (!o) return loading("flights", "flights");
  const a = airport(), t0 = Date.parse(o.t0), base = S.meta.base_rate;
  const mean = o.flights.length ? o.flights.reduce((s, f) => s + f.p, 0) / o.flights.length : 0;
  const hours = Array.from({ length: 10 }, (_, i) => { const lo = t0 + (i + 3) * 3600e3, fs = o.flights.filter(f => { const t = Date.parse(f.t); return t >= lo - 1800e3 && t < lo + 1800e3; });
    return { t: new Date(lo).toISOString(), n: fs.length, m: fs.length ? fs.reduce((s, f) => s + f.p, 0) / fs.length : 0 }; });
  const top = Math.max(0.3, ...hours.map(h => h.m));
  const drv = {}; o.flights.forEach(f => f.drivers.forEach(([n]) => { drv[n] = (drv[n] || 0) + 1; }));
  const drivers = Object.entries(drv).sort((x, y) => y[1] - x[1]).slice(0, 8), dTop = Math.max(1, ...drivers.map(d => d[1]));
  return `<div class="page scroll" data-view="flights">
    <header class="page-h"><div><h1>Delay risk</h1><p class="lede">Sampled ${a.iata} flights scheduled 3 to 12 hours after ${day(o.t0)} ${hhmm(o.t0)}. Risk is the calibrated probability of a delay over 45 minutes.</p></div>
      <div class="page-stats"><div><div class="readout">${o.n_flights}</div><div class="label">Sampled flights</div></div>
        <div><div class="readout">${o.n_high}</div><div class="label">At risk ≥ ${pct(S.thr)}%</div></div>
        <div><div class="readout">${pct(mean)}<small>%</small></div><div class="label">Mean risk · 2025 average ${pct(base)}%</div></div></div></header>
    <div class="ins-grid">
      <section class="glass card"><div class="ph">Mean risk by scheduled hour<span class="meta">bar colour: at or above ${pct(S.thr)}%</span></div>
        <div class="land" style="grid-template-columns:repeat(10,1fr)">${hours.map(h => `<div class="land-cell" title="${hhmm(h.t)} · ${h.n} flights · mean risk ${pct(h.m)}%"><span class="n mono">${h.n || ""}</span><span class="bar ${h.m >= S.thr ? "severe" : ""}" style="height:${(h.m / top * 100).toFixed(0)}%"></span></div>`).join("")}
          ${hours.map(h => `<span class="label land-y mono">${h.t.slice(11, 13)}Z</span>`).join("")}</div>
        <div class="legend label"><span>Numbers above the bars are flight counts.</span></div></section>
      <section class="glass card"><div class="ph">What pushes the risk up<span class="meta">flights with this driver in their top three</span></div>
        <div class="m-bars">${drivers.length ? drivers.map(([n, c]) => `<div><span class="label">${esc(n)}</span><span class="mb"><i class="accent" style="width:${(c / dTop * 100).toFixed(0)}%"></i></span><span class="mono">${c}</span></div>`).join("") : '<p class="empty-note">No flights in this window.</p>'}</div></section>
      <section class="glass card span2"><div class="ph">Every sampled flight in the window<span class="meta">sorted by risk</span></div>
        <div class="scroll" style="max-height:480px"><table class="evtab"><thead><tr><th>Scheduled</th><th>Type</th><th>Route</th><th>Airline</th><th>Risk</th><th>Main drivers</th>${S.reveal ? "<th>Actual delay</th>" : ""}</tr></thead><tbody>
        ${o.flights.map(f => `<tr${S.sel === f.id ? ' style="background:rgba(0,133,202,.08)"' : ""}><td class="mono">${day(f.t)} ${hhmm(f.t)}</td><td>${f.dir === "arr" ? "Arrival" : "Departure"}</td><td class="mono">${esc(f.other)}</td><td class="mono">${esc(f.carrier)}</td>
          <td class="mono"><span class="mb"><i class="${f.p >= S.thr ? "bad" : f.p >= 0.2 ? "accent" : ""}" style="width:${Math.min(100, f.p * 200).toFixed(0)}%"></i></span>${pct(f.p)}%</td>
          <td>${f.drivers.map(d => esc(d[0])).join(", ") || "—"}</td>${S.reveal ? `<td class="mono${f.severe ? " t-stop" : ""}">${n0(f.delay_min)} min</td>` : ""}</tr>`).join("")}
        </tbody></table></div></section>
    </div></div>`;
}

// ---------- history ----------
function historyView() {
  const h = S.hist; if (!h) return loading("history", "history");
  const low = h.daily.reduce((s, d) => s + d.low, 0), days = h.daily.filter(d => d.low).length;
  const obs = h.delay.length ? h.delay.reduce((s, d) => s + d.obs * d.n, 0) / Math.max(1, h.delay.reduce((s, d) => s + d.n, 0)) : null;
  const top = Math.max(6, ...h.daily.map(d => d.low)), hm = Math.max(0.05, ...h.heat.flat().filter(v => v != null));
  return `<div class="page scroll" data-view="history">
    <header class="page-h"><div><h1>History</h1><p class="lede">${h.iata} in the 30 days before ${day(h.t0)} ${hhmm(h.t0)}, and when capacity is usually reduced across the 2025 replay period.</p></div>
      <div class="page-stats"><div><div class="readout">${low}</div><div class="label">Reduced hours, last 30 days</div></div>
        <div><div class="readout">${days}</div><div class="label">Days with a reduction</div></div>
        <div><div class="readout">${obs == null ? "—" : pct(obs)}<small>%</small></div><div class="label">Severe-delay share, sampled flights</div></div></div></header>
    <div class="ins-grid">
      <section class="glass card"><div class="ph">Hours per day at ≥ 20% below best case<span class="meta">last 30 days</span></div>
        <div class="land" style="grid-template-columns:repeat(${h.daily.length},1fr)">${h.daily.map(d => `<div class="land-cell" title="${day(d.day)} · ${d.low} reduced hours · mean ${pct(d.mean)}% of best case"><span class="bar ${d.low >= 8 ? "severe" : d.low ? "moderate" : ""}" style="height:${(d.low / top * 100).toFixed(0)}%"></span></div>`).join("")}
          ${h.daily.map((d, i) => `<span class="label land-y mono">${i % 5 ? "" : day(d.day).slice(0, 2)}</span>`).join("")}</div></section>
      <section class="glass card"><div class="ph">Severe-delay share per day<span class="meta">observed vs model, 3 h ahead</span></div>${delayChart(h.delay)}</section>
      <section class="glass card span2"><div class="ph">When is capacity reduced?<span class="meta">share of 2025 hours ≥ 20% below best case · local time</span></div>
        <div class="heatgrid"><span></span>${Array.from({ length: 24 }, (_, i) => `<span class="label mono land-y">${i % 3 ? "" : String(i).padStart(2, "0")}</span>`).join("")}
          ${h.heat.map((row, d) => `<span class="label">${["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]}</span>${row.map((v, hr) => `<i title="${["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]} ${String(hr).padStart(2, "0")}:00 · ${v == null ? "no data" : `${pct(v)}% of hours reduced`}" style="background:rgba(0,133,202,${v == null ? 0.03 : (0.05 + 0.85 * v / hm).toFixed(2)})"></i>`).join("")}`).join("")}
        </div></section>
    </div></div>`;
}
function delayChart(rows) {
  if (!rows.length) return '<p class="empty-note">No sampled flights in this period.</p>';
  const W = 560, Hh = 220, P = { l: 34, r: 10, t: 10, b: 22 }, n = rows.length;
  const yMax = Math.max(0.3, ...rows.map(r => Math.max(r.obs, r.pred))), X = i => P.l + i / Math.max(1, n - 1) * (W - P.l - P.r), Y = v => P.t + (1 - v / yMax) * (Hh - P.t - P.b);
  const path = k => rows.map((r, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(r[k]).toFixed(1)}`).join("");
  const grid = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].filter(v => v <= yMax).map(v => `<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 6}" y="${Y(v) + 4}" text-anchor="end">${pct(v)}%</text>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Severe-delay share per day, observed and predicted">${grid}
      <path d="${path("obs")}" fill="none" stroke="var(--stop)" stroke-width="1.8"/>${rows.map((r, i) => `<circle cx="${X(i)}" cy="${Y(r.obs)}" r="2.6" fill="var(--stop)"><title>${day(r.day)} · observed ${pct(r.obs)}% of ${r.n} flights</title></circle>`).join("")}
      <path d="${path("pred")}" fill="none" stroke="var(--sky)" stroke-width="2.6"/>
      ${rows.map((r, i) => i % 7 ? "" : `<text x="${X(i)}" y="${Hh - 5}" text-anchor="middle">${day(r.day)}</text>`).join("")}
    </svg><div class="legend label"><span><i style="background:var(--stop)"></i>Observed share</span><span><i style="background:var(--sky)"></i>Model risk</span></div>`;
}

// ---------- evaluation ----------
function evalView() {
  const e = S.ev, leads = `<span class="seg" role="group" aria-label="Lead time">${[3, 6, 9, 12].map(l => `<button type="button" data-lead="${l}" aria-pressed="${S.lead === l}">+${l} h</button>`).join("")}</span>`;
  if (!e) return loading("eval", "evaluation");
  const bars = (rows, digits, good) => { const top = Math.max(...rows.map(r => r[1] ?? 0)) || 1;
    return `<div class="m-bars">${rows.map(([l, v, tone]) => `<div><span class="label">${l}</span><span class="mb"><i class="${tone || ""}" style="width:${v == null ? 0 : Math.max(2, v / top * 100).toFixed(0)}%"></i></span><span class="mono">${v == null ? "—" : digits === "%" ? `${pct(v)}%` : v.toFixed(digits)}</span></div>`).join("")}</div>`; };
  const ours = e.aar[0], d = e.delay;
  const metric = (label, value, unit, rows, digits, note) => `<div class="glass card metric"><div class="label">${label}</div><div class="readout">${value}<small>${unit}</small></div>${bars(rows, digits)}${note ? `<div class="label">${note}</div>` : ""}</div>`;
  const tr = e.traffic;
  return `<div class="page scroll" data-view="eval">
    <header class="page-h"><div><h1>Evaluation</h1><p class="lede">Every number is measured on the held-out 2025 data (Jan to Jul), which no model saw during training or tuning. Shorter RMSE bars are better; longer PR-AUC bars are better.</p></div>${leads}</header>
    <div class="metrics">
      ${metric("Capacity forecast error (RMSE, arrivals per hour)", ours.rmse.toFixed(1), "/h", e.aar.map(r => [r.label, r.rmse, r.model === "lightgbm" ? "good" : ""]), 1, "Lower is better.")}
      ${metric("Spotting a ≥ 20% capacity cut (PR-AUC)", ours.pr_auc.toFixed(2), "", e.aar.map(r => [r.label, r.pr_auc, r.model === "lightgbm" ? "good" : ""]), 2, `Cuts occur in ${pct(e.event_rate)}% of hours; a random guess scores that.`)}
      ${metric("Severe-delay ranking (PR-AUC)", d.full.toFixed(2), `vs ${d.chance.toFixed(2)} chance`, [["Full model", d.full, "good"], ["Schedule only", d.sched], ["LightGBM", d.lightgbm], ["Chance", d.chance]], 3, "Logistic regression was picked out-of-fold on 2024.")}
      ${metric("80% forecast range: outcomes inside", pct(e.band), "%", [["Achieved", e.band, e.band >= 0.8 ? "good" : "bad"], ["Promised", 0.8]], "%", "Above 80% means the range is a little wide: the safe side.")}
    </div>
    <div class="ins-grid">
      ${e.tune ? `<section class="glass card"><div class="ph">What each data source adds to delay risk<span class="meta">PR-AUC · +${e.lead} h</span></div>
          ${bars(e.ladder.map(s => [s.label, s.pr_auc, s.final ? "good" : "accent"]), 3)}
          <div class="label" style="margin-top:10px">Gain over schedule-only +${e.boot.gain.toFixed(3)}, day-block bootstrap 95% interval ${e.boot.lo.toFixed(3)} to ${e.boot.hi.toFixed(3)}. Weather earns its place; BTS airport traffic adds little.</div></section>` : ""}
      ${tr ? `<section class="glass card"><div class="ph">Does the AAR rule match real ${airport().iata} traffic?<span class="meta">busy hours · BTS</span></div>
          ${bars(tr.bands.map(b => [b.label, b.share, "accent"]), "%")}
          <div class="label" style="margin-top:10px">Share of scheduled arrivals that actually landed, by the rule's AAR band. Rank correlation ${tr.spearman}: the direction holds, the effect is modest.</div></section>` : ""}
      ${e.tune ? `<section class="glass card"><div class="ph">Is the risk well calibrated?<span class="meta">10 risk groups · +${e.lead} h</span></div>${relChart(e.reliability)}</section>
        <section class="glass card"><div class="ph">Where the delay model fails<span class="meta">shown on purpose</span></div>
          <div class="fails">${e.failures.slice(0, 6).map(f => `<div class="fail"><p class="q mono">${esc(f.carrier)} ${f.dir === "arr" ? "from" : "to"} ${esc(f.other)} · ${esc(f.hub)}</p>
            <p><span class="pill ${f.kind === "missed" ? "stop" : "warn"}">${f.kind}</span> <span class="label">${day(f.t)} ${hhmm(f.t)}</span></p><p>Delay ${n0(f.delay_min)} min at risk ${pct(f.p)}%.</p></div>`).join("")}</div></section>`
        : `<section class="glass card span2"><p class="empty-note">The data-source ladder, calibration and failure cases are reported at +3 h and +12 h, the two tuning leads.</p></section>`}
      <section class="glass card span2"><div class="ph">Known limits<span class="meta">read before acting</span></div><ul class="limits">
        <li>The delay model cannot anticipate multi-hour disruptions with no weather precursor, such as maintenance.</li>
        <li>In the very worst capacity drops the AAR model is no more accurate than the TAF-implied rate.</li>
        <li>AAR labels are a weather-derived proxy of FAA called rates, not logged facility data.</li>
        <li>The flight file is a sample of about 3% of traffic; BTS covers reporting carriers only.</li></ul></section>
    </div></div>`;
}
function relChart(rows) {
  const W = 420, Hh = 260, P = { l: 38, r: 12, t: 10, b: 26 }, top = Math.max(0.1, ...rows.map(r => Math.max(r.pred, r.obs))) * 1.08;
  const X = v => P.l + v / top * (W - P.l - P.r), Y = v => P.t + (1 - v / top) * (Hh - P.t - P.b);
  const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5].filter(v => v <= top);
  return `<svg class="chart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Predicted against observed severe-delay rate">
      ${ticks.map(v => `<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 6}" y="${Y(v) + 4}" text-anchor="end">${pct(v)}%</text><text x="${X(v)}" y="${Hh - 8}" text-anchor="middle">${pct(v)}%</text>`).join("")}
      <line x1="${X(0)}" y1="${Y(0)}" x2="${X(top)}" y2="${Y(top)}" stroke="var(--ink-3)" stroke-dasharray="4 4"/>
      <path d="${rows.map((r, i) => `${i ? "L" : "M"}${X(r.pred).toFixed(1)} ${Y(r.obs).toFixed(1)}`).join("")}" fill="none" stroke="var(--sky)" stroke-width="2.6"/>
      ${rows.map(r => `<circle cx="${X(r.pred)}" cy="${Y(r.obs)}" r="3.4" fill="#fff" stroke="var(--sky)" stroke-width="2"><title>predicted ${pct(r.pred)}% · observed ${pct(r.obs)}% · ${r.n} flights</title></circle>`).join("")}
    </svg><div class="legend label"><span><i style="background:var(--sky)"></i>Model, predicted vs observed</span><span><i style="background:var(--ink-3)"></i>Perfect calibration</span></div>`;
}

// ---------- storms: full-BTS pipeline evidence (reports/jho_*.csv via /api/storm) ----------
function stormView() {
  const s = S.st, leads = `<span class="seg" role="group" aria-label="Lead time">${[3, 6, 9, 12].map(l => `<button type="button" data-lead="${l}" aria-pressed="${S.lead === l}">+${l} h</button>`).join("")}</span>`;
  if (!s) return loading("storms", "storm-hour results");
  const bars = (rows, digits) => { const top = Math.max(...rows.map(r => r[1] ?? 0)) || 1;
    return `<div class="m-bars">${rows.map(([l, v, tone]) => `<div><span class="label">${l}</span><span class="mb"><i class="${tone || ""}" style="width:${v == null ? 0 : Math.max(2, v / top * 100).toFixed(0)}%"></i></span><span class="mono">${v == null ? "—" : digits === "%" ? `${pct(v)}%` : v.toFixed(digits)}</span></div>`).join("")}</div>`; };
  const metric = (label, value, unit, body, note) => `<div class="glass card metric"><div class="label">${label}</div><div class="readout">${value}<small>${unit}</small></div>${body}${note ? `<div class="label">${note}</div>` : ""}</div>`;
  const tone = v => v === "model better" ? "good" : v === "model worse" ? "bad" : "accent";
  const pill = v => `<span class="pill ${v === "model better" ? "ok" : v === "model worse" ? "stop" : "warn"}">${esc(v)}</span>`;
  const sg = v => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
  const a = s.arr, now = s.delay.filter(d => d.lead === S.lead);
  return `<div class="page scroll" data-view="storms">
    <header class="page-h"><div><h1>Before severe weather</h1><p class="lede">How forecasts made 3 to 12 hours ahead held up in ${s.iata}'s storm hours: hours with a NOAA Storm Event (thunderstorm wind, hail, tornado, flash flood) within 50 km.
      NOAA logs are written after the event, so they only mark which 2025 test hours were storms and never feed a model. Scored on actual arrivals from the full BTS archive (3.26 M flights).</p></div>${leads}</header>
    <div class="metrics">
      ${metric(`Arrivals error in storm hours · RMSE · ${s.hours} hours`, a.model.toFixed(1), "/h",
               bars([["Model", a.model, "good"], ["Climatology", a.clim], ["Schedule", a.sched], ["Persistence", a.persist]], 1),
               `Normal hours ${a.all_hours.toFixed(1)}/h: errors are about ${(a.model / a.all_hours).toFixed(1)}× larger in storms. Lower is better.`)}
      ${metric("90% range: outcomes inside, storm hours", pct(a.cover), "%", bars([["Storm hours", a.cover, "bad"], ["All hours", a.cover_all, "good"], ["Promised", 0.9]], "%"),
               "Failure case: in storms the range is far too narrow. Read a storm-hour forecast as a warning, not an exact count.")}
      ${now.map(d => metric(`Severe-delay ranking in storms · ${d.dir}s · PR-AUC`, d.model.toFixed(2), ` vs ${d.clim.toFixed(2)}`,
               bars([["Model", d.model, tone(d.verdict)], ["Climatology", d.clim], ["Severe share", d.rate], ["Mean predicted", d.mean_p, "bad"]], 3),
               `${d.flights} flights in ${d.hours} storm hours. Gain 95% interval ${sg(d.lo)} to ${sg(d.hi)}: ${d.verdict}.`)).join("")}
    </div>
    <div class="ins-grid">
      <section class="glass card"><div class="ph">Arrivals error by lead time<span class="meta">RMSE, arrivals per hour</span></div>${stormLeadChart(s.by_lead)}</section>
      <section class="glass card"><div class="ph">Storm-hour delay ranking vs climatology<span class="meta">all leads · 95% interval</span></div>
        <div class="scroll" style="max-height:300px"><table class="evtab"><thead><tr><th>Type</th><th>Lead</th><th>Flights</th><th>Model</th><th>Climatology</th><th>Gain, 95% interval</th><th>Verdict</th></tr></thead><tbody>
        ${s.delay.map(d => `<tr${d.lead === S.lead ? ' style="background:rgba(0,133,202,.08)"' : ""}><td>${d.dir === "arrival" ? "Arrival" : "Departure"}</td><td class="mono">+${d.lead} h</td><td class="mono">${d.flights}</td>
          <td class="mono">${d.model.toFixed(3)}</td><td class="mono">${d.clim.toFixed(3)}</td><td class="mono">${sg(d.lo)} to ${sg(d.hi)}</td><td>${pill(d.verdict)}</td></tr>`).join("")}
        </tbody></table></div></section>
      ${s.fullbts?.length ? `<section class="glass card span2"><div class="ph">Same classifier on the full BTS archive, all 2025 hours<span class="meta">+${s.lead} h · LightGBM</span></div>
        <table class="evtab"><thead><tr><th>Type</th><th>Test flights</th><th>Severe share</th><th>PR-AUC model</th><th>PR-AUC climatology</th><th>Precision</th><th>Recall</th></tr></thead><tbody>
        ${s.fullbts.map(f => `<tr><td>${f.dir === "arrival" ? "Arrival" : "Departure"}</td><td class="mono">${n0(f.n)}</td><td class="mono">${pct(f.rate)}%</td><td class="mono">${f.model.toFixed(3)}</td>
          <td class="mono">${f.clim.toFixed(3)}</td><td class="mono">${f.precision.toFixed(3)}</td><td class="mono">${f.recall.toFixed(3)}</td></tr>`).join("")}</tbody></table>
        <div class="label" style="margin-top:10px">Trained on 2023-01 to 2024-06, threshold chosen on 2024 H2, scored once on 2025. About 45,000 test flights per row, so these intervals are tight.</div></section>` : ""}
      <section class="glass card span2"><div class="ph">Known limits in storm hours<span class="meta">read before acting</span></div><ul class="limits">
        <li>Few storm hours: ${s.hours} at ${s.iata} in the 2025 test period, so storm results are less certain than the overall ones.</li>
        <li>In storms the model's probabilities run well below the observed severe-delay share. Treat any rise in risk as significant.</li>
        <li>A TAF thunderstorm-timing feature (thunder within 2 h of the target hour) was tested and rejected on 2024 validation data.</li>
        <li>Source: jho/aerocast_data (run_all.py), copied into reports/jho_*.csv. This target is real arrivals, so its RMSE is not comparable with the AAR RMSE on the Evaluation page.</li></ul></section>
    </div></div>`;
}
function stormLeadChart(rows) {
  const W = 520, Hh = 240, P = { l: 34, r: 10, t: 12, b: 26 }, top = Math.ceil(Math.max(...rows.flatMap(r => [r.storm, r.clim, r.all])) / 5) * 5;
  const gw = (W - P.l - P.r) / rows.length, bw = gw / 4, Y = v => P.t + (1 - v / top) * (Hh - P.t - P.b);
  const grid = []; for (let v = 0; v <= top; v += 5) grid.push(`<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
  const g = rows.map((r, i) => { const x = P.l + i * gw + gw / 2;
    return `<rect x="${(x - bw - 2).toFixed(1)}" y="${Y(r.all)}" width="${bw}" height="${Y(0) - Y(r.all)}" rx="3" fill="var(--sky)"><title>+${r.lead} h · all hours ${r.all}/h</title></rect>
      <rect x="${(x + 2).toFixed(1)}" y="${Y(r.storm)}" width="${bw}" height="${Y(0) - Y(r.storm)}" rx="3" fill="var(--stop)"><title>+${r.lead} h · storm hours ${r.storm}/h</title></rect>
      <line x1="${(x - bw - 6).toFixed(1)}" x2="${(x + bw + 6).toFixed(1)}" y1="${Y(r.clim)}" y2="${Y(r.clim)}" stroke="var(--ink-2)" stroke-width="2" stroke-dasharray="3 3"><title>+${r.lead} h · climatology in storms ${r.clim}/h</title></line>
      <text x="${x}" y="${Hh - 8}" text-anchor="middle">+${r.lead} h</text>`; }).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Arrivals forecast error by lead time, all hours and storm hours">${grid.join("")}${g}</svg>
    <div class="legend label"><span><i style="background:var(--sky)"></i>All hours</span><span><i style="background:var(--stop)"></i>Storm hours</span><span><i style="background:var(--ink-2)"></i>Climatology, storm hours</span></div>`;
}

// ---------- tour (Torque Stripe's spotlight pattern) ----------
const STEPS = [
  { title: "Flowcast", text: "A replay of the held-out 2025 period at Atlanta and Dallas/Fort Worth. Pick an issue time and see what the models would have said 3 to 12 hours ahead, then reveal what happened." },
  { sel: '[data-tour="ask"]', view: "outlook", title: "Pick an airport and an issue time", text: "Every forecast uses only the weather, TAFs and flights published before this time." },
  { sel: '[data-tour="chips"]', view: "outlook", title: "Jump to a notable event", text: "The worst capacity dips of 2025. Each chip issues the forecast six hours before the dip." },
  { sel: '[data-tour="capacity"]', view: "outlook", title: "Capacity, hour by hour", text: "Forecast arrival rate with its 80% range and the TAF-implied rate. Hover an hour to light its arc on the ring around the airfield." },
  { sel: '[data-tour="weather"]', view: "outlook", title: "The weather the controller has", text: "The latest METAR. The stage shows it too: wind streaks, the cloud deck at the reported ceiling, haze for visibility, rain, snow and lightning." },
  { sel: '[data-tour="radar"]', view: "outlook", title: "Delay radar", text: "Sampled flights in the window. Centre is soonest, angle is the route bearing, colour is the severe-delay risk." },
  { sel: '[data-tab="eval"]', title: "Measured, not claimed", text: "RMSE, PR-AUC, calibration and the known failures, all on data the models never saw." },
  { sel: '[data-tab="storms"]', title: "Before severe weather", text: "The same kind of forecast scored only on hours with a NOAA storm nearby, on the full BTS archive, including where it falls short." },
];
function openTour(i = 0) {
  const s = STEPS[i]; if (s.view && S.view !== s.view) go(s.view);
  setTimeout(() => {
    const box = s.sel ? $(s.sel)?.getBoundingClientRect() : null, last = i === STEPS.length - 1;
    const pos = box ? `left:${Math.min(innerWidth - 380, Math.max(16, box.left))}px;top:${box.bottom + 16 > innerHeight - 230 ? Math.max(16, box.top - 230) : box.bottom + 16}px` : "";
    $("#overlay").innerHTML = `<div class="tour" role="dialog" aria-modal="true" aria-labelledby="tour-t">
      ${box ? `<div class="tour-ring" style="left:${box.left - 6}px;top:${box.top - 6}px;width:${box.width + 12}px;height:${box.height + 12}px"></div>` : ""}
      <div class="tour-card glass strong${box ? " placed" : ""}" style="${pos}"><div class="label">Step ${i + 1} of ${STEPS.length}</div><h3 id="tour-t">${s.title}</h3><p>${s.text}</p>
        <div class="tour-acts"><button class="glass-btn" data-tour-close>Skip</button><span style="flex:1"></span>
          ${i ? `<button class="glass-btn" data-tour-step="${i - 1}">Back</button>` : ""}<button class="glass-btn pri" ${last ? "data-tour-close" : `data-tour-step="${i + 1}"`}>${last ? "Start" : "Next"}</button></div></div></div>`;
    $("#overlay .tour-acts .pri")?.focus();
  }, 60);
}
function closeTour() { $("#overlay").innerHTML = ""; store.set("cr_tour_v1", "1"); }

// ---------- events ----------
document.addEventListener("click", e => {
  const t = e.target.closest("[data-go],[data-tour-open],[data-tour-step],[data-tour-close],[data-ev],[data-sens],[data-reveal],[data-cam],[data-render],[data-flight],[data-lead],[data-explain]");
  if (!t) return;
  const d = t.dataset;
  if (d.go) { e.preventDefault(); go(d.go); }
  else if ("tourOpen" in d) openTour();
  else if (d.tourStep) openTour(+d.tourStep);
  else if ("tourClose" in d) closeTour();
  else if (d.ev) { const ev = S.meta.events[+d.ev]; S.icao = ev.icao; S.t0 = ev.t0; forecast(); }
  else if (d.sens) { S.sens = d.sens; forecast(); }
  else if ("reveal" in d) { S.reveal = !S.reveal; render(); }
  else if (d.cam) { S.cam = d.cam; stage?.setView(d.cam); document.querySelectorAll("[data-cam]").forEach(b => b.setAttribute("aria-pressed", b.dataset.cam === d.cam)); }
  else if (d.render) { S.render = d.render; stage?.setRender(d.render); document.querySelectorAll("[data-render]").forEach(b => b.setAttribute("aria-pressed", b.dataset.render === d.render)); }
  else if (d.lead) { S.lead = +d.lead; S.ev = null; S.st = null; render(); ensure(S.view); }
  else if ("explain" in d) {
    t.disabled = true; t.textContent = "Asking Claude…";
    api("explain", { airport: S.icao, t0: S.t0, sens: S.sens, thr: S.thr }).then(r => { S.expl = r.text; render(); })
      .catch(err => { toast(err.message); t.disabled = false; t.textContent = "Explain this outlook in plain English"; });
  }
  else if (d.flight) {
    S.sel = +d.flight; const el = document.getElementById(`f${d.flight}`);
    if (S.view === "outlook" && el) { document.querySelectorAll(".src.cur").forEach(x => x.classList.remove("cur")); el.classList.add("cur"); el.scrollIntoView({ behavior: "smooth", block: "center" }); }
    else go("flights");
  }
});
document.addEventListener("submit", e => {
  if (!e.target.matches(".askbar")) return;
  e.preventDefault(); const f = new FormData(e.target);
  S.icao = f.get("airport"); S.t0 = `${f.get("date")}T${String(f.get("hour")).padStart(2, "0")}:00Z`; forecast();
});
document.addEventListener("change", e => {
  if (e.target.matches('.askbar [name="airport"]')) { S.icao = e.target.value; forecast(); }
  else if (e.target.matches("[data-thr]")) { S.thr = +e.target.value; forecast(); }
});
document.addEventListener("mouseover", e => { const li = e.target.closest("li[data-hour]"); if (li) { stage?.highlight(+li.dataset.hour); li.classList.add("hot"); } });
document.addEventListener("mouseout", e => { const li = e.target.closest("li[data-hour]"); if (li && !li.contains(e.relatedTarget)) { stage?.highlight(null); li.classList.remove("hot"); } });
document.addEventListener("keydown", e => { if (e.key === "Escape" && $("#overlay .tour")) closeTour(); });
addEventListener("hashchange", () => { const h = location.hash.replace(/^#\/?/, ""); if (VIEWS.some(v => v[0] === h) && h !== S.view) go(h); });
function paintAskButton() { const b = $(".askbar .glass-btn.pri"); if (b) { b.disabled = S.busy; b.textContent = S.busy ? "Forecasting…" : "Forecast"; } }

// ---------- boot ----------
(async () => {
  const noTour = new URLSearchParams(location.search).has("notour");
  try { S.meta = await api("meta"); } catch (e) { $("#view").innerHTML = `<div class="page"><p class="empty-note">Could not reach the dashboard API: ${esc(e.message)}. Start it with python dashboard/server.py.</p></div>`; return; }
  readUrl(); render();
  // the 3D stage is decoration: if WebGL or the CDN is unavailable the dashboard still works
  import("./stage.js").then(m => { stage = m.createStage($("#stage canvas"), $("#stage"), { zoom: innerWidth < 860 ? 4.4 : 3.3 }); stage.setActive(S.view === "outlook"); syncStage(false); })
    .catch(err => console.warn("3D stage unavailable:", err));
  await forecast(); ensure(S.view);
  if (!store.get("cr_tour_v1") && !noTour) openTour();
})();
