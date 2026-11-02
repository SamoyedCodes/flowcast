/* Flowcast dashboard. Plain ES modules, no build step; data comes from dashboard/server.py.
   Layout and interaction patterns follow Torque Stripe (MIT, (c) 2026 Aljabri Alam, github.com/aljabrialam/torque-stripe).
   Written for a first-time reader: plain words and local airport times up front, the technical evidence one "details" click away. */
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = { get: k => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };
async function api(path, q = {}) {
  const r = await fetch(`/api/${path}?${new URLSearchParams(q)}`).catch(() => { throw new Error("The Flowcast server did not respond. Check your connection, then try again."); });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `The server answered ${r.status}${r.statusText ? ` (${r.statusText})` : ""}. Try again in a moment.`);
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
  doc: svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>'),
  wind: svg('<path d="M3 8h10.5a2.5 2.5 0 1 0-2.5-2.5M3 12h15.5a2.5 2.5 0 1 1-2.5 2.5M3 16h7"/>'),
  storm: svg('<path d="M7 16.5a4.5 4.5 0 1 1 1.2-8.8A5.5 5.5 0 0 1 18.6 9 3.8 3.8 0 0 1 18 16.5"/><path d="m12.5 12-2.5 4.5h3.5L11 21"/>'),
  down: svg('<path d="m6 9 6 6 6-6"/>'),
};
const LOGO = `<svg width="30" height="30" viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="26" fill="#00205B"/><rect x="28" y="12" width="8" height="40" rx="1.5" fill="#fff"/><path d="M32 16v5M32 27v5M32 38v5" stroke="#00205B" stroke-width="1.8"/><path d="M32 6a26 26 0 0 1 26 26" fill="none" stroke="#0085CA" stroke-width="6" stroke-linecap="round"/></svg>`;

// ---------- plain words ----------
const VIEWS = [["outlook", "Forecast", I.outlook], ["flights", "Flights", I.flights], ["history", "Past 30 days", I.history], ["eval", "Accuracy", I.eval]];
const SEV = { clear: "Normal", moderate: "Reduced", severe: "Severe" };
const CITY = { ATL: "Atlanta", DFW: "Dallas–Fort Worth" };
const AIRLINES = { AA: "American", DL: "Delta", UA: "United", WN: "Southwest", B6: "JetBlue", AS: "Alaska", NK: "Spirit", F9: "Frontier", G4: "Allegiant",
                   HA: "Hawaiian", MQ: "Envoy", OO: "SkyWest", "9E": "Endeavor", YX: "Republic", OH: "PSA", YV: "Mesa", QX: "Horizon", EV: "ExpressJet" };
const HAZ = { thunder: "thunderstorms", low_vis: "low cloud or poor visibility", xwind: "strong crosswind on the runway", tail: "wind from behind on the runway" };
const WX = { TS: "thunderstorm", SH: "showers", FZ: "freezing", RA: "rain", SN: "snow", DZ: "drizzle", BR: "mist", FG: "fog", HZ: "haze", GR: "hail", GS: "hail",
             PL: "ice pellets", UP: "rain or snow", SQ: "squalls", VC: "nearby", FU: "smoke" };
const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];

// ---------- state ----------
const S = { view: "outlook", meta: null, icao: "KATL", t0: "", sens: "balanced", thr: 0.25, reveal: true, out: null, hist: null, ev: null,
            st: null, expl: null, lead: 3, busy: false, sel: null, cam: "3d", open: new Set(),
            err: null,    // forecast failure, shown under the ask bar: { when, msg }
            fail: null }; // History or Accuracy failure: { view, msg }
const THRS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5];
let stage = null;

const hhmm = s => s ? `${s.slice(11, 16)} UTC` : "—";   // for the technical layer only
const day = s => new Date(s).toLocaleDateString("en-US", { day: "numeric", month: "short", timeZone: "UTC" });
const pct = p => p == null ? "—" : Math.round(p * 100);
const n0 = v => v == null ? "—" : Math.round(v).toLocaleString("en-US");
const airport = () => S.meta.airports.find(a => a.icao === S.icao);
const city = (a = airport()) => CITY[a.iata] || a.iata;
const zone = iata => S.meta.airports.find(a => a.iata === iata)?.tz || "UTC";
// local airport time: "2 AM", "2:05 AM", "Sat, Mar 15"
const hr = (s, z = airport().tz) => new Date(s).toLocaleTimeString("en-US", { hour: "numeric", timeZone: z });
const clock = (s, z = airport().tz) => s ? new Date(s).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: z }) : "—";
const dayL = (s, z = airport().tz) => new Date(s).toLocaleDateString("en-US", { weekday: "short", day: "numeric", month: "short", timeZone: z });
const later = (s, h) => new Date(Date.parse(s) + h * 3600e3).toISOString();
const riskTone = p => p >= S.thr ? "stop" : p >= 0.2 ? "warn" : "mute";
const flightName = f => `${esc(AIRLINES[f.carrier] || f.carrier)} ${f.dir === "arr" ? "from" : "to"} <span class="mono">${esc(f.other)}</span>`;
const reasons = f => [...new Set(f.drivers.map(d => d[0]))];
const outcome = f => f.delay_min == null ? "No record" : `${f.delay_min <= 0 ? "On time" : `${n0(f.delay_min)} min late`}${f.severe ? ' · <b class="t-stop">long delay</b>' : ""}`;
const wxText = s => { const t = s.split(" ").map(w => (w[0] === "-" ? "light " : w[0] === "+" ? "heavy " : "") + (w.replace(/^[-+]/, "").match(/../g) || []).map(c => WX[c] || c).join(" ")).join(", ");
  return t[0].toUpperCase() + t.slice(1); };
const more = (k, title, body, cls = "") => `<details class="more ${cls}" data-k="${k}"${S.open.has(k) ? " open" : ""}><summary>${title}${I.down}</summary><div class="more-b">${body}</div></details>`;
const bars = (rows, digits) => { const top = Math.max(...rows.map(r => r[1] ?? 0)) || 1;
  return `<div class="m-bars">${rows.map(([l, v, tone]) => `<div><span class="label">${l}</span><span class="mb"><i class="${tone || ""}" style="width:${v == null ? 0 : Math.max(2, v / top * 100).toFixed(0)}%"></i></span><span class="mono">${v == null ? "—" : digits === "%" ? `${pct(v)}%` : v.toFixed(digits)}</span></div>`).join("")}</div>`; };
const metric = (label, value, unit, body, note) => `<div class="glass card metric"><div class="label">${label}</div><div class="readout">${value}<small>${unit}</small></div>${body}${note ? `<div class="label">${note}</div>` : ""}</div>`;

function readUrl() {
  const q = new URLSearchParams(location.search), h = location.hash.replace(/^#\/?/, "").replace("storms", "eval");
  if (VIEWS.some(v => v[0] === h)) S.view = h;
  if (q.get("airport") && S.meta.airports.some(a => a.icao === q.get("airport"))) S.icao = q.get("airport");
  const t = q.get("t0")?.slice(0, 13);   // strict: Date.parse happily reads "garbage" as the year 2000
  S.t0 = /^\d{4}-\d\d-\d\dT\d\d$/.test(t) ? `${t}:00Z` : S.meta.default.t0;
  if (q.get("sens") === "cautious") S.sens = "cautious";
  if (THRS.includes(+q.get("thr"))) S.thr = +q.get("thr");
}
// settings ride along only when changed, so a shared link shows the same answer
function writeUrl() {
  const extra = (S.sens === "cautious" ? "&sens=cautious" : "") + (S.thr !== 0.25 ? `&thr=${S.thr}` : "");
  history.replaceState(null, "", `?airport=${S.icao}&t0=${S.t0.slice(0, 13)}${extra}#/${S.view}`);
}

// ---------- data ----------
let seq = 0;   // newest request wins, so a slow answer never overwrites a newer one
async function forecast() {
  const id = ++seq; S.busy = true; paintAskButton();
  try {
    const out = await api("outlook", { airport: S.icao, t0: S.t0, sens: S.sens, thr: S.thr });
    if (id !== seq) return;
    S.out = out; S.err = null; S.fail = null;
    S.t0 = S.out.t0; S.hist = null; S.ev = null; S.st = null; S.expl = null; S.sel = null; writeUrl(); syncStage(true);
    const v = verdict(S.out); $("#live").textContent = `${city()}: ${v.title}. ${v.sub}`;
  } catch (e) {
    if (id !== seq) return;
    S.err = { when: `${dayL(S.t0)}, ${clock(S.t0)} in ${city()}.`, msg: e.message };
    if (S.out) { S.icao = S.out.icao; S.t0 = S.out.t0; }   // the inputs go back to the forecast still on screen
  }
  S.busy = false; render();
}
function syncStage(scan) {
  if (!stage || !S.out) return;
  const a = airport(); stage.setAirport(a.iata, a.runway_heading); stage.setWeather(S.out.now);
  stage.setRing(S.out.hours.map(h => h.sev)); if (scan) stage.scan();
}
async function ensure(view) {
  const icao = S.icao, t0 = S.t0, lead = S.lead, stale = () => icao !== S.icao || t0 !== S.t0 || lead !== S.lead;   // the user moved on meanwhile
  try {
    if (view === "history" && !S.hist) { const h = await api("history", { airport: icao, t0 }); if (!stale()) { S.hist = h; render(); } }
    if (view === "eval" && (!S.ev || S.ev.lead !== S.lead || S.ev.icao !== S.icao)) {
      const q = { airport: icao, lead };
      const [ev, st] = await Promise.all([api("eval", q), api("storm", q).catch(() => null)]);   // storm results are extra; Accuracy works without them
      if (!stale()) { S.ev = { ...ev, icao }; S.st = st; render(); }
    }
  } catch (e) { if (!stale()) { S.fail = { view, msg: e.message }; render(); } }
}
function toast(msg) {
  const t = document.createElement("div"); t.className = "toast glass strong"; t.textContent = msg; document.body.append(t); setTimeout(() => t.remove(), 4200);
  $("#alerts").textContent = msg;   // a live region that already exists, so screen readers announce it
}

// ---------- shell ----------
function go(v) { S.view = v; S.fail = null; writeUrl(); render(); ensure(v); }   // arriving retries, so an old error never shows over the new load
function renderShell() {
  $("#rail").innerHTML = `<a class="rail-logo" href="#/outlook" data-go="outlook" aria-label="Flowcast, forecast">${LOGO}</a>`
    + VIEWS.map(([id, l, ic]) => `<button class="rail-btn" data-go="${id}" data-tab="${id}"${S.view === id ? ' aria-current="page"' : ""}>${ic}<span>${l}</span></button>`).join("")
    + `<span class="rail-gap"></span><button class="rail-btn" data-tour-open aria-label="Help: show the guided tour">${I.help}<span>Help</span></button>`;
  $("#topbar").innerHTML = `<span class="tb-item brand"><b>Flowcast</b></span>
    <span class="pill" title="Uses January to July 2025, which the models never saw while they were built">Replay of 2025 · not live</span>`;
  $("#stage").classList.toggle("dim", S.view !== "outlook"); stage?.setActive(S.view === "outlook");
}
function render() {
  renderShell();
  const v = $("#view"), keep = v.firstElementChild?.dataset.view === S.view ? v.firstElementChild.scrollTop : 0;
  v.innerHTML = { outlook: outlookView, flights: flightsView, history: historyView, eval: evalView }[S.view]();
  if (v.firstElementChild) v.firstElementChild.scrollTop = keep;
  chipCue();
}
// chip row: the arrow shows only while some chips are out of view, and at the end it points back to the start
function chipCue(c = $(".chips")) {
  if (!c) return;
  c.classList.toggle("over", c.scrollWidth > c.clientWidth + 4);
  c.classList.toggle("end", c.scrollLeft + c.clientWidth >= c.scrollWidth - 4);
}
// a page waiting on data: its header stays put, and a failure says so with a way to retry
const loading = (view, what, head = "") => {
  const msg = view === "flights" ? S.err?.msg : S.fail?.view === view && S.fail.msg;
  return `<div class="page scroll" data-view="${view}"${msg ? "" : ' aria-busy="true"'}>${head}${msg
    ? `<div class="empty-note" role="alert"><p>Could not load ${what}. ${esc(msg)}</p><button type="button" class="glass-btn" data-retry>Try again</button></div>`
    : `<p class="empty-note">Loading ${what}…</p>`}</div>`;
};

// ---------- forecast: one answer over the 3D airfield, then what to do ----------
function outlookView() {
  const m = S.meta, o = S.out, a = airport();
  const date = S.t0.slice(0, 10), hour = +S.t0.slice(11, 13);
  const askbar = `<form class="askbar glass strong" data-tour="ask">
      <label class="ab-model"><span class="label">Airport</span>
        <select name="airport">${m.airports.map(x => `<option value="${x.icao}"${x.icao === S.icao ? " selected" : ""}>${city(x)} (${x.iata})</option>`).join("")}</select></label>
      <div class="ab-q"><span class="label">Forecast time (UTC), any hour from January to July 2025</span>
        <span class="row"><input type="date" name="date" class="mono" aria-label="Forecast date (UTC)" min="${m.start.slice(0, 10)}" max="${m.end.slice(0, 10)}" value="${date}" required />
        <select name="hour" class="mono" aria-label="Forecast hour (UTC)">${Array.from({ length: 24 }, (_, h) => `<option value="${h}"${h === hour ? " selected" : ""}>${String(h).padStart(2, "0")}:00</option>`).join("")}</select>
        ${o ? `<span class="label">${clock(o.t0)} in ${city(a)}</span>` : ""}</span></div>
      <button class="glass-btn pri" type="submit"${S.busy ? " disabled" : ""}>${S.busy ? "Loading…" : "Show forecast"}</button>
    </form>`;
  const err = S.err && `<p class="askerr glass strong" role="alert"><span class="dot stop"></span><span><b>${S.err.when}</b> ${esc(S.err.msg)}</span></p>`;
  const chips = `<div class="chips" data-tour="chips"><span class="label chips-l">Replay a bad day</span>${m.events.map((e, i) => { const d = dayL(e.at, zone(e.iata));
    return `<button class="chip glass" data-ev="${i}" aria-label="Replay ${e.iata}, ${d}: landings fell to ${n0(e.aar)} an hour" title="Landings fell to ${n0(e.aar)} an hour. Starts the forecast 6 hours before."><span class="mono k">${e.iata}</span>${d}<span class="mono v">${n0(e.aar)}/hr</span></button>`; }).join("")}
    <button type="button" class="chips-more glass" data-chips-more tabindex="-1" aria-hidden="true" title="More bad days">${I.down}</button></div>`;
  if (!o) return `<div class="outlook" data-view="outlook"><section class="console">${askbar}${err || chips}</section></div>`;

  const k = o.verdict.kind, v = verdict(o), low = o.hours.reduce((x, h) => (h.pred < x.pred ? h : x), o.hours[0]);
  const answer = `<section class="hud left glass strong answer" aria-labelledby="verdict" data-tour="answer" id="answer-top">
      <div class="pillrow"><span class="pill ${k}"><span class="dot ${k}"></span>${SEV[k]}</span><span class="label">${city(a)} · ${dayL(o.t0)}, ${clock(o.t0)}</span>
        ${S.sens === "cautious" ? '<span class="pill" title="Warns when even the low end of the forecast range drops. Change it under Alert settings.">Cautious alerts</span>' : ""}</div>
      <h1 class="headline" id="verdict">${v.title}</h1><p class="lede">${v.sub}</p>
      <div class="label">Each bar is one hour, 3 to 12 hours ahead</div>
      <div class="segs" role="group" aria-label="Forecast by hour">${o.hours.map((h, i) => `<button type="button" class="${h.sev}" data-hour="${i}" tabindex="${i ? -1 : 0}" title="${hr(h.hour)}: about ${n0(h.pred)} landings"
        aria-label="${hr(h.hour)}: about ${n0(h.pred)} landings an hour, ${SEV[h.sev].toLowerCase()}"></button>`).join("")}</div>
      <div class="ticks label"><span>${hr(o.hours[0].hour)}</span><span>${hr(o.hours.at(-1).hour)}</span></div>
      <div class="legend label">${Object.entries(SEV).map(([s, l]) => `<span><i class="box ${s}"></i>${l}</span>`).join("")}</div>
      <div class="readouts">
        <div><div class="readout">${n0(low.pred)}<small>/hr</small></div><div class="label">Fewest landings in one hour (up to ${o.aar_vfr} in good weather)</div></div>
        <div><div class="readout">${o.n_high}<small>of ${o.n_flights}</small></div><div class="label">Sampled flights with a ${pct(S.thr)}%+ chance of a 45+ min delay</div></div>
      </div>${truth(o)}</section>`;
  const w = o.now;
  const weather = `<aside class="hud right glass" aria-label="Weather now" data-tour="weather">
      <div class="ph">${I.wind}Weather now<span class="meta">${w ? `as of ${clock(w.hour)}` : ""}</span></div>
      <div class="pad">${w ? weatherBody(w) : '<p class="label">No weather report in the hour before this time.</p>'}</div>
    </aside>`;
  const tag = `<div class="hud bl"><div class="readout">${a.iata}</div><div class="label">${esc(a.name)}</div><div class="label attrib">Airport map © OpenStreetMap contributors</div></div>`;
  const next = `<button type="button" class="hud bc glass-btn" data-scroll="answer">What to do, and the details${I.down}</button>`;
  const cam = `<div class="hud br seg glass" role="group" aria-label="3D view">${[["3d", "3D"], ["top", "Top"], ["side", "Side"]].map(([c, l]) => `<button type="button" data-cam="${c}" aria-pressed="${S.cam === c}">${l}</button>`).join("")}</div>`;
  return `<div class="outlook" data-view="outlook"><section class="console">${tag}${answer}${askbar}${err || chips}${weather}${next}${cam}</section>${advisory(o)}</div>`;
}

function verdict(o) {
  const w = o.windows.find(x => x.severity === "severe") || o.windows[0];
  if (!w) return { title: "Arrivals should run normally", sub: `No weather slowdown is forecast between ${hr(o.hours[0].hour)} and ${hr(later(o.hours.at(-1).hour, 1))}.` };
  const span = `${hr(w.start)} to ${hr(w.end)}`, why = w.codes?.length ? ` Likely cause: ${w.codes.map(c => HAZ[c]).join(" and ")}.` : "";
  if (w.pct < 20)   // cautious setting: the central forecast is fine, only the low end of the range dips
    return { title: `Possible slowdown from ${span}`, sub: `Probably fine, but the low end of the forecast dips to ${w.lo} landings an hour. Treat it as a heads-up.${why}` };
  // cautious judges severity on the low end, so say the low end; otherwise "Big slowdown" sits over a mild central number
  const low = o.sens === "cautious" ? `, and it could fall to ${w.lo}` : "";
  return { title: `${w.severity === "severe" ? "Big slowdown" : "Fewer arrivals"} from ${span}`,
           sub: `About ${w.min_aar} planes an hour can land${low}, down from ${o.aar_vfr} in good weather (${w.pct}% fewer).${why}` };
}

// replay only: set the forecast against what the airport really managed, misses included
function truth(o) {
  const v = o.aar_vfr, hs = o.hours.filter(h => h.actual != null);
  if (!S.reveal || !hs.length) return "";
  const w = hs.reduce((x, h) => (h.actual < x.actual ? h : x)), cut = w.actual <= 0.8 * v, warned = o.windows.length > 0;
  const tafCut = hs.some(h => h.taf != null && h.taf <= 0.8 * v);
  const said = cut ? `landings fell to ${n0(w.actual)} an hour at ${hr(w.hour)}` : `landings stayed near normal, at least ${n0(w.actual)} an hour`;
  const call = cut ? (warned ? "Flowcast warned of it." : `Flowcast missed this slowdown${tafCut ? ", but the weather-forecast rule flagged one" : ""}.`)
    : (warned ? "This warning did not come true." : "As forecast.");
  const tone = cut && !warned ? "stop" : warned && !cut ? "warn" : "ok";
  return `<p class="truth"><span class="dot ${tone}"></span><span><b>What really happened:</b> ${said}. ${call}</span></p>`;
}

function weatherBody(w) {
  const CAT = { VFR: ["Good flying weather", "clear"], MVFR: ["Slightly reduced visibility", "warn"], IFR: ["Low cloud or poor visibility", "stop"], LIFR: ["Very low cloud or fog", "stop"] };
  const [say, tone] = CAT[w.category] || ["Conditions not reported", "mute"];
  const x = Math.abs(w.crosswind_kt ?? 0), c = w.ceiling_ft, v = w.vis_mi;
  const wind = w.wind_kt == null ? "—" : w.wind_kt < 1 ? "Calm" : `${w.wind_kt}${w.gust_kt ? `–${w.gust_kt}` : ""}<small>knots</small>`;
  const windSub = [!(w.wind_kt >= 1) ? "" : w.wind_dir == null ? "Direction varies" : `From the ${COMPASS[Math.round(w.wind_dir / 45) % 8]}`, x >= 10 ? `${x} knots across the runway` : ""].filter(Boolean).join(", ");
  const tile = (l, val, t = "", sub = "") => `<div class="tile"><div class="label">${l}</div><div class="tv t-${t}">${val}</div>${sub}</div>`;
  return `<p class="wx-say"><span class="dot ${tone}"></span>${say}</p>
    <div class="tiles">
      ${tile("Wind", wind, x >= 15 ? "stop" : x >= 10 ? "warn" : "", windSub ? `<div class="label">${windSub}</div>` : "")}
      ${tile("Cloud base", c == null || c >= 12000 ? "High or none" : `${n0(c)}<small>ft</small>`, c != null && c < 1000 ? "stop" : c != null && c < 3000 ? "warn" : "")}
      ${tile("Visibility", v == null ? "—" : `${v >= 10 ? "10+" : v}<small>miles</small>`, v != null && v < 1 ? "stop" : v != null && v < 3 ? "warn" : "")}
      ${tile("Rain or storms", w.wx ? esc(wxText(w.wx)) : "None", w.thunder ? "stop" : w.wx ? "warn" : "mute")}
    </div>`;
}

function advisory(o) {
  const act = o.windows.length
    ? o.windows.map(w => `<div class="act"><p class="act-h"><span class="dot ${w.severity}"></span><b>${hr(w.start)} to ${hr(w.end)}</b><span class="label">${w.severity === "severe" ? "big slowdown" : "fewer arrivals"}, down to about ${w.min_aar} an hour</span></p>
        <ul class="steps">${w.advice.slice(1, -1).map(s => `<li>${esc(s)}</li>`).join("")}</ul></div>`).join("")
      + more("terms", "What these terms mean", `<dl class="terms">
          <dt>TMU</dt><dd>Traffic management unit: the team that matches the number of arriving flights to what the airport can handle.</dd>
          <dt>Ground delay program</dt><dd>Holding flights at their departure airport, so they wait on the ground instead of circling in the air.</dd>
          <dt>Arrival metering</dt><dd>Spacing arriving flights out so they reach the runway at a steady rate.</dd>
          <dt>Gate hold</dt><dd>Keeping a departing plane at the gate, engines off, until there is room for it.</dd>
          <dt>Runway configuration</dt><dd>Which runways are in use, and in which direction. Planes land into the wind, so a change in wind can force a switch.</dd></dl>`)
    : '<p class="lede">No action needed. Check again when the next weather report or forecast comes in.</p>';
  const explain = !S.meta.explain ? "" : S.expl
    ? `<div class="ai"><p>${esc(S.expl)}</p><p class="label">Summary by Claude, written only from the numbers on this page. It explains; it does not forecast or decide.</p></div>`
    : `<button type="button" class="glass-btn" data-explain${S.busy ? " disabled" : ""}>Explain this forecast in plain English</button>`;
  const hourly = `<div class="tablewrap"><table class="evtab"><thead><tr><th>Time</th><th>Forecast</th><th>Likely range</th><th>Weather-forecast rule</th>${S.reveal ? "<th>What happened</th>" : ""}</tr></thead><tbody>
      ${o.hours.map(h => `<tr><td class="mono">${hr(h.hour)}</td><td class="mono t-${h.sev === "clear" ? "" : h.sev}">${n0(h.pred)}</td><td class="mono">${n0(h.lo)}–${n0(h.hi)}</td><td class="mono">${n0(h.taf)}</td>${S.reveal ? `<td class="mono">${n0(h.actual)}</td>` : ""}</tr>`).join("")}</tbody></table></div>
    <p class="label">Landings per hour. The likely range holds the real value about 8 times in 10. The weather-forecast rule turns the official airport forecast (TAF) straight into a landing rate: it is the simple method Flowcast has to beat.</p>`;
  const settings = `<div class="set"><span>Alert sensitivity</span><span class="seg" role="group" aria-label="Alert sensitivity">${[["balanced", "Balanced"], ["cautious", "Cautious"]].map(([s, l]) => `<button type="button" data-sens="${s}" aria-pressed="${S.sens === s}">${l}</button>`).join("")}</span></div>
    <p class="label">Balanced warns when the forecast drops 20% or more below good-weather capacity. Cautious also warns when only the low end of the range drops: earlier warnings, but in 2025 testing about half of them did not come true.</p>
    <div class="set"><label for="thr">Flag a flight when its chance of a 45+ min delay is at least</label><select id="thr" class="seg mono" data-thr>${THRS.map(t => `<option value="${t}"${t === S.thr ? " selected" : ""}>${pct(t)}%</option>`).join("")}</select></div>`;
  const trace = `<ol class="trace">
      <li>Weather: the latest airport weather report (METAR) published before ${hhmm(o.t0)}, used only if under 2 hours old, and the official forecast (TAF) only if issued before then and valid for the hour.</li>
      <li>Landings: a LightGBM model forecasts the arrival acceptance rate (AAR) as a share of the ${o.aar_vfr}/hr good-weather rate, 3 to 12 hours ahead. Quantile models give the 80% likely range.</li>
      <li>Alerts: an hour is reduced at 80% of the good-weather rate or less, and severe at 60% or less, judged on the ${S.sens === "balanced" ? "central forecast" : "low end of the range"}.</li>
      <li>Delay risk: a calibrated logistic regression for each lead time. The reasons are the inputs that push a flight's risk above the average flight's.</li>
      <li>The models were trained on data before 2025 only. This is a replay of the held-out 2025 period.</li></ol>
    <p class="label">Suggested actions come from fixed, readable rules (src/advisory.py). The models only supply the forecasts.</p>`;
  const shown = o.flights.slice(0, 6);
  const flights = shown.length ? shown.map(f => { const r = reasons(f); return `<button class="src${S.sel === f.id ? " cur" : ""}" data-flight="${f.id}">
      <span class="src-h"><b class="readout t-${riskTone(f.p)}">${pct(f.p)}<small>%</small></b><span class="src-n"><span>${flightName(f)}</span><span class="label">${f.dir === "arr" ? "Lands" : "Leaves"} ${clock(f.t)}</span></span></span>
      <span class="src-p">${r.length ? `Why: ${esc(r.slice(0, 2).join(", "))}` : "Nothing unusual: close to an average flight."}</span>
      ${S.reveal ? `<span class="label">What happened: ${outcome(f)}</span>` : ""}</button>`; }).join("")
    : '<p class="empty-note">No sampled flights are scheduled 3 to 12 hours after this time.</p>';
  return `<section class="results" id="answer"><div class="answer-grid">
    <article class="glass strong card" data-tour="advice">
      <div class="ph">${I.doc}What to do</div>
      ${act}
      <p class="label note">Flowcast suggests; a qualified person decides. Always check the latest weather reports before acting.</p>
      ${explain}
      <div class="ph sec">${I.history}Landings per hour<span class="meta"><button type="button" class="glass-btn sm" data-reveal aria-pressed="${S.reveal}">${S.reveal ? "Hide" : "Show"} what really happened</button></span></div>
      ${aarChart(o)}
      ${more("hours", "Hour by hour", hourly)}${more("settings", "Alert settings", settings)}${more("how", "How this forecast was made", trace)}
    </article>
    <aside class="glass strong card sources" aria-label="Flights at risk" data-tour="flights">
      <div class="ph">${I.flights}Flights at risk<span class="meta">chance of a 45+ min delay</span></div>
      ${flights}
      ${o.flights.length > shown.length ? `<button class="glass-btn" data-go="flights">See all ${o.flights.length} flights</button>` : ""}
      <p class="label">A sample of real flights, so compare percentages, not counts.</p>
    </aside>
  </div></section>`;
}

function aarChart(o) {
  const sm = innerWidth < 600, W = sm ? 340 : 760, Hh = sm ? 220 : 260, P = { l: 34, r: 10, t: 12, b: 24 };
  const t0 = Date.parse(o.t0), x0 = t0 - 24 * 3600e3, x1 = t0 + 13 * 3600e3;
  const X = t => P.l + (Date.parse(t) - x0) / (x1 - x0) * (W - P.l - P.r);
  const vals = [...o.hist.map(h => h.aar), ...o.hours.flatMap(h => [h.lo, h.hi, S.reveal ? h.actual : null])].filter(v => v != null);
  const yMax = Math.ceil(Math.max(o.aar_vfr * 1.06, ...vals) / 20) * 20, yMin = Math.max(0, Math.floor(Math.min(o.aar_vfr * 0.5, ...vals) / 20) * 20);
  const Y = v => P.t + (1 - (v - yMin) / (yMax - yMin)) * (Hh - P.t - P.b);
  const path = pts => pts.filter(p => p[1] != null).map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)} ${Y(p[1]).toFixed(1)}`).join("");
  const band = o.hours.map(h => [h.hour, h.hi]).concat(o.hours.slice().reverse().map(h => [h.hour, h.lo]));
  const grid = []; for (let v = yMin; v <= yMax; v += 20) grid.push(`<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`);
  const ticks = []; for (let k = -24; k <= 12; k += sm ? 12 : 6) ticks.push(`<text x="${X(later(o.t0, k))}" y="${Hh - 6}" text-anchor="middle">${k === 0 ? "now" : hr(later(o.t0, k))}</text>`);
  // the bright ring colour draws the line; the text takes the darker amber that passes contrast
  const thr = [[0.8, "var(--ring-moderate)", "var(--amber)", "Reduced"], [0.6, "var(--stop)", "var(--stop)", "Severe"]].map(([f, c, ink, l]) => `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(f * o.aar_vfr)}" y2="${Y(f * o.aar_vfr)}" stroke="${c}" stroke-dasharray="4 4" opacity=".8"/><text x="${P.l + 4}" y="${Y(f * o.aar_vfr) - 4}" style="fill:${ink}">${l}</text>`);
  const dots = o.hours.map(h => `<circle cx="${X(h.hour)}" cy="${Y(h.pred)}" r="3.4" fill="#fff" stroke="var(--sky)" stroke-width="2"><title>${hr(h.hour)} · forecast ${h.pred} (likely ${h.lo}–${h.hi})${S.reveal ? ` · really ${h.actual}` : ""}</title></circle>`).join("");
  const legend = [["var(--ink-3)", "Past 24 hours"], ["var(--sky)", "Forecast"], ["rgba(0,133,202,.25)", "Likely range", true]]
    .concat(S.reveal ? [["var(--ink)", "What really happened"]] : []).map(([c, l, box]) => `<span><i class="${box ? "box" : ""}" style="background:${c}"></i>${l}</span>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Planes landing per hour: the last 24 hours and the forecast for the next 12">
      ${grid.join("")}${thr.join("")}
      <line x1="${X(o.t0)}" x2="${X(o.t0)}" y1="${P.t}" y2="${Hh - P.b}" stroke="var(--navy)" stroke-dasharray="2 3" opacity=".5"/>
      <path d="${path(band)}Z" fill="rgba(0,133,202,.14)"/>
      <path d="${path(o.hist.map(h => [h.hour, h.aar]))}" fill="none" stroke="var(--ink-3)" stroke-width="2"/>
      ${S.reveal ? `<path d="${path(o.hours.map(h => [h.hour, h.actual]))}" fill="none" stroke="var(--ink)" stroke-width="2"/>` : ""}
      <path d="${path(o.hours.map(h => [h.hour, h.pred]))}" fill="none" stroke="var(--sky)" stroke-width="2.6"/>
      ${dots}${ticks.join("")}
    </svg><div class="legend label">${legend}</div>`;
}

// ---------- flights ----------
function flightsView() {
  const o = S.out; if (!o) return loading("flights", "flights");
  const a = airport(), base = S.meta.base_rate;
  const mean = o.flights.length ? o.flights.reduce((s, f) => s + f.p, 0) / o.flights.length : 0;
  const drv = {}; o.flights.forEach(f => reasons(f).forEach(n => { drv[n] = (drv[n] || 0) + 1; }));
  const drivers = Object.entries(drv).sort((x, y) => y[1] - x[1]).slice(0, 8), dTop = Math.max(1, ...drivers.map(d => d[1]));
  return `<div class="page scroll" data-view="flights">
    <header class="page-h"><div><h1>Flights</h1><p class="lede">A sample of real ${city(a)} flights scheduled 3 to 12 hours after ${clock(o.t0)} on ${dayL(o.t0)}. Each percentage is the chance of a delay of 45 minutes or more.</p></div>
      <div class="page-stats"><div><div class="readout">${o.n_flights}</div><div class="label">Sampled flights</div></div>
        <div><div class="readout">${o.n_high}</div><div class="label">At risk (${pct(S.thr)}%+ chance)</div></div>
        <div><div class="readout">${pct(mean)}<small>%</small></div><div class="label">Average chance, usually ${pct(base)}%</div></div></div></header>
    <div class="ins-grid flights-grid">
      <section class="glass card"><div class="ph">Every flight<span class="meta">highest chance first</span></div>
        <div class="scroll tablewrap flight-list"><table class="evtab stack"><thead><tr><th>Time</th><th>Flight</th><th>Chance</th><th>Main reasons</th>${S.reveal ? "<th>What happened</th>" : ""}</tr></thead><tbody>
        ${o.flights.map(f => `<tr${S.sel === f.id ? ' class="cur"' : ""}><td class="mono">${clock(f.t)}</td><td>${flightName(f)}</td>
          <td class="mono"><span class="mb"><i class="${f.p >= S.thr ? "bad" : f.p >= 0.2 ? "accent" : ""}" style="width:${Math.min(100, f.p * 200).toFixed(0)}%"></i></span>${pct(f.p)}%</td>
          <td>${esc(reasons(f).join(", ")) || "—"}</td>${S.reveal ? `<td class="mono">${outcome(f)}</td>` : ""}</tr>`).join("")}
        </tbody></table></div></section>
      <section class="glass card"><div class="ph">Most common reasons<span class="meta">flights listing it</span></div>
        <div class="m-bars">${drivers.length ? drivers.map(([n, c]) => `<div><span class="label">${esc(n)}</span><span class="mb"><i class="accent" style="width:${(c / dTop * 100).toFixed(0)}%"></i></span><span class="mono">${c}</span></div>`).join("") : '<p class="empty-note">No flights in this window.</p>'}</div></section>
    </div></div>`;
}

// ---------- past 30 days ----------
function historyView() {
  const h = S.hist; if (!h) return loading("history", "the past 30 days");
  const low = h.daily.reduce((s, d) => s + d.low, 0), days = h.daily.filter(d => d.low).length;
  const obs = h.delay.length ? h.delay.reduce((s, d) => s + d.obs * d.n, 0) / Math.max(1, h.delay.reduce((s, d) => s + d.n, 0)) : null;
  const top = Math.max(6, ...h.daily.map(d => d.low)), hm = Math.max(0.05, ...h.heat.flat().filter(v => v != null));
  const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  // the grids are pictures to screen readers (their numbers live in hover titles), so each gets a one-line summary
  const worst = h.daily.reduce((x, d) => (d.low > x.low ? d : x), h.daily[0] || { low: 0 });
  const hot = h.heat.flatMap((row, d) => row.map((v, hh) => [v, d, hh])).filter(x => x[0]).sort((x, y) => y[0] - x[0]).slice(0, 3);
  return `<div class="page scroll" data-view="history">
    <header class="page-h"><div><h1>Past 30 days</h1><p class="lede">${CITY[h.iata] || h.iata} in the 30 days before ${dayL(h.t0)}, and the times of day when landings are usually cut, across January to July 2025.</p></div>
      <div class="page-stats"><div><div class="readout">${low}</div><div class="label">Hours with fewer landings</div></div>
        <div><div class="readout">${days}</div><div class="label">Days affected</div></div>
        <div><div class="readout">${obs == null ? "—" : pct(obs)}<small>%</small></div><div class="label">Flights 45+ min late</div></div></div></header>
    <div class="ins-grid">
      <section class="glass card"><div class="ph">Hours with fewer landings, each day</div>
        <div class="land" style="grid-template-columns:repeat(${h.daily.length},minmax(0,1fr))" role="img" aria-label="${low} hours with 20%+ fewer landings${worst.low ? `. Worst day: ${day(worst.day)}, ${worst.low} hours` : ""}">${h.daily.map(d => `<div class="land-cell" title="${day(d.day)} · ${d.low} hours with 20%+ fewer landings"><span class="bar ${d.low >= 8 ? "severe" : d.low ? "moderate" : ""}" style="height:${(d.low / top * 100).toFixed(0)}%"></span></div>`).join("")}
          ${h.daily.map((d, i) => `<span class="label land-y mono">${i % 7 ? "" : day(d.day)}</span>`).join("")}</div>
        <div class="legend label"><span><i class="box moderate"></i>Some hours</span><span><i class="box severe"></i>8 hours or more</span></div></section>
      <section class="glass card"><div class="ph">Flights 45+ min late, each day</div>${delayChart(h.delay)}</section>
      <section class="glass card span2"><div class="ph">When are landings usually cut?<span class="meta">local time · Jan to Jul 2025</span></div>
        <div class="heatgrid" role="img" aria-label="${hot.length ? `Share of hours with 20%+ fewer landings, highest: ${hot.map(([v, d, hh]) => `${DOW[d]} ${String(hh).padStart(2, "0")}:00, ${pct(v)}%`).join("; ")}` : "No cuts on record"}"><span></span>${Array.from({ length: 24 }, (_, i) => `<span class="label mono land-y">${i % 3 ? "" : String(i).padStart(2, "0")}</span>`).join("")}
          ${h.heat.map((row, d) => `<span class="label">${DOW[d]}</span>${row.map((v, hh) => `<i title="${DOW[d]} ${String(hh).padStart(2, "0")}:00 · ${v == null ? "no data" : `${pct(v)}% of these hours had 20%+ fewer landings`}" ${v == null ? "" : ` style="--v:${(v / hm).toFixed(2)}"`}></i>`).join("")}`).join("")}
        </div>
        <div class="legend label"><span>Rarely cut<i class="ramp"></i>Often cut</span></div></section>
    </div></div>`;
}
function delayChart(rows) {
  if (!rows.length) return '<p class="empty-note">No sampled flights in this period.</p>';
  const W = innerWidth < 600 ? 340 : 560, Hh = 220, P = { l: 34, r: 10, t: 10, b: 22 }, n = rows.length;   // narrow viewBox on phones keeps axis text near 11px
  const yMax = Math.max(0.3, ...rows.map(r => Math.max(r.obs, r.pred))), X = i => P.l + i / Math.max(1, n - 1) * (W - P.l - P.r), Y = v => P.t + (1 - v / yMax) * (Hh - P.t - P.b);
  const path = k => rows.map((r, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(r[k]).toFixed(1)}`).join("");
  const grid = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].filter(v => v <= yMax).map(v => `<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 6}" y="${Y(v) + 4}" text-anchor="end">${pct(v)}%</text>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Share of flights 45 or more minutes late each day, real and predicted">${grid}
      <path d="${path("obs")}" fill="none" stroke="var(--ink)" stroke-width="1.8"/>${rows.map((r, i) => `<circle cx="${X(i)}" cy="${Y(r.obs)}" r="2.6" fill="var(--ink)"><title>${day(r.day)} · ${pct(r.obs)}% of ${r.n} flights</title></circle>`).join("")}
      <path d="${path("pred")}" fill="none" stroke="var(--sky)" stroke-width="2.6"/>
      ${rows.map((r, i) => i % 7 ? "" : `<text x="${X(i)}" y="${Hh - 5}" text-anchor="middle">${day(r.day)}</text>`).join("")}
    </svg><div class="legend label"><span><i style="background:var(--ink)"></i>Real share</span><span><i style="background:var(--sky)"></i>Flowcast's prediction, 3 hours ahead</span></div>`;
}

// ---------- accuracy: plain answers first, the full evaluation (incl. storm hours) behind "details" ----------
function evalView() {
  const e = S.ev, s = S.st, a = airport();
  const leads = `<div class="lead-pick"><span class="label">Forecast made</span><span class="seg" role="group" aria-label="Hours ahead">${[3, 6, 9, 12].map(l => `<button type="button" data-lead="${l}" aria-pressed="${S.lead === l}">${l} h</button>`).join("")}</span><span class="label">ahead</span></div>`;
  const head = `<header class="page-h"><div><h1>Accuracy</h1><p class="lede">How Flowcast did on January to July 2025, data the models never saw while they were built. Each result is compared with simpler ways of forecasting.</p></div>${leads}</header>`;
  if (!e) return loading("eval", "accuracy results", head);
  const ours = e.aar[0], d = e.delay, st = s?.arr;
  const NAME = { lightgbm: "Flowcast", taf_implied: "Weather-forecast rule", persistence: "No change", climatology: "Typical for the hour" };
  const cmp = k => e.aar.map(r => [NAME[r.model] || r.label, r[k], r.model === "lightgbm" ? "good" : ""]);
  const limits = ["It cannot foresee long disruptions that give no weather warning, such as technical faults.",
    "In the very worst slowdowns it does no better than the official weather-forecast rule.",
    "The landing rates it learns from are estimated from the weather using FAA rates, not taken from control-tower logs.",
    "The flight data is a sample of about 3% of traffic, and airline data covers reporting carriers only.",
    ...(s ? [`Storms are rare: ${s.hours} storm hours at ${s.iata} in the test period, so storm results are less certain.`,
             "In storms, its delay chances run well below what really happens. Treat any rise as significant."] : [])];
  // failure cases exist only at the two tuning leads; show two of each kind for this airport
  const fl = e.failures || [], fails = [...fl.filter(f => f.kind === "missed").slice(0, 2), ...fl.filter(f => f.kind !== "missed").slice(0, 2)].map(f => `<div class="fail"><p class="q">${esc(AIRLINES[f.carrier] || f.carrier)} ${f.dir === "arr" ? "from" : "to"} <span class="mono">${esc(f.other)}</span> at ${esc(f.hub)}</p>
      <p class="label">${dayL(f.t, zone(f.hub))}</p><p>${f.kind === "missed" ? `${n0(f.delay_min)} min late, but Flowcast gave it only ${pct(f.p)}%.` : `Flagged at ${pct(f.p)}%, but ${f.delay_min <= 0 ? "it was on time" : `only ${n0(f.delay_min)} min late`}.`}</p></div>`).join("");
  return `<div class="page scroll" data-view="eval">${head}
    <div class="metrics">
      ${metric("How far off is the landings forecast?", ours.rmse.toFixed(1), "/hr", bars(cmp("rmse"), 1), "The typical miss, in landings per hour. Shorter bars are better.")}
      ${metric("Does it see slowdowns coming?", ours.pr_auc.toFixed(2), "of 1", bars(cmp("pr_auc"), 2), `A 0-to-1 score for flagging hours with 20%+ fewer landings. Guessing scores ${e.event_rate.toFixed(2)}.`)}
      ${metric("Can it spot flights that will be late?", d.full.toFixed(2), "of 1", bars([["Flowcast", d.full, "good"], ["Schedule only", d.sched], ["Guessing", d.chance]], 2), "The same kind of score, for delays of 45+ minutes. Longer bars are better.")}
      ${metric("Is its likely range honest?", pct(e.band), "%", bars([["Real value inside", e.band, e.band >= 0.8 ? "good" : "bad"], ["Promised", 0.8]], "%"), "How often the real number fell inside the range. Over 80% means it errs on the safe side.")}
    </div>
    <div class="ins-grid">
      ${st ? `<section class="glass card storm"><div class="ph">${I.storm}In storms, trust it less</div>
        <div class="readout t-stop">${(st.model / st.all_hours).toFixed(1)}<small>× bigger misses</small></div>
        <p>In ${s.hours} storm hours at ${s.iata}, arrivals forecasts missed by about ${st.model.toFixed(0)} an hour, against ${st.all_hours.toFixed(0)} in normal hours. Their 90% range held the real number only ${pct(st.cover)}% of the time. Read a storm-hour forecast as an early warning, not an exact count.</p>
        <p class="label">From a separate storm check that scores real arrivals with its own 90% range, so these numbers do not compare directly with the cards above.</p></section>` : ""}
      <section class="glass card${st ? "" : " span2"}"><div class="ph">Where it goes wrong</div><ul class="limits">${limits.map(l => `<li>${l}</li>`).join("")}</ul></section>
      <section class="glass card span2"><div class="ph">Biggest misses<span class="meta">the model's worst calls, shown on purpose</span></div>${fails ? `<div class="fails">${fails}</div>` : '<p class="empty-note">Worst calls are listed for forecasts made 3 or 12 hours ahead. Pick one of those above.</p>'}</section>
      ${more("tech", "Technical results for specialists", techResults(e, s, a), "glass card span2")}
    </div></div>`;
}
function techResults(e, s, a) {
  const d = e.delay, tr = e.traffic, now = s ? s.delay.filter(x => x.lead === S.lead) : [], st = s?.arr;
  const sg = v => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
  const tone = v => v === "model better" ? "good" : v === "model worse" ? "bad" : "accent";
  const pill = v => `<span class="pill ${v === "model better" ? "ok" : v === "model worse" ? "stop" : "warn"}">${esc(v)}</span>`;
  const sec = (title, meta, body, cls = "") => `<section class="tsec ${cls}"><div class="ph">${title}${meta ? `<span class="meta">${meta}</span>` : ""}</div>${body}</section>`;
  return `<p class="label">Metric names: the landings error is RMSE (root mean square error) of the arrival acceptance rate (AAR); the 0-to-1 scores are PR-AUC (area under the precision-recall curve).
      The delay model is a calibrated logistic regression, picked out-of-fold on 2024; LightGBM scored ${d.lightgbm.toFixed(3)} at +${e.lead} h.</p>
    <div class="tgrid">
      ${e.tune ? sec("What each data source adds to delay risk", `PR-AUC · +${e.lead} h`, `${bars(e.ladder.map(x => [x.label, x.pr_auc, x.final ? "good" : "accent"]), 3)}
          <div class="label">Gain over schedule-only +${e.boot.gain.toFixed(3)}, day-block bootstrap 95% interval ${e.boot.lo.toFixed(3)} to ${e.boot.hi.toFixed(3)}. Weather earns its place; BTS airport traffic adds little.</div>`)
        + sec("Is the risk well calibrated?", `10 risk groups · +${e.lead} h`, relChart(e.reliability))
        : `<p class="empty-note span2">The data-source ladder and calibration are reported at +3 h and +12 h, the two tuning leads.</p>`}
      ${tr ? sec(`Does the AAR rule match real ${a.iata} traffic?`, "busy hours · BTS", `${bars(tr.bands.map(b => [b.label, b.share, "accent"]), "%")}
          <div class="label">Share of scheduled arrivals that actually landed, by the rule's AAR band. Rank correlation ${tr.spearman}: the direction holds, the effect is modest.</div>`) : ""}
      ${st ? sec(`Arrivals error in storm hours · ${s.hours} hours`, "RMSE, real arrivals", `${bars([["Model", st.model, "good"], ["Climatology", st.clim], ["Schedule", st.sched], ["Persistence", st.persist]], 1)}
          <div class="label">Storm hours: a NOAA Storm Event (thunderstorm wind, hail, tornado, flash flood) within 50 km. NOAA logs only mark which test hours were storms; they never feed a model. Scored on the full BTS archive (3.26 M flights). Normal hours ${st.all_hours.toFixed(1)}/h.</div>`)
        + sec("90% range: outcomes inside", "storm hours", bars([["Storm hours", st.cover, "bad"], ["All hours", st.cover_all, "good"], ["Promised", 0.9]], "%"))
        + now.map(x => sec(`Severe-delay ranking in storms · ${x.dir}s`, "PR-AUC", `${bars([["Model", x.model, tone(x.verdict)], ["Climatology", x.clim], ["Severe share", x.rate], ["Mean predicted", x.mean_p, "bad"]], 3)}
          <div class="label">${x.flights} flights in ${x.hours} storm hours. Gain 95% interval ${sg(x.lo)} to ${sg(x.hi)}: ${x.verdict}.</div>`)).join("")
        + sec("Arrivals error by lead time", "RMSE, arrivals per hour", stormLeadChart(s.by_lead))
        + sec("Storm-hour delay ranking vs climatology", "all leads · 95% interval", `<div class="tablewrap"><table class="evtab"><thead><tr><th>Type</th><th>Lead</th><th>Flights</th><th>Model</th><th>Climatology</th><th>Gain, 95% interval</th><th>Verdict</th></tr></thead><tbody>
          ${s.delay.map(x => `<tr${x.lead === S.lead ? ' class="cur"' : ""}><td>${x.dir === "arrival" ? "Arrival" : "Departure"}</td><td class="mono">+${x.lead} h</td><td class="mono">${x.flights}</td>
            <td class="mono">${x.model.toFixed(3)}</td><td class="mono">${x.clim.toFixed(3)}</td><td class="mono">${sg(x.lo)} to ${sg(x.hi)}</td><td>${pill(x.verdict)}</td></tr>`).join("")}</tbody></table></div>`, "span2") : ""}
      ${s?.fullbts?.length ? sec("Same classifier on the full BTS archive, all 2025 hours", `+${s.lead} h · LightGBM`, `<div class="tablewrap"><table class="evtab"><thead><tr><th>Type</th><th>Test flights</th><th>Severe share</th><th>PR-AUC model</th><th>PR-AUC climatology</th><th>Precision</th><th>Recall</th></tr></thead><tbody>
          ${s.fullbts.map(f => `<tr><td>${f.dir === "arrival" ? "Arrival" : "Departure"}</td><td class="mono">${n0(f.n)}</td><td class="mono">${pct(f.rate)}%</td><td class="mono">${f.model.toFixed(3)}</td>
            <td class="mono">${f.clim.toFixed(3)}</td><td class="mono">${f.precision.toFixed(3)}</td><td class="mono">${f.recall.toFixed(3)}</td></tr>`).join("")}</tbody></table></div>
          <div class="label">Trained on 2023-01 to 2024-06, threshold chosen on 2024 H2, scored once on 2025. About 45,000 test flights per row, so these intervals are tight.</div>`, "span2") : ""}
      ${s ? `<p class="label span2">A TAF thunderstorm-timing feature (thunder within 2 h of the target hour) was tested and rejected on 2024 validation data. Storm results come from jho/aerocast_data (run_all.py), copied into reports/jho_*.csv; their target is real arrivals, so their RMSE is not comparable with the AAR RMSE above.</p>` : ""}
    </div>`;
}
function relChart(rows) {
  const W = innerWidth < 600 ? 340 : 420, Hh = 260, P = { l: 38, r: 12, t: 10, b: 26 }, top = Math.max(0.1, ...rows.map(r => Math.max(r.pred, r.obs))) * 1.08;
  const X = v => P.l + v / top * (W - P.l - P.r), Y = v => P.t + (1 - v / top) * (Hh - P.t - P.b);
  const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5].filter(v => v <= top);
  return `<svg class="chart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Predicted against observed severe-delay rate">
      ${ticks.map(v => `<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 6}" y="${Y(v) + 4}" text-anchor="end">${pct(v)}%</text><text x="${X(v)}" y="${Hh - 8}" text-anchor="middle">${pct(v)}%</text>`).join("")}
      <line x1="${X(0)}" y1="${Y(0)}" x2="${X(top)}" y2="${Y(top)}" stroke="var(--ink-3)" stroke-dasharray="4 4"/>
      <path d="${rows.map((r, i) => `${i ? "L" : "M"}${X(r.pred).toFixed(1)} ${Y(r.obs).toFixed(1)}`).join("")}" fill="none" stroke="var(--sky)" stroke-width="2.6"/>
      ${rows.map(r => `<circle cx="${X(r.pred)}" cy="${Y(r.obs)}" r="3.4" fill="#fff" stroke="var(--sky)" stroke-width="2"><title>predicted ${pct(r.pred)}% · observed ${pct(r.obs)}% · ${r.n} flights</title></circle>`).join("")}
    </svg><div class="legend label"><span><i style="background:var(--sky)"></i>Model, predicted vs observed</span><span><i style="background:var(--ink-3)"></i>Perfect calibration</span></div>`;
}
function stormLeadChart(rows) {
  const W = innerWidth < 600 ? 340 : 520, Hh = 240, P = { l: 34, r: 10, t: 12, b: 26 }, top = Math.ceil(Math.max(...rows.flatMap(r => [r.storm, r.clim, r.all])) / 5) * 5;
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
  { title: "Welcome to Flowcast", text: "Flowcast predicts when bad weather will cut the number of planes that can land at Atlanta and Dallas–Fort Worth, 3 to 12 hours ahead. This is a replay of 2025: pick a moment, see what it would have said, then compare with what really happened." },
  { sel: '[data-tour="answer"]', view: "outlook", title: "The answer", text: "Blue means normal, amber means fewer landings, red means a big slowdown. Each bar is one hour, 3 to 12 hours ahead." },
  { sel: '[data-tour="ask"]', view: "outlook", title: "Pick an airport and a time", text: "The forecast only uses weather reports and flight data that existed at that moment." },
  { sel: '[data-tour="chips"]', view: "outlook", title: "Or replay a bad day", text: "The biggest slowdowns of 2025. Each one starts the forecast 6 hours before it happened." },
  { sel: '[data-tour="weather"]', view: "outlook", title: "Weather right now", text: "The latest airport weather report. The 3D scene shows it too: wind, cloud height, rain and lightning." },
  { sel: '[data-tab="eval"]', title: "Checked, not just claimed", text: "See how accurate the forecasts were on data the models never saw, including where they go wrong." },
];
let tourFrom = null;   // focus goes back here when the tour closes
function openTour(i = 0) {
  if (!$("#overlay .tour")) tourFrom = document.activeElement;
  const s = STEPS[i]; if (s.view && S.view !== s.view) go(s.view);
  setTimeout(() => {
    const el = s.sel ? $(s.sel) : null; el?.scrollIntoView({ block: "nearest" });
    const box = el?.getBoundingClientRect(), last = i === STEPS.length - 1, cw = Math.min(360, innerWidth - 32);
    const pos = box ? `left:${Math.max(16, Math.min(innerWidth - cw - 16, box.left))}px;top:${box.bottom + 16 > innerHeight - 230 ? Math.max(16, box.top - 230) : box.bottom + 16}px` : "";
    $("#overlay").innerHTML = `<div class="tour" role="dialog" aria-modal="true" aria-labelledby="tour-t">
      ${box ? `<div class="tour-ring" style="left:${box.left - 6}px;top:${box.top - 6}px;width:${box.width + 12}px;height:${box.height + 12}px"></div>` : ""}
      <div class="tour-card glass strong${box ? " placed" : ""}" style="${pos}"><div class="label">Step ${i + 1} of ${STEPS.length}</div><h3 id="tour-t">${s.title}</h3><p>${s.text}</p>
        <div class="tour-acts"><button class="glass-btn" data-tour-close>Skip</button><span style="flex:1"></span>
          ${i ? `<button class="glass-btn" data-tour-step="${i - 1}">Back</button>` : ""}<button class="glass-btn pri" ${last ? "data-tour-close" : `data-tour-step="${i + 1}"`}>${last ? "Start" : "Next"}</button></div></div></div>`;
    $("#overlay .tour-acts .pri")?.focus();
  }, 60);
}
function closeTour() {
  $("#overlay").innerHTML = ""; store.set("cr_tour_v2", "1");
  (tourFrom?.isConnected && tourFrom !== document.body ? tourFrom : $("[data-tour-open]"))?.focus();   // re-renders can detach the opener
}

// ---------- events ----------
const toAnswer = () => { if (innerWidth < 860) $("#answer-top")?.scrollIntoView({ behavior: "smooth", block: "start" }); };
document.addEventListener("click", e => {
  const t = e.target.closest("[data-go],[data-tour-open],[data-tour-step],[data-tour-close],[data-ev],[data-sens],[data-reveal],[data-cam],[data-flight],[data-lead],[data-explain],[data-scroll],[data-retry],[data-chips-more]");
  if (!t) return;
  const d = t.dataset;
  if (d.go) { e.preventDefault(); go(d.go); }
  else if ("tourOpen" in d) openTour();
  else if (d.tourStep) openTour(+d.tourStep);
  else if ("tourClose" in d) closeTour();
  else if (d.ev) { const ev = S.meta.events[+d.ev]; S.icao = ev.icao; S.t0 = ev.t0; forecast().then(toAnswer); }
  else if (d.sens) { S.sens = d.sens; forecast(); }
  else if ("reveal" in d) { S.reveal = !S.reveal; render(); }
  else if (d.cam) { S.cam = d.cam; stage?.setView(d.cam); document.querySelectorAll("[data-cam]").forEach(b => b.setAttribute("aria-pressed", b.dataset.cam === d.cam)); }
  else if (d.scroll) $(`#${d.scroll}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  else if (d.lead) { S.lead = +d.lead; S.ev = null; S.st = null; S.fail = null; render(); ensure(S.view); }
  else if ("explain" in d) {
    t.disabled = true; t.textContent = "Asking Claude…";
    api("explain", { airport: S.icao, t0: S.t0, sens: S.sens, thr: S.thr }).then(r => { S.expl = r.text; render(); })
      .catch(err => { toast(err.message); t.disabled = false; t.textContent = "Explain this forecast in plain English"; });
  }
  else if (d.flight) { S.sel = +d.flight; go("flights"); }
  else if ("retry" in d) { S.fail = S.err = null; render(); S.out ? ensure(S.view) : forecast(); }
  else if ("chipsMore" in d) { const c = t.parentElement; c.scrollTo({ left: c.classList.contains("end") ? 0 : c.scrollLeft + c.clientWidth * 0.8, behavior: "smooth" }); }
});
document.addEventListener("submit", e => {
  if (!e.target.matches(".askbar")) return;
  e.preventDefault(); const f = new FormData(e.target);
  S.icao = f.get("airport"); S.t0 = `${f.get("date")}T${String(f.get("hour")).padStart(2, "0")}:00Z`; forecast().then(toAnswer);
});
document.addEventListener("change", e => {
  if (e.target.matches('.askbar [name="airport"]')) { S.icao = e.target.value; forecast().then(toAnswer); }
  else if (e.target.matches("[data-thr]")) { S.thr = +e.target.value; forecast(); }
});
document.addEventListener("toggle", e => { const k = e.target.dataset?.k; if (k) S.open[e.target.open ? "add" : "delete"](k); }, true);   // keep details open across re-renders
// hour bars: hover and keyboard focus both light that hour on the 3D ring
const hot = (el, on) => { stage?.highlight(on ? +el.dataset.hour : null); el.classList.toggle("hot", on); };
document.addEventListener("mouseover", e => { const el = e.target.closest("[data-hour]"); if (el) hot(el, true); });
document.addEventListener("mouseout", e => { const el = e.target.closest("[data-hour]"); if (el && !el.contains(e.relatedTarget)) hot(el, false); });
document.addEventListener("focusin", e => { if (e.target.matches?.("[data-hour]")) hot(e.target, true); });
document.addEventListener("focusout", e => { if (e.target.matches?.("[data-hour]")) hot(e.target, false); });
document.addEventListener("keydown", e => {
  const tour = $("#overlay .tour");
  if (tour && e.key === "Escape") closeTour();
  else if (tour && e.key === "Tab") {   // keep focus inside the tour card
    const b = [...tour.querySelectorAll("button")], i = b.indexOf(document.activeElement);
    if (i < 0 || (e.shiftKey ? i === 0 : i === b.length - 1)) { e.preventDefault(); b.at(e.shiftKey ? -1 : 0).focus(); }
  }
  else if (e.target.matches?.("[data-hour]") && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {   // roving tab stop: Tab returns to the last hour visited
    const n = e.target[e.key === "ArrowRight" ? "nextElementSibling" : "previousElementSibling"];
    if (n) { e.target.tabIndex = -1; n.tabIndex = 0; n.focus(); }
  }
});
document.addEventListener("scroll", e => { if (e.target.classList?.contains("chips")) chipCue(e.target); }, true);
addEventListener("resize", () => chipCue());
addEventListener("hashchange", () => { const h = location.hash.replace(/^#\/?/, "").replace("storms", "eval"); if (VIEWS.some(v => v[0] === h) && h !== S.view) go(h); });
function paintAskButton() { const b = $(".askbar .glass-btn.pri"); if (b) { b.disabled = S.busy; b.textContent = S.busy ? "Loading…" : "Show forecast"; } }

// ---------- boot ----------
(async () => {
  const noTour = new URLSearchParams(location.search).has("notour");
  // screen-reader announcements; made here, not in index.html, so a cached page can never lack them
  document.body.insertAdjacentHTML("beforeend", '<div id="live" class="sr-only" aria-live="polite"></div><div id="alerts" class="sr-only" role="alert"></div>');
  try { S.meta = await api("meta"); } catch (e) {
    $("#view").innerHTML = `<div class="page"><div class="empty-note" role="alert"><p>Flowcast could not load its data. ${esc(e.message)}</p>
      <p>Reload the page to try again. Running it yourself? Start the server with <span class="mono">python dashboard/server.py</span>.</p></div></div>`;
    return;
  }
  readUrl(); render();
  // the 3D stage is decoration: if WebGL or the CDN is unavailable the dashboard still works
  import("./stage.js").then(m => { stage = m.createStage($("#stage canvas"), $("#stage"), { zoom: innerWidth < 860 ? 4.4 : 3.3 }); stage.setActive(S.view === "outlook"); syncStage(false); })
    .catch(err => console.warn("3D stage unavailable:", err));
  await forecast(); ensure(S.view);
  if (!store.get("cr_tour_v2") && !noTour) openTour();
})();
