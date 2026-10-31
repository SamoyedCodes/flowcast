/* Capacity & Risk stage: the airfield, its latest METAR and the next hours of capacity, drawn with three.js.
   Adapted from Torque Stripe's inspection stage (MIT, (c) 2026 Aljabri Alam, github.com/aljabrialam/torque-stripe):
   the materials, lighting, lathe/panel helpers, narrow-body silhouette, camera and loop come from there.
   Decoration only: it shows the forecast and the weather, it never makes either. Not to scale. */
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import FIELDS from "./airfields.js";   // real runways, taxiways, aprons, terminals (OpenStreetMap), see dashboard/build_airfields.py

const GROUND = -2.2, RR = 12.6, BOX = 17, FIT = 11.4;   // FIT: farthest runway end, just inside the capacity ring

export function createStage(cv, host, opts = {}) {
  const T = THREE, reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888";
  const rnd = (a, b) => a + Math.random() * (b - a);
  let zoom = opts.zoom ?? 3.3, spinSpeed = opts.spinSpeed ?? 0.035, lookY = opts.lookY ?? -2.4;
  const renderer = new T.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 0.92;
  const scene = new T.Scene(), cam = new T.PerspectiveCamera(32, 1, 0.1, 300);
  const pmrem = new T.PMREMGenerator(renderer); scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  const hemi = new T.HemisphereLight(0xeef3ff, 0x1a1d22, 0.55); scene.add(hemi);
  const key = new T.DirectionalLight(0xffffff, 2.1); key.position.set(-6, 7, 5); scene.add(key);
  const rim = new T.DirectionalLight(0xa8ccff, 1.5); rim.position.set(6, 2.5, -7); scene.add(rim);
  const fill = new T.DirectionalLight(0xc9b8a4, 0.5); fill.position.set(2, -3, 6); scene.add(fill);
  scene.fog = new T.Fog(0xe6eef8, 30, 80);

  let solid = true; const solids = [];
  const M = { wire: new T.LineBasicMaterial({ transparent: true, opacity: 0.72 }), dim: new T.LineBasicMaterial({ transparent: true, opacity: 0.55 }),
              edge: new T.LineBasicMaterial({ transparent: true, opacity: 0.4 }), floor: new T.LineBasicMaterial({ transparent: true, opacity: 0.5 }),
              mark: new T.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95 }),
              lights: new T.PointsMaterial({ size: 0.07, transparent: true, opacity: 0.9 }),
              scan: new T.LineBasicMaterial({ transparent: true, opacity: 0 }),
              scanFill: new T.MeshBasicMaterial({ transparent: true, opacity: 0, side: T.DoubleSide, depthWrite: false }),
              wind: new T.LineBasicMaterial({ transparent: true, opacity: 0.5 }), rain: new T.LineBasicMaterial({ transparent: true, opacity: 0.4 }),
              snow: new T.PointsMaterial({ size: 0.13, transparent: true, opacity: 0.95, depthWrite: false }),
              bolt: new T.LineBasicMaterial({ transparent: true, opacity: 0 }),
              hull: new T.MeshPhysicalMaterial({ color: 0x98a3b3, metalness: 0.6, roughness: 0.34, clearcoat: 0.5, clearcoatRoughness: 0.25, envMapIntensity: 0.9, side: T.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
              hullDim: new T.MeshPhysicalMaterial({ color: 0x6f747a, metalness: 0.4, roughness: 0.5, envMapIntensity: 0.6, side: T.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
              fem: new T.MeshBasicMaterial({ color: 0x00205b, wireframe: true, transparent: true, opacity: 0.035, depthWrite: false }),
              intake: new T.MeshStandardMaterial({ color: 0x120c08, metalness: 0.2, roughness: 0.8, side: T.DoubleSide }),
              asphalt: new T.MeshStandardMaterial({ color: 0x8796ab, roughness: 0.92, metalness: 0.05, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
              apron: new T.MeshStandardMaterial({ color: 0xb4bfcd, roughness: 0.95, metalness: 0.02, transparent: true, opacity: 0.7, depthWrite: false }),
              bldg: new T.MeshStandardMaterial({ color: 0xe8ecf2, roughness: 0.55, metalness: 0.15, envMapIntensity: 0.8, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
              taxi: new T.LineBasicMaterial({ transparent: true, opacity: 0.3 }) };
  const skin = (geo, mat) => { const m = new T.Mesh(geo, mat === M.dim ? M.hullDim : M.hull); m.add(new T.Mesh(geo, M.fem)); m.visible = solid; solids.push(m); return m; };
  function applyRender() { M.wire.opacity = solid ? 0.07 : 0.72; M.dim.opacity = solid ? 0.05 : 0.55; M.edge.opacity = solid ? 0.35 : 0.85; solids.forEach(m => m.visible = solid); }
  function colors() {
    M.wire.color.set(css("--wire")); M.dim.color.set(css("--wire-dim")); M.edge.color.set(css("--wire")); M.floor.color.set(css("--wire-dim")); M.taxi.color.set(css("--wire"));
    M.wind.color.set(css("--sky")); M.rain.color.set(css("--wire")); M.snow.color.set(css("--wire-dim")); M.bolt.color.set(css("--sky")); M.lights.color.set(css("--sky"));
    M.scan.color.set(css("--sky")); M.scanFill.color.set(css("--sky")); scene.fog.color.set(css("--stage")); paintRing(); applyRender();
  }

  // ---------- geometry helpers (from Torque Stripe) ----------
  const seg = (arr, mat) => { const g = new T.BufferGeometry(); g.setAttribute("position", new T.Float32BufferAttribute(arr, 3)); return new T.LineSegments(g, mat || M.wire); };
  function lathe(profile, long = 16, ringEvery = 1, mat) {   // profile: [[x, r], ...] along X
    const a = [];
    for (let k = 0; k < long; k++) { const t = k / long * Math.PI * 2, c = Math.cos(t), s = Math.sin(t);
      for (let i = 0; i < profile.length - 1; i++) { const [x0, r0] = profile[i], [x1, r1] = profile[i + 1]; a.push(x0, r0 * c, r0 * s, x1, r1 * c, r1 * s); } }
    profile.forEach(([x, r], i) => { if (i % ringEvery || r < 0.02) return;
      for (let k = 0; k < 40; k++) { const t0 = k / 40 * Math.PI * 2, t1 = (k + 1) / 40 * Math.PI * 2; a.push(x, r * Math.cos(t0), r * Math.sin(t0), x, r * Math.cos(t1), r * Math.sin(t1)); } });
    const curve = new T.CatmullRomCurve3(profile.map(([x, r]) => new T.Vector3(x, r, 0)), false, "centripetal");
    const pts = curve.getPoints(Math.max(24, profile.length * 10)).map(v => new T.Vector2(Math.max(v.y, 0.001), v.x));
    const geo = new T.LatheGeometry(pts, Math.max(long, 24) * 2); geo.rotateZ(-Math.PI / 2);
    const g = new T.Group(); g.add(skin(geo, mat), seg(a, mat)); return g;
  }
  function panel(pts, ribs = 0, mat) {   // closed quad of [x,y,z] + optional ribs between edge 0-1 and 3-2
    const a = []; pts.forEach((p, i) => { const q = pts[(i + 1) % pts.length]; a.push(...p, ...q); });
    for (let i = 1; i <= ribs; i++) { const t = i / (ribs + 1), L = (p, q) => p.map((v, j) => v + (q[j] - v) * t); a.push(...L(pts[0], pts[1]), ...L(pts[3], pts[2])); }
    const N = 10, pos = [], idx = [], L = (p, q, t) => p.map((v, j) => v + (q[j] - v) * t);
    for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) pos.push(...L(L(pts[0], pts[1], i / N), L(pts[3], pts[2], i / N), j / N));
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const a0 = j * (N + 1) + i, b0 = a0 + 1, c0 = a0 + N + 1, d0 = c0 + 1; idx.push(a0, b0, d0, a0, d0, c0); }
    const geo = new T.BufferGeometry(); geo.setAttribute("position", new T.Float32BufferAttribute(pos, 3)); geo.setIndex(idx); geo.computeVertexNormals();
    const g = new T.Group(); g.add(skin(geo, mat), seg(a, mat)); return g;
  }
  const mirrorZ = pts => pts.map(([x, y, z]) => [x, y, -z]);
  const circle = (r, n = 64, y = 0) => { const a = []; for (let k = 0; k < n; k++) { const t0 = k / n * Math.PI * 2, t1 = (k + 1) / n * Math.PI * 2; a.push(r * Math.cos(t0), y, r * Math.sin(t0), r * Math.cos(t1), y, r * Math.sin(t1)); } return a; };
  function nacelle(len, r) {
    const g = new T.Group(); r *= 1.15;
    g.add(lathe([[-len / 2, r * 0.82], [-len / 2 + 0.1, r], [len * 0.25, r * 0.95], [len / 2, r * 0.6]], 14, 2, M.wire));
    const d = new T.Mesh(new T.CircleGeometry(r * 0.8, 32), M.intake); d.rotation.y = Math.PI / 2; d.position.x = -len / 2 + 0.06; g.add(d);
    return g;
  }
  function airliner() {   // Torque Stripe's twin-engine narrow-body, nose towards -x
    const S = { len: 10, rad: 0.6, span: 9.6, wingX: 0.2, sweep: 1.9, chord: 1.9, taper: 0.3, fin: 1.6, hspan: 3.6, engR: 0.3, engL: 1.4 };
    const g = new T.Group(), L = S.len / 2, R = S.rad;
    g.add(lathe([[-L, 0], [-L + .15, R * .36], [-L + .4, R * .62], [-L + .8, R * .84], [-L + 1.4, R * .97], [-L + 2.5, R], [L - 2.0, R], [L - 1.2, R * .82], [L - 0.5, R * .45], [L, R * .14]], 18, 1));
    const wy = -R * 0.3, wx = S.wingX, span = S.span / 2, cr = S.chord, ct = cr * S.taper, dih = 0.07;
    const wing = [[wx - cr / 2, wy, R * 0.9], [wx - cr / 2 + S.sweep, wy + span * dih, span], [wx - cr / 2 + S.sweep + ct, wy + span * dih, span], [wx + cr / 2, wy, R * 0.9]];
    g.add(panel(wing, 7), panel(mirrorZ(wing), 7));
    const tx = L - 1.3, fh = S.fin;
    g.add(panel([[tx - 0.7, R * 0.7, 0], [tx + 0.4, R * 0.7 + fh, 0], [tx + 0.95, R * 0.7 + fh, 0], [L - 0.15, R * 0.4, 0]], 3));
    const hy = R * 0.3, hs = S.hspan / 2, ht = [[tx - 0.2, hy, 0.15], [tx + 0.5, hy + 0.05, hs], [tx + 0.9, hy + 0.05, hs], [tx + 0.75, hy, 0.15]];
    g.add(panel(ht, 3), panel(mirrorZ(ht), 3));
    for (const z of [1.75, -1.75]) { const n = nacelle(S.engL, S.engR); n.position.set(-0.3, -0.62, z); g.add(n); }
    return g;
  }

  // ---------- terrain: a wide valley around the field, ridges fading into the haze ----------
  const floor = (() => {
    const N = 84, S = 1.4, H = (x, z) => { const d = Math.hypot(x, z), ridge = Math.max(0, d - 18) / 10;
      return GROUND - 0.02 + ridge * ridge * 2.2 * (0.55 + 0.45 * Math.sin(x * 0.23 + Math.cos(z * 0.17) * 2) * Math.cos(z * 0.21 - x * 0.07)); };
    const a = [], P = (i, j) => { const x = (i - N / 2) * S, z = (j - N / 2) * S; return [x, H(x, z), z]; };
    for (let i = 0; i <= N; i++) for (let j = 0; j < N; j++) { a.push(...P(i, j), ...P(i, j + 1)); a.push(...P(j, i), ...P(j + 1, i)); }
    return seg(a, M.floor);
  })();
  scene.add(floor);

  // ---------- the field: the airport's real layout (north = -z, fitted to the ring), approach lights, the aircraft on final ----------
  // Everything on the field shares one scale, upm (scene units per metre), so it is true to life except the aircraft,
  // drawn PLANE_X times life size (an A320 is 37.6 m nose to tail) so it still reads at dashboard zoom.
  const AC_LEN = 37.6, PLANE_X = 5, BLDG_H = 18, GLIDE = Math.tan(3 * Math.PI / 180), FINAL = 3000, TD = 300, ROLL = 1500, ALS = 730;
  const field = new T.Group(); field.position.y = GROUND; scene.add(field);
  const plane = new T.Group(), craft = airliner(); craft.rotation.z = -0.05; plane.add(craft); field.add(plane);
  const approach = new T.Points(new T.BufferGeometry(), M.lights); field.add(approach);
  const rabbit = new T.Mesh(new T.SphereGeometry(0.05, 12, 8), new T.MeshBasicMaterial({ color: 0xffffff })); field.add(rabbit);
  let fieldGroup = null, rwys = [], heading = 0, landDir = 1, thr = [0, 0], dir = [1, 0], upm = FIT / 2500, gearY = 0;
  function runway(len, w) {   // scene units; markings at their FAA sizes: 120 ft centreline dashes, 150 ft threshold bars
    const g = new T.Group(), slab = new T.Mesh(new T.PlaneGeometry(len, w), M.asphalt);
    slab.rotation.x = -Math.PI / 2; slab.visible = solid; solids.push(slab); g.add(slab);
    const x = len / 2, z = w / 2, y = 0.012, m = upm;
    g.add(seg([-x, y, -z, x, y, -z, x, y, -z, x, y, z, x, y, z, -x, y, z, -x, y, z, -x, y, -z], M.edge));
    const a = []; for (let p = -x + 120 * m; p < x - 120 * m; p += 61 * m) a.push(p, y, 0, p + 37 * m, y, 0);
    for (const s of [-1, 1]) for (let i = 0; i < 8; i++) { const zz = (i - 3.5) * w * 0.11; a.push(s * (x - 6 * m), y, zz, s * (x - 52 * m), y, zz); }
    g.add(seg(a, M.mark));
    return g;
  }
  function footprint(ring, s, h) {   // ring: [x, y, ...] metres (x east, y north); h > 0 extrudes a building
    const shape = new T.Shape(); for (let i = 0; i < ring.length; i += 2) shape[i ? "lineTo" : "moveTo"](ring[i] * s, ring[i + 1] * s);
    const geo = h ? new T.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false }) : new T.ShapeGeometry(shape);
    geo.rotateX(-Math.PI / 2);   // shape y (north) -> -z, extrusion -> up
    const m = new T.Mesh(geo, h ? M.bldg : M.apron); m.visible = solid; solids.push(m);
    const g = new T.Group(); g.add(m, new T.LineSegments(new T.EdgesGeometry(geo, 30), h ? M.edge : M.dim)); return g;
  }
  function setAirport(iata, hdg) {
    const f = FIELDS[iata] || Object.values(FIELDS)[0], s = upm = FIT / f.r, P = (x, y) => [x * s, -y * s];
    craft.scale.setScalar(AC_LEN / 10 * PLANE_X * s); gearY = craft.scale.x;   // the model's engines hang 1 unit below its axis
    if (fieldGroup) {   // remove() alone keeps the GPU buffers
      field.remove(fieldGroup); const old = new Set(); fieldGroup.traverse(o => { old.add(o); o.geometry?.dispose(); });
      solids.splice(0, solids.length, ...solids.filter(m => !old.has(m)));
    }
    fieldGroup = new T.Group();
    for (const ring of f.aprons) { const a = footprint(ring, s, 0); a.position.y = 0.004; fieldGroup.add(a); }
    const tx = [];
    for (const w of f.taxiways) for (let i = 2; i < w.length; i += 2) tx.push(w[i - 2] * s, 0.008, -w[i - 1] * s, w[i] * s, 0.008, -w[i + 1] * s);
    fieldGroup.add(seg(tx, M.taxi));
    rwys = f.runways.map(([ref, x1, y1, x2, y2, w]) => {
      const a = P(x1, y1), b = P(x2, y2), len = Math.hypot(b[0] - a[0], b[1] - a[1]), r = runway(len, w * s);
      r.position.set((a[0] + b[0]) / 2, 0, (a[1] + b[1]) / 2); r.rotation.y = Math.atan2(a[1] - b[1], b[0] - a[0]); fieldGroup.add(r);
      return { ref, a, b, len };
    });
    for (const ring of f.terminals) fieldGroup.add(footprint(ring, s, BLDG_H * s));
    field.add(fieldGroup); heading = hdg || 0;
    placeApproach(); kick();
  }
  function placeApproach() {   // land on the runway end numbered nearest the landing heading (the outlook's "landing runway NN"), longest first
    const land = (landDir > 0 ? heading : heading + 180) % 360, off = n => Math.abs(((n * 10 - land) % 360 + 540) % 360 - 180);
    let best = null;
    for (const r of rwys) r.ref.split("/").forEach((end, k) => { const sc = off(parseInt(end) || 0) * 100 - r.len; if (!best || sc < best.sc) best = { sc, r, k }; });
    if (!best) return;
    const { r, k } = best, [p, q] = k ? [r.b, r.a] : [r.a, r.b];
    thr = p; dir = [(q[0] - p[0]) / r.len, (q[1] - p[1]) / r.len];
    const a = []; for (let d = 30; d <= ALS; d += 30) a.push(...along(-d * upm, 0.02));   // approach light bars every 100 ft
    approach.geometry.dispose(); approach.geometry = new T.BufferGeometry(); approach.geometry.setAttribute("position", new T.Float32BufferAttribute(a, 3));
    plane.rotation.y = Math.atan2(dir[1], -dir[0]);   // the airliner's nose is local -x
  }
  const along = (d, y) => [thr[0] + dir[0] * d, y, thr[1] + dir[1] * d];
  function flyPlane(u) {   // u in [0, 1): 0..0.7 final down a 3° glideslope to touchdown TD m past the threshold, 0.7..1 rollout
    if (u < 0.7) { const d = -FINAL + (FINAL + TD) * u / 0.7; plane.position.set(...along(d * upm, gearY + (TD - d) * GLIDE * upm)); craft.rotation.z = -0.05; }
    else { const k = (u - 0.7) / 0.3, e = 1 - (1 - k) * (1 - k); plane.position.set(...along((TD + ROLL * e) * upm, gearY)); craft.rotation.z = -0.05 * (1 - k); }
  }

  // ---------- capacity ring: one arc per forecast hour (+3 h at north, clockwise), coloured by severity ----------
  const ring = new T.Group(); ring.position.y = GROUND + 0.02; scene.add(ring);
  ring.add(seg(circle(RR + 0.7, 96), M.dim), seg(circle(RR - 0.7, 96), M.dim));
  const arcs = [], labels = [];
  let sevs = Array(10).fill(null), hot = null;
  for (let i = 0; i < 10; i++) {
    const b0 = i * 36 + 1.8, geo = new T.RingGeometry(RR - 0.45, RR + 0.45, 20, 1, Math.PI / 2 - (b0 + 32.4) * Math.PI / 180, 32.4 * Math.PI / 180);
    geo.rotateX(-Math.PI / 2);
    const m = new T.Mesh(geo, new T.MeshBasicMaterial({ transparent: true, opacity: 0.25, side: T.DoubleSide, depthWrite: false })); arcs.push(m); ring.add(m);
    const lb = label(`+${i + 3}h`); const b = (i * 36 + 18) * Math.PI / 180; lb.position.set((RR + 1.8) * Math.sin(b), 0.5, -(RR + 1.8) * Math.cos(b)); labels.push(lb); ring.add(lb);
  }
  const north = label("N", true); north.position.set(0, 0.5, -(RR + 3.2)); ring.add(north);
  const windArrow = new T.Mesh(new T.ConeGeometry(0.55, 1.5, 3), new T.MeshBasicMaterial({ transparent: true, opacity: 0.9 })); ring.add(windArrow);
  function label(text, strong) {
    const c = document.createElement("canvas"); c.width = 160; c.height = 64; const g = c.getContext("2d");
    g.font = `500 ${strong ? 38 : 30}px "IBM Plex Mono", ui-monospace, monospace`; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillStyle = strong ? "#00205b" : "#3c4f72"; g.fillText(text, 80, 34);
    const s = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(c), transparent: true, depthWrite: false })); s.scale.set(2.2, 0.88, 1); return s;
  }
  function paintRing() {
    arcs.forEach((m, i) => { const s = sevs[i]; m.material.color.set(css(s ? `--ring-${s}` : "--wire-dim"));
      m.material.opacity = !s ? 0.18 : hot === null ? 0.78 : hot === i ? 0.95 : 0.3; m.position.y = hot === i ? 0.12 : 0; });
    windArrow.material.color.set(css("--navy"));
  }

  // ---------- weather: wind streaks, precipitation, cloud deck at the ceiling, lightning ----------
  let wx = { dir: null, kt: 0, gust: 0, ceil: null, vis: 10, precip: null, amount: 0, thunder: false };
  const NW = 420, wPos = new Float32Array(NW * 6), wP = Array.from({ length: NW }, () => [rnd(-BOX, BOX), rnd(GROUND + 0.4, GROUND + 6.5), rnd(-BOX, BOX), rnd(0, 6.3)]);
  const wGeo = new T.BufferGeometry(); wGeo.setAttribute("position", new T.BufferAttribute(wPos, 3)); scene.add(new T.LineSegments(wGeo, M.wind));
  const NR = 1400, rPos = new Float32Array(NR * 6), sPos = new Float32Array(NR * 3), rP = Array.from({ length: NR }, () => [rnd(-15, 15), rnd(GROUND, GROUND + 9), rnd(-15, 15), rnd(0, 6.3)]);
  const rGeo = new T.BufferGeometry(); rGeo.setAttribute("position", new T.BufferAttribute(rPos, 3)); const rain = new T.LineSegments(rGeo, M.rain); scene.add(rain);
  const sGeo = new T.BufferGeometry(); sGeo.setAttribute("position", new T.BufferAttribute(sPos, 3)); const snow = new T.Points(sGeo, M.snow); scene.add(snow);
  const deck = new T.Group(), cloudTex = cloudTexture();
  for (const [dy, op] of [[0, 0.75], [-0.35, 0.45]]) {
    const m = new T.Mesh(new T.PlaneGeometry(90, 90), new T.MeshBasicMaterial({ map: cloudTex, color: 0xf3f6fa, transparent: true, opacity: op, depthWrite: false, side: T.DoubleSide }));
    m.rotation.x = -Math.PI / 2; m.position.y = dy; deck.add(m);
  }
  deck.add(seg(circle(BOX, 96), M.dim)); scene.add(deck);
  const bolt = seg([], M.bolt); scene.add(bolt);
  let flashT = 0, nextFlash = 2;
  function cloudTexture() {
    const c = document.createElement("canvas"); c.width = c.height = 256; const g = c.getContext("2d");
    for (let i = 0; i < 110; i++) { const x = Math.random() * 256, y = Math.random() * 256, r = 12 + Math.random() * 42;
      for (const [dx, dy] of [[0, 0], [256, 0], [-256, 0], [0, 256], [0, -256]]) {
        const gr = g.createRadialGradient(x + dx, y + dy, 0, x + dx, y + dy, r); gr.addColorStop(0, "rgba(255,255,255,.5)"); gr.addColorStop(1, "rgba(255,255,255,0)");
        g.fillStyle = gr; g.beginPath(); g.arc(x + dx, y + dy, r, 0, 7); g.fill(); } }
    const t = new T.CanvasTexture(c); t.wrapS = t.wrapT = T.RepeatWrapping; t.repeat.set(3, 3); return t;
  }
  const cloudY = () => wx.ceil == null ? GROUND + 9 : GROUND + Math.max(0.7, Math.min(8.5, wx.ceil / 400));
  function setWeather(now) {
    const code = (now && now.wx) || "";
    wx = { dir: now?.wind_dir ?? null, kt: now?.wind_kt ?? 0, gust: now?.gust_kt ?? 0, vis: now?.vis_mi ?? 10,
           ceil: now?.ceiling_ft != null && now.ceiling_ft < 12000 ? now.ceiling_ft : null, thunder: !!now?.thunder || /TS/.test(code),
           precip: /SN|SG|PL|IC|GS/.test(code) ? "snow" : /RA|DZ|SH|UP/.test(code) ? "rain" : null,
           amount: /\+/.test(code) ? 1 : /-|DZ/.test(code) ? 0.4 : 0.7 };
    landDir = (now?.headwind_kt ?? 1) >= 0 ? 1 : -1; placeApproach();
    deck.visible = wx.ceil != null; deck.position.y = cloudY();
        rain.visible = wx.precip === "rain"; snow.visible = wx.precip === "snow";
    const n = Math.round(NR * wx.amount); rGeo.setDrawRange(0, n * 2); sGeo.setDrawRange(0, n);
    const b = (wx.dir ?? 0) * Math.PI / 180; windArrow.visible = wx.dir != null && wx.kt > 0;
    windArrow.position.set((RR + 3.4) * Math.sin(b), 0.6, -(RR + 3.4) * Math.cos(b));
    windArrow.rotation.set(Math.PI / 2, 0, 0); windArrow.rotateOnWorldAxis(new T.Vector3(0, 1, 0), -b + Math.PI);
    kick();
  }

  // ---------- scan sweep across the field when a new forecast arrives ----------
  const scanner = new T.Group();
  scanner.add(seg([0, 0, -BOX, 0, 4, -BOX, 0, 4, -BOX, 0, 4, BOX, 0, 4, BOX, 0, 0, BOX, 0, 0, BOX, 0, 0, -BOX], M.scan));
  const sp = new T.Mesh(new T.PlaneGeometry(BOX * 2, 4), M.scanFill); sp.rotation.y = Math.PI / 2; sp.position.y = 2; scanner.add(sp);
  field.add(scanner);
  let scanT = -1;

  // ---------- camera, drag, resize (from Torque Stripe) ----------
  let az = -0.9, el = 0.5, dragging = false, px = 0, py = 0, autoT = 0, view = "3d", tAz = null, tEl = null;
  function setView(v) { view = v; if (v === "top") { tAz = Math.round(az / (Math.PI * 2)) * Math.PI * 2 + Math.PI / 2; tEl = 1.5; }   // camera south of the field: north is up
    else if (v === "side") { tAz = Math.round(az / (Math.PI * 2)) * Math.PI * 2 + Math.PI / 2; tEl = 0.08; } else { tAz = null; tEl = 0.5; } kick(); }
  cv.addEventListener("pointerdown", e => { dragging = true; px = e.clientX; py = e.clientY; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => { if (!dragging) return; tAz = null; tEl = null; az -= (e.clientX - px) * 0.008; el = Math.max(0.08, Math.min(1.5, el + (e.clientY - py) * 0.006)); px = e.clientX; py = e.clientY; autoT = 2.5; kick(); });
  cv.addEventListener("pointerup", () => dragging = false);
  function size() { const r = host.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); cam.aspect = r.width / Math.max(1, r.height); cam.updateProjectionMatrix(); kick(); }
  const ro = new ResizeObserver(size); ro.observe(host);

  // ---------- loop: pauses in background tabs and while another view covers the stage ----------
  let raf = 0, last = performance.now(), clock = 0, active = true;
  document.addEventListener("visibilitychange", kick);
  function kick() { if (!raf && !document.hidden) { last = performance.now(); raf = requestAnimationFrame(frame); } }
  function frame(now) {
    raf = 0; const dt = Math.min(0.05, (now - last) / 1000); last = now; const anim = !reduce && active; if (anim) clock += dt;
    if (anim && view === "3d" && tAz === null) { if (autoT > 0) autoT -= dt; else az += dt * spinSpeed; }
    let easing = false;
    if (tAz !== null && !dragging) { az += (tAz - az) * Math.min(1, dt * 4); if (Math.abs(tAz - az) > 0.002) easing = true; }
    if (tEl !== null && !dragging) { el += (tEl - el) * Math.min(1, dt * 4); if (Math.abs(tEl - el) > 0.002) easing = true; else if (view === "3d") tEl = null; }
    flyPlane(reduce ? 0.55 : (clock / 18) % 1);
    rabbit.position.set(...along(-(ALS - (ALS - 60) * ((clock * 2.2) % 1)) * upm, 0.03));
    // wind: streaks drift downwind, gusts pulse their speed
    const b = (wx.dir ?? 225) * Math.PI / 180, ux = -Math.sin(b), uz = Math.cos(b), calm = wx.dir == null || wx.kt < 3;
    M.wind.opacity = calm ? 0.2 : Math.min(0.85, 0.4 + wx.kt / 35);
    for (let i = 0; i < NW; i++) {
      const p = wP[i], s = calm ? 0.8 : wx.kt + (wx.gust > wx.kt ? (wx.gust - wx.kt) * (0.5 + 0.5 * Math.sin(clock * 1.7 + p[3])) : 0), v = s * 0.14;
      if (anim) { p[0] += ux * v * dt; p[2] += uz * v * dt; p[0] = ((p[0] + BOX) % (2 * BOX) + 2 * BOX) % (2 * BOX) - BOX; p[2] = ((p[2] + BOX) % (2 * BOX) + 2 * BOX) % (2 * BOX) - BOX; }
      const len = 0.6 + s * 0.09;
      wPos.set([p[0], p[1], p[2], p[0] - ux * len, p[1], p[2] - uz * len], i * 6);
    }
    wGeo.attributes.position.needsUpdate = true;
    if (wx.precip) {
      const top = cloudY(), drift = calm ? 0 : wx.kt * 0.04, snowy = wx.precip === "snow";
      for (let i = 0; i < NR; i++) { const p = rP[i];
        if (anim) { p[1] -= (snowy ? 0.9 : 11) * dt; p[0] += ux * drift * (snowy ? 2 : 1) * dt + (snowy ? Math.sin(clock + p[3]) * 0.3 * dt : 0); p[2] += uz * drift * dt; }
        if (p[1] < GROUND) { p[1] = top; p[0] = rnd(-15, 15); p[2] = rnd(-15, 15); }
        if (snowy) sPos.set([p[0], p[1], p[2]], i * 3); else rPos.set([p[0], p[1], p[2], p[0] + ux * drift * 0.04, p[1] + 0.42, p[2] + uz * drift * 0.04], i * 6); }
      (snowy ? sGeo : rGeo).attributes.position.needsUpdate = true;
    }
    if (deck.visible && anim) { cloudTex.offset.x += ux * (wx.kt || 2) * 0.00012 * dt * 60; cloudTex.offset.y -= uz * (wx.kt || 2) * 0.00012 * dt * 60; }
    // lightning: a flash and a bolt every few seconds when thunder is reported
    if (wx.thunder && anim) { nextFlash -= dt; if (nextFlash <= 0) { nextFlash = rnd(2.5, 7); flashT = 0.35; strike(); } }
    if (flashT > 0) { flashT -= dt; const f = Math.max(0, flashT / 0.35); hemi.intensity = 0.55 + 2.6 * f; M.bolt.opacity = f; } else { hemi.intensity = 0.55; M.bolt.opacity = 0; }
    if (scanT >= 0) {
      scanT += dt / 1.1; scanner.position.x = -BOX + 2 * BOX * Math.min(1, scanT);
      const f = Math.sin(Math.min(1, scanT) * Math.PI); M.scan.opacity = 0.9 * f; M.scanFill.opacity = 0.08 * f;
      if (scanT >= 1) { scanT = -1; M.scan.opacity = 0; M.scanFill.opacity = 0; }
    }
    const R = (view === "top" ? 21 : 15.5) * zoom, vis = Math.max(0.25, Math.min(10, wx.vis ?? 10)) / 10;
    scene.fog.near = R - 12 + 18 * vis; scene.fog.far = R + 10 + 45 * vis;
    cam.position.set(R * Math.cos(el) * Math.cos(az), R * Math.sin(el) + 1, R * Math.cos(el) * Math.sin(az));
    cam.lookAt(0, lookY, 0); renderer.render(scene, cam);
    if ((anim || scanT >= 0 || dragging || easing) && !document.hidden) raf = requestAnimationFrame(frame);
  }
  function strike() {
    const x = rnd(-10, 10), z = rnd(-10, 10), top = cloudY(), a = []; let p = [x, top, z];
    for (let i = 1; i <= 9; i++) { const q = [x + rnd(-0.8, 0.8), top - (top - GROUND) * i / 9, z + rnd(-0.8, 0.8)]; a.push(...p, ...q); p = q; }
    bolt.geometry.dispose(); bolt.geometry.setAttribute("position", new T.Float32BufferAttribute(a, 3));
  }

  colors(); size(); setAirport("ATL", 80); setWeather(null);
  return {
    setAirport, setWeather,
    setRing: s => { sevs = s.slice(0, 10); paintRing(); kick(); },
    highlight: i => { hot = i; paintRing(); kick(); },
    setView, setRender: mode => { solid = mode !== "wire"; applyRender(); kick(); },
    scan: () => { if (!reduce) { scanT = 0; kick(); } },
    setActive: on => { active = on; kick(); },
    dispose: () => { cancelAnimationFrame(raf); ro.disconnect(); document.removeEventListener("visibilitychange", kick); renderer.dispose(); },
  };
}
