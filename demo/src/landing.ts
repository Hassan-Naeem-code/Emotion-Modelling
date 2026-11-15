/**
 * Landing-page motion. Everything here is decorative and simulated: the hero
 * "scanner" animates a dot-mesh face, and its smile and activation drive the
 * valence-arousal readout, a PPG-style pulse trace and a blink counter, so a
 * visitor sees what the real demo does before allowing the camera.
 * Under prefers-reduced-motion each piece draws one still frame instead.
 */

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = <T extends Element = HTMLElement>(s: string) => document.querySelector(s) as T | null;
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** Size a canvas to its CSS box at device pixel ratio; returns CSS width/height. */
function fit(c: HTMLCanvasElement, ctx: CanvasRenderingContext2D): [number, number] {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = c.clientWidth;
  const h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [w, h];
}

/* ------------------------------------------------------------------ */
/* Background: drifting neural mesh that leans toward the cursor.      */
/* ------------------------------------------------------------------ */
const mouse = { x: -9999, y: -9999 };
window.addEventListener("pointermove", (e) => {
  mouse.x = e.clientX;
  mouse.y = e.clientY;
});

function meshLayer() {
  const c = $<HTMLCanvasElement>("#mesh");
  if (!c) return () => {};
  const ctx = c.getContext("2d")!;
  let pts: { x: number; y: number; vx: number; vy: number }[] = [];
  const seed = () => {
    const [w, h] = fit(c, ctx);
    const n = Math.round(clamp((w * h) / 17000, 30, 95));
    pts = Array.from({ length: n }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
    }));
  };
  seed();
  window.addEventListener("resize", seed);
  return () => {
    const [w, h] = fit(c, ctx);
    ctx.clearRect(0, 0, w, h);
    for (const p of pts) {
      const dx = mouse.x - p.x;
      const dy = mouse.y - p.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < 200 * 200) {
        p.vx += dx * 0.00002;
        p.vy += dy * 0.00002;
      }
      p.vx = clamp(p.vx, -0.5, 0.5);
      p.vy = clamp(p.vy, -0.5, 0.5);
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x;
        const dy = pts[i].y - pts[j].y;
        const d = Math.hypot(dx, dy);
        if (d < 140) {
          ctx.strokeStyle = `rgba(88,166,255,${0.16 * (1 - d / 140)})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.stroke();
        }
      }
    }
    for (const p of pts) {
      const near = Math.hypot(mouse.x - p.x, mouse.y - p.y) < 160;
      ctx.fillStyle = near ? "rgba(210,168,255,0.9)" : "rgba(139,148,158,0.5)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, near ? 2 : 1.3, 0, Math.PI * 2);
      ctx.fill();
    }
  };
}

/* ------------------------------------------------------------------ */
/* Hero scanner: simulated face, VA region, pulse and blinks.          */
/* ------------------------------------------------------------------ */

// Face mesh in face-local units (face spans roughly -1..1). Built once.
type P = [number, number];
const outline: P[] = Array.from({ length: 48 }, (_, i) => {
  const a = (i / 48) * Math.PI * 2;
  const y = Math.sin(a);
  // Narrower toward the chin.
  const taper = y > 0 ? 1 - 0.22 * y * y : 1;
  return [Math.cos(a) * 0.74 * taper, y * 1.0];
});
const inner: P[] = [];
for (let gy = -0.8; gy <= 0.85; gy += 0.16) {
  for (let gx = -0.7; gx <= 0.7; gx += 0.16) {
    const jx = gx + 0.05 * Math.sin(gy * 13 + gx * 7);
    const jy = gy + 0.05 * Math.cos(gx * 11 - gy * 5);
    const taper = jy > 0 ? 1 - 0.22 * jy * jy : 1;
    if ((jx / (0.7 * taper)) ** 2 + (jy / 0.95) ** 2 < 0.92) inner.push([jx, jy]);
  }
}
const neighbours: number[][] = inner.map((p, i) =>
  inner
    .map((q, j) => [j, Math.hypot(p[0] - q[0], p[1] - q[1])] as [number, number])
    .filter(([j]) => j !== i)
    .sort((a, b) => a[1] - b[1])
    .slice(0, 3)
    .map(([j]) => j),
);

interface Sim {
  smile: number; // -1..1 (frown..smile) -> valence
  arousal: number; // -1..1
  yaw: number; // head turn
  eyeOpen: number; // 0..1
  gaze: number; // -1..1
  unsure: boolean;
}

function scanner() {
  const faceC = $<HTMLCanvasElement>("#face");
  const ecgC = $<HTMLCanvasElement>("#ecg");
  if (!faceC || !ecgC) return () => {};
  const fctx = faceC.getContext("2d")!;
  const ectx = ecgC.getContext("2d")!;
  const box = $("#va-box")!;
  const dot = $("#va-dot")!;
  const ring = $("#va-ring")!;
  const vaNote = $("#va-note")!;
  const chip = $("#chip-display")!;
  const bpmEl = $("#bpm")!;
  const heart = $("#heart")!;
  const blinksEl = $("#blinks")!;
  const lid = $("#eyelid")!;
  const iris = $("#iris")!;
  const pupil = $("#pupil")!;

  let nextBlink = 1.5;
  let blinkEnd = -1;
  const intervals = [3.6, 4.1, 3.2, 3.9, 3.5];
  let lastBlink = 0;
  let beatPhase = 0;
  const beats: number[] = [];
  let lastBpmShown = -1;
  let lastT = 0;

  const state = (t: number): Sim => {
    // A 14 s loop with a scripted "head turned away" moment where the model abstains.
    const cyc = t % 14;
    const unsure = cyc > 10.2 && cyc < 12.4;
    const turn = unsure ? Math.sin(((cyc - 10.2) / 2.2) * Math.PI) : 0;
    if (t > nextBlink && blinkEnd < t - 0.2) {
      blinkEnd = t + 0.16;
      intervals.push(t - lastBlink);
      if (intervals.length > 8) intervals.shift();
      lastBlink = t;
      nextBlink = t + 2.4 + Math.random() * 2.8;
    }
    const blinkPos = blinkEnd > t ? 1 - Math.sin(((blinkEnd - t) / 0.16) * Math.PI) : 1;
    return {
      smile: clamp(0.65 * Math.sin(t * 0.42) + 0.25 * Math.sin(t * 1.07 + 1), -1, 1),
      arousal: clamp(0.4 * Math.sin(t * 0.3 + 0.6) + 0.18 * Math.sin(t * 0.93), -1, 1),
      yaw: 0.08 * Math.sin(t * 0.5) + 0.75 * turn,
      eyeOpen: blinkPos,
      gaze: 0.6 * Math.sin(t * 0.37),
      unsure,
    };
  };

  const drawFace = (t: number, s: Sim) => {
    const [w, h] = fit(faceC, fctx);
    fctx.clearRect(0, 0, w, h);
    const R = Math.min(w, h) * 0.36;
    const cx = w / 2 + s.yaw * R * 0.5;
    const cy = h * 0.5 + Math.sin(t * 0.8) * 3;
    const sx = 1 - Math.abs(s.yaw) * 0.35; // foreshortening when turned
    const tr = ([x, y]: P): P => [cx + (x * sx + s.yaw * 0.25 * (1 - x * x)) * R, cy + y * R];
    const beamY = ((t * 0.45) % 1.3) * h * 1.1 - h * 0.1;
    const lit = (y: number) => Math.max(0, 1 - Math.abs(y - beamY) / 40);
    const base = s.unsure ? "227,179,65" : "88,166,255";

    // Mesh edges.
    fctx.lineWidth = 0.7;
    inner.forEach((p, i) => {
      const a = tr(p);
      for (const j of neighbours[i]) {
        const b = tr(inner[j]);
        fctx.strokeStyle = `rgba(${base},${0.12 + 0.35 * lit(a[1])})`;
        fctx.beginPath();
        fctx.moveTo(a[0], a[1]);
        fctx.lineTo(b[0], b[1]);
        fctx.stroke();
      }
    });
    const dots = (pts: P[], color: string, r: number, line = false) => {
      const q = pts.map(tr);
      if (line) {
        fctx.strokeStyle = color;
        fctx.lineWidth = 1.2;
        fctx.beginPath();
        q.forEach(([x, y], k) => (k ? fctx.lineTo(x, y) : fctx.moveTo(x, y)));
        fctx.stroke();
      }
      for (const [x, y] of q) {
        const l = lit(y);
        fctx.fillStyle = color;
        fctx.globalAlpha = 0.55 + 0.45 * l;
        fctx.beginPath();
        fctx.arc(x, y, r + 1.4 * l, 0, Math.PI * 2);
        fctx.fill();
      }
      fctx.globalAlpha = 1;
    };
    dots(inner, `rgba(${base},0.9)`, 1.1);
    dots([...outline, outline[0]], `rgba(${base},0.95)`, 1.4, true);

    // Eyes (ellipses that close on blink).
    const feat = s.unsure ? "rgba(227,179,65,1)" : "rgba(210,168,255,1)";
    for (const side of [-1, 1]) {
      const ex = side * 0.3;
      const ey = -0.18;
      const eye: P[] = Array.from({ length: 14 }, (_, i) => {
        const a = (i / 14) * Math.PI * 2;
        return [ex + Math.cos(a) * 0.15, ey + Math.sin(a) * 0.075 * s.eyeOpen];
      });
      dots([...eye, eye[0]], feat, 1.3, true);
      if (s.eyeOpen > 0.4) dots([[ex + s.gaze * 0.06, ey]], feat, 3);
      // Brows rise with activation.
      const by = ey - 0.17 - 0.06 * s.arousal;
      dots(
        Array.from({ length: 6 }, (_, i) => [ex - 0.17 + i * 0.068, by - 0.04 * Math.sin((i / 5) * Math.PI)] as P),
        feat,
        1.2,
        true,
      );
    }
    // Nose.
    dots([[0, -0.1], [0.02, 0.05], [0.05, 0.2], [0, 0.25], [-0.05, 0.2]], `rgba(${base},0.9)`, 1.1, true);
    // Mouth: corners lift with the smile, opens with activation.
    const open = 0.02 + 0.06 * Math.max(0, s.arousal);
    const mouth = (sign: number): P[] =>
      Array.from({ length: 11 }, (_, i) => {
        const u = i / 10 - 0.5;
        const bow = 1 - 4 * u * u; // 1 in the middle, 0 at the corners
        return [u * 0.56, 0.5 + s.smile * 0.14 * bow + sign * open * bow];
      });
    dots(mouth(-1), feat, 1.3, true);
    dots(mouth(1), feat, 1.3, true);

    // Scan beam.
    const g = fctx.createLinearGradient(0, beamY - 30, 0, beamY + 30);
    g.addColorStop(0, `rgba(${base},0)`);
    g.addColorStop(0.5, `rgba(${base},0.18)`);
    g.addColorStop(1, `rgba(${base},0)`);
    fctx.fillStyle = g;
    fctx.fillRect(0, beamY - 30, w, 60);
    fctx.fillStyle = `rgba(${base},0.7)`;
    fctx.fillRect(0, beamY, w, 1);
  };

  const ppg = (tau: number) =>
    Math.exp(-(((tau - 0.12) / 0.06) ** 2)) + 0.38 * Math.exp(-(((tau - 0.36) / 0.07) ** 2));

  const drawEcg = (t: number) => {
    const [w, h] = fit(ecgC, ectx);
    ectx.clearRect(0, 0, w, h);
    const span = 3;
    ectx.strokeStyle = "#ff7b72";
    ectx.lineWidth = 1.8;
    ectx.shadowColor = "rgba(255,123,114,0.7)";
    ectx.shadowBlur = 6;
    ectx.beginPath();
    for (let x = 0; x <= w; x += 1.5) {
      const tt = t - span + (x / w) * span;
      let v = 0;
      for (const b of beats) if (tt - b > -0.05 && tt - b < 0.7) v += ppg(tt - b);
      const y = h - 6 - v * (h - 12) * 0.85;
      if (x) ectx.lineTo(x, y);
      else ectx.moveTo(x, y);
    }
    ectx.stroke();
    ectx.shadowBlur = 0;
  };

  const updateReadouts = (t: number, s: Sim) => {
    // Valence-arousal plane: SVG is 100x100, axes through the centre.
    const v = 0.7 * s.smile;
    const a = 0.1 + 0.6 * s.arousal;
    const half = s.unsure ? 30 : 7 + 3.5 * Math.abs(Math.sin(t * 0.5));
    const halfY = s.unsure ? 28 : 6 + 3 * Math.abs(Math.cos(t * 0.4));
    const px = 50 + v * 44;
    const py = 50 - a * 44;
    box.setAttribute("x", String(clamp(px - half, 2, 98)));
    box.setAttribute("y", String(clamp(py - halfY, 2, 98)));
    box.setAttribute("width", String(clamp(px + half, 2, 98) - clamp(px - half, 2, 98)));
    box.setAttribute("height", String(clamp(py + halfY, 2, 98) - clamp(py - halfY, 2, 98)));
    for (const el of [dot, ring]) {
      el.setAttribute("cx", px.toFixed(1));
      el.setAttribute("cy", py.toFixed(1));
    }
    box.classList.toggle("unsure", s.unsure);
    vaNote.classList.toggle("unsure", s.unsure);
    chip.classList.toggle("unsure", s.unsure);
    vaNote.textContent = s.unsure ? "not sure: head turned" : "90% region";
    const word = v > 0.15 ? "pleasant" : v < -0.15 ? "unpleasant" : "neutral";
    chip.textContent = s.unsure ? "displays: not sure" : `displays: ${word}`;

    // Pulse.
    const bpm = 72 + 5 * Math.sin(t * 0.11) + 4 * s.arousal;
    // Step the beat clock in small increments so beats stay evenly spaced even
    // after the loop paused (tab hidden or scanner scrolled off-screen).
    let tau = Math.max(lastT, t - 4);
    while (tau < t) {
      const step = Math.min(1 / 60, t - tau);
      tau += step;
      beatPhase += step * (bpm / 60);
      if (beatPhase >= 1) {
        beatPhase -= 1;
        beats.push(tau);
        if (t - tau < 0.1) {
          heart.classList.add("beat");
          setTimeout(() => heart.classList.remove("beat"), 140);
        }
      }
    }
    while (beats.length && beats[0] < t - 4) beats.shift();
    if (Math.floor(t) !== lastBpmShown) {
      lastBpmShown = Math.floor(t);
      bpmEl.textContent = s.unsure ? "—" : String(Math.round(bpm));
    }

    // Eye + blink rate.
    lid.classList.toggle("shut", s.eyeOpen < 0.5);
    const gx = 30 + s.gaze * 9;
    iris.setAttribute("cx", gx.toFixed(1));
    pupil.setAttribute("cx", gx.toFixed(1));
    const mean = intervals.reduce((x, y) => x + y, 0) / intervals.length;
    blinksEl.textContent = String(Math.round(60 / mean));
  };

  return (t: number) => {
    const s = state(t);
    drawFace(t, s);
    updateReadouts(t, s);
    drawEcg(t);
    lastT = t;
  };
}

/* ------------------------------------------------------------------ */
/* Typewriter in the headline.                                         */
/* ------------------------------------------------------------------ */
function rotator() {
  const el = $("#rot");
  if (!el || reduced) return;
  const words = ["a smile.", "a frown.", "surprise.", "a polite grin.", "a poker face."];
  let wi = 0;
  let ci = words[0].length;
  let deleting = false;
  el.textContent = words[0];
  const tick = () => {
    const w = words[wi];
    if (!deleting) {
      ci++;
      el.textContent = w.slice(0, ci);
      if (ci >= w.length) {
        deleting = true;
        return void setTimeout(tick, 1800);
      }
    } else {
      ci--;
      el.textContent = w.slice(0, ci);
      if (ci <= 0) {
        deleting = false;
        wi = (wi + 1) % words.length;
      }
    }
    setTimeout(tick, deleting ? 45 : 85);
  };
  setTimeout(() => {
    deleting = true;
    tick();
  }, 2200);
}

/* ------------------------------------------------------------------ */
/* Illustrative timeline (lines draw in when scrolled into view).      */
/* ------------------------------------------------------------------ */
function buildTimeline() {
  const svg = $<SVGSVGElement>("#tl");
  if (!svg) return;
  const NS = "http://www.w3.org/2000/svg";
  const W = 800;
  const L = 74;
  const rows = [
    { label: "Valence", color: "#58a6ff", f: (x: number) => 0.35 * Math.sin(x * 0.018) + 0.2 * Math.sin(x * 0.051 + 1), band: 0.18 },
    { label: "Arousal", color: "#d2a8ff", f: (x: number) => 0.3 * Math.sin(x * 0.012 + 2) + 0.15 * Math.sin(x * 0.07), band: 0.2 },
    { label: "Pulse", color: "#ff7b72", f: (x: number) => 0.2 * Math.sin(x * 0.009 + 0.5) + 0.06 * Math.sin(x * 0.11), band: 0 },
  ];
  const rh = 68;
  const gap = 10;
  const gapFrom = 470;
  const gapTo = 545;
  const el = (tag: string, attrs: Record<string, string | number>) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
    svg.appendChild(e);
    return e;
  };
  rows.forEach((r, i) => {
    const top = i * (rh + gap) + 4;
    const y = (v: number) => top + rh / 2 - v * (rh / 2) * 0.85;
    el("rect", { x: L, y: top, width: W - L - 4, height: rh, rx: 8, class: "row-bg" });
    const t = el("text", { x: 6, y: top + rh / 2 + 4, class: "lab" });
    t.textContent = r.label;
    const segs: [number, number][] = [
      [L + 6, gapFrom],
      [gapTo, W - 10],
    ];
    for (const [a, b] of segs) {
      if (r.band) {
        let d = "";
        for (let x = a; x <= b; x += 4) d += `${x === a ? "M" : "L"}${x},${y(r.f(x) + r.band)}`;
        for (let x = b; x >= a; x -= 4) d += `L${x},${y(r.f(x) - r.band)}`;
        el("path", { d: d + "Z", fill: r.color, "fill-opacity": 0.14, class: "band" });
      }
      let d = "";
      for (let x = a; x <= b; x += 4) d += `${x === a ? "M" : "L"}${x},${y(r.f(x))}`;
      el("path", { d, stroke: r.color, class: "line", pathLength: 1 });
    }
  });
  const H = rows.length * (rh + gap) - gap + 4;
  el("rect", { x: gapFrom + 4, y: 2, width: gapTo - gapFrom - 8, height: H, rx: 6, class: "gapzone" });
  const gl = el("text", { x: (gapFrom + gapTo) / 2, y: H + 16, "text-anchor": "middle", class: "gaplab" });
  gl.textContent = "not sure (looked away)";
}

/* ------------------------------------------------------------------ */
/* Scroll effects: reveal, count-up, progress bar, sticky nav.         */
/* ------------------------------------------------------------------ */
function countUp(span: HTMLElement) {
  const ends = (span.dataset.count ?? "0").split(",").map(Number);
  const fmt = (k: number) => ends.map((e) => Math.round(e * k)).join("–");
  if (reduced) return void (span.textContent = fmt(1));
  const t0 = performance.now();
  const step = (now: number) => {
    const k = clamp((now - t0) / 1600, 0, 1);
    span.textContent = fmt(1 - (1 - k) ** 3);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function scrollFx() {
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.add("in");
        e.target.querySelectorAll<HTMLElement>("[data-count]").forEach(countUp);
        io.unobserve(e.target);
      }
    },
    { threshold: 0.15, rootMargin: "0px 0px -40px 0px" },
  );
  document.querySelectorAll(".reveal").forEach((n) => io.observe(n));
  // The hero is on screen at load: reveal it on the next frame without waiting for the observer.
  requestAnimationFrame(() => document.querySelectorAll(".hero .reveal").forEach((n) => n.classList.add("in")));

  const bar = $("#progress")!;
  const nav = $(".nav")!;
  const onScroll = () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? scrollY / max : 0})`;
    nav.classList.toggle("scrolled", scrollY > 10);
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

/** 3D tilt and cursor spotlight on the feature cards. */
function tilt() {
  if (reduced || !matchMedia("(hover: hover)").matches) return;
  document.querySelectorAll<HTMLElement>(".tilt").forEach((card) => {
    card.addEventListener("pointermove", (e) => {
      const r = card.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width;
      const y = (e.clientY - r.top) / r.height;
      card.style.setProperty("--mx", `${x * 100}%`);
      card.style.setProperty("--my", `${y * 100}%`);
      card.style.transform = `perspective(800px) rotateY(${(x - 0.5) * 10}deg) rotateX(${(0.5 - y) * 10}deg) translateY(-4px)`;
    });
    card.addEventListener("pointerleave", () => (card.style.transform = ""));
  });
}

/* ------------------------------------------------------------------ */
buildTimeline();
scrollFx();
tilt();
rotator();

const drawMesh = meshLayer();
const drawScanner = scanner();
if (reduced) {
  drawMesh();
  drawScanner(2.0);
} else {
  const hero = $(".scanner");
  let heroVisible = true;
  if (hero) new IntersectionObserver(([e]) => (heroVisible = e.isIntersecting)).observe(hero);
  const t0 = performance.now();
  const loop = (now: number) => {
    const t = (now - t0) / 1000;
    drawMesh();
    if (heroVisible) drawScanner(t);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
