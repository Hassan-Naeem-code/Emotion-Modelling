const $ = (id) => document.getElementById(id);
const f3 = (x) => (x == null ? "–" : x.toFixed(3));
const sd = (x) => (x == null ? "" : `<span class="sd">±${x.toFixed(3)}</span>`);
const NOM = 0.9;

function cell(v, s, opt = {}) {
  if (v == null) return "<td>–</td>";
  const cls = opt.cov && v < NOM ? ' class="below"' : "";
  return `<td${cls}>${f3(v)}${s != null && opt.sd !== false ? sd(s) : ""}</td>`;
}

function table(el, cols, data, showSeeds) {
  const head = "<thead><tr><th>Method</th>" + cols.map((c) => `<th>${c.h}</th>`).join("") + "</tr></thead>";
  let body = "";
  for (const m of data.methods) {
    body += "<tr class=\"fresh\"><td>" + m.label + (m.n > 1 ? ` <span class="sd">n=${m.n}</span>` : "") + "</td>" +
      cols.map((c) => { const v = m.summary[c.k]; return v ? cell(v[0], m.n > 1 ? v[1] : null, c) : "<td>–</td>"; }).join("") + "</tr>";
    if (showSeeds && m.n > 1) {
      for (const [s, r] of Object.entries(m.per_seed)) {
        body += `<tr class="seed fresh"><td>seed ${s}</td>` + cols.map((c) => cell(r[c.k], null, c)).join("") + "</tr>";
      }
    }
  }
  el.innerHTML = head + "<tbody>" + body + "</tbody>";
}

const COV = [
  { h: "Per-axis joint", k: "cov_perdim_joint", cov: true }, { h: "Per-axis area", k: "area_perdim" },
  { h: "Bonferroni joint", k: "cov_bonf", cov: true }, { h: "Bonferroni area", k: "area_bonf" },
  { h: "Joint (ours)", k: "cov_joint", cov: true }, { h: "Area (ours)", k: "area_joint" },
  { h: "Worst expr.", k: "worst_seen_joint", cov: true }, { h: "Shift", k: "shift_cov_joint", cov: true },
  { h: "Contempt", k: "contempt_cov_joint", cov: true },
];
const ACC = [
  { h: "CCC-V", k: "ccc_v" }, { h: "CCC-A", k: "ccc_a" }, { h: "RMSE", k: "rmse" },
  { h: "NLL", k: "nll" }, { h: "CRPS", k: "crps" }, { h: "ECE", k: "ece" }, { h: "AURC", k: "aurc" },
  { h: "Shift CCC", k: "shift_ccc" },
];

function chart(data, key) {
  const m = data.methods.find((x) => x.key === key);
  const svg = $("chart");
  const W = 900, H = 300, L = 44, B = 40, T = 12;
  const em = data.emotions, n = em.length, gw = (W - L - 8) / n, bw = gw * 0.34;
  const y = (v) => T + (1 - v) * (H - T - B);
  let s = `<defs><linearGradient id="gj" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#79c0ff"/><stop offset="1" stop-color="#1f6feb"/></linearGradient>
    <linearGradient id="gp" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8b949e"/><stop offset="1" stop-color="#484f58"/></linearGradient></defs>`;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    s += `<line x1="${L}" x2="${W - 8}" y1="${y(t)}" y2="${y(t)}" stroke="rgba(240,246,252,.08)"/>`;
    s += `<text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>`;
  }
  const series = [["cov_perdim_by_emotion", "url(#gp)"], ["cov_joint_by_emotion", "url(#gj)"]];
  em.forEach((e, i) => {
    series.forEach(([k, col], j) => {
      const d = m.by_emotion[k]?.[e];
      if (!d) return;
      const x = L + i * gw + gw / 2 - bw + j * bw + 1;
      s += `<rect x="${x}" y="${y(d[0])}" width="${bw - 2}" height="${y(0) - y(d[0])}" fill="${col}" rx="3" class="bar" style="animation-delay:${(i * 0.07 + j * 0.03).toFixed(2)}s"><title>${e}: ${d[0].toFixed(3)} ± ${d[1].toFixed(3)}</title></rect>`;
      if (d[1] > 0) {
        const cx = x + (bw - 2) / 2;
        s += `<line x1="${cx}" x2="${cx}" y1="${y(Math.min(1, d[0] + d[1]))}" y2="${y(d[0] - d[1])}" stroke="var(--line)" opacity=".6" class="whisk"/>`;
      }
    });
    s += `<text x="${L + i * gw + gw / 2}" y="${H - B + 18}" text-anchor="middle">${e}${e === "contempt" ? " (unseen)" : ""}</text>`;
  });
  s += `<line x1="${L}" x2="${W - 8}" y1="${y(NOM)}" y2="${y(NOM)}" stroke="var(--warn)" stroke-dasharray="5 4"/><text x="${W - 10}" y="${y(NOM) - 6}" text-anchor="end" style="fill:var(--warn)">nominal 90%</text>`;
  svg.innerHTML = s;
}

function cards(data) {
  const ev = data.methods.find((m) => m.key === "v2_evidential").summary;
  const all = data.methods.filter((m) => m.summary.cov_perdim_joint);
  const lo = Math.min(...all.map((m) => m.summary.cov_perdim_joint[0]));
  const hi = Math.max(...all.map((m) => m.summary.cov_perdim_joint[0]));
  const c = all.map((m) => m.summary.contempt_cov_joint[0]);
  const shrink = 1 - ev.area_joint[0] / ev.area_bonf[0];
  $("cards").innerHTML = [
    [`${(lo * 100).toFixed(1)}–${(hi * 100).toFixed(1)}%`, "Per-axis 90% intervals: how often the true VA point is inside both, across all methods"],
    [`${(ev.cov_joint[0] * 100).toFixed(1)}%`, `Our joint region (evidential), ${(shrink * 100).toFixed(0)}% smaller than Bonferroni for the same guarantee`],
    [`${(Math.min(...c) * 100).toFixed(0)}–${(Math.max(...c) * 100).toFixed(0)}%`, "Coverage on contempt, the expression no model saw in training or calibration"],
  ].map(([b, t], i) => `<div class="card reveal ${["bad", "", "warn"][i]}" style="--d:${i * 0.1}s"><div class="big" data-final="${b}">${b.replace(/\d+(\.\d+)?/g, (m) => m.replace(/\d/g, "0"))}</div><div class="sub">${t}</div></div>`).join("");
  $("cards").querySelectorAll(".card").forEach((c) => fx.observe(c));
}

function novelty(data) {
  const nv = data.novelty, el = $("nov");
  if (!nv) { el.innerHTML = "<tbody><tr><td>Novelty results not included in this build.</td></tr></tbody>"; return; }
  const NAMES = { sigma: "Predictive uncertainty (Σσ)", maha: "Mahalanobis", maha_cls: "Class-cond. Mahalanobis", knn: "k-NN (k=50)" };
  const MN = { v2_evidential: "Evidential", v2_gaussian_nll: "Gaussian", v2_baseline: "Deterministic" };
  let body = "";
  for (const [mk, ml] of Object.entries(MN)) {
    const r = nv[mk]; if (!r) continue;
    for (const [dk, dl] of Object.entries(NAMES)) {
      const d = r[dk]; if (!d) continue;
      body += `<tr><td>${ml} · ${dl}</td>${cell(d.auroc[0], d.auroc[1])}${cell(d.tpr[0], d.tpr[1])}${cell(d.fpr[0], d.fpr[1])}${cell(d.shift_auroc[0], d.shift_auroc[1])}</tr>`;
    }
  }
  el.innerHTML = "<thead><tr><th>Model · detector</th><th>Contempt AUROC</th><th>Contempt flagged</th><th>Seen flagged</th><th>Corruption AUROC</th></tr></thead><tbody>" + body + "</tbody>";
  $("nov-note").textContent = "Contempt faces' nearest training images are mostly neutral and happy: they look in-distribution, so detectors that only look at the input cannot single them out.";
}

fetch("/emotion-modeling/results/data.json").then((r) => r.json()).then((data) => {
  cards(data);
  const draw = () => {
    table($("cov"), COV, data, $("seeds-cov").checked);
    table($("acc"), ACC, data, $("seeds-acc").checked);
  };
  $("seeds-cov").onchange = draw; $("seeds-acc").onchange = draw; draw();
  const sel = $("method");
  sel.innerHTML = data.methods.map((m) => `<option value="${m.key}">${m.label}</option>`).join("");
  sel.value = "v2_evidential";
  sel.onchange = () => chart(data, sel.value); chart(data, sel.value);
  novelty(data);
}).catch((e) => { document.querySelector("main").insertAdjacentHTML("beforeend", `<p class="below">Could not load data.json: ${e}</p>`); });

/* ---------------- Motion (mirrors the landing page) ---------------- */
var fx; // assigned below, used by cards()
(function motion() {
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Count every number inside a string up from zero, keeping its decimals.
  function countUp(el) {
    const final = el.dataset.final;
    if (!final) return;
    if (reduced) { el.textContent = final; return; }
    const t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / 1500), e = 1 - (1 - k) ** 3;
      el.textContent = final.replace(/\d+(\.\d+)?/g, (m) => {
        const d = (m.split(".")[1] || "").length;
        return (parseFloat(m) * e).toFixed(d);
      });
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  fx = new IntersectionObserver((es) => {
    for (const e of es) {
      if (!e.isIntersecting) continue;
      e.target.classList.add("in");
      e.target.querySelectorAll("[data-final]").forEach(countUp);
      fx.unobserve(e.target);
    }
  }, { threshold: 0.1, rootMargin: "0px 0px -30px 0px" });
  document.querySelectorAll(".reveal").forEach((n) => fx.observe(n));
  requestAnimationFrame(() => document.querySelectorAll("header.hero .reveal").forEach((n) => n.classList.add("in")));

  // Scroll progress + nav border.
  const bar = document.getElementById("progress"), nav = document.querySelector(".nav");
  const onScroll = () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? scrollY / max : 0})`;
    nav.classList.toggle("scrolled", scrollY > 10);
  };
  addEventListener("scroll", onScroll, { passive: true }); onScroll();

  // Copy the terminal commands (without prompts or comments).
  const copy = document.getElementById("copy");
  copy.onclick = async () => {
    const text = document.getElementById("cmds").innerText.split("\n")
      .map((l) => l.replace(/^\$ /, "").replace(/\s+#.*$/, "")).join("\n");
    try { await navigator.clipboard.writeText(text); copy.textContent = "Copied ✓"; }
    catch { copy.textContent = "Select & copy"; }
    setTimeout(() => (copy.textContent = "Copy"), 1800);
  };

  // Drifting dot network behind the page.
  const c = document.getElementById("mesh"), ctx = c.getContext("2d");
  const mouse = { x: -9999, y: -9999 };
  addEventListener("pointermove", (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
  let pts = [], W = 0, H = 0;
  const fit = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = c.clientWidth; H = c.clientHeight;
    c.width = W * dpr; c.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(Math.max(30, Math.min(90, (W * H) / 17000)));
    pts = Array.from({ length: n }, () => ({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25 }));
  };
  fit(); addEventListener("resize", fit);
  const draw = () => {
    ctx.clearRect(0, 0, W, H);
    for (const p of pts) {
      const dx = mouse.x - p.x, dy = mouse.y - p.y;
      if (dx * dx + dy * dy < 40000) { p.vx += dx * 2e-5; p.vy += dy * 2e-5; }
      p.vx = Math.max(-0.5, Math.min(0.5, p.vx)); p.vy = Math.max(-0.5, Math.min(0.5, p.vy));
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    }
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const d = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
      if (d < 140) {
        ctx.strokeStyle = `rgba(88,166,255,${0.14 * (1 - d / 140)})`;
        ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke();
      }
    }
    ctx.fillStyle = "rgba(139,148,158,.5)";
    for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 1.3, 0, 6.283); ctx.fill(); }
    if (!reduced) requestAnimationFrame(draw);
  };
  draw();
})();
