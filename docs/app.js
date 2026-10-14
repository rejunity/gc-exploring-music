// Static front-end for docs/batches.json. No build step, no backend.
const $ = (s) => document.querySelector(s);
const REPO = "rejunity/gc-exploring-music";
let BRANCH = new URLSearchParams(location.search).get("branch") || "main";
let BASE = "";   // prefix for clip paths: "" on main (served by Pages), raw.githubusercontent.com for exp/* branches
const KEY = "gc-ratings-v1";
const PAIRS_PER_SESSION = 12;
let DATA = { repo: "", batches: [] };
let session = null; // { batch, rater, pairs, i }

async function loadBranch(branch) {
  BRANCH = branch;
  BASE = branch === "main" ? "" : `https://raw.githubusercontent.com/${REPO}/${branch}/docs/`;
  const url = branch === "main" ? "batches.json" : `${BASE}batches.json?t=${Date.now()}`;
  try {
    DATA = await (await fetch(url, { cache: "no-store" })).json();
  } catch (e) {
    DATA = { repo: REPO, batches: [] };
  }
  const u = new URL(location); branch === "main" ? u.searchParams.delete("branch") : u.searchParams.set("branch", branch);
  history.replaceState(null, "", u);
  renderListen();
  const sel = $("#rate-batch"); sel.innerHTML = "";
  [...DATA.batches].reverse().forEach((b) => sel.add(new Option(`${b.id} — ${b.title}`, b.id)));
}

async function loadBranches() {
  const sel = $("#branch");
  let names = ["main"];
  try {
    const r = await fetch(`https://api.github.com/repos/${REPO}/branches?per_page=100`);
    if (r.ok) names = names.concat((await r.json()).map((b) => b.name).filter((n) => n.startsWith("exp/")));
  } catch (e) { /* offline or rate-limited: main only */ }
  if (!names.includes(BRANCH)) names.push(BRANCH);
  names.forEach((n) => sel.add(new Option(n, n)));
  sel.value = BRANCH;
  sel.onchange = () => loadBranch(sel.value);
}

async function init() {
  await loadBranches();
  await loadBranch(BRANCH);
  $("#repo-link").href = `https://github.com/${REPO}`;
  $("#rater").value = localStorage.getItem("gc-rater") || "";
  updateCount();
  $("#tab-listen").onclick = () => showTab("listen");
  $("#tab-rate").onclick = () => showTab("rate");
  $("#tab-generate").onclick = () => showTab("generate");
  ["#g-branch", "#g-mode", "#g-iters", "#g-seed", "#g-continue", "#g-title"].forEach((id) => ($(id).oninput = updateGenerate));
  $("#g-mode").onchange = updateGenerate; $("#g-continue").onchange = updateGenerate;
  if (BRANCH.startsWith("exp/")) $("#g-branch").value = BRANCH.slice(4);
  updateGenerate();
  $("#rate-start").onclick = startSession;
  document.querySelectorAll(".choices button").forEach((b) => (b.onclick = () => choose(b.dataset.choice)));
  $("#pair-next").onclick = nextPair;
  $("#mark-sent").onclick = () => { if (confirm("Clear the local copy of your ratings? Only do this after GitHub shows your new issue.")) { localStorage.removeItem(KEY); updateCount(); } };
  $("#download-json").onclick = downloadJson;
  $("#clear-ratings").onclick = () => { if (confirm("Delete unsent ratings?")) { localStorage.removeItem(KEY); updateCount(); } };
  if (location.hash === "#rate") showTab("rate");
  if (location.hash === "#generate") showTab("generate");
}

function showTab(name) {
  for (const t of ["listen", "rate", "generate"]) {
    $("#" + t).hidden = name !== t;
    $("#tab-" + t).classList.toggle("active", name === t);
  }
  location.hash = name === "listen" ? "" : "#" + name;
}

function genParams() {
  const slug = $("#g-branch").value.trim().toLowerCase().replace(/^exp\//, "").replace(/\s+/g, "-").replace(/[^a-z0-9._-]/g, "");
  return { branch: "exp/" + (slug || "try"), mode: $("#g-mode").value, iters: +$("#g-iters").value || 200,
           seed: +$("#g-seed").value || 0, continue: $("#g-continue").checked, title: $("#g-title").value.trim() };
}

function updateGenerate() {
  const p = genParams();
  const perIter = p.mode === "critic" ? 7.5 : 1;
  const max = p.mode === "critic" ? 800 : 2000;
  const mins = Math.round((p.iters * perIter) / 60 + 4);
  $("#g-estimate").textContent = p.iters > max ? `Too many iterations for ${p.mode} mode (max ${max}).`
    : `Estimated run time ~${mins} min (incl. ~4 min setup). Result: 8 clips on ${p.branch}.`;
  const title = `evolve: ${p.branch} ${p.mode} ${p.iters}${p.title ? " — " + p.title : ""}`;
  const body = "Started from the site's Generate tab. Parameters:\n\n```json\n" + JSON.stringify(p, null, 1) + "\n```";
  $("#g-issue").href = `https://github.com/${REPO}/issues/new?labels=evolve&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
  $("#g-issue").classList.toggle("disabled", p.iters > max);
  $("#g-dispatch").href = `https://github.com/${REPO}/actions/workflows/evolve.yml`;
  $("#g-runs").href = `https://github.com/${REPO}/actions/workflows/evolve.yml`;
  $("#g-values").textContent = `branch=${p.branch}, iters=${p.iters}, seed=${p.seed}, mode=${p.mode}, continue=${p.continue}`;
}

function renderListen() {
  const root = $("#batches");
  root.innerHTML = "";
  [...DATA.batches].reverse().forEach((b) => {
    const div = document.createElement("div");
    div.className = "card batch";
    const best = b.clips.filter((c) => c.fitness != null).sort((x, y) => y.fitness - x.fitness)[0];
    div.innerHTML = `<h2>${b.title}</h2><div class="meta">${b.id} · ${b.date}${b.notes ? " · " + b.notes : ""}</div>`;
    b.clips.forEach((c) => {
      const el = document.createElement("div");
      el.className = "clip";
      const star = best && c.id === best.id ? `<span class="star" title="best fitness in batch">★</span> ` : "";
      const info = [c.fitness != null ? `fitness ${c.fitness}` : null,
        c.p_real != null ? `critic "sounds real" ${c.p_real}` : null,
        c.amt_gap != null ? `AMT gap ${c.amt_gap > 0 ? "+" : ""}${c.amt_gap}` : null,
        c.desc ? `density ${c.desc.density} notes/bar · melodic entropy ${c.desc.entropy} bits · range ${c.desc.range} st` : null,
        c.chords ? `chords: ${c.chords}` : null].filter(Boolean).join(" · ");
      el.innerHTML = `<div class="title">${star}${c.id}<span class="label">${c.label}</span></div>
        <a href="${BASE}${c.mid}" download>.mid</a>
        ${info ? `<div class="info">${info}</div>` : ""}
        ${c.credit ? `<div class="info">${c.credit.url ? `<a href="${c.credit.url}" target="_blank" rel="noopener">${c.credit.text}</a>` : c.credit.text}</div>` : ""}
        <midi-player sound-font src="${BASE}${c.mid}"></midi-player>`;
      div.appendChild(el);
    });
    root.appendChild(div);
  });
}

function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

function startSession() {
  const batch = DATA.batches.find((b) => b.id === $("#rate-batch").value);
  const rater = $("#rater").value.trim() || "anon";
  localStorage.setItem("gc-rater", rater);
  // all unordered pairs, shuffled, random A/B side, capped at 20
  const pairs = [];
  for (let i = 0; i < batch.clips.length; i++) for (let j = i + 1; j < batch.clips.length; j++) pairs.push(Math.random() < 0.5 ? [i, j] : [j, i]);
  session = { batch, rater, pairs: shuffle(pairs).slice(0, PAIRS_PER_SESSION), i: 0 };
  $("#pair").hidden = false;
  $("#pair-done").hidden = true;
  document.querySelectorAll(".choices button").forEach((x) => (x.hidden = false));
  showPair();
}

function showPair() {
  const { batch, pairs, i } = session;
  const [a, b] = pairs[i].map((k) => batch.clips[k]);
  $("#pair-count").textContent = `pair ${i + 1} / ${pairs.length}`;
  $("#pair-batch").textContent = batch.id;
  $("#audio-a").src = BASE + a.mid; $("#audio-b").src = BASE + b.mid;
  $("#reveal-a").textContent = ""; $("#reveal-b").textContent = "";
  document.querySelectorAll(".choices button").forEach((x) => (x.disabled = false));
  $("#pair-next").hidden = true;
}

function choose(choice) {
  const { batch, pairs, i, rater } = session;
  const [a, b] = pairs[i].map((k) => batch.clips[k]);
  const ratings = JSON.parse(localStorage.getItem(KEY) || "[]");
  ratings.push({ t: new Date().toISOString(), rater, branch: BRANCH, batch: batch.id, a: a.id, b: b.id, choice });
  localStorage.setItem(KEY, JSON.stringify(ratings));
  $("#reveal-a").textContent = `${a.id} (${a.label})`; $("#reveal-b").textContent = `${b.id} (${b.label})`;
  document.querySelectorAll(".choices button").forEach((x) => (x.disabled = true));
  $("#pair-next").hidden = false;
  updateCount();
}

function nextPair() {
  session.i++;
  if (session.i >= session.pairs.length) {
    document.querySelectorAll(".choices button").forEach((x) => (x.hidden = true));
    $("#pair-next").hidden = true; $("#pair-done").hidden = false;
    $("#submit").scrollIntoView({ behavior: "smooth" });
    return;
  }
  showPair();
}

function updateCount() {
  const ratings = JSON.parse(localStorage.getItem(KEY) || "[]");
  $("#n-ratings").textContent = ratings.length;
  const link = $("#submit-link");
  const url = ratings.length ? issueUrl(ratings) : "#";
  const tooLong = url.length > 7500;
  link.href = tooLong ? "#" : url;
  link.classList.toggle("disabled", !ratings.length || tooLong);
  link.textContent = tooLong ? "Too many ratings for one issue: use Download JSON" : "Open GitHub with my ratings ↗";
  $("#mark-sent").disabled = !ratings.length;
}

function issueUrl(ratings) {
  const rater = localStorage.getItem("gc-rater") || "anon";
  const title = `ratings: ${rater} ${new Date().toISOString().slice(0, 10)} (${ratings.length})`;
  const body = "```json\n" + JSON.stringify(ratings) + "\n```";
  return `https://github.com/${REPO}/issues/new?labels=ratings&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
}

function downloadJson() {
  const ratings = localStorage.getItem(KEY) || "[]";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([ratings], { type: "application/json" }));
  a.download = `ratings-${localStorage.getItem("gc-rater") || "anon"}-${Date.now()}.json`;
  a.click();
}

init();
