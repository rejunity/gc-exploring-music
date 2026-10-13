// Static front-end for docs/batches.json. No build step, no backend.
const $ = (s) => document.querySelector(s);
const KEY = "gc-ratings-v1";
let DATA = { repo: "", batches: [] };
let session = null; // { batch, rater, pairs, i }

async function init() {
  DATA = await (await fetch("batches.json", { cache: "no-store" })).json();
  $("#repo-link").href = `https://github.com/${DATA.repo}`;
  renderListen();
  const sel = $("#rate-batch");
  [...DATA.batches].reverse().forEach((b) => sel.add(new Option(`${b.id} — ${b.title}`, b.id)));
  $("#rater").value = localStorage.getItem("gc-rater") || "";
  updateCount();
  $("#tab-listen").onclick = () => showTab("listen");
  $("#tab-rate").onclick = () => showTab("rate");
  $("#rate-start").onclick = startSession;
  document.querySelectorAll(".choices button").forEach((b) => (b.onclick = () => choose(b.dataset.choice)));
  $("#pair-next").onclick = nextPair;
  $("#submit-issue").onclick = submitIssue;
  $("#download-json").onclick = downloadJson;
  $("#clear-ratings").onclick = () => { if (confirm("Delete unsent ratings?")) { localStorage.removeItem(KEY); updateCount(); } };
  if (location.hash === "#rate") showTab("rate");
}

function showTab(name) {
  $("#listen").hidden = name !== "listen";
  $("#rate").hidden = name !== "rate";
  $("#tab-listen").classList.toggle("active", name === "listen");
  $("#tab-rate").classList.toggle("active", name === "rate");
  location.hash = name === "rate" ? "#rate" : "";
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
        c.desc ? `density ${c.desc.density} notes/bar · melodic entropy ${c.desc.entropy} bits · range ${c.desc.range} st` : null,
        c.chords ? `chords: ${c.chords}` : null].filter(Boolean).join(" · ");
      el.innerHTML = `<div class="title">${star}${c.id}<span class="label">${c.label}</span></div>
        <a href="${c.mid}" download>.mid</a>
        ${info ? `<div class="info">${info}</div>` : ""}
        ${c.credit ? `<div class="info">${c.credit.url ? `<a href="${c.credit.url}" target="_blank" rel="noopener">${c.credit.text}</a>` : c.credit.text}</div>` : ""}
        <midi-player sound-font src="${c.mid}"></midi-player>`;
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
  session = { batch, rater, pairs: shuffle(pairs).slice(0, 20), i: 0 };
  $("#pair").hidden = false;
  showPair();
}

function showPair() {
  const { batch, pairs, i } = session;
  const [a, b] = pairs[i].map((k) => batch.clips[k]);
  $("#pair-count").textContent = `pair ${i + 1} / ${pairs.length}`;
  $("#pair-batch").textContent = batch.id;
  $("#audio-a").src = a.mid; $("#audio-b").src = b.mid;
  $("#reveal-a").textContent = ""; $("#reveal-b").textContent = "";
  document.querySelectorAll(".choices button").forEach((x) => (x.disabled = false));
  $("#pair-next").hidden = true;
}

function choose(choice) {
  const { batch, pairs, i, rater } = session;
  const [a, b] = pairs[i].map((k) => batch.clips[k]);
  const ratings = JSON.parse(localStorage.getItem(KEY) || "[]");
  ratings.push({ t: new Date().toISOString(), rater, batch: batch.id, a: a.id, b: b.id, choice });
  localStorage.setItem(KEY, JSON.stringify(ratings));
  $("#reveal-a").textContent = `${a.id} (${a.label})`; $("#reveal-b").textContent = `${b.id} (${b.label})`;
  document.querySelectorAll(".choices button").forEach((x) => (x.disabled = true));
  $("#pair-next").hidden = false;
  updateCount();
}

function nextPair() {
  session.i++;
  if (session.i >= session.pairs.length) { $("#pair").hidden = true; alert("Batch done. Thank you! Now submit your ratings below."); return; }
  showPair();
}

function updateCount() { $("#n-ratings").textContent = JSON.parse(localStorage.getItem(KEY) || "[]").length; }

function submitIssue() {
  const ratings = JSON.parse(localStorage.getItem(KEY) || "[]");
  if (!ratings.length) return alert("No ratings yet.");
  const rater = localStorage.getItem("gc-rater") || "anon";
  const title = `ratings: ${rater} ${new Date().toISOString().slice(0, 10)} (${ratings.length})`;
  const body = "```json\n" + JSON.stringify(ratings) + "\n```";
  const url = `https://github.com/${DATA.repo}/issues/new?labels=ratings&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
  if (url.length > 7500) return alert("Too many ratings for one issue URL; use Download JSON instead.");
  window.open(url, "_blank");
  if (confirm("Did you press 'Submit new issue' on GitHub? OK clears the local copy.")) { localStorage.removeItem(KEY); updateCount(); }
}

function downloadJson() {
  const ratings = localStorage.getItem(KEY) || "[]";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([ratings], { type: "application/json" }));
  a.download = `ratings-${localStorage.getItem("gc-rater") || "anon"}-${Date.now()}.json`;
  a.click();
}

init();
