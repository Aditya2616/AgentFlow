// app.js: wires the controls, the live SSE stream, and recorded replay to the engine and graph.
import { applyEvent, displayTimes, emptyState, TOOLS } from "./engine.js";
import { Graph } from "./graph.js";

const $ = (s) => document.querySelector(s);
const state = emptyState();
const graph = new Graph($("#graph"));
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmt = (s) => `${s.toFixed(1)} s`;
let backend = false;
let mode = "recorded";
let player = null;     // recorded replay
let source = null;     // live EventSource
let liveRun = null;
let recordings = [];
let suiteTasks = [];
let planSig = "";

// ---------------------------------------------------------------- rendering
const rowsEl = $("#rows");
const rowEls = new Map();

function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function upsertRow(row) {
  let li = rowEls.get(row.id);
  const nearBottom = rowsEl.scrollHeight - rowsEl.scrollTop - rowsEl.clientHeight < 80;
  if (!li) {
    li = document.createElement("li");
    li.addEventListener("click", () => {
      const open = li.classList.toggle("open");
      row.open = open;
      renderFull(li, row);
    });
    rowEls.set(row.id, li);
    rowsEl.appendChild(li);
    if (nearBottom) requestAnimationFrame(() => (rowsEl.scrollTop = rowsEl.scrollHeight));
  }
  li.className = `row k-${row.kind} tone-${row.tone || "none"}${row.running ? " running" : ""}${row.open ? " open" : ""}`;
  li.innerHTML = `<span class="tag">${escapeHtml(row.label)}</span><span class="detail">${escapeHtml(row.detail)}</span>` +
    `<span class="ms">${row.ms != null ? row.ms + " ms" : ""}</span><span class="t">${row.t.toFixed(1)}s</span>`;
  if (row.open) renderFull(li, row);
}

function renderFull(li, row) {
  li.querySelector(".full")?.remove();
  if (!row.open) return;
  const d = document.createElement("div");
  d.className = "full";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(row.events.map((e) => ({ event: e.event, t: e.t, status: e.status, ...e.attributes })), null, 2);
  d.appendChild(pre);
  li.appendChild(d);
}

function renderPlan() {
  const sig = JSON.stringify(state.plan.map((s) => [s.id, s.state]));
  if (sig === planSig) return;
  planSig = sig;
  const host = $("#plan");
  host.innerHTML = "";
  state.plan.forEach((s) => {
    const d = document.createElement("span");
    d.className = `step ${s.state}`;
    d.title = s.goal;
    d.innerHTML = `<span class="mark"></span><span class="n">${escapeHtml(s.id)}</span><span class="tool">${escapeHtml(s.tool)}</span><span class="g">${escapeHtml(s.goal)}</span>`;
    host.appendChild(d);
  });
}

function renderAnswer() {
  const box = $("#answer");
  if (state.status === "idle" || state.status === "running") { box.hidden = true; return; }
  box.hidden = false;
  const ok = state.status === "completed";
  const chip = $("#status-chip");
  chip.textContent = state.status;
  chip.className = "chip " + (ok ? "ok" : "bad");
  $("#answer-text").textContent = state.answer || "(no answer)";
  const c = state.counters;
  $("#stats").innerHTML = [["tool calls", c.tools], ["tool errors", c.errors], ["retries", c.retries], ["re-plans", c.replans], ["tokens", c.tokens]]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

function updateSummary() {
  const n = state.rows.length;
  $("#trace-sum").textContent = `${n} events · ${state.counters.tools} tool calls`;
}

function handle(cues, instant) {
  for (const c of cues) {
    switch (c.type) {
      case "reset": graph.reset(); rowsEl.innerHTML = ""; rowEls.clear(); planSig = ""; break;
      case "row": upsertRow(c.row); break;
      case "packet": graph.packet(c.from, c.to, c.tone, instant); break;
      case "pulse": graph.pulse(c.node, c.tone, instant); break;
      case "think": graph.think(c.on, c.phase, c.step); break;
      case "toolDone": graph.toolDone(c.tool, c.ms, c.ok); break;
      case "verdict": graph.verdict(c.verdict, c.forced); break;
      default: break;
    }
  }
  renderPlan(); renderAnswer(); updateSummary();
}

function feed(ev, instant = false) { handle(applyEvent(state, ev), instant); }

// ---------------------------------------------------------------- recorded replay
class Player {
  constructor(events) {
    this.events = events;
    this.disp = displayTimes(events);
    this.total = this.disp.length ? this.disp[this.disp.length - 1] + 1.0 : 0;
    this.i = 0; this.pos = 0; this.playing = false; this.speed = 1; this.last = 0;
    this.tick = this.tick.bind(this);
  }
  seek(p, keepPlaying = false) {
    this.pos = Math.max(0, Math.min(this.total, p));
    // a fresh start event resets the engine, so rebuild by replaying everything up to p instantly
    this.i = 0;
    state.rows = []; Object.assign(state, emptyState());
    graph.reset(); rowsEl.innerHTML = ""; rowEls.clear(); planSig = "";
    while (this.i < this.events.length && this.disp[this.i] <= this.pos) feed(this.events[this.i++], true);
    graph.think(false);
    ui();
    if (!keepPlaying) this.pause();
  }
  play() {
    if (this.pos >= this.total) this.seek(0, true);
    this.playing = true; this.last = performance.now(); requestAnimationFrame(this.tick); ui();
  }
  pause() { this.playing = false; ui(); }
  tick(now) {
    if (!this.playing) return;
    this.pos += ((now - this.last) / 1000) * this.speed; this.last = now;
    while (this.i < this.events.length && this.disp[this.i] <= this.pos) feed(this.events[this.i++], false);
    if (this.pos >= this.total) { this.pos = this.total; this.playing = false; }
    ui();
    if (this.playing) requestAnimationFrame(this.tick);
  }
}

function ui() {
  const p = player;
  $("#play-icon").setAttribute("d", p && p.playing ? "M7 5h4v14H7zM13 5h4v14h-4z" : "M8 5v14l11-7z");
  $("#play-btn").setAttribute("aria-label", p && p.playing ? "Pause" : "Play");
  if (!p) return;
  $("#scrubber").value = p.total ? String(Math.round((p.pos / p.total) * 1000)) : "0";
  const evAt = p.events[Math.max(0, p.i - 1)];
  $("#t-now").textContent = evAt ? `run time ${fmt(evAt.t)}` : "0.0 s";
  $("#t-total").textContent = `of ${fmt(p.events[p.events.length - 1]?.t ?? 0)}`;
}

async function loadRecording(rec, autoplay) {
  stopLive();
  const res = await fetch(`recordings/${rec.file}`);
  const events = (await res.text()).split("\n").filter(Boolean).map((l) => JSON.parse(l));
  player = new Player(events);
  player.speed = Number($("#speed .on").dataset.speed);
  badge(`Recorded run · ${rec.model} · ${rec.date}`, false);
  $("#rec-note").textContent = rec.note || "";
  $("#dl").href = `recordings/${rec.file}`; $("#dl").hidden = false;
  $("#dl").setAttribute("download", rec.file);
  $("#transport").classList.remove("live-mode");
  if (reduced || !autoplay) player.seek(player.total); else { player.seek(0, true); player.play(); }
}

function badge(text, live) {
  const b = $("#badge");
  b.textContent = text; b.className = "badge" + (live ? " live" : "");
}

// ---------------------------------------------------------------- live
function stopLive() { if (source) { source.close(); source = null; } }

async function startLive(e) {
  e.preventDefault();
  if (!backend) return;
  const sel = $("#task-select").value;
  const task = sel === "__custom" ? $("#task-custom").value.trim() : suiteTasks.find((t) => t.id === sel)?.prompt;
  if (!task || task.length < 3) { $("#live-note").textContent = "Write a task first."; return; }
  const llm = $("#llm-select").value;
  const body = { task, llm, verifier: $("#verifier-on").checked, faults: Number($("#fault-select").value), mistake: $("#mistake-select").value };
  $("#run-btn").disabled = true; $("#live-note").textContent = "";
  player = null; stopLive();
  state.rows = []; Object.assign(state, emptyState()); graph.reset(); rowsEl.innerHTML = ""; rowEls.clear(); planSig = ""; renderPlan(); renderAnswer();
  try {
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
    liveRun = await r.json();
  } catch (err) {
    $("#live-note").textContent = `Could not start the run: ${err.message}`; $("#run-btn").disabled = false; return;
  }
  $("#transport").classList.add("live-mode");
  badge(`Live · ${llm === "scripted" ? "scripted model" : llm}`, true);
  $("#dl").href = `/api/runs/${liveRun.run_id}/trace`; $("#dl").hidden = false; $("#dl").setAttribute("download", `${liveRun.run_id}.jsonl`);
  $("#live-note").textContent = llm === "scripted" ? "" : "Small models on CPU are slow: expect tens of seconds per model call.";
  source = new EventSource(`/api/runs/${liveRun.run_id}/events`);
  source.addEventListener("trace", (m) => feed(JSON.parse(m.data)));
  source.addEventListener("done", () => { stopLive(); $("#run-btn").disabled = false; graph.think(false); });
  source.onerror = () => { if (source && source.readyState === EventSource.CLOSED) { $("#run-btn").disabled = false; } };
}

// ---------------------------------------------------------------- controls
function setMode(m) {
  mode = m;
  for (const b of document.querySelectorAll(".seg [data-mode]")) b.setAttribute("aria-selected", String(b.dataset.mode === m));
  $("#recorded-panel").hidden = m !== "recorded";
  $("#live-panel").hidden = m !== "live";
  if (m === "live") {
    if (player) player.pause();
    $("#transport").classList.add("live-mode");
    badge(backend ? "Live · ready" : "Live runs need the local server", false);
  } else {
    $("#transport").classList.toggle("live-mode", !player);
    if (recordings.length) loadRecording(recordings[$("#rec-select").selectedIndex], false);
  }
}

function wire() {
  $("#tab-recorded").addEventListener("click", () => setMode("recorded"));
  $("#tab-live").addEventListener("click", () => { if (backend) setMode("live"); });
  $("#rec-select").addEventListener("change", (e) => loadRecording(recordings[e.target.selectedIndex], true));
  $("#play-btn").addEventListener("click", () => { if (!player) return; player.playing ? player.pause() : player.play(); });
  $("#scrubber").addEventListener("input", (e) => { if (player) player.seek((Number(e.target.value) / 1000) * player.total, player.playing); });
  $("#speed").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of $("#speed").children) x.classList.toggle("on", x === b);
    const sp = Number(b.dataset.speed); graph.speed = sp; if (player) player.speed = sp;
  });
  $("#filters").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of $("#filters").children) x.classList.toggle("on", x === b);
    rowsEl.dataset.filter = b.dataset.f;
  });
  $("#sheet-handle").addEventListener("click", () => {
    const open = $("#trace").classList.toggle("open");
    $("#sheet-handle").setAttribute("aria-expanded", String(open));
  });
  $("#live-panel").addEventListener("submit", startLive);
  $("#task-select").addEventListener("change", (e) => { $("#custom-field").hidden = e.target.value !== "__custom"; });
  $("#llm-select").addEventListener("change", (e) => { $("#mistake-field").hidden = e.target.value !== "scripted"; });
}

async function boot() {
  wire();
  try {
    const rl = await fetch("recordings/index.json");
    recordings = (await rl.json()).recordings;
  } catch { recordings = []; }
  const sel = $("#rec-select");
  recordings.forEach((r) => { const o = document.createElement("option"); o.textContent = r.title; sel.appendChild(o); });

  try {
    const h = await fetch("/health", { cache: "no-store" });
    backend = h.ok && (await h.json()).status === "ok";
    if (backend) {
      suiteTasks = await (await fetch("/api/tasks")).json();
      const ts = $("#task-select");
      const bands = { single: "One tool", two_step: "Two steps", multi: "Three steps", adversarial: "Should be refused" };
      for (const [band, label] of Object.entries(bands)) {
        const g = document.createElement("optgroup"); g.label = label;
        suiteTasks.filter((t) => t.band === band).forEach((t) => { const o = document.createElement("option"); o.value = t.id; o.textContent = `${t.title}: ${t.prompt}`; g.appendChild(o); });
        ts.appendChild(g);
      }
      const c = document.createElement("option"); c.value = "__custom"; c.textContent = "Write my own task…"; ts.appendChild(c);
      const health = await (await fetch("/health")).json();
      const ls = $("#llm-select");
      [["local", `${health.local_model.split("/").pop()} (local, CPU)`], ["local-large", `${health.large_model.split("/").pop()} (local, larger)`], ["scripted", "Scripted model (shows the loop mechanics)"]]
        .forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; ls.appendChild(o); });
    }
  } catch { backend = false; }
  if (!backend) {
    const t = $("#tab-live"); t.disabled = true; t.title = "Live runs need the local server (pip install, then agentflow serve)";
    t.textContent = "Live run (local only)"; t.style.opacity = ".55";
  }
  if (recordings.length) {
    await loadRecording(recordings[0], true);
  } else {
    badge("No recordings found", false);
  }
}
boot();
export { TOOLS };
