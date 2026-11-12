// engine.js: turns trace events into UI state and animation cues.
// It is the single source of truth for both the live stream and recorded replay:
// the same events, in the same order, always produce the same state.

export const TOOLS = ["search", "python", "sql", "email", "docs"];

export function emptyState() {
  return {
    task: "", model: "", verifierOn: true, status: "idle", answer: "", t: 0,
    plan: [], replans: 0,
    rows: [], rowById: new Map(),
    counters: { tools: 0, retries: 0, replans: 0, tokens: 0, errors: 0 },
    stepState: {},
  };
}

const short = (v, n = 90) => {
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return s && s.length > n ? s.slice(0, n - 1) + "…" : s ?? "";
};

// The first line of an arguments object that is worth showing in the log.
export function argSummary(args) {
  if (!args || typeof args !== "object") return "";
  const v = args.query ?? args.code ?? args.filename ?? (args.to ? `${args.to}: ${args.subject}` : null);
  return short(v ?? args, 80).replace(/\s+/g, " ");
}

export function resultSummary(res) {
  if (res == null) return "";
  if (typeof res === "string") return short(res.replace(/\s+/g, " "), 90);
  if (Array.isArray(res)) return res.length ? short(res[0].text ?? res[0], 90) : "no hits";
  if (res.rows) return short(res.rows.slice(0, 3), 80);
  return short(res, 90);
}

/**
 * Apply one event to the state. Returns a list of cues for the view:
 *   {type:"packet", from, to, tone, tool?}  {type:"pulse", node, tone}  {type:"plan"}  {type:"row", row}
 *   {type:"status"}  {type:"reset"}
 */
export function applyEvent(state, ev) {
  const cues = [];
  const a = ev.attributes || {};
  state.t = ev.t;
  const isStart = ev.event === "start";

  const addRow = (id, kind, label, detail, tone = "") => {
    const row = { id, kind, label, detail, tone, t: ev.t, ms: null, events: [ev], open: false, running: true };
    state.rows.push(row);
    state.rowById.set(id, row);
    cues.push({ type: "row", row });
    return row;
  };
  const finishRow = (id, detail, tone) => {
    const row = state.rowById.get(id);
    if (!row) return null;
    row.running = false;
    row.ms = Math.round((ev.t - row.t) * 1000);
    if (detail != null) row.detail = detail;
    if (tone) row.tone = tone;
    row.events.push(ev);
    cues.push({ type: "row", row });
    return row;
  };

  switch (ev.name) {
    case "invoke_agent":
      if (isStart) {
        Object.assign(state, emptyState());
        state.t = ev.t;
        state.task = a["agentflow.task"] || "";
        state.model = a["gen_ai.request.model"] || "";
        state.verifierOn = a["agentflow.verifier_enabled"] !== false;
        state.status = "running";
        cues.push({ type: "reset" });
        addRow(ev.span_id, "run", "RUN", short(state.task, 140));
      } else {
        state.status = a["agentflow.status"] || (ev.status === "error" ? "error" : "completed");
        state.answer = a["agentflow.answer"] || a["agentflow.error"] || "";
        finishRow(ev.span_id, `${state.status}`, state.status === "completed" ? "ok" : "bad");
        cues.push({ type: "status" });
      }
      break;
    case "chat": {
      const phase = a["agentflow.phase"] || "";
      if (isStart) {
        addRow(ev.span_id, "llm", `LLM ${phase}`, "thinking…");
        state.pendingPhase = { phase, step: a["agentflow.step_id"] };
        cues.push({ type: "think", on: true, phase, step: a["agentflow.step_id"] });
      } else {
        const tokens = (a["gen_ai.usage.input_tokens"] || 0) + (a["gen_ai.usage.output_tokens"] || 0);
        state.counters.tokens += tokens;
        const text = a["agentflow.output"] || (a["agentflow.tool_calls"]?.[0] ? JSON.stringify(a["agentflow.tool_calls"][0].args) : "");
        finishRow(ev.span_id, short(text, 110) + (tokens ? `  (${tokens} tok)` : ""));
        cues.push({ type: "think", on: false });
      }
      break;
    }
    case "agentflow.plan":
      if (isStart) {
        const prevStates = state.stepState;
        state.plan = (a["agentflow.plan"] || []).map((s) => ({ ...s, state: prevStates[s.id] || "pending" }));
        state.replans = a["agentflow.replan"] || 0;
        if (state.replans > state.counters.replans) state.counters.replans = state.replans;
        const label = state.replans ? `REPLAN ${state.replans}` : "PLAN";
        const row = addRow(ev.span_id, "plan", label, state.plan.map((s) => `${s.id}:${s.tool}`).join(" → "),
          state.replans ? "warn" : "");
        row.running = false;
        if (a["agentflow.plan_fallback"]) row.detail += "  (fallback router)";
        cues.push({ type: "plan" });
      }
      break;
    case "execute_tool": {
      const tool = a["gen_ai.tool.name"] || "";
      if (isStart) {
        const step = a["agentflow.step_id"];
        state.stepState[step] = "running";
        setStep(state, step, "running");
        state.counters.tools += 1;
        addRow(ev.span_id, "tool", `TOOL ${tool}`, argSummary(a["agentflow.arguments"]) +
          (a["agentflow.reused_args"] ? "  (same arguments)" : ""));
        state.rowById.get(ev.span_id).tool = tool;
        cues.push({ type: "packet", from: "planner", to: tool, tone: "go" }, { type: "pulse", node: tool, tone: "go" });
      } else {
        const row = state.rowById.get(ev.span_id);
        const bad = ev.status === "error";
        if (bad) state.counters.errors += 1;
        const detail = bad ? `${a["agentflow.error_kind"] || "error"}: ${short(a["agentflow.error"], 100)}`
          : resultSummary(a["agentflow.result"]);
        finishRow(ev.span_id, (row ? row.detail + "\n" : "") + "→ " + detail, bad ? "bad" : "ok");
        if (row) row.result = detail;
        cues.push({ type: "toolDone", tool: row?.tool || tool, ms: a["agentflow.latency_ms"] ?? row?.ms, ok: !bad });
        cues.push({ type: "packet", from: row?.tool || tool, to: "verifier", tone: bad ? "bad" : "ok" },
          { type: "pulse", node: row?.tool || tool, tone: bad ? "bad" : "ok" });
      }
      break;
    }
    case "agentflow.verify": {
      const step = a["agentflow.step_id"];
      if (isStart) {
        addRow(ev.span_id, "verify", `VERIFY ${step}`, "checking…");
        cues.push({ type: "pulse", node: "verifier", tone: "go" });
      } else {
        const v = a["agentflow.verdict"] || "pass";
        const tone = v === "pass" ? "ok" : v === "fail" ? "bad" : "warn";
        const forced = a["agentflow.verifier_enabled"] === false;
        const would = a["agentflow.would_flag"];
        let detail = v + (a["agentflow.reason"] && v !== "pass" ? `: ${short(a["agentflow.reason"], 120)}` : "");
        if (forced) detail = "verifier off, passed through" + (would ? " (it would have flagged this)" : "");
        finishRow(ev.span_id, detail, forced && would ? "warn" : tone);
        if (v === "retry") state.counters.retries += 1;
        state.stepState[step] = v === "pass" ? "passed" : v === "fail" ? "failed" : "retry";
        setStep(state, step, state.stepState[step]);
        cues.push({ type: "packet", from: "verifier", to: "planner", tone, label: v },
          { type: "verdict", verdict: v, forced });
      }
      break;
    }
    case "agentflow.finish":
      if (isStart) addRow(ev.span_id, "plan", "FINISH", short(a["agentflow.output"] || "", 120), "bad").running = false;
      break;
    default:
      break;
  }
  return cues;
}

function setStep(state, id, st) {
  const s = state.plan.find((p) => p.id === id);
  if (s) s.state = st;
}

/** Build a display timeline: real timestamps, compressed so long model waits do not stall a replay. */
export function displayTimes(events) {
  const out = [];
  let acc = 0, prev = null;
  for (const ev of events) {
    if (prev != null) {
      const gap = Math.max(0, ev.t - prev);
      acc += Math.min(1.3, 0.22 + Math.sqrt(gap) * 0.28);
    }
    out.push(acc);
    prev = ev.t;
  }
  return out;
}
