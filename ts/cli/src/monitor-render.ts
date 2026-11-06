import type { HostMetricsSnapshot } from "./host-metrics.js";
import type { MonitorHistorySample } from "./monitor-history.js";
import type { MonitorSnapshot } from "./monitor-types.js";
import { monitorGpuReading } from "./monitor-gpu.js";

type Tone = "text" | "muted" | "border" | "cyan" | "blue" | "violet" | "green" | "amber" | "red";
type Span = { text: string; tone: Tone; bold: boolean };
type Line = Span[];
const palette: Record<Tone, number> = { text: 252, muted: 245, border: 240, cyan: 81, blue: 111, violet: 141, green: 114, amber: 221, red: 203 };
const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });
const s = (text: string, tone: Tone = "text", bold = false): Span => ({ text, tone, bold });
const line = (text: string, tone: Tone = "text"): Line => [s(text, tone)];

export type MonitorRenderOptions = {
  columns?: number;
  rows?: number;
  intervalMs?: number;
  interactive?: boolean;
  color?: boolean;
  history?: readonly MonitorHistorySample[];
  nextStep?: string;
};

/** Treat every external value as text, including terminal and bidi controls. */
function clean(text: string): string {
  return text.replace(/[\u0000-\u001f\u007f-\u009f\u2028-\u202e\u2066-\u2069]/g, " ");
}

function cellWidth(grapheme: string): number {
  const code = grapheme.codePointAt(0) ?? 0;
  if (/^[\p{Mark}\u200b-\u200f\ufeff]+$/u.test(grapheme)) return 0;
  if (/\p{Emoji_Presentation}|\uFE0F/u.test(grapheme)) return 2;
  return code >= 0x1100 && (code <= 0x115f || code === 0x2329 || code === 0x232a ||
    (code >= 0x2e80 && code <= 0xa4cf && code !== 0x303f) ||
    (code >= 0xac00 && code <= 0xd7a3) || (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0xfe10 && code <= 0xfe19) || (code >= 0xfe30 && code <= 0xfe6f) ||
    (code >= 0xff00 && code <= 0xff60) || (code >= 0xffe0 && code <= 0xffe6) ||
    (code >= 0x20000 && code <= 0x3fffd)) ? 2 : 1;
}

/** Terminal cells, not UTF-16 units: long model names must not wrap a panel. */
export function monitorTextWidth(text: string): number {
  return Array.from(segmenter.segment(clean(text))).reduce((n, part) => n + cellWidth(part.segment), 0);
}

function widthOf(spans: Line): number { return spans.reduce((n, span) => n + monitorTextWidth(span.text), 0); }

function clipped(spans: Line, width: number): Line {
  const result: Line = [];
  const overflow = widthOf(spans) > width;
  let remaining = Math.max(0, width - (overflow ? 1 : 0));
  outer: for (const span of spans) {
    let text = "";
    for (const { segment } of segmenter.segment(clean(span.text))) {
      const cells = cellWidth(segment);
      if (cells > remaining) { if (text) result.push({ ...span, text }); break outer; }
      text += segment;
      remaining -= cells;
    }
    if (text) result.push({ ...span, text });
  }
  if (overflow && width > 0) result.push(s("…", "muted"));
  return result;
}

function padded(spans: Line, width: number): Line {
  const result = clipped(spans, width);
  return [...result, s(" ".repeat(Math.max(0, width - widthOf(result))))];
}

function paint(spans: Line, color: boolean): string {
  return spans.map((span) => color && span.text
    ? `\u001b[${span.bold ? "1;" : ""}38;5;${palette[span.tone]}m${clean(span.text)}\u001b[0m`
    : clean(span.text)).join("");
}

export function formatBytes(bytes: number | null): string {
  if (bytes === null || !Number.isFinite(bytes)) return "unavailable";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++; }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function percent(value: number | null): string { return value === null || !Number.isFinite(value) ? "unavailable" : `${value.toFixed(1)}%`; }
function power(value: number | null): string { return value === null || !Number.isFinite(value) ? "unavailable" : value >= 1000 ? `${(value / 1000).toFixed(2)} W` : `${value.toFixed(0)} mW`; }
function age(value: number | null): string { return value === null ? "no sample" : `${(value / 1000).toFixed(1)}s old`; }
function ratio(value: number | null, total: number | null): number | null {
  return value === null || total === null || !Number.isFinite(value) || !Number.isFinite(total) || total <= 0 ? null : value / total * 100;
}
function uptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  return `${hours >= 24 ? `${Math.floor(hours / 24)}d ` : ""}${hours % 24}h ${Math.floor(seconds / 60) % 60}m`;
}

function bar(value: number | null, width: number, tone: Tone): Line {
  if (value === null || !Number.isFinite(value)) return [s("·".repeat(width), "muted")];
  const fill = Math.round(Math.min(100, Math.max(0, value)) / 100 * width);
  return [s("━".repeat(fill), tone, true), s("─".repeat(width - fill), "border")];
}

function meter(label: string, value: number | null, detail: string, width: number, tone: Tone): Line {
  const labelWidth = 12;
  const barWidth = Math.max(3, Math.min(28, width - 39));
  return [...padded(line(label), labelWidth), s(" "), ...bar(value, barWidth, tone), s(" "), s(detail, tone)];
}

/** Missing samples are dots; a reported zero is the bottom block. */
export function renderMonitorSparkline(values: readonly (number | null)[], width: number, ceiling = 100): string {
  const size = Math.max(0, Math.floor(width));
  if (size === 0) return "";
  const selected = values.slice(-size);
  const blocks = "▁▂▃▄▅▆▇█";
  const scale = Number.isFinite(ceiling) && ceiling > 0 ? ceiling : 1;
  return "·".repeat(Math.max(0, size - selected.length)) + selected.map((value) =>
    value === null || !Number.isFinite(value) || value < 0 ? "·" : blocks[Math.min(7, Math.max(0, Math.ceil(value / scale * 8) - 1))]!).join("");
}

function trend(label: string, values: readonly (number | null)[], width: number, tone: Tone, ceiling = 100): Line {
  const prefix = `${label} `;
  return [s(prefix, "muted"), s(renderMonitorSparkline(values, Math.max(0, Math.min(60, width - monitorTextWidth(prefix))), ceiling), tone)];
}

function panel(title: string, status: Span, body: Line[], width: number, height: number): Line[] {
  const inner = width - 4;
  const heading = clipped([s(` ${title} `, "cyan", true)], Math.max(0, width - monitorTextWidth(status.text) - 7));
  const tag = [s(" "), status, s(" ")];
  const top = [s("╭─", "border"), ...heading, s("─".repeat(Math.max(0, width - 3 - widthOf(heading) - widthOf(tag))), "border"), ...tag, s("╮", "border")];
  const lines = [top];
  for (let row = 0; row < height - 2; row++) lines.push([s("│ ", "border"), ...padded(body[row] ?? [], inner), s(" │", "border")]);
  lines.push(line(`╰${"─".repeat(width - 2)}╯`, "border"));
  return lines;
}

function systemBody(snapshot: MonitorSnapshot, width: number, capacity: number, history: readonly MonitorHistorySample[]): Line[] {
  const host = snapshot.host;
  if (host === null) return [line("Host unavailable", "amber"), line(snapshot.host_error ?? "Waiting for a host sample", "muted")];
  const error = snapshot.host_error ?? (host.errors.length ? host.errors.join("; ") : null);
  const memory = ratio(host.memory_used_bytes, host.memory_total_bytes);
  const body: Line[] = [
    meter("CPU", host.cpu_utilization_percent, `${percent(host.cpu_utilization_percent)} · ${host.logical_cpu_count} cores`, width, "cyan"),
    meter(host.platform === "darwin" ? "Unified RAM" : "RAM", memory, `${formatBytes(host.memory_used_bytes)} / ${formatBytes(host.memory_total_bytes)}`, width, "blue"),
    meter("Swap", ratio(host.swap_used_bytes, host.swap_total_bytes), `${formatBytes(host.swap_used_bytes)} / ${formatBytes(host.swap_total_bytes)}`, width, "violet"),
    error ? line(`Host error: ${error}`, "amber") : line(`Wired ${formatBytes(host.wired_bytes)} · Compressed ${formatBytes(host.compressed_bytes)}`, "muted"),
  ];
  if (capacity > body.length) body.push(trend("CPU", history.map((item) => item.cpu_percent), width, "cyan"));
  if (capacity > body.length) body.push(line("RAM = non-free, including caches; not memory pressure", "muted"));
  if (capacity > body.length) body.push(trend("RAM", history.map((item) => item.memory_percent), width, "blue"));
  if (error && capacity > body.length) body.push(line(`Wired ${formatBytes(host.wired_bytes)} · Compressed ${formatBytes(host.compressed_bytes)}`, "muted"));
  if (capacity > body.length) body.push(line(`Load 1/5/15m  ${host.load_average.map((value) => value.toFixed(2)).join(" / ")}`, "muted"));
  if (capacity > body.length) body.push(line(`Free ${formatBytes(host.memory_free_bytes)} · Uptime ${uptime(host.uptime_seconds)}`, "muted"));
  if (host.memory_available_bytes !== null && capacity > body.length) body.push(line(`OS available ${formatBytes(host.memory_available_bytes)}`, "muted"));
  return body;
}

function gpuFrequency(a: HostMetricsSnapshot["accelerators"] | undefined): string {
  if (a?.gpu_frequency_mhz != null) {
    return `${a.source === "ioreport" ? "GPU state avg" : "GPU"} ${a.gpu_frequency_mhz.toFixed(0)} MHz`;
  }
  return a?.source === "ioreport" && a.gpu_active_percent === 0 ? "GPU idle" : "GPU frequency unavailable";
}

function acceleratorsBody(host: HostMetricsSnapshot | null, width: number, capacity: number, history: readonly MonitorHistorySample[]): Line[] {
  if (host === null) return [line("Hardware sample unavailable", "amber")];
  const a = host.accelerators;
  const reading = monitorGpuReading(host);
  const frequency = gpuFrequency(a);
  const thermal: Line = [s("Thermal ", "muted"), s(a.thermal_pressure ?? "unavailable", /serious|critical/i.test(a.thermal_pressure ?? "") ? "red" : "text")];
  const source = a.source === "ioreport" ? "IOReport" : "recorded powermetrics";
  const note = a.error
    ? `Power ${a.status} · ${age(a.age_ms)} · ${a.error}`
    : `Power ${a.status} · ${age(a.age_ms)} · ${source}${a.warnings?.length ? ` · ${a.warnings.join("; ")}` : ""}`;
  const gpu = meter(reading.label, reading.percent, percent(reading.percent), width, "violet");
  const gpuNote = reading.error ? `GPU ${reading.status} · ${reading.error}` : reading.description;
  if (capacity < 8) {
    const body = [gpu, line(`Est. power  CPU ${power(a.cpu_power_mw)} · GPU ${power(a.gpu_power_mw)} · ANE ${power(a.ane_power_mw)}`, "blue")];
    if (capacity >= 4 && !(reading.error && host.gpu)) body.push([...thermal, s(` · ${frequency}`, "muted")]);
    body.push(line(reading.error && host.gpu ? gpuNote : note, reading.error || a.error || a.warnings?.length ? "amber" : "muted"));
    if (reading.error && host.gpu && capacity > body.length) body.push(line(note, a.error ? "amber" : "muted"));
    if (capacity > body.length) body.push(trend("GPU", history.map((item) => item.gpu_percent), width, "violet"));
    if (capacity > body.length && !(reading.error && host.gpu)) body.push(line(gpuNote, reading.error ? "amber" : "muted"));
    if (capacity > body.length && (capacity < 4 || reading.error && host.gpu)) body.push([...thermal, s(` · ${frequency}`, "muted")]);
    if (capacity > body.length) body.push(line("ANE = estimated power, not occupancy", "muted"));
    return body;
  }
  const powers = [a.cpu_power_mw, a.gpu_power_mw, a.ane_power_mw];
  const measuredPowers = powers.filter((value): value is number => value !== null && Number.isFinite(value) && value >= 0);
  const scale = Math.max(1000, ...measuredPowers);
  const body = [gpu, trend("GPU", history.map((item) => item.gpu_percent), width, "violet"),
    reading.error && host.gpu ? line(gpuNote, "amber") : line(reading.description, "muted"),
    line(measuredPowers.length ? `ESTIMATED POWER · full bar ${power(scale)}` : "ESTIMATED POWER · unavailable", "muted"),
    meter("CPU", ratio(a.cpu_power_mw, scale), power(a.cpu_power_mw), width, "cyan"),
    meter("GPU", ratio(a.gpu_power_mw, scale), power(a.gpu_power_mw), width, "blue"),
    meter("ANE", ratio(a.ane_power_mw, scale), power(a.ane_power_mw), width, "violet"),
    line(note, a.error || a.warnings?.length ? "amber" : "muted"),
  ];
  const aneHistory = history.map((item) => item.ane_power_mw);
  const aneValues = aneHistory.filter((value): value is number => value !== null && Number.isFinite(value) && value >= 0);
  const aneScale = Math.max(1, ...aneValues);
  if (capacity > body.length) body.push(aneValues.length
    ? trend(`ANE 0–${power(aneScale)}`, aneHistory, width, "violet", aneScale)
    : line("ANE trend unavailable · power, not occupancy", "muted"));
  if (aneValues.length && capacity > body.length) body.push(line(`ANE trend 0–${power(aneScale)} · power, not occupancy`, "muted"));
  if (capacity > body.length) body.push([...thermal, s(` · ${frequency}`, "muted")]);
  for (const cluster of a.cpu_clusters) {
    if (capacity <= body.length) break;
    const frequency = cluster.frequency_mhz === null
      ? a.source === "ioreport" ? ` · ${cluster.active_percent === 0 ? "idle" : "frequency unavailable"}` : ""
      : ` · ${cluster.frequency_mhz.toFixed(0)} MHz${a.source === "ioreport" ? " state avg" : ""}`;
    body.push(meter(cluster.name, cluster.active_percent, `${percent(cluster.active_percent)}${frequency}`, width, "cyan"));
  }
  const rate = (value: number | null): string => value === null ? "unavailable" : `${formatBytes(value)}/s`;
  if (capacity > body.length && (a.disk_read_bytes_per_second !== null || a.disk_write_bytes_per_second !== null)) {
    body.push(line(`Disk R/W  ${rate(a.disk_read_bytes_per_second)} / ${rate(a.disk_write_bytes_per_second)}`, "muted"));
  }
  if (capacity > body.length && (a.network_in_bytes_per_second !== null || a.network_out_bytes_per_second !== null)) {
    body.push(line(`Network ↓/↑  ${rate(a.network_in_bytes_per_second)} / ${rate(a.network_out_bytes_per_second)}`, "muted"));
  }
  return body;
}

function pressureTone(pressure: string, emergency: boolean): Tone {
  return emergency || /emergency/i.test(pressure) ? "red" : /hard|soft/i.test(pressure) ? "amber" : /^(ok|normal)$/i.test(pressure) ? "green" : "muted";
}

function runtimeBody(snapshot: MonitorSnapshot, width: number, capacity: number, history: readonly MonitorHistorySample[], nextStep?: string): Line[] {
  const { alloy } = snapshot;
  const metrics = alloy.metrics;
  if (metrics === null) return [
    [s("METRICS UNAVAILABLE", "amber", true), s(" · System monitoring continues", "muted")],
    line(`Connection: ${alloy.error ?? "Waiting for daemon.metrics"}`, "muted"),
    line(nextStep ?? "Check alloyd and its socket; select another with --socket PATH.", "text"),
    line("The monitor reconnects automatically; it does not start inference.", "muted"),
  ];
  const { admission, execution, residency, memory_guard: guard } = metrics;
  const elapsed = alloy.sample_age_ms ?? 0;
  const guardAge = age(guard.sample_age_ms === null ? null : guard.sample_age_ms + elapsed);
  const guardDetail = `Guard ${guard.tier}: ${formatBytes(guard.usage_bytes)} / ${formatBytes(guard.hard_limit_bytes)} · ${guard.binding ?? "binding unavailable"} · ${guardAge}`;
  const errors: Line = [];
  if (alloy.error) errors.push(...clipped(line(`Connection: ${alloy.error}`, "amber"), guard.last_error ? Math.floor(width / 2) : width));
  if (guard.last_error) errors.push(...line(`${errors.length ? " · " : ""}Guard error: ${guard.last_error}`, "amber"));
  const residencyLine = line(`${errors.length ? `Guard ${guardAge} · ` : ""}Residency ${age(residency.sample_age_ms + elapsed)} · Models ${residency.models_resident} · Worker ${residency.worker_alive ? "alive" : "stopped"}`, "muted");
  const body: Line[] = [
    [s("Admission ", "muted"), s(admission.admissions_paused ? "PAUSED" : "OPEN", admission.admissions_paused ? "amber" : "green", true),
      s(" · Pressure ", "muted"), s(guard.emergency ? "EMERGENCY" : guard.pressure.toUpperCase(), pressureTone(guard.pressure, guard.emergency), true),
      s(` · ${admission.active_reservations} reservations`, "muted")],
    meter("Ledger", ratio(admission.reserved_bytes, admission.budget_bytes), `${formatBytes(admission.reserved_bytes)} / ${formatBytes(admission.budget_bytes)} reserved`, width, "cyan"),
    execution.cancellation_requested
      ? [s("Cancellation requested; cleanup pending", "amber", true), s(` · Request ${execution.active_request_id ?? "idle"}`, "muted")]
      : [s(execution.active_request_id === null ? "IDLE" : "RUNNING", execution.active_request_id === null ? "muted" : "green", true), s(` · Request ${execution.active_request_id ?? "none"}`)],
    errors.length ? errors : line(guardDetail, "muted"),
    nextStep ? line(nextStep) : residencyLine,
  ];
  if (nextStep && capacity > body.length) body.push(residencyLine);
  if (errors.length && capacity > body.length) body.push(line(guardDetail, "muted"));
  if (capacity > body.length) body.push(line(`Available ${formatBytes(admission.available_bytes)}${guard.cache_reclaim_requested ? " · Cache reclaim requested" : ""} · PID ${metrics.process_id}`, "muted"));
  if (capacity > body.length) body.push(trend("Reserved", history.map((item) => item.reserved_percent), width, "cyan"));
  for (const model of residency.models) {
    if (capacity <= body.length) break;
    body.push([s(model.artifact_id, "text"), s(` · weights ${formatBytes(model.weights_reserved_bytes)} · KV ${formatBytes(model.kv_reserved_bytes)}`, "muted"),
      s(`${model.pinned ? " · pinned" : ""}${model.leased ? " · leased" : ""}`, "blue")]);
  }
  return body;
}

function compact(snapshot: MonitorSnapshot, width: number, height: number, history: readonly MonitorHistorySample[], nextStep?: string): Line[] {
  const host = snapshot.host;
  const a = host?.accelerators;
  const gpu = monitorGpuReading(host);
  const metrics = snapshot.alloy.metrics;
  const framed = width >= 44 && height >= 13;
  const inner = framed ? width - 4 : width;
  const rows: Array<{ spans: Line; priority: number }> = [];
  const add = (spans: Line, priority: number): void => { rows.push({ spans, priority }); };
  // Select by importance before restoring display order. Short windows must not
  // hide cancellation/pressure or the freshness labels for retained readings.
  add(line(`HOST ${snapshot.host_error ? "STALE · " : ""}${age(snapshot.host_sample_age_ms)} · ${host?.hostname ?? "unavailable"}`, "muted"), 110);
  add(meter("CPU", host?.cpu_utilization_percent ?? null, percent(host?.cpu_utilization_percent ?? null), inner, "cyan"), 100);
  add(meter("RAM non-free", ratio(host?.memory_used_bytes ?? null, host?.memory_total_bytes ?? null),
    `${formatBytes(host?.memory_used_bytes ?? null)} / ${formatBytes(host?.memory_total_bytes ?? null)}`, inner, "blue"), 99);
  add(trend("CPU history", history.map((sample) => sample.cpu_percent), inner, "cyan"), 80);
  add(line(`${host?.gpu ? "GPU" : "ACCELERATORS"} ${gpu.status} · ${age(gpu.age_ms)}`, gpu.status === "ready" ? "muted" : "amber"), 109);
  add(meter(gpu.label, gpu.percent, percent(gpu.percent), inner, "violet"), host?.gpu ? 108 : 98);
  add(trend("GPU history", history.map((sample) => sample.gpu_percent), inner, "violet"), 79);
  add(line(`ANE power ${power(a?.ane_power_mw ?? null)} · CPU ${power(a?.cpu_power_mw ?? null)} · GPU ${power(a?.gpu_power_mw ?? null)}`, "blue"), 97);
  if (host?.gpu) {
    if (gpu.error) add(line(`GPU: ${gpu.error}`, "amber"), 190);
    add(line(`Power ${a?.status ?? "unavailable"} · ${age(a?.age_ms ?? null)}`, a?.status === "ready" ? "muted" : "amber"), 106);
  }
  if (a?.error) add(line(`Power: ${a.error}`, "amber"), host?.gpu ? 103 : 190);
  if (a?.warnings?.length) add(line(`Counters: ${a.warnings.join("; ")}`, "amber"), 102);
  if (snapshot.host_error || host?.errors.length) add(line(`Host error: ${snapshot.host_error ?? host?.errors.join("; ")}`, "amber"), 190);
  add(line(`ALLOY ${snapshot.alloy.status}${snapshot.alloy.status === "unavailable" && metrics ? " (stale)" : ""} · ${age(snapshot.alloy.sample_age_ms)}`, snapshot.alloy.status === "connected" ? "green" : "amber"), 240);
  if (metrics) {
    const { admission, memory_guard: guard, execution } = metrics;
    add(line(`Admission ${admission.admissions_paused ? "PAUSED" : "OPEN"} · Pressure ${guard.emergency ? "EMERGENCY" : guard.pressure}`, pressureTone(guard.pressure, guard.emergency)), 220);
    add(line(execution.cancellation_requested ? "Cancellation requested; cleanup pending" : `Request ${execution.active_request_id ?? "idle"}`, execution.cancellation_requested ? "amber" : "text"), 210);
    add(meter("Ledger", ratio(admission.reserved_bytes, admission.budget_bytes), `${formatBytes(admission.reserved_bytes)} / ${formatBytes(admission.budget_bytes)} reserved`, inner, "cyan"), 85);
    if (guard.last_error) add(line(`Guard error: ${guard.last_error}`, "amber"), 190);
  }
  if (snapshot.alloy.error) add(line(`Connection: ${snapshot.alloy.error}`, "amber"), 200);
  if (nextStep) add(line(nextStep), 205);
  add(line(`Swap ${formatBytes(host?.swap_used_bytes ?? null)} · Wired ${formatBytes(host?.wired_bytes ?? null)} · Compressed ${formatBytes(host?.compressed_bytes ?? null)}`, "muted"), 75);
  add(line(`Thermal ${a?.thermal_pressure ?? "unavailable"} · ${gpuFrequency(a)}`, "muted"), 70);
  add(line("Full panels: 60 columns × 22 terminal rows", "muted"), 1);
  const budget = Math.max(0, height - (framed ? 3 : 2));
  const visible = rows.map((row, index) => ({ ...row, index }))
    .sort((a, b) => b.priority - a.priority || a.index - b.index).slice(0, budget)
    .sort((a, b) => a.index - b.index).map((row) => row.spans);
  return framed
    ? panel("ALLOY MONITOR", s("compact · read-only", "muted"), visible, width, visible.length + 2)
    : [line("ALLOY MONITOR · read-only", "cyan"), ...visible].map((row) => clipped(row, width));
}

/** Pure responsive renderer. Color is explicit; JSON never passes through it. */
export function renderMonitorSnapshot(snapshot: MonitorSnapshot, options: MonitorRenderOptions = {}): string {
  const width = Math.max(1, Math.min(240, Math.floor(options.columns ?? 80)));
  const height = Math.max(1, Math.min(1000, Math.floor(options.rows ?? 24)));
  const history = options.history ?? [];
  const host = snapshot.host;
  const metrics = snapshot.alloy.metrics;
  const footer = options.interactive === false ? "Read-only snapshot · unavailable ≠ zero · RAM includes caches"
    : `q quit · Ctrl+C exit · ${(options.intervalMs ?? 1000) / 1000}s · · = missing · RAM non-free incl. caches`;
  let lines: Line[];
  if (width < 60 || height < 21) {
    lines = compact(snapshot, width, height, history, options.nextStep);
  } else {
    const clock = new Date(snapshot.sampled_at_unix_ms).toLocaleTimeString("en-GB", { hour12: false });
    const left = [s(" ALLOY ", "cyan", true), s(" / LIVE MONITOR", "text", true)];
    const right = line(`READ ONLY  ${clock} `, "muted");
    const range = history.length < 2 ? "collecting history" : `history ${Math.round((history.at(-1)!.at_unix_ms - history[0]!.at_unix_ms) / 1000)}s`;
    lines = [[...left, s(" ".repeat(Math.max(1, width - widthOf(left) - widthOf(right)))), ...right],
      line(` ${host?.hostname ?? "Host unavailable"} · ${host ? `${host.platform}/${host.architecture} · ` : ""}system-wide · ${range}`, "muted")];
    const hostTag = s(`${snapshot.host_error ? "STALE" : host === null ? "OFFLINE" : "HOST"} ${age(snapshot.host_sample_age_ms)}`, snapshot.host_error || host === null ? "amber" : "green");
    const gpu = monitorGpuReading(host);
    const powerTag = s(`${host?.gpu ? "GPU " : ""}${gpu.status} · ${age(gpu.age_ms)}`, gpu.status === "ready" ? "green" : "amber");
    const runtimeTag = s(`${snapshot.alloy.status}${snapshot.alloy.status === "unavailable" && metrics ? " (stale)" : ""} · ${age(snapshot.alloy.sample_age_ms)}`, snapshot.alloy.status === "connected" ? "green" : "amber");
    if (width >= 108) {
      const sideHeight = Math.min(17, Math.max(9, height - 13));
      const runtimeHeight = Math.min(14, height - sideHeight - 4);
      const leftWidth = Math.floor((width - 2) / 2);
      const rightWidth = width - leftWidth - 2;
      const system = panel("SYSTEM", hostTag, systemBody(snapshot, leftWidth - 4, sideHeight - 2, history), leftWidth, sideHeight);
      const accelerators = panel("ACCELERATORS", powerTag, acceleratorsBody(host, rightWidth - 4, sideHeight - 2, history), rightWidth, sideHeight);
      for (let row = 0; row < sideHeight; row++) lines.push([...system[row]!, s("  "), ...accelerators[row]!]);
      lines.push([]);
      lines.push(...panel("ALLOY RUNTIME", runtimeTag, runtimeBody(snapshot, width - 4, runtimeHeight - 2, history, options.nextStep), width, runtimeHeight));
    } else {
      const sizes = [6, 5, 7];
      const caps = [13, 14, 14];
      let extra = height - 21;
      while (extra > 0 && sizes.some((size, index) => size < caps[index]!)) {
        for (let i = 0; i < 3 && extra > 0; i++) if (sizes[i]! < caps[i]!) { sizes[i] = sizes[i]! + 1; extra--; }
      }
      lines.push(...panel("SYSTEM", hostTag, systemBody(snapshot, width - 4, sizes[0]! - 2, history), width, sizes[0]!));
      lines.push(...panel("ACCELERATORS", powerTag, acceleratorsBody(host, width - 4, sizes[1]! - 2, history), width, sizes[1]!));
      lines.push(...panel("ALLOY RUNTIME", runtimeTag, runtimeBody(snapshot, width - 4, sizes[2]! - 2, history, options.nextStep), width, sizes[2]!));
    }
  }
  lines = lines.slice(0, Math.max(0, height - 1));
  lines.push(line(footer, "muted"));
  return lines.map((row) => paint(clipped(row, width), options.color === true)).join("\n");
}
