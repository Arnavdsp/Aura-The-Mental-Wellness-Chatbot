/**
 * Aura — client application.
 *
 * Plain ES modules, no build step and no framework. The whole UI is small
 * enough that a framework would cost more than it saves, and a hackathon judge
 * (or a maintainer) can read the entire client in one sitting.
 *
 * Structure: a single `state` object, small pure render functions, and one
 * transport module that speaks the SSE protocol from `/api/chat/stream`.
 */

const $ = (id) => document.getElementById(id);

/* ── State ─────────────────────────────────────────────────────────────── */

const state = {
  sessionId: null,
  pendingAttachments: [], // { id, kind, previewUrl, name, inline }
  attachmentSeq: 0,
  sending: false,
  speak: false,
  recorder: null,
  health: null,
};

const el = {
  thread: $("thread"),
  welcome: $("welcome"),
  form: $("composer"),
  input: $("composer-input"),
  send: $("send-btn"),
  mic: $("mic-btn"),
  attach: $("attach-btn"),
  file: $("file-input"),
  tray: $("attachment-tray"),
  notices: $("notices"),
  speakToggle: $("speak-toggle"),
  recorder: $("recorder"),
  recorderTime: $("recorder-time"),
  recorderViz: $("recorder-viz"),
  recordStop: $("record-stop"),
  recordCancel: $("record-cancel"),
  insights: $("insights"),
  insightsToggle: $("insights-toggle"),
  main: document.querySelector(".main"),
  status: $("status"),
  themeToggle: $("theme-toggle"),
  crisis: $("crisis"),
  crisisList: $("crisis-list"),
  crisisClose: $("crisis-close"),
  moodEmoji: $("mood-emoji"),
  moodLabel: $("mood-label"),
  moodMeta: $("mood-meta"),
  moodDirection: $("mood-direction"),
  sparkline: $("sparkline"),
  topics: $("topics"),
  resources: $("resources"),
  clear: $("clear-btn"),
  toast: $("toast"),
};

/* ── Small utilities ───────────────────────────────────────────────────── */

const escapeHtml = (value) =>
  String(value).replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));

/**
 * Render the small markdown subset the coach actually emits: paragraphs,
 * bullet lists, bold, italic and inline code.
 *
 * Everything is HTML-escaped before any tag is introduced, so this can never
 * inject markup — which matters because the same renderer draws model output
 * and crisis resources.
 */
function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  const out = [];
  let list = null;
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${inline(paragraph.join("<br>"))}</p>`);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
  };

  for (const line of lines) {
    const bullet = line.match(/^\s*[-•*]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      (list ??= []).push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }
    // An indented line after a bullet is a continuation of that bullet.
    if (list && /^\s{2,}\S/.test(line)) {
      list[list.length - 1] = list.at(-1).replace(
        /<\/li>$/, `<br>${inline(line.trim())}</li>`
      );
      continue;
    }
    flushList();
    if (line.trim() === "") flushParagraph();
    else paragraph.push(line);
  }
  flushParagraph();
  flushList();
  return out.join("");
}

function inline(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<![*\w])_([^_]+)_(?!\w)/g, "<em>$1</em>");
}

const timeLabel = () =>
  new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

let toastTimer;
function toast(message) {
  el.toast.textContent = message;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, 4200);
}

const MOOD_FACES = {
  joy: "☺", calm: "◡", hopeful: "✦", neutral: "·", tired: "◠",
  sad: "◡", lonely: "◌", anxious: "◍", overwhelmed: "◉", angry: "◆",
  fearful: "◈", surprise: "◇", disgust: "◇",
};

function moodColor(valence) {
  if (valence <= -0.25) return "var(--mood-low)";
  if (valence >= 0.25) return "var(--mood-high)";
  return "var(--mood-mid)";
}

function scrollToEnd(force = false) {
  const nearBottom =
    el.thread.scrollHeight - el.thread.scrollTop - el.thread.clientHeight < 160;
  if (force || nearBottom) el.thread.scrollTop = el.thread.scrollHeight;
}

/* ── Message rendering ─────────────────────────────────────────────────── */

function hideWelcome() {
  if (el.welcome && !el.welcome.hidden) {
    el.welcome.hidden = true;
    el.welcome.remove();
  }
}

function addMessage({ role, text = "", attachments = [], affect = null, crisis = false }) {
  hideWelcome();

  const wrapper = document.createElement("article");
  wrapper.className = `msg msg--${role === "user" ? "user" : "coach"}`;
  if (crisis) wrapper.classList.add("msg--crisis");

  const avatar = document.createElement("div");
  avatar.className = "msg__avatar";
  avatar.textContent = role === "user" ? "You" : "A";
  avatar.setAttribute("aria-hidden", "true");

  const body = document.createElement("div");
  body.className = "msg__body";

  if (attachments.length) {
    const strip = document.createElement("div");
    strip.className = "msg__attachments";
    for (const item of attachments) {
      if (item.kind === "image" && item.previewUrl) {
        const img = document.createElement("img");
        img.className = "msg__image";
        img.src = item.previewUrl;
        img.alt = item.name || "Image you shared";
        img.loading = "lazy";
        strip.append(img);
      } else if (item.kind === "audio") {
        const pill = document.createElement("span");
        pill.className = "msg__voice";
        pill.textContent = item.transcript
          ? `🎙 "${item.transcript}"`
          : "🎙 Voice message";
        strip.append(pill);
      }
    }
    body.append(strip);
  }

  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.innerHTML = text ? renderMarkdown(text) : "";
  body.append(bubble);

  const meta = document.createElement("div");
  meta.className = "msg__meta";
  meta.append(Object.assign(document.createElement("span"), { textContent: timeLabel() }));
  if (affect) tagAffect(meta, affect);
  body.append(meta);

  wrapper.append(avatar, body);
  el.thread.append(wrapper);
  scrollToEnd(role === "user");
  return { wrapper, bubble, meta, body };
}

/**
 * Show how the message was read, on the message itself.
 *
 * Making the affect estimate visible (and hedged with a confidence figure)
 * keeps the user in control of it: a wrong read is something they can see and
 * correct, rather than something that silently steers the conversation.
 */
function tagAffect(meta, affect) {
  if (!affect || affect.source === "none" || affect.confidence <= 0.3) return;
  if (meta.querySelector(".affect-tag")) return;
  const tag = document.createElement("span");
  tag.className = "affect-tag";
  tag.style.setProperty("--tag-color", moodColor(affect.valence));
  tag.textContent = affect.label;
  tag.title =
    `Heard as ${affect.label} from your ${affect.source === "fused" ? "words and voice" : affect.source}` +
    ` · ${(affect.confidence * 100).toFixed(0)}% confident. Tell me if I've got it wrong.`;
  meta.append(tag);
}

function addTypingIndicator() {
  const node = addMessage({ role: "coach" });
  node.bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  return node;
}

function renderFollowups(node, suggestions) {
  if (!suggestions?.length) return;
  const row = document.createElement("div");
  row.className = "followups";
  for (const suggestion of suggestions.slice(0, 3)) {
    const button = document.createElement("button");
    button.className = "followup";
    button.type = "button";
    button.textContent = suggestion;
    button.addEventListener("click", () => {
      row.remove();
      submitMessage(suggestion);
    });
    row.append(button);
  }
  node.body.append(row);
  scrollToEnd();
}

function attachAudioPlayer(node, url) {
  const player = document.createElement("audio");
  player.className = "msg__audio";
  player.controls = true;
  player.preload = "none";
  player.src = url;
  node.body.insertBefore(player, node.meta);
  player.play().catch(() => { /* autoplay blocked — the control is still there */ });
}

/* ── Transport ─────────────────────────────────────────────────────────── */

/** Consume an SSE body, dispatching `event:`/`data:` pairs to `onEvent`. */
async function consumeStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      let name = "message";
      const payload = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) payload.push(line.slice(5).trim());
      }
      if (!payload.length) continue;
      try {
        onEvent(name, JSON.parse(payload.join("\n")));
      } catch (error) {
        console.warn("bad SSE frame", error);
      }
    }
  }
}

const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024;

/**
 * Read a file into the shape `/api/chat` accepts inline.
 *
 * The turn carries its own bytes rather than an id from a prior upload: on a
 * serverless host the two requests can land on different instances, and a
 * staged upload the second instance never saw is a photo that vanishes.
 */
async function readAttachment(file, kind) {
  if (file.size > MAX_ATTACHMENT_BYTES) {
    throw new Error("That file is larger than the 25 MB limit.");
  }
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return {
    kind,
    media_type: file.type || (kind === "audio" ? "audio/webm" : "application/octet-stream"),
    filename: file.name || `clip.${kind === "audio" ? "webm" : "png"}`,
    data: btoa(binary),
  };
}

/* ── Sending a turn ────────────────────────────────────────────────────── */

async function submitMessage(text) {
  if (state.sending) return;
  const message = (text ?? el.input.value).trim();
  const attachments = [...state.pendingAttachments];
  if (!message && !attachments.length) return;

  state.sending = true;
  setComposerEnabled(false);
  el.notices.textContent = "";

  const userNode = addMessage({ role: "user", text: message, attachments });
  el.input.value = "";
  autoGrow();
  clearAttachments();

  const placeholder = addTypingIndicator();
  let streamed = "";
  let sawToken = false;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        message,
        speak: state.speak,
        attachments: attachments.map((a) => a.inline),
      }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Server responded ${response.status}`);
    }

    await consumeStream(response, (name, data) => {
      if (name === "meta") {
        state.sessionId = data.session_id;
        applySafety(data.safety);
        applyAffect(data.affect);
        tagAffect(userNode.meta, data.affect);
        annotateVoiceAttachments(userNode, data.user_turn);
        showNotices(data.notices);
        if (data.safety?.risk === "crisis") placeholder.wrapper.classList.add("msg--crisis");
      } else if (name === "token") {
        if (!sawToken) { placeholder.bubble.innerHTML = ""; sawToken = true; }
        streamed += data.text;
        placeholder.bubble.innerHTML = renderMarkdown(streamed);
        scrollToEnd();
      } else if (name === "done") {
        if (data.reply?.text) placeholder.bubble.innerHTML = renderMarkdown(data.reply.text);
        if (data.audio_url) attachAudioPlayer(placeholder, data.audio_url);
        renderFollowups(placeholder, data.suggestions);
        refreshInsights();
      } else if (name === "error") {
        throw new Error(data.message);
      }
    });
  } catch (error) {
    console.error(error);
    placeholder.bubble.innerHTML = renderMarkdown(
      "I couldn't reach my thinking just then. Could you try that again?"
    );
    toast(error.message || "Something went wrong.");
  } finally {
    state.sending = false;
    setComposerEnabled(true);
    el.input.focus();
  }
}

/** Replace the "Voice message" placeholder with what we actually heard. */
function annotateVoiceAttachments(node, userTurn) {
  const transcripts = (userTurn?.attachments || [])
    .filter((a) => a.kind === "audio" && a.transcript)
    .map((a) => a.transcript);
  if (!transcripts.length) return;
  const pills = node.body.querySelectorAll(".msg__voice");
  pills.forEach((pill, index) => {
    if (transcripts[index]) pill.textContent = `🎙 "${transcripts[index]}"`;
  });
}

function setComposerEnabled(enabled) {
  el.send.disabled = !enabled;
  el.input.disabled = !enabled;
  el.mic.disabled = !enabled;
  el.attach.disabled = !enabled;
}

function showNotices(notices) {
  el.notices.textContent = notices?.length ? notices.join(" ") : "";
}

/* ── Attachments ───────────────────────────────────────────────────────── */

function renderTray() {
  el.tray.innerHTML = "";
  el.tray.hidden = state.pendingAttachments.length === 0;
  for (const item of state.pendingAttachments) {
    const box = document.createElement("div");
    box.className = "attachment";
    box.title = item.name || "attachment";
    if (item.previewUrl) {
      const img = document.createElement("img");
      img.src = item.previewUrl;
      img.alt = "";
      box.append(img);
    } else {
      box.textContent = "🎙";
      box.style.display = "grid";
      box.style.placeItems = "center";
      box.style.fontSize = "22px";
    }
    const remove = document.createElement("button");
    remove.className = "attachment__remove";
    remove.type = "button";
    remove.innerHTML = "&times;";
    remove.setAttribute("aria-label", `Remove ${item.name || "attachment"}`);
    remove.addEventListener("click", () => {
      state.pendingAttachments = state.pendingAttachments.filter((a) => a.id !== item.id);
      if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
      renderTray();
    });
    box.append(remove);
    el.tray.append(box);
  }
}

function clearAttachments() {
  state.pendingAttachments = [];
  renderTray();
}

async function handleFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith("image/")) {
      toast("I can look at images — other files I can't read yet.");
      continue;
    }
    try {
      state.pendingAttachments.push({
        id: `att-${state.attachmentSeq++}`,
        kind: "image",
        name: file.name,
        previewUrl: URL.createObjectURL(file),
        inline: await readAttachment(file, "image"),
      });
      renderTray();
    } catch (error) {
      toast(error.message);
    }
  }
}

/* ── Voice recording ───────────────────────────────────────────────────── */

async function startRecording() {
  if (state.recorder) return stopRecording(true);

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    toast("This browser can't record audio. You can still type.");
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
  } catch {
    toast("I need microphone permission to listen.");
    return;
  }

  const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]
    .find((type) => MediaRecorder.isTypeSupported?.(type)) || "";

  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) chunks.push(event.data);
  });

  const started = Date.now();
  const timer = setInterval(() => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    el.recorderTime.textContent =
      `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    if (seconds >= 180) stopRecording(); // hard cap, keeps uploads sane
  }, 250);

  const visualiser = startVisualiser(stream);

  state.recorder = {
    recorder, stream, chunks, timer, visualiser,
    mimeType: mimeType || "audio/webm",
  };

  recorder.start();
  el.recorder.hidden = false;
  el.mic.classList.add("is-active");
  el.recorderTime.textContent = "0:00";
}

function stopRecording(cancel = false) {
  const session = state.recorder;
  if (!session) return;
  state.recorder = null;

  clearInterval(session.timer);
  session.visualiser?.stop();
  el.recorder.hidden = true;
  el.mic.classList.remove("is-active");

  session.recorder.addEventListener("stop", async () => {
    session.stream.getTracks().forEach((track) => track.stop());
    if (cancel || !session.chunks.length) return;

    const blob = new Blob(session.chunks, { type: session.mimeType });
    if (blob.size < 1200) {
      toast("That recording was too short for me to hear.");
      return;
    }
    try {
      const file = new File([blob], "voice.webm", { type: session.mimeType });
      state.pendingAttachments.push({
        id: `att-${state.attachmentSeq++}`,
        kind: "audio",
        name: "Voice message",
        inline: await readAttachment(file, "audio"),
      });
      renderTray();
      await submitMessage("");
    } catch (error) {
      toast(error.message);
    }
  }, { once: true });

  if (session.recorder.state !== "inactive") session.recorder.stop();
}

/** Live waveform, so the user can see that we're actually hearing them. */
function startVisualiser(stream) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return null;

  const context = new AudioCtx();
  const analyser = context.createAnalyser();
  analyser.fftSize = 512;
  context.createMediaStreamSource(stream).connect(analyser);

  const canvas = el.recorderViz;
  const ctx = canvas.getContext("2d");
  const buffer = new Uint8Array(analyser.frequencyBinCount);
  const bars = 42;
  let frame;

  const draw = () => {
    frame = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(buffer);
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--accent").trim();

    const step = Math.floor(buffer.length / bars) || 1;
    const barWidth = width / bars;
    for (let i = 0; i < bars; i += 1) {
      const value = buffer[i * step] / 255;
      const barHeight = Math.max(2, value * height * 0.92);
      ctx.globalAlpha = 0.35 + value * 0.65;
      ctx.fillRect(
        i * barWidth + barWidth * 0.22,
        (height - barHeight) / 2,
        barWidth * 0.56,
        barHeight
      );
    }
  };
  draw();

  return {
    stop() {
      cancelAnimationFrame(frame);
      context.close().catch(() => {});
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    },
  };
}

/* ── Insights panel ────────────────────────────────────────────────────── */

function applyAffect(affect) {
  if (!affect || affect.source === "none" || affect.confidence < 0.25) return;
  el.moodEmoji.textContent = MOOD_FACES[affect.label] || "·";
  el.moodEmoji.style.color = moodColor(affect.valence);
  el.moodLabel.textContent = affect.label;
  el.moodMeta.textContent =
    `read from ${affect.source === "fused" ? "your words and voice" : `your ${affect.source}`}` +
    ` · ${(affect.confidence * 100).toFixed(0)}% confident`;
}

function applySafety(safety) {
  if (!safety || safety.risk === "none" || safety.risk === "low") return;
  if (!safety.resources?.length) return;
  el.crisisList.innerHTML = safety.resources
    .map((r) => `<li><strong>${escapeHtml(r.name)}</strong> — ${escapeHtml(r.contact)}</li>`)
    .join("");
  el.crisis.hidden = false;
}

function drawSparkline(values) {
  const svg = el.sparkline;
  svg.innerHTML = "";
  if (!values || values.length < 2) return;

  const width = 240;
  const height = 56;
  const points = values.slice(-24).map((value, index, array) => [
    array.length === 1 ? width : (index / (array.length - 1)) * width,
    height / 2 - (Math.max(-1, Math.min(1, value)) * height) / 2.4,
  ]);

  const ns = "http://www.w3.org/2000/svg";
  const path = points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`)
    .join(" ");

  const baseline = document.createElementNS(ns, "line");
  baseline.setAttribute("x1", "0"); baseline.setAttribute("x2", String(width));
  baseline.setAttribute("y1", String(height / 2)); baseline.setAttribute("y2", String(height / 2));
  baseline.setAttribute("stroke", "var(--border)");
  baseline.setAttribute("stroke-dasharray", "3 4");
  svg.append(baseline);

  const area = document.createElementNS(ns, "path");
  area.setAttribute("d", `${path} L${width},${height / 2} L0,${height / 2} Z`);
  area.setAttribute("fill", moodColor(values.at(-1)));
  area.setAttribute("opacity", "0.12");
  svg.append(area);

  const line = document.createElementNS(ns, "path");
  line.setAttribute("d", path);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", moodColor(values.at(-1)));
  line.setAttribute("stroke-width", "2");
  line.setAttribute("stroke-linecap", "round");
  line.setAttribute("stroke-linejoin", "round");
  svg.append(line);

  const [cx, cy] = points.at(-1);
  const dot = document.createElementNS(ns, "circle");
  dot.setAttribute("cx", cx.toFixed(1)); dot.setAttribute("cy", cy.toFixed(1));
  dot.setAttribute("r", "3.2");
  dot.setAttribute("fill", moodColor(values.at(-1)));
  svg.append(dot);
}

function renderTopics(graph) {
  const entries = Object.entries(graph?.topics || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    el.topics.innerHTML = '<p class="empty">Themes will appear here as we talk.</p>';
    return;
  }
  const max = entries[0][1];
  el.topics.innerHTML = entries
    .slice(0, 6)
    .map(([name, count]) => {
      const links = (graph.links || [])
        .filter((link) => link.source === name || link.target === name)
        .slice(0, 2)
        .map((link) => (link.source === name ? link.target : link.source));
      const linkNote = links.length
        ? `<span class="topic__links">often alongside ${links.join(", ").replace(/_/g, " ")}</span>`
        : "";
      return `
        <div class="topic">
          <div class="topic__head">
            <span class="topic__name">${escapeHtml(name.replace(/_/g, " "))}</span>
            <span class="topic__count">${count}&times;</span>
          </div>
          <div class="topic__bar"><div class="topic__fill" style="width:${(count / max) * 100}%"></div></div>
          ${linkNote}
        </div>`;
    })
    .join("");
}

async function refreshInsights() {
  if (!state.sessionId) return;
  try {
    const response = await fetch(`/api/sessions/${state.sessionId}`);
    if (!response.ok) return;
    const data = await response.json();
    drawSparkline(data.insights.mood_trend);
    renderTopics(data.insights.graph);
    const direction = data.insights.mood_direction;
    el.moodDirection.textContent = direction === "unknown" ? "" : direction;
  } catch { /* the panel is decoration; never block the conversation on it */ }
}

async function loadResources() {
  try {
    const response = await fetch("/api/resources");
    const resources = await response.json();
    el.resources.innerHTML = resources
      .map((r) => {
        const link = r.url
          ? ` <a href="${escapeHtml(r.url)}" target="_blank" rel="noopener noreferrer">↗</a>`
          : "";
        return `<li><strong>${escapeHtml(r.name)}${link}</strong><span>${escapeHtml(r.contact)}</span></li>`;
      })
      .join("");
  } catch { /* offline is survivable here */ }
}

async function loadHealth() {
  const dot = el.status.querySelector(".status__dot");
  const label = el.status.querySelector(".status__label");
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    state.health = health;
    dot.dataset.state = health.status === "ok" ? "ok" : "degraded";
    label.textContent = health.engine === "echo" ? "demo engine" : health.engine;
    el.status.title =
      `${health.engine} · ${health.engine_ready ? "ready" : "warming up"}\n` +
      Object.entries(health.capabilities)
        .map(([key, on]) => `${on ? "✓" : "✕"} ${key}`)
        .join("\n");

    if (!health.capabilities.audio_out) {
      el.speakToggle.disabled = true;
      el.speakToggle.closest(".switch").title =
        "Spoken replies need a text-to-speech backend on the server.";
    }
  } catch {
    dot.dataset.state = "down";
    label.textContent = "offline";
  }
}

/* ── Composer behaviour ────────────────────────────────────────────────── */

function autoGrow() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 168)}px`;
}

/* ── Theme ─────────────────────────────────────────────────────────────── */

const THEME_KEY = "aura.theme";

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }
}

function initTheme() {
  let stored = "auto";
  try { stored = localStorage.getItem(THEME_KEY) || "auto"; } catch { /* ignore */ }
  document.documentElement.dataset.theme = stored;
}

function cycleTheme() {
  const current = document.documentElement.dataset.theme;
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const effective = current === "auto" ? (prefersDark ? "dark" : "light") : current;
  applyTheme(effective === "dark" ? "light" : "dark");
}

/* ── Wiring ────────────────────────────────────────────────────────────── */

function init() {
  initTheme();

  el.form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitMessage();
  });

  el.input.addEventListener("input", autoGrow);
  el.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submitMessage();
    }
  });

  el.attach.addEventListener("click", () => el.file.click());
  el.file.addEventListener("change", () => {
    handleFiles([...el.file.files]);
    el.file.value = "";
  });

  el.mic.addEventListener("click", () => startRecording());
  el.recordStop.addEventListener("click", () => stopRecording(false));
  el.recordCancel.addEventListener("click", () => stopRecording(true));

  el.speakToggle.addEventListener("change", () => {
    state.speak = el.speakToggle.checked;
  });

  el.insightsToggle.addEventListener("click", () => {
    const open = el.main.classList.toggle("is-open");
    el.insightsToggle.setAttribute("aria-expanded", String(open));
    if (open) refreshInsights();
  });

  el.themeToggle.addEventListener("click", cycleTheme);
  el.crisisClose.addEventListener("click", () => { el.crisis.hidden = true; });

  el.clear.addEventListener("click", async () => {
    if (state.sessionId) {
      await fetch(`/api/sessions/${state.sessionId}`, { method: "DELETE" }).catch(() => {});
    }
    state.sessionId = null;
    el.thread.innerHTML = "";
    el.crisis.hidden = true;
    drawSparkline([]);
    renderTopics(null);
    el.moodLabel.textContent = "Not enough to go on yet";
    el.moodMeta.textContent = "Say something and I'll listen for tone.";
    el.moodEmoji.textContent = "·";
    el.moodDirection.textContent = "";
    location.reload();
  });

  for (const chip of document.querySelectorAll("#starter-chips .chip")) {
    chip.addEventListener("click", () => submitMessage(chip.dataset.prompt));
  }

  // Drag-and-drop an image anywhere on the page.
  document.addEventListener("dragover", (event) => event.preventDefault());
  document.addEventListener("drop", (event) => {
    event.preventDefault();
    if (event.dataTransfer?.files?.length) handleFiles([...event.dataTransfer.files]);
  });

  // Paste an image straight from the clipboard.
  el.input.addEventListener("paste", (event) => {
    const files = [...(event.clipboardData?.files || [])];
    if (files.length) {
      event.preventDefault();
      handleFiles(files);
    }
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.recorder) stopRecording(true);
  });

  loadHealth();
  loadResources();
  autoGrow();
  el.input.focus();
}

init();
