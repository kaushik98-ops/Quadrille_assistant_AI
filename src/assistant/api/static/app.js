"use strict";

const $ = (id) => document.getElementById(id);
const log = $("log");
let apiKey = sessionStorage.getItem("apiKey") || "";
let conversationId = newId();
let busy = false;

function newId() {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return "web-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function setStatus(text) { $("status").textContent = text; }
function scrollDown() { log.scrollTop = log.scrollHeight; }

// Everything is built with textContent, never innerHTML: model output and document text are untrusted.
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addMessage(role, text = "") {
  const div = el("div", "msg " + role);
  const meta = el("div", "meta");
  const body = el("div", "body", text);
  div.append(meta, body);
  log.append(div);
  scrollDown();
  return { div, meta, body };
}

async function errorText(res) {
  if (res.status === 401) return "Invalid or missing API key.";
  if (res.status === 429) return "Rate limit reached. Try again in " + (res.headers.get("Retry-After") || "a few") + " seconds.";
  try {
    const data = await res.json();
    if (typeof data.detail === "string") return data.detail;
  } catch (_) { /* not JSON */ }
  return "Request failed (" + res.status + ").";
}

async function post(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": "Bearer " + apiKey },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res;
}

// Parses a server-sent-events stream: blocks separated by a blank line.
async function* readEvents(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

function renderSources(div, sources) {
  const box = el("div", "sources");
  if (!sources.length) {
    box.textContent = "No matching passages in your documents.";
  } else {
    box.append(el("div", null, "Sources"));
    for (const s of sources) {
      box.append(el("div", null, `[${s.n}] ${s.source}, page ${s.page} (score ${s.score.toFixed(2)})`));
    }
  }
  div.append(box);
}

function renderConfirm(div, data) {
  const box = el("div", "confirm");
  box.append(el("div", "title", "The assistant wants to run: " + data.tool));
  box.append(el("pre", null, JSON.stringify(data.args, null, 2)));
  const approve = el("button", null, "Approve");
  const deny = el("button", "secondary", "Deny");
  const answer = (approved) => {
    approve.disabled = true;
    deny.disabled = true;
    box.append(el("div", "meta", approved ? "Approved." : "Denied."));
    const bot = addMessage("bot");
    run(() => post("/chat/confirm", { conversation_id: conversationId, run_id: data.run_id, approved }), bot);
  };
  approve.onclick = () => answer(true);
  deny.onclick = () => answer(false);
  box.append(approve, deny);
  div.append(box);
}

async function consume(res, bot) {
  for await (const { event, data } of readEvents(res)) {
    if (event === "route") {
      let label = `route: ${data.route} (${data.confidence.toFixed(2)})`;
      if (data.reason !== "model") label += ` · ${data.reason}`;
      bot.meta.textContent = label;
    } else if (event === "sources") {
      renderSources(bot.div, data.sources);
    } else if (event === "text") {
      bot.body.textContent += data.text;
    } else if (event === "confirm") {
      renderConfirm(bot.div, data);
    } else if (event === "error") {
      bot.body.textContent += (bot.body.textContent ? "\n" : "") + data.message;
      bot.div.classList.add("error");
    }
    scrollDown();
  }
}

async function run(makeRequest, bot) {
  busy = true;
  $("send").disabled = true;
  try {
    await consume(await makeRequest(), bot);
  } catch (err) {
    bot.body.textContent = err.message;
    bot.div.classList.add("error");
  } finally {
    busy = false;
    $("send").disabled = false;
  }
}

async function send() {
  const text = $("input").value.trim();
  if (!text || busy) return;
  if (!apiKey) { setStatus("Enter your API key first."); return; }
  $("input").value = "";
  addMessage("user", text);
  const bot = addMessage("bot");
  await run(() => post("/chat", { conversation_id: conversationId, message: text }), bot);
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); send(); });
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});

$("save-key").addEventListener("click", () => {
  apiKey = $("key").value.trim();
  $("key").value = "";
  sessionStorage.setItem("apiKey", apiKey);  // kept for this browser tab only
  setStatus(apiKey ? "Key saved for this tab." : "Key cleared.");
});

$("new-chat").addEventListener("click", () => {
  conversationId = newId();
  log.replaceChildren();
  setStatus("New conversation started.");
});

$("pdf").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  if (!apiKey) { setStatus("Enter your API key first."); return; }
  setStatus("Uploading and indexing " + file.name + "...");
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/documents", { method: "POST", headers: { "Authorization": "Bearer " + apiKey }, body: form });
    if (!res.ok) throw new Error(await errorText(res));
    const data = await res.json();
    setStatus(`${data.filename}: ${data.chunks} chunks indexed.`);
  } catch (err) {
    setStatus(err.message);
  }
});

$("list-docs").addEventListener("click", async () => {
  if (!apiKey) { setStatus("Enter your API key first."); return; }
  try {
    const res = await fetch("/documents", { headers: { "Authorization": "Bearer " + apiKey } });
    if (!res.ok) throw new Error(await errorText(res));
    const { documents } = await res.json();
    setStatus(documents.length ? documents.map((d) => `${d.name} (${d.chunks})`).join(", ") : "No documents indexed yet.");
  } catch (err) {
    setStatus(err.message);
  }
});

if (apiKey) setStatus("Key loaded for this tab.");
