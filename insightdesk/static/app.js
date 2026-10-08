/* InsightDesk front end — plain JS, no build step. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n, d = 0) => (n == null || isNaN(n) ? "—" : Number(n).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d }));

  const S = {
    view: "data", datasets: [], ds: null, settings: null,
    provider: localStorage.getItem("provider") || "",
    rowSet: "actionable", rowOffset: 0, rowTotal: 0,
    reportId: null, templateId: null, templates: [], tplFile: null, dataFile: null,
  };

  // ------------------------------------------------------------ api
  async function api(path, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    const adm = sessionStorage.getItem("adminToken");
    if (adm) headers["X-Admin-Token"] = adm;
    const send = { ...opts, headers, credentials: "same-origin" };
    if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; send.body = JSON.stringify(opts.json); }
    const res = await fetch(path, send);
    if (res.status === 401 && path !== "/api/login") { await showLogin(); return api(path, opts); }
    if (!res.ok) {
      let msg = res.statusText;
      try { const j = await res.json(); msg = j.detail || msg; } catch (_) {}
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    const ct = res.headers.get("content-type") || "";
    return ct.includes("application/json") ? res.json() : res.text();
  }

  // Sign-in screen. Resolves once the password is accepted.
  let loginWait = null;
  function showLogin() {
    if (loginWait) return loginWait;
    $("#login").hidden = false; $("#loginErr").textContent = ""; $("#loginPw").value = ""; setTimeout(() => $("#loginPw").focus(), 50);
    loginWait = new Promise((resolve) => {
      const form = $("#loginForm");
      const onSubmit = async (e) => {
        e.preventDefault(); $("#loginBtn").disabled = true; $("#loginErr").textContent = "";
        try {
          await api("/api/login", { method: "POST", json: { password: $("#loginPw").value } });
          form.removeEventListener("submit", onSubmit); $("#login").hidden = true; loginWait = null; resolve();
        } catch (err) { $("#loginErr").textContent = err.message; $("#loginPw").select(); }
        $("#loginBtn").disabled = false;
      };
      form.addEventListener("submit", onSubmit);
    });
    return loginWait;
  }
  $("#logoutBtn").addEventListener("click", async () => { await fetch("/api/logout", { method: "POST" }); location.reload(); });

  function toast(msg, err = false) {
    const t = $("#toast"); t.textContent = msg; t.className = "toast" + (err ? " err" : ""); t.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), err ? 6000 : 3000);
  }

  async function download(path, fallbackName) {
    let res = await fetch(path, { credentials: "same-origin" });
    if (res.status === 401) { await showLogin(); res = await fetch(path, { credentials: "same-origin" }); }
    if (!res.ok) return toast("Download failed", true);
    const cd = res.headers.get("content-disposition") || ""; const m = cd.match(/filename="([^"]+)"/);
    const blob = await res.blob(); const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = m ? m[1] : fallbackName; a.click(); URL.revokeObjectURL(a.href);
  }

  // ------------------------------------------------------------ jobs
  function watchJob(jobId, box, onDone) {
    box.hidden = false; box.classList.remove("err");
    const tick = async () => {
      let j;
      try { j = await api(`/api/jobs/${jobId}`); } catch (e) { box.classList.add("err"); box.textContent = e.message; return; }
      const steps = j.progress.map((p) => `<li>${esc(p.msg)}</li>`).join("");
      if (j.status === "running") {
        box.innerHTML = `<div class="now"><span class="spin"></span>Working…</div><ol>${steps}</ol>`;
        setTimeout(tick, 1200);
      } else if (j.status === "error") {
        box.classList.add("err"); box.innerHTML = `<div class="now">Stopped: ${esc(j.error)}</div><ol>${steps}</ol>`;
      } else {
        box.innerHTML = `<div class="now">Done</div><ol>${steps}</ol>`;
        setTimeout(() => (box.hidden = true), 4000);
        onDone && onDone(j.result);
      }
    };
    tick();
  }

  // ------------------------------------------------------------ navigation
  function show(view) {
    S.view = view;
    $$(".rail button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
    $$(".view").forEach((v) => (v.hidden = v.id !== `view-${view}`));
    if (view === "report") loadReports();
    if (view === "template") loadTemplates();
    if (view === "ask") loadChat();
    if (view === "settings") loadSettings();
  }
  $$(".rail button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));

  // ------------------------------------------------------------ model choice
  const PNAME = { claude: "Claude (Anthropic)", azure_openai: "OpenAI (Azure)" };
  const configured = () => Object.entries(S.settings?.providers || {}).filter(([, v]) => v.configured).map(([k]) => k);

  function renderProvider() {
    const st = S.settings?.providers || {};
    const ok = configured();
    if (!ok.includes(S.provider)) S.provider = ok.includes(S.settings?.default_provider) ? S.settings.default_provider : ok[0] || "";
    $("#railFoot").textContent = ok.length ? "" : "No AI key found on the server. Add one in Settings or as an environment variable.";
    const sel = $("#askProv");
    sel.innerHTML = ["claude", "azure_openai"].map((p) => `<option value="${p}" ${st[p]?.configured ? "" : "disabled"}>${PNAME[p]}${st[p]?.configured ? "" : " — not set up"}</option>`).join("");
    if (S.provider) sel.value = S.provider;
  }
  $("#askProv").addEventListener("change", (e) => { S.provider = e.target.value; localStorage.setItem("provider", S.provider); });

  // Ask the user which model to use. Resolves to "claude" | "azure_openai", or null if cancelled.
  function chooseProvider(title, sub) {
    const st = S.settings?.providers || {};
    if (!configured().length) { toast("No AI key is set up on the server. Add one in Settings.", true); return Promise.resolve(null); }
    const dlg = $("#provDlg");
    $("#provDlgTitle").textContent = title;
    $("#provDlgSub").textContent = sub || "Both use the same computed figures; only the writing differs.";
    $$(".prov-opt", dlg).forEach((b) => {
      const p = b.dataset.p, on = !!st[p]?.configured;
      b.disabled = !on;
      b.classList.toggle("last", p === S.provider);
      $(`[data-model="${p}"]`, dlg).textContent = p === "claude" ? st.claude?.model || "" : st.azure_openai?.deployment || "";
      $(`[data-state="${p}"]`, dlg).textContent = on ? (p === S.provider ? "Ready · used last time" : "Ready") : "Not set up on the server";
    });
    return new Promise((resolve) => {
      dlg.addEventListener("close", () => {
        const v = dlg.returnValue;
        if (v === "claude" || v === "azure_openai") { S.provider = v; localStorage.setItem("provider", v); renderProvider(); resolve(v); }
        else resolve(null);
      }, { once: true });
      dlg.returnValue = "";
      dlg.showModal();
      const pre = $(`.prov-opt[data-p="${S.provider}"]`, dlg);
      (pre && !pre.disabled ? pre : $(".prov-opt:not(:disabled)", dlg))?.focus();
    });
  }

  // ------------------------------------------------------------ datasets
  async function loadDatasets(selectId) {
    S.datasets = await api("/api/datasets");
    const sel = $("#dsSelect");
    sel.innerHTML = S.datasets.length ? S.datasets.map((d) => `<option value="${d.id}">${esc(d.name)} · ${d.status === "ready" ? fmt(d.rows) + " rows" : d.status}</option>`).join("")
      : `<option value="">No dataset yet</option>`;
    const want = selectId || localStorage.getItem("ds") || (S.datasets[0] && S.datasets[0].id);
    if (want && S.datasets.find((d) => d.id === want)) sel.value = want;
    await selectDataset(sel.value);
    renderDsList();
  }
  $("#dsSelect").addEventListener("change", (e) => selectDataset(e.target.value));

  async function selectDataset(id) {
    if (!id) { S.ds = null; $("#dsEmpty").hidden = false; $("#dsDetail").hidden = true; return; }
    localStorage.setItem("ds", id);
    S.ds = await api(`/api/datasets/${id}`);
    $("#dsEmpty").hidden = true; $("#dsDetail").hidden = false;
    renderDataset();
    if (S.view === "report") loadReports();
    if (S.view === "ask") loadChat();
  }

  const REASON_COLORS = { "Duplicate": "#8A97A6", "Spam & promotional": "#B42318", "Bot & automated": "#7A1A12", "Pure noise": "#A9B4C2",
    "Contentless engagement": "#C3CBD5", "Off-topic / brand-mismatch": "#B54708", "Non-opinion / factual noise": "#5B6776", "Below actionability threshold": "#D2B48C" };

  function renderDataset() {
    const d = S.ds, m = d.meta || {};
    $("#dsTitle").textContent = d.name;
    $("#dsPeriod").textContent = d.status !== "ready" ? `Status: ${d.status}` :
      `${m.period?.[0] || "?"} to ${m.period?.[1] || "?"} · focus brand ${m.primary_entity || "—"} · ${fmt(m.rows)} rows`;
    const h = m.data_health;
    if (!h) { $("#strip").innerHTML = ""; $("#figs").innerHTML = ""; $("#stripLegend").innerHTML = ""; return; }
    const tot = h.total_ingested || 1;
    const seg = [[`Actionable`, h.actionable_items, "#0F5C63"], ...Object.entries(h.excluded_by_reason).map(([k, v]) => [k, v, REASON_COLORS[k] || "#999"])];
    $("#strip").innerHTML = seg.map(([k, v, c]) => `<div style="width:${(100 * v) / tot}%;background:${c}" title="${esc(k)}: ${fmt(v)}">${k === "Actionable" && v / tot > 0.18 ? `<span>${h.signal_rate_pct}% actionable</span>` : ""}</div>`).join("");
    $("#stripLegend").innerHTML = seg.map(([k, v, c]) => `<span><i style="background:${c}"></i>${esc(k)} ${fmt(v)}</span>`).join("");
    const o = m.overview || {};
    const figs = [["Ingested", fmt(h.total_ingested)], ["Actionable", fmt(h.actionable_items)], ["Focus-brand items", fmt(o.items)],
      ["Negative share", o.neg_share_pct != null ? o.neg_share_pct + "%" : "—", "neg"], ["Net sentiment (reach-wtd)", o.nssw != null ? (o.nssw > 0 ? "+" : "") + o.nssw : "—"],
      ["Social engagement / post", fmt(o.eng_per_post, 1)]];
    $("#figs").innerHTML = figs.map(([a, b, c]) => `<div><dt>${a}</dt><dd class="${c || ""}">${b}</dd></div>`).join("");
    renderConfig();
    S.rowOffset = 0; loadRows();
  }

  function renderConfig() {
    const d = S.ds, c = d.config || {}, m = d.meta || {};
    const ents = (m.entities || []).map((e) => e.entity);
    $("#cfgPrimary").innerHTML = ents.map((e) => `<option ${e === (c.primary_entity || m.primary_entity) ? "selected" : ""}>${esc(e)}</option>`).join("") || "<option>Unspecified</option>";
    $("#cfgBrand").value = (c.brand_terms || []).join(", ");
    $("#cfgOwned").value = (c.owned_handles || []).join(", ");
    $("#cfgExclude").value = (c.exclude_terms || []).join(", ");
    $("#cfgFrame").value = (c.frame_terms || []).join(", ");
    $("#cfgTheta").value = c.theta ?? 0.5; $("#cfgThetaOut").textContent = Number(c.theta ?? 0.5).toFixed(2);
    $("#cfgAi").checked = !!c.ai_enrich; $("#cfgAiN").value = c.ai_sample || 400;
    const cols = m.columns || [], help = m.field_help || {}, map = m.mapping || {};
    const prof = Object.fromEntries((m.column_profile || []).map((p) => [p.column, p]));
    $("#mapTable").innerHTML = `<tr><th>Field</th><th>Source column</th><th>Example value</th></tr>` + Object.keys(help).map((f) => {
      const cur = map[f]?.column || "";
      const ex = cur && prof[cur] ? prof[cur].sample?.[0] || "" : "";
      return `<tr><td>${esc(help[f])}</td><td><select data-field="${f}"><option value="">(not used / auto)</option>${cols.map((col) => `<option ${col === cur ? "selected" : ""}>${esc(col)}</option>`).join("")}</select></td><td class="muted">${esc(ex)}</td></tr>`;
    }).join("");
  }
  $("#cfgTheta").addEventListener("input", (e) => ($("#cfgThetaOut").textContent = Number(e.target.value).toFixed(2)));
  const splitList = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);

  $("#rerunBtn").addEventListener("click", async () => {
    const auto = S.ds.meta.mapping || {};
    const overrides = {};
    $$("#mapTable select").forEach((s) => { if ((auto[s.dataset.field]?.column || "") !== s.value) overrides[s.dataset.field] = s.value; });
    const body = { primary_entity: $("#cfgPrimary").value, brand_terms: splitList($("#cfgBrand").value), owned_handles: splitList($("#cfgOwned").value),
      exclude_terms: splitList($("#cfgExclude").value), frame_terms: splitList($("#cfgFrame").value), theta: Number($("#cfgTheta").value),
      ai_enrich: $("#cfgAi").checked, ai_sample: Number($("#cfgAiN").value), provider: null,
      mapping_overrides: { ...(S.ds.config.mapping_overrides || {}), ...overrides } };
    if (body.ai_enrich) {
      body.provider = await chooseProvider("Which AI should classify the sample?", `Labels sentiment, emotion and lenses for up to ${body.ai_sample} items.`);
      if (!body.provider) return;
    }
    try {
      const r = await api(`/api/datasets/${S.ds.id}/config`, { method: "POST", json: body });
      watchJob(r.job_id, $("#dataJob"), () => loadDatasets(S.ds.id));
      $("#dataJob").scrollIntoView({ behavior: "smooth", block: "center" });
    } catch (e) { toast(e.message, true); }
  });

  // rows browser
  $$(".tabs button[data-set]").forEach((b) => b.addEventListener("click", () => {
    $$(".tabs button[data-set]").forEach((x) => x.classList.toggle("on", x === b));
    S.rowSet = b.dataset.set; S.rowOffset = 0;
    const sel = $("#reasonSel"); sel.hidden = S.rowSet !== "excluded";
    if (!sel.hidden) {
      const r = S.ds.meta.data_health?.excluded_by_reason || {};
      sel.innerHTML = `<option value="">All reasons</option>` + Object.entries(r).map(([k, v]) => `<option value="${esc(k)}">${esc(k)} (${fmt(v)})</option>`).join("");
    }
    loadRows();
  }));
  $("#reasonSel").addEventListener("change", () => { S.rowOffset = 0; loadRows(); });
  let searchT; $("#rowSearch").addEventListener("input", () => { clearTimeout(searchT); searchT = setTimeout(() => { S.rowOffset = 0; loadRows(); }, 300); });
  $("#prevBtn").addEventListener("click", () => { S.rowOffset = Math.max(0, S.rowOffset - 50); loadRows(); });
  $("#nextBtn").addEventListener("click", () => { if (S.rowOffset + 50 < S.rowTotal) { S.rowOffset += 50; loadRows(); } });
  $("#exportBtn").addEventListener("click", () => download(`/api/datasets/${S.ds.id}/export?set=${S.rowSet}`, "export.csv"));

  async function loadRows() {
    if (!S.ds || S.ds.status !== "ready") { $("#rowsTable").innerHTML = ""; return; }
    const p = new URLSearchParams({ set: S.rowSet, offset: S.rowOffset, limit: 50, q: $("#rowSearch").value, reason: S.rowSet === "excluded" ? $("#reasonSel").value : "" });
    const r = await api(`/api/datasets/${S.ds.id}/rows?${p}`);
    S.rowTotal = r.total;
    const excl = S.rowSet === "excluded";
    $("#rowsTable").innerHTML = `<tr><th>Date</th><th>Platform</th><th>Author</th><th>Sentiment</th><th>Eng.</th><th>${excl ? "Reason" : "Lenses"}</th><th>Score</th><th>Text</th></tr>` +
      r.rows.map((x) => `<tr><td>${esc((x.date || "").slice(0, 10))}</td><td>${esc(x.platform)}</td><td>${esc(x.author)}<br><small class="muted">${esc(x.author_type || "")}</small></td>
        <td class="s-${x.sentiment}">${x.sentiment}</td><td>${fmt(x.engagement)}</td>
        <td>${excl ? esc(x.exclude_reason) : (x.lenses || []).map((l) => `<span class="tag">${esc(l)}</span>`).join("")}</td>
        <td>${x.actionability != null ? Number(x.actionability).toFixed(2) : ""}</td>
        <td class="txt">${x.url ? `<a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.text)}</a>` : esc(x.text)}</td></tr>`).join("") ||
      `<tr><td colspan="8" class="muted">No rows match.</td></tr>`;
    $("#pageInfo").textContent = r.total ? `${fmt(S.rowOffset + 1)}–${fmt(Math.min(S.rowOffset + 50, r.total))} of ${fmt(r.total)}` : "";
  }

  function renderDsList() {
    $("#dsList").innerHTML = `<tr><th>Name</th><th>File</th><th>Rows</th><th>Period</th><th>Status</th><th></th></tr>` + S.datasets.map((d) =>
      `<tr class="${S.ds && d.id === S.ds.id ? "on" : ""}"><td>${esc(d.name)}</td><td class="muted">${esc(d.filename)}</td><td>${fmt(d.rows)}</td>
       <td>${esc((d.period || []).join(" to "))}</td><td>${esc(d.status)}</td>
       <td><button class="ghost" data-open="${d.id}">Open</button><button class="ghost danger" data-del="${d.id}">Delete</button></td></tr>`).join("");
  }
  $("#dsList").addEventListener("click", async (e) => {
    const o = e.target.dataset.open, del = e.target.dataset.del;
    if (o) { $("#dsSelect").value = o; selectDataset(o); }
    if (del && confirm("Delete this dataset, its reports and its conversation?")) {
      await api(`/api/datasets/${del}`, { method: "DELETE" }); localStorage.removeItem("ds"); loadDatasets();
    }
  });

  // upload
  function wireDrop(zone, input, onFile) {
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
    zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("drag"); if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); });
    input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
  }
  wireDrop($("#dropzone"), $("#dataFile"), async (file) => {
    const fd = new FormData(); fd.append("file", file); fd.append("name", $("#dataName").value || file.name.replace(/\.[^.]+$/, ""));
    fd.append("config", JSON.stringify({}));
    const box = $("#dataJob"); box.hidden = false; box.innerHTML = `<span class="spin"></span>Uploading ${esc(file.name)}…`;
    try {
      const r = await api("/api/datasets", { method: "POST", body: fd });
      $("#dataName").value = "";
      await loadDatasets(r.dataset_id);
      watchJob(r.job_id, box, () => loadDatasets(r.dataset_id));
    } catch (e) { box.classList.add("err"); box.textContent = e.message; }
  });

  // ------------------------------------------------------------ reports
  async function loadTemplatesInto(sel) {
    S.templates = await api("/api/templates");
    const cur = sel.value;
    sel.innerHTML = `<option value="">Reputation risk scan (built in)</option>` + S.templates.map((t) => `<option value="${t.id}">${esc(t.name)} (from ${esc(t.source_filename || "sample")})</option>`).join("");
    if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
  }

  async function loadReports() {
    await loadTemplatesInto($("#repTemplate"));
    if (!S.ds) { $("#repList").innerHTML = `<li class="muted">Pick a dataset first.</li>`; return; }
    const reps = await api(`/api/reports?dataset_id=${S.ds.id}`);
    $("#repList").innerHTML = reps.map((r) => `<li class="${r.id === S.reportId ? "on" : ""}"><button data-rep="${r.id}">${esc(r.title)}
      <small>${new Date(r.created_at * 1000).toLocaleString()} · ${r.kind === "template" ? "from sample" : "built in"} · ${PNAME[r.provider] || esc(r.provider)}</small></button></li>`).join("")
      || `<li class="muted" style="padding:8px 0">No reports yet for this dataset.</li>`;
    if (S.reportId && reps.some((r) => r.id === S.reportId)) openReport(S.reportId);
    else $("#viewer").hidden = true;
  }
  $("#repList").addEventListener("click", (e) => { const b = e.target.closest("[data-rep]"); if (b) openReport(b.dataset.rep); });

  async function openReport(id) {
    S.reportId = id;
    $$("#repList li").forEach((li) => li.classList.toggle("on", li.querySelector(`[data-rep="${id}"]`) != null));
    const [meta, html] = await Promise.all([api(`/api/reports/${id}`), api(`/api/reports/${id}/html`)]);
    $("#viewer").hidden = false; $("#repTitle").textContent = meta.title;
    $("#repWho").textContent = `Written with ${PNAME[meta.provider] || meta.provider}`;
    $("#repFrame").srcdoc = html;
    const q = meta.quality || {};
    const un = (q.unverified || []).length;
    const qc = $("#repQc"); qc.className = "qc" + (un ? " warn" : "");
    qc.textContent = `${q.verified ?? 0} of ${q.checked ?? 0} figures traced to data`;
    $("#qcBody").innerHTML = (un ? `<p>Figures in the narrative that could not be matched to the computed data — review before sharing:</p><ul>${q.unverified.map((u) => `<li><strong>${esc(u.number)}</strong> — “…${esc(u.context)}…”</li>`).join("")}</ul>` : "<p>All checked figures match the computed data.</p>")
      + ((q.unavailable_kpis || []).length ? `<p>KPIs from the sample that this data cannot support: ${q.unavailable_kpis.map(esc).join(", ")}</p>` : "");
  }

  $("#genBtn").addEventListener("click", async () => {
    if (!S.ds || S.ds.status !== "ready") return toast("Pick a dataset that has finished analysing", true);
    const provider = await chooseProvider("Which AI should write this report?");
    if (!provider) return;
    const btn = $("#genBtn"); btn.disabled = true;
    try {
      const r = await api("/api/reports", { method: "POST", json: { dataset_id: S.ds.id, provider, template_id: $("#repTemplate").value || null, focus: $("#repFocus").value } });
      watchJob(r.job_id, $("#repJob"), (res) => { btn.disabled = false; S.reportId = res.report_id; loadReports(); });
      const poll = setInterval(async () => { const j = await api(`/api/jobs/${r.job_id}`).catch(() => null); if (!j || j.status !== "running") { clearInterval(poll); btn.disabled = false; } }, 2000);
    } catch (e) { btn.disabled = false; toast(e.message, true); }
  });
  $("#dlHtml").addEventListener("click", () => download(`/api/reports/${S.reportId}/html?download=1`, "report.html"));
  $("#printBtn").addEventListener("click", () => $("#repFrame").contentWindow.print());
  $("#delRep").addEventListener("click", async () => {
    if (!confirm("Delete this report?")) return;
    await api(`/api/reports/${S.reportId}`, { method: "DELETE" }); S.reportId = null; loadReports();
  });

  // ------------------------------------------------------------ templates
  wireDrop($("#tplDrop"), $("#tplFile"), (f) => { S.tplFile = f; $("#tplFileName").textContent = `Selected: ${f.name}`; if (!$("#tplName").value) $("#tplName").value = f.name.replace(/\.[^.]+$/, ""); });
  $("#tplBtn").addEventListener("click", async () => {
    if (!S.tplFile) return toast("Choose a sample report first", true);
    if (!S.ds || S.ds.status !== "ready") return toast("Pick an analysed dataset — the layout is mapped onto its fields", true);
    const provider = await chooseProvider("Which AI should read this sample report?", "It studies the layout, KPIs and charts, then maps them to your data.");
    if (!provider) return;
    const fd = new FormData(); fd.append("file", S.tplFile); fd.append("dataset_id", S.ds.id); fd.append("name", $("#tplName").value); fd.append("provider", provider);
    try {
      const r = await api("/api/templates", { method: "POST", body: fd });
      watchJob(r.job_id, $("#tplJob"), (res) => { S.templateId = res.template_id; loadTemplates(); });
    } catch (e) { toast(e.message, true); }
  });

  async function loadTemplates() {
    S.templates = await api("/api/templates");
    $("#tplList").innerHTML = S.templates.map((t) => `<li class="${t.id === S.templateId ? "on" : ""}"><button data-tpl="${t.id}">${esc(t.name)}<small>${t.section_count} sections · ${esc(t.source_filename || "")}</small></button></li>`).join("")
      || `<li class="muted" style="padding:8px 0">No saved layouts yet.</li>`;
    if (S.templateId) openTemplate(S.templateId); else $("#tplDetail").hidden = true;
  }
  $("#tplList").addEventListener("click", (e) => { const b = e.target.closest("[data-tpl]"); if (b) openTemplate(b.dataset.tpl); });

  async function openTemplate(id) {
    S.templateId = id;
    $$("#tplList li").forEach((li) => li.classList.toggle("on", li.querySelector(`[data-tpl="${id}"]`) != null));
    let t; try { t = await api(`/api/templates/${id}`); } catch (_) { $("#tplDetail").hidden = true; return; }
    const bp = t.blueprint;
    $("#tplDetail").hidden = false; $("#tplTitle").textContent = t.name; $("#tplDesc").textContent = bp.description || "";
    $("#tplSections").innerHTML = (bp.sections || []).map((s) => `<li><strong>${esc(s.title)}</strong> <span class="muted">${esc(s.purpose || "")}</span><br>
      ${(s.kpis || []).map((k) => `<span class="k ${k.available === false ? "na" : ""}" title="${esc(k.note || k.metric || "custom query")}">${esc(k.label || k.source_label)}${k.available === false ? " — not in data" : ""}</span>`).join("")}
      <div class="v">${(s.visuals || []).map((v) => v.builtin ? esc(v.builtin.replace(/_/g, " ")) : esc(v.title || "custom chart")).join(" · ")}</div></li>`).join("");
    $("#tplJson").value = JSON.stringify(bp, null, 2);
  }
  $("#tplSave").addEventListener("click", async () => {
    let bp; try { bp = JSON.parse($("#tplJson").value); } catch (e) { return toast("That JSON is not valid: " + e.message, true); }
    try { await api(`/api/templates/${S.templateId}`, { method: "PUT", json: { blueprint: bp } }); toast("Layout saved"); loadTemplates(); } catch (e) { toast(e.message, true); }
  });
  $("#tplDel").addEventListener("click", async () => {
    if (!confirm("Delete this layout?")) return;
    await api(`/api/templates/${S.templateId}`, { method: "DELETE" }); S.templateId = null; loadTemplates();
  });
  $("#tplUse").addEventListener("click", async () => {
    show("report"); await loadTemplatesInto($("#repTemplate")); $("#repTemplate").value = S.templateId; $("#genBtn").click();
  });

  // ------------------------------------------------------------ ask
  function md(src) {
    const blocks = String(src || "").replace(/\r/g, "").split(/\n{2,}/);
    const inline = (t) => esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>").replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return blocks.map((b) => {
      const lines = b.split("\n");
      if (lines.length >= 2 && /^\s*\|.*\|\s*$/.test(lines[0]) && /^\s*\|?\s*:?-{2,}/.test(lines[1])) {
        const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        return `<div class="scroll"><table><tr>${cells(lines[0]).map((c) => `<th>${inline(c)}</th>`).join("")}</tr>${lines.slice(2).map((l) => `<tr>${cells(l).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</table></div>`;
      }
      if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*[-*•]\s+/, ""))}</li>`).join("")}</ul>`;
      if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) return `<ol>${lines.map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s+/, ""))}</li>`).join("")}</ol>`;
      const h = b.match(/^#{1,4}\s+(.*)/); if (h && lines.length === 1) return `<h4>${inline(h[1])}</h4>`;
      return `<p>${lines.map((l) => inline(l.replace(/^#{1,4}\s+/, ""))).join("<br>")}</p>`;
    }).join("");
  }

  function evidenceHtml(ev) {
    if (!ev || !ev.length) return "";
    const parts = ev.filter((e) => e.columns).map((e) => `<div class="ev"><code>${esc(e.tool === "search_posts" ? "Posts matching: " + JSON.stringify(e.args.keywords || []) : JSON.stringify(e.args))}</code>
      <div class="scroll"><table><tr>${e.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${e.rows.slice(0, 25).map((r) => `<tr>${r.map((v) => `<td>${esc(Array.isArray(v) ? v.join(", ") : v)}</td>`).join("")}</tr>`).join("")}</table></div>
      ${e.base_items != null ? `<div class="muted">Based on ${fmt(e.base_items)} items</div>` : ""}</div>`).join("");
    return parts ? `<details><summary>How this was calculated</summary>${parts}</details>` : "";
  }

  function addMsg(role, content, ev, provider) {
    const div = document.createElement("div");
    div.className = "msg " + (role === "user" ? "user" : "bot");
    div.innerHTML = role === "user" ? esc(content) : md(content) + evidenceHtml(ev) + (provider ? `<div class="meta">Answered with ${PNAME[provider] || provider}</div>` : "");
    $("#chat").appendChild(div); $("#chat").scrollTop = $("#chat").scrollHeight; return div;
  }

  const SUGGEST = ["What are the top 3 risks in this data?", "Which perception lens is rising fastest, and why?",
    "Show weekly negative share for the focus brand", "Who are the most active authors on negative posts?",
    "Compare negative share across all brands", "Give 5 example complaints about fees and charges"];

  async function loadChat() {
    $("#chat").innerHTML = "";
    if (!S.ds) { $("#chat").innerHTML = `<div class="empty">Pick a dataset to start asking questions.</div>`; return; }
    const h = await api(`/api/chat/${S.ds.id}`);
    if (!h.length) $("#chat").innerHTML = `<div class="empty">Ask a question about <strong>${esc(S.ds.name)}</strong>. Answers are calculated from the data and show how they were worked out.</div>`;
    h.forEach((m) => addMsg(m.role, m.content, m.evidence, m.provider));
    $("#chips").innerHTML = SUGGEST.map((s) => `<button type="button">${esc(s)}</button>`).join("");
  }
  $("#chips").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") { $("#askInput").value = e.target.textContent; $("#askForm").requestSubmit(); } });
  $("#askInput").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#askForm").requestSubmit(); } });
  $("#askForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#askInput").value.trim(); if (!q || !S.ds) return;
    const empty = $("#chat .empty"); if (empty) empty.remove();
    addMsg("user", q); $("#askInput").value = "";
    const wait = addMsg("bot", ""); wait.innerHTML = `<span class="typing"><span class="spin"></span>Analysing the data…</span>`;
    $("#askBtn").disabled = true;
    try {
      const r = await api(`/api/chat/${S.ds.id}`, { method: "POST", json: { question: q, provider: $("#askProv").value } });
      wait.innerHTML = md(r.answer) + evidenceHtml(r.evidence) + `<div class="meta">Answered with ${PNAME[r.provider] || r.provider}</div>`;
    } catch (err) { wait.innerHTML = `<p class="s-Negative">${esc(err.message)}</p>`; }
    $("#askBtn").disabled = false; $("#chat").scrollTop = $("#chat").scrollHeight;
  });
  $("#clearChat").addEventListener("click", async () => { if (S.ds && confirm("Clear this conversation?")) { await api(`/api/chat/${S.ds.id}`, { method: "DELETE" }); loadChat(); } });

  // ------------------------------------------------------------ settings
  async function loadSettings() {
    const s = (S.settings = await api("/api/settings"));
    const c = s.providers.claude, a = s.providers.azure_openai;
    $("#sClaudeKey").value = ""; $("#sClaudeKey").placeholder = c.key_masked || (c.key_from_env ? "Set via environment variable" : "sk-ant-…");
    $("#sClaudeState").textContent = c.configured ? "Configured. Leave blank to keep the current key." : "Not configured.";
    $("#sClaudeModel").value = c.model;
    $("#sAzEndpoint").value = a.endpoint; $("#sAzDeploy").value = a.deployment; $("#sAzVersion").value = a.api_version;
    $("#sAzKey").value = ""; $("#sAzKey").placeholder = a.key_masked || (a.key_from_env ? "Set via environment variable" : "Azure key");
    $("#sAzState").textContent = a.configured ? "Configured. Leave blank to keep the current key." : "Not configured.";
    $("#sDefault").value = s.default_provider; $("#sOrg").value = s.org_name;
    $("#sLenses").value = JSON.stringify(s.lens_taxonomy, null, 2);
    $("#adminBox").hidden = !s.admin_required; $("#adminToken").value = sessionStorage.getItem("adminToken") || "";
    renderProvider();
  }
  $("#resetLenses").addEventListener("click", async () => {
    if (!confirm("Restore the default lenses?")) return;
    await api("/api/settings", { method: "POST", json: { lens_taxonomy: [] } }); loadSettings(); toast("Default lenses restored");
  });
  $("#saveSettings").addEventListener("click", async () => {
    let lenses; try { lenses = JSON.parse($("#sLenses").value); } catch (e) { return toast("Lens JSON is not valid: " + e.message, true); }
    if ($("#adminToken").value) sessionStorage.setItem("adminToken", $("#adminToken").value);
    const body = { anthropic_api_key: $("#sClaudeKey").value, anthropic_model: $("#sClaudeModel").value,
      azure_openai_endpoint: $("#sAzEndpoint").value, azure_openai_deployment: $("#sAzDeploy").value,
      azure_openai_api_version: $("#sAzVersion").value, azure_openai_api_key: $("#sAzKey").value,
      default_provider: $("#sDefault").value, org_name: $("#sOrg").value, lens_taxonomy: lenses };
    try { await api("/api/settings", { method: "POST", json: body }); await loadSettings(); $("#saveOut").textContent = "Saved"; $("#saveOut").className = "test-out ok"; toast("Settings saved"); }
    catch (e) { $("#saveOut").textContent = e.message; $("#saveOut").className = "test-out err"; }
  });
  $$("[data-test]").forEach((b) => b.addEventListener("click", async () => {
    const out = $(`#test-${b.dataset.test}`); out.className = "test-out"; out.textContent = "Testing…";
    try { const r = await api("/api/settings/test", { method: "POST", json: { provider: b.dataset.test } }); out.className = "test-out ok"; out.textContent = `Connected in ${r.seconds}s`; }
    catch (e) { out.className = "test-out err"; out.textContent = e.message.slice(0, 160); }
  }));

  // ------------------------------------------------------------ boot
  (async () => {
    try {
      const sess = await api("/api/session");
      $("#logoutBtn").hidden = !sess.auth_required;
      $("#ephNote").hidden = !sess.ephemeral;
      if (sess.max_upload_mb) $("#uploadHint").textContent = `CSV, Excel, JSON. Up to ${sess.max_upload_mb} MB.`;
      if (sess.auth_required && !sess.signed_in) await showLogin();
      S.settings = await api("/api/settings"); renderProvider();
      await loadDatasets();
    } catch (e) { toast(e.message, true); }
  })();
})();
