# InsightDesk — AI reputation intelligence reports

Upload a social-listening / media export (Signal AI, Meltwater, Brandwatch, Talkwalker or any CSV/Excel), and InsightDesk:

1. **Cleans it** — the Step 0 relevance & noise gate (duplicates, spam, bots, pure noise, contentless replies, brand mismatch, job/ticker noise). Nothing is deleted; every removed row keeps its reason in a queryable *excluded bin*.
2. **Analyses it** — perception lenses, sentiment (NSS / reach-weighted NSSw), emotions, momentum (velocity z), priority, weekly trajectory, spikes, share of voice vs peers, and the advanced formulas (C, E, journalist trend, V, PT, R).
3. **Writes a report** — Claude or Azure OpenAI writes the narrative from the computed figures; every number in the narrative is checked back against the data.
4. **Copies any report layout** — upload a sample report (PDF, PowerPoint, Word, Excel or an image). The AI reads its sections, KPIs and charts and maps them to your new data; KPIs the new data can't support are marked, not faked.
5. **Answers questions** — the *Ask the data* tab is a chat that calculates answers with a safe query engine and shows how each answer was worked out.

---

## Run locally

```bash
cp .env.example .env        # set APP_SECRET_KEY at minimum
./run.sh                    # creates a venv, installs, starts on http://localhost:8000
```

Then open **Settings**, paste your Claude key and/or Azure OpenAI key, press *Test connection*, and *Save settings*.

## Deploy on Render (free plan, from GitHub)

1. Push this folder to a private GitHub repo.
2. In Render: **New + → Blueprint**, pick the repo. `render.yaml` creates a **free** web service.
3. When asked, fill in `APP_PASSWORD`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY` (leave `ADMIN_TOKEN` empty if you don't need it). Click **Apply**.
4. Open the `onrender.com` link and sign in.

What the free plan means:
- **Storage is temporary.** Uploads, reports, saved layouts and chat history are wiped when the app restarts, redeploys or sleeps. The app shows a yellow note about this. Download reports (HTML / Save as PDF) right after creating them. Keys and password live in Render's Environment, so they are never lost.
- **It sleeps** after about 15 minutes without visits; the first visit afterwards takes 30–60 seconds to wake up.
- **512 MB memory.** Uploads are capped at 40 MB (`MAX_UPLOAD_MB`). Very large exports may run out of memory — split them by month if that happens.
- Keep **one worker**. Background jobs run as threads in the web process.

To move to a paid plan later: set `plan: starter`, remove `EPHEMERAL_STORAGE`, raise `MAX_UPLOAD_MB`, and add a disk:
```yaml
    disk:
      name: insightdesk-data
      mountPath: /var/data
      sizeGB: 5
```
plus env var `DATA_DIR=/var/data`.

## How API keys are handled

- Saved from the Settings tab → encrypted with Fernet (AES + HMAC) using `APP_SECRET_KEY`, stored in SQLite on the server.
- The browser only ever receives a masked version (`sk-a••••x9Qa`).
- Environment variables (`ANTHROPIC_API_KEY`, `AZURE_OPENAI_*`) work as fallbacks.
- `APP_PASSWORD` shows a sign-in screen (signed, HttpOnly session cookie, 5 wrong tries → 5-minute lockout). `ADMIN_TOKEN` (optional) is additionally required to change keys in Settings. Scripts can send the password as header `X-Access-Token`.
- Recommended for production: set `ANTHROPIC_API_KEY` and `AZURE_OPENAI_API_KEY` (plus endpoint/deployment/version) as server environment variables. Both models are then available to every user without anyone entering keys.
- **The user chooses the model for every AI action.** Clicking *Create report*, *Learn this layout*, or re-running with AI classification opens a dialog: Claude (Anthropic) or OpenAI (Azure). An option is greyed out if its key isn't set on the server. The chat has its own "Answer with" picker. Each report records which model wrote it.
- The API enforces this too: `POST /api/reports` and `POST /api/templates` reject requests without `provider` = `claude` or `azure_openai`.

## Architecture

```
static/            index.html · app.css · app.js        (no build step)
app/
  main.py          FastAPI routes, background jobs, auth
  ingest.py        read CSV/Excel/JSON/markdown tables; auto column mapping (name + content checks)
  taxonomy.py      seed lenses, emotion lexicon, risk/complaint/misinformation flags
  cleaning.py      Step 0 noise gate + Actionability score
  enrich.py        optional AI classification of a sample (sentiment, emotion, lenses, emergent topics)
  analytics.py     all calculations → facts pack + metric catalog
  query_engine.py  safe JSON query language (no code execution) used by chat and templates
  reporting.py     blueprints, data resolution, narrative writing, number check, HTML rendering
  template_extract.py   read sample reports (PDF text + page images, PPTX incl. chart data, DOCX, XLSX, images)
  chat.py          tool-calling Q&A
  llm.py           one interface for Claude and Azure OpenAI (text, vision, tool use, JSON)
  storage.py       SQLite: settings, datasets, reports, templates, chat
```

**Numbers are never generated by the model.** Python computes every KPI, chart and table; the model only writes words around them. After writing, `verify_numbers` extracts each figure from the narrative and matches it to the computed data (±1.2%). The report screen shows "x of y figures traced", and lists any that didn't match.

## Methodology → code

| Methodology step | Where |
|---|---|
| Step 0 noise gate, Actionability = w1·rel + w2·opinion + w3·specificity + w4·(1−spam) + w5·(1−bot), θ | `cleaning.py` (θ is a slider in Analysis settings) |
| Signal rate = kept ÷ ingested | `cleaning.gate_summary` |
| Step 1 seed lenses + emergent candidates | `taxonomy.DEFAULT_LENSES` (editable in Settings), `analytics.lens_table` |
| Step 2 multi-label assignment | keyword match by default; classifier via **AI classification** toggle |
| Step 3 Volume, Share, WVolume = Σ(1 + ln(1+reach) + w·eng) | `analytics.lens_table` |
| Step 4 NSS, %Neg, NSSw (reach-weighted) | `analytics.sentiment_stats` |
| Step 5 emotions + Complaint / Risk / Misinformation flags | `taxonomy.tag`, `enrich.py` |
| Step 6 Tone (±0.15), dominant emotion, momentum (velocity z), "Denotes" line | `analytics.lens_table` |
| Step 7 Priority = 0.30·%Neg + 0.20·normVol + 0.20·σ(z) + 0.15·reach share + 0.15·risk share | `analytics.lens_table` |
| Narrative pull-through PT | needs "Our key messages" in Analysis settings |
| Newsroom escalation E, Journalist warming/cooling, Organic vs coordinated C, Vacuum V, Resilience R | `analytics.advanced_formulas` |

Two deliberate design choices:
- **Coverage vs actionable.** Volume, share of voice and the sentiment trajectory use the de-duplicated *coverage* set (news headlines count as coverage even without an opinion). Lenses use the stricter *actionable* set, as the methodology intends.
- **Syndicated news isn't "bot".** Near-duplicate detection only flags social posts (same text from ≥5 authors on the same day), so a wire story carried by 40 outlets stays in coverage.

Proxies to be aware of: relevance uses brand-term matching rather than embedding cosine similarity; competitor presence in V is measured only when the file has several brands; author type (media / brand-owned / individual) is inferred from names and handles.

## Data tips

- `########` in Excel is a too-narrow column, not data. InsightDesk treats it as empty — export the real values.
- If columns look shifted (e.g. the *title* column holds URLs), the auto-mapper rejects them and picks the right column by content. Check **Analysis settings → Column mapping** after the first upload and fix anything that looks wrong.
- Sentiment from the provider is used when present; otherwise a lexicon fills it in, and the AI classification toggle upgrades a sample.

## API

Interactive docs at `/api/docs`. Main routes: `POST /api/datasets` (upload), `POST /api/datasets/{id}/config` (re-run), `GET /api/datasets/{id}/rows|export|facts`, `POST /api/reports`, `GET /api/reports/{id}/html`, `POST /api/templates`, `POST /api/chat/{id}`, `GET /api/jobs/{id}`, `GET/POST /api/settings`.

## Next steps worth considering

Scheduled weekly runs with email delivery · user accounts and roles (SSO) · Word/PowerPoint export of reports · embedding-based lens assignment (Azure `text-embedding-3` deployment) · alert rules (negative share > 15% / week).
