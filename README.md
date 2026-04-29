# deal_research_workflow

Streamlined research workflow for new potential deals. Four phases, each
with an AI chat that can manipulate state through typed tool calls:

  1. **org_select**       — name a company, AI proposes matching orgs from our DB
  2. **entity_select**    — pick the entities (docs / emails / slack / calendar)
                            that should make it into the data room
  3. **data_room_setup**  — choose preset / custom questions, build the room
  4. **data_room_view**   — overview + Q&A on the built room with citations

Sessions are versioned (every change is an append-only row), so undo / redo
and deep-link sharing work for free.

## Stack

- Backend: FastAPI + Pydantic + psycopg2 (Python 3.13)
- Frontend: Vite + React + TypeScript + Tailwind + TanStack Query + Zustand
- DB: Neon Postgres, schema `research` (cross-schema reads from `dealcloud.*`)
- LLM: Anthropic Claude (Sonnet 4.6 chat agent, Haiku 4.5 cheap subtasks)
- Auth: Entra ID (shared with org_history_viewer) — stubbed in V0

## Local dev

```bash
# backend
cd backend
python -m venv .venv && source .venv/Scripts/activate     # Windows bash
pip install -r requirements.txt
cp .env.example .env                                       # fill DATABASE_URL etc
psql "$DATABASE_URL" -f migrations/001_initial.sql         # apply schema once
uvicorn app.main:app --reload --port 8000

# frontend
cd frontend
npm install
npm run dev                                                # http://localhost:5173
```

## Repo layout

```
backend/
  app/
    main.py            FastAPI app + middleware
    config.py          env-driven settings
    db.py              connection helper
    auth.py            current_user dependency (stub for V0)
    routes/            one file per resource (sessions, versions, orgs, ...)
    models/            Pydantic shapes
    services/          business logic separated from routes
  migrations/
    001_initial.sql    research schema

frontend/
  src/
    routes/            page components per URL
    components/        UI building blocks
    lib/               API client + TanStack Query hooks
    stores/            Zustand UI stores
```
