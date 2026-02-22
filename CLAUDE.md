# Coronary RWS Analyser v2

## Project Overview
Desktop medical imaging application for coronary artery analysis using RWS (Radial Wall Strain) and QFR (Quantitative Flow Ratio). Research tool only — **NOT validated for clinical decision-making**.

## Architecture
- **Backend**: Python 3.11+ / FastAPI, 3-layer architecture (api/ → services/ → core/ + infra/)
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS
- **State**: Zustand stores with typed event bus for cross-store communication
- **Persistence**: SQLite (backend) + localStorage (frontend settings)
- **Image transport**: Binary PNG for frames, base64 only for small masks

## Directory Structure
```
python-backend/          # FastAPI backend
  app/
    api/routes/          # Thin HTTP handlers
    services/            # Orchestration layer
    core/                # Pure computation (no I/O, no logging)
    infra/               # I/O: persistence, ML engines, DICOM
    models/              # Pydantic schemas (domain, requests, responses)
    config/              # Settings
  tests/
    unit/core/           # Core algorithm tests
    integration/api/     # API endpoint tests

src/                     # React frontend
  components/            # UI components by domain
  stores/                # Zustand stores
  lib/api/               # Axios client + TanStack Query
  lib/eventBus.ts        # Cross-store typed events
  hooks/                 # Custom React hooks
  types/                 # Shared TypeScript interfaces

mds/                     # Specification documents
```

## Key Conventions

### Backend
- **core/ is pure**: No I/O, no logging, no file access. Takes data in, returns data out.
- **Session-based state**: All mutable state via SessionStore, injected with FastAPI Depends(). No module-level globals.
- **Error format**: `{"error": {"code": "ERROR_CODE", "message": "...", "details": {}}}`
- **Session header**: `X-Session-ID` required on all endpoints except /health and /dicom/upload
- **Frame responses**: Binary PNG with `Content-Type: image/png`, not base64 JSON

### Frontend
- **Path aliases**: `@/*` maps to `src/*`
- **No getState()**: Cross-store communication via eventBus only
- **Theme**: CSS custom properties + Tailwind `darkMode: 'class'`
- **Disclaimer**: "For Research Use Only" must appear in status bar and all exports

### Testing
- Backend: `cd python-backend && python -m pytest tests/ -v --cov=app`
- Frontend: `npm run test`
- Core algorithms: synthetic data, known-geometry → known-result pattern

### Running
```bash
# Backend
cd python-backend && uvicorn app.main:app --reload --port 8000

# Frontend
npm run dev
```

## Spec Documents (mds/)
- `SPEC_PART1_PRODUCT_UX.md` — Product/UX spec (definitive for frontend)
- `SPEC_PART2_ARCHITECTURE.md` — Architecture spec (definitive for backend + frontend)
- `SPEC_PART3_QUALITY.md` — Quality, testing, security specs
- `CONTEXT.md` — v1.1 feature catalog
- `CONTEXT_ARCHITECTURE.md` — v1.1 architecture deep-dive
