<p align="center">
  <h1 align="center">🫀 Coronary Clear Vision V2</h1>
  <p align="center">
    <strong>AI-Powered Coronary Artery Analysis Platform</strong>
  </p>
  <p align="center">
    Quantitative Flow Ratio (QFR) · Radial Wall Strain (RWS) · Quantitative Coronary Analysis (QCA)
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React">
    <img src="https://img.shields.io/badge/TypeScript-5.5-3178C6?logo=typescript&logoColor=white" alt="TypeScript">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  </p>
  <p align="center">
    <a href="https://coronaryanalyser.com"><strong>🌐 Live Demo → coronaryanalyser.com</strong></a>
  </p>
</p>

---

> [!CAUTION]
> **For Research Use Only.** This software is a research tool and has **NOT** been validated or approved for clinical decision-making. It must not be used for patient diagnosis or treatment planning.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Docker Deployment](#docker-deployment)
- [Core Algorithms](#core-algorithms)
- [API Reference](#api-reference)
- [Frontend Panels](#frontend-panels)
- [Testing](#testing)
- [Environment Variables](#environment-variables)

---

## Overview

**Coronary Clear Vision V2** is a web-based medical imaging platform designed for the quantitative analysis of coronary arteries from standard angiographic DICOM sequences. It leverages deep learning–based vessel segmentation, computational fluid dynamics–inspired flow estimation, and biomechanical wall motion analysis to provide researchers with non-invasive functional and morphological assessments of coronary arteries.

The platform processes conventional coronary angiography DICOM files and delivers:

- **Wire-free QFR** — Pressure-drop estimation across stenotic lesions without requiring a physical pressure wire
- **Radial Wall Strain (RWS)** — Frame-by-frame vessel wall deformation tracking throughout the cardiac cycle
- **Quantitative Coronary Analysis (QCA)** — Automated lumen diameter, area stenosis, and reference vessel measurements
- **3D Stereo Reconstruction** — Biplane-calibrated 3D vessel geometry from two angiographic projections
- **ECG-Gated Analysis** — Automated cardiac phase detection from embedded or overlaid ECG traces

---

## Key Features

### 🧠 AI-Powered Segmentation
- **nnU-Net v2** integration for state-of-the-art coronary vessel segmentation
- **AngioPy** — Custom neural network engine for angiographic image analysis
- **SeedModel** — Seed-point based segmentation refinement architecture
- Interactive mask editing tools for manual correction

### 📐 Quantitative Flow Ratio (QFR)
- Dual-view angiographic projection support
- Automated centerline extraction with bifurcation detection
- Stereoscopic 3D vessel reconstruction from biplane views
- Computational flow simulation using fluid dynamics principles
- Pressure drop estimation along the vessel length
- Interactive 3D mesh visualization (Three.js / React Three Fiber)

### 📊 Radial Wall Strain (RWS)
- Frame-by-frame vessel wall motion tracking
- Radial displacement quantification at user-defined cross-sections
- Cardiac cycle–synchronized strain curves
- Outlier filtering for robust measurements

### 🔬 Quantitative Coronary Analysis (QCA)
- Automated lumen diameter measurement
- Reference vessel diameter detection
- Percentage diameter and area stenosis calculations
- Lesion length measurement

### ❤️ ECG Analysis
- Automatic ECG curve extraction from DICOM overlays
- R-peak detection and cardiac phase identification
- End-diastolic frame selection for optimal analysis
- Siemens curved ECG display filter support

### 🎬 DICOM Viewer
- Full DICOM series playback with frame-rate control
- Zoom, pan, and window/level adjustments
- Frame-by-frame navigation
- Annotation and measurement overlay tools

### 🔐 Authentication & Security
- JWT-based user authentication
- User registration with email verification (via Resend)
- Password reset flow
- Session-based state management

### 🚀 Deployment
- Docker Compose production stack
- Cloudflare Tunnel integration for secure remote access
- Nginx reverse proxy with optimized configuration
- Health check monitoring

---

## Architecture

The application follows a clean **three-layer architecture** with strict separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend (React + TS)                  │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ │
│  │  Panels  │ │   Viewer   │ │  Charts  │ │  Stores  │ │
│  │ (QFR,RWS │ │ (Dual-view │ │(Recharts)│ │(Zustand) │ │
│  │ QCA,ECG) │ │  3D Mesh)  │ │          │ │          │ │
│  └──────────┘ └────────────┘ └──────────┘ └──────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │ REST API (JSON + Binary PNG)
┌────────────────────────┴────────────────────────────────┐
│                Backend (FastAPI + Python)                 │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  API Layer (routes/)       Thin HTTP handlers       │ │
│  ├─────────────────────────────────────────────────────┤ │
│  │  Service Layer (services/) Orchestration & workflow  │ │
│  ├─────────────────────────────────────────────────────┤ │
│  │  Core Layer (core/)        Pure computation (no I/O)│ │
│  ├─────────────────────────────────────────────────────┤ │
│  │  Infra Layer (infra/)      I/O, ML, DICOM, DB      │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    │
│  │  SQLite   │  │ ML Models │  │  DICOM Processing │    │
│  │ (Sessions)│  │(nnU-Net,  │  │  (pydicom, GDCM)  │    │
│  │           │  │ AngioPy)  │  │                   │    │
│  └───────────┘  └───────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Pure core** | `core/` contains zero I/O — no file access, no logging, no network calls. Takes data in, returns data out. |
| **Session isolation** | All mutable state flows through `SessionStore`, injected via FastAPI `Depends()`. No module-level globals. |
| **Binary transport** | Frame images served as binary PNG (`Content-Type: image/png`), not base64 JSON, for performance. |
| **Cross-store events** | Frontend stores communicate via a typed `eventBus`, not direct `getState()` calls. |

---

## Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| Framework | FastAPI ≥ 0.109 |
| Runtime | Python 3.11+ / Uvicorn |
| DICOM | pydicom + python-gdcm |
| Image Processing | OpenCV, scikit-image, Pillow |
| Scientific Computing | NumPy, SciPy |
| Deep Learning | PyTorch ≥ 2.0, nnU-Net v2, segmentation-models-pytorch, timm |
| Database | SQLite |
| Auth | python-jose (JWT), passlib + bcrypt |
| Email | Resend SDK |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | React 18 + TypeScript |
| Build Tool | Vite 5 |
| Styling | Tailwind CSS 3.4 |
| State Management | Zustand 5 |
| Data Fetching | TanStack React Query 5 |
| HTTP Client | Axios |
| 3D Visualization | Three.js + React Three Fiber + Drei |
| Charts | Recharts |
| Icons | Lucide React |
| Routing | React Router v7 |

### Infrastructure
| Component | Technology |
|-----------|-----------|
| Containerization | Docker + Docker Compose |
| Reverse Proxy | Nginx |
| Secure Tunnel | Cloudflare Tunnel (cloudflared) |

---

## Project Structure

```
coronary_rws_analyserV2/
│
├── python-backend/                 # FastAPI backend application
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/             # HTTP endpoint handlers
│   │   │   │   ├── auth.py         # Registration, login, email verification
│   │   │   │   ├── dicom.py        # DICOM upload & frame serving
│   │   │   │   ├── segmentation.py # Vessel segmentation endpoints
│   │   │   │   ├── qfr.py          # QFR analysis pipeline
│   │   │   │   ├── rws.py          # RWS computation endpoints
│   │   │   │   ├── qca.py          # QCA measurement endpoints
│   │   │   │   ├── ecg.py          # ECG extraction & analysis
│   │   │   │   ├── calibration.py  # Pixel-to-mm calibration
│   │   │   │   ├── tracking.py     # Vessel tracking across frames
│   │   │   │   ├── mask_edit.py    # Interactive mask editing
│   │   │   │   ├── motion.py       # Motion analysis
│   │   │   │   └── export.py       # Data export
│   │   │   ├── middleware/         # CORS, error handling
│   │   │   └── dependencies.py    # FastAPI dependency injection
│   │   │
│   │   ├── core/                   # Pure computational algorithms
│   │   │   ├── qfr_calculator.py   # QFR pressure-drop computation
│   │   │   ├── rws_calculator.py   # Radial wall strain computation
│   │   │   ├── qca_engine.py       # Quantitative coronary analysis
│   │   │   ├── centerline_extractor.py  # Vessel centerline extraction
│   │   │   ├── stereo_reconstructor.py  # 3D stereo reconstruction
│   │   │   ├── vessel_mesher.py    # 3D vessel mesh generation
│   │   │   ├── tracking_engine.py  # Frame-to-frame vessel tracking
│   │   │   ├── ecg_analyzer.py     # ECG signal processing
│   │   │   ├── calibration.py      # Spatial calibration
│   │   │   ├── mask_ops.py         # Segmentation mask operations
│   │   │   ├── motion_analyzer.py  # Motion quantification
│   │   │   ├── outlier_filter.py   # Statistical outlier removal
│   │   │   ├── siemens_ecg_filter.py  # Siemens ECG display filter
│   │   │   ├── auth.py             # Auth utilities
│   │   │   └── email_service.py    # Email templates
│   │   │
│   │   ├── infra/                  # I/O and external integrations
│   │   │   ├── dicom_handler.py    # DICOM file parsing & frame extraction
│   │   │   ├── ecg_parser.py       # ECG trace extraction from DICOM
│   │   │   ├── ml_engines/         # Machine learning model engines
│   │   │   │   ├── nnunet.py       # nnU-Net v2 inference wrapper
│   │   │   │   ├── angiopy.py      # Custom angiography CNN
│   │   │   │   ├── seedmodel.py    # Seed-point segmentation model
│   │   │   │   └── registry.py     # Model registry & selection
│   │   │   └── persistence/        # Database & storage
│   │   │
│   │   ├── models/                 # Pydantic schemas
│   │   ├── services/               # Business logic orchestration
│   │   └── config/                 # Application settings
│   │
│   ├── tests/                      # Test suite
│   │   ├── unit/core/              # Core algorithm unit tests
│   │   └── integration/api/        # API integration tests
│   │
│   ├── Dockerfile                  # Backend container definition
│   ├── requirements.txt            # Python dependencies
│   └── pyproject.toml              # Project metadata
│
├── src/                            # React frontend application
│   ├── components/
│   │   ├── panels/                 # Analysis tool panels
│   │   │   ├── QFRPanel.tsx        # QFR analysis controls
│   │   │   ├── RWSPanel.tsx        # RWS analysis controls
│   │   │   ├── QCAPanel.tsx        # QCA measurement controls
│   │   │   ├── ECGPanel.tsx        # ECG display panel
│   │   │   ├── SegmentationPanel.tsx  # Segmentation controls
│   │   │   ├── TrackingPanel.tsx   # Vessel tracking panel
│   │   │   ├── CalibrationPanel.tsx  # Calibration tools
│   │   │   └── ExportPanel.tsx     # Data export panel
│   │   ├── viewer/                 # Image viewer components
│   │   │   ├── QFRDualViewer.tsx   # Side-by-side dual-projection viewer
│   │   │   ├── QFRMesh3DViewer.tsx # Interactive 3D vessel mesh viewer
│   │   │   ├── ViewerContainer.tsx # Main viewer wrapper
│   │   │   ├── AnnotationLayer.tsx # Drawing & measurement overlays
│   │   │   ├── OverlayLayer.tsx    # Segmentation & info overlays
│   │   │   ├── SegmentationLayer.tsx  # Mask visualization
│   │   │   └── VideoLayer.tsx      # DICOM frame playback
│   │   ├── charts/                 # Data visualization
│   │   ├── common/                 # Shared UI components
│   │   ├── controls/               # Playback & interaction controls
│   │   └── layout/                 # Page layout components
│   │
│   ├── stores/                     # Zustand state management
│   │   ├── qfrStore.ts             # QFR analysis state
│   │   ├── rwsStore.ts             # RWS analysis state
│   │   ├── analysisStore.ts        # General analysis state
│   │   ├── trackingStore.ts        # Vessel tracking state
│   │   ├── playerStore.ts          # Video playback state
│   │   ├── timingStore.ts          # ECG timing state
│   │   ├── maskEditStore.ts        # Mask editing state
│   │   ├── sessionStore.ts         # Session management
│   │   ├── studyStore.ts           # DICOM study state
│   │   ├── settingsStore.ts        # User preferences
│   │   └── toolStore.ts            # Active tool state
│   │
│   ├── lib/
│   │   ├── api/                    # API client & React Query hooks
│   │   ├── AuthContext.tsx         # Authentication context provider
│   │   ├── AuthGate.tsx            # Route protection
│   │   ├── eventBus.ts             # Typed cross-store event bus
│   │   └── canvas/                 # Canvas rendering utilities
│   │
│   ├── pages/                      # Route pages
│   │   ├── LandingPage.tsx         # Marketing / home page
│   │   ├── LoginPage.tsx           # User login
│   │   ├── RegisterPage.tsx        # User registration
│   │   └── ...                     # Password reset, email verification, etc.
│   │
│   ├── hooks/                      # Custom React hooks
│   └── types/                      # TypeScript type definitions
│
├── docker-compose.prod.yml         # Production Docker stack
├── docker-compose.dev.yml          # Development Docker stack
├── Dockerfile.frontend             # Frontend container (Nginx)
├── nginx.conf                      # Nginx reverse proxy config
├── deploy.sh                       # One-command deployment script
├── .env.example                    # Environment variable template
├── package.json                    # Frontend dependencies
├── vite.config.ts                  # Vite build configuration
├── tailwind.config.js              # Tailwind CSS configuration
├── tsconfig.json                   # TypeScript configuration
└── mds/                            # Specification documents
```

---

## Getting Started

### Prerequisites

- **Python** 3.11+
- **Node.js** 18+ and npm
- **Docker** & **Docker Compose** (for containerized deployment)
- **CUDA-capable GPU** (optional, recommended for ML inference)

### Local Development

#### 1. Clone the repository

```bash
git clone -b qfr https://github.com/drfatihkoksal/Coronary_Clear_Vision_V2.git
cd Coronary_Clear_Vision_V2
```

#### 2. Set up the backend

```bash
cd python-backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (choose one):
# GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Start the backend server
uvicorn app.main:app --reload --port 8000
```

#### 3. Set up the frontend

```bash
# From the project root
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173` and will proxy API requests to the backend at `http://localhost:8000`.

### Docker Deployment

#### 1. Configure environment

```bash
cp .env.example .env
# Edit .env with your configuration:
#   - JWT_SECRET (generate with: openssl rand -hex 32)
#   - TUNNEL_TOKEN (from Cloudflare dashboard, optional)
#   - RESEND_API_KEY (for email verification)
```

#### 2. Deploy with Docker Compose

```bash
# One-command deployment
./deploy.sh

# Or manually:
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

#### 3. Access the application

- **Local**: http://localhost
- **Remote**: Via your configured Cloudflare Tunnel hostname

#### Management commands

```bash
# View logs
docker compose -f docker-compose.prod.yml logs -f

# Stop services
docker compose -f docker-compose.prod.yml down

# Check health
curl http://localhost/api/health
```

---

## Core Algorithms

### Quantitative Flow Ratio (QFR)

The QFR pipeline estimates the pressure drop across coronary stenoses without a physical pressure wire:

```
DICOM Input → Segmentation → Centerline Extraction → Calibration
                                     ↓
              3D Reconstruction ← Dual-View Matching
                    ↓
            Vessel Meshing → Flow Simulation → QFR Value
```

1. **Vessel Segmentation** — nnU-Net or AngioPy segments the coronary lumen from each frame
2. **Centerline Extraction** — Skeletonization + graph-based extraction with bifurcation handling
3. **Spatial Calibration** — Known catheter diameter or pixel spacing from DICOM metadata
4. **Stereo Reconstruction** — Epipolar geometry–based 3D reconstruction from two angiographic projections taken ≥25° apart
5. **Vessel Meshing** — 3D tubular mesh generation from reconstructed centerlines and lumen boundaries
6. **Flow Computation** — Thrombolysis in Myocardial Infarction (TIMI) frame count–based flow estimation combined with vessel geometry for pressure-drop calculation

### Radial Wall Strain (RWS)

RWS quantifies vessel wall deformation through the cardiac cycle:

1. **Contour Tracking** — Segmentation masks tracked across consecutive frames
2. **Radial Displacement** — Wall boundary displacement measured perpendicular to the centerline at defined cross-sections
3. **Strain Calculation** — Displacement normalized to reference diameter, producing strain curves
4. **ECG Gating** — Results synchronized with cardiac phase for physiological interpretation

### Quantitative Coronary Analysis (QCA)

Automated morphological measurements:

- Reference vessel diameter (proximal and distal)
- Minimum lumen diameter (MLD) at stenosis
- Percentage diameter stenosis (%DS)
- Percentage area stenosis (%AS)
- Lesion length

### ECG Analysis

- Automatic detection of ECG traces from DICOM pixel data or overlays
- R-peak detection using signal processing (peak finding + amplitude thresholding)
- Cardiac cycle segmentation (systole / diastole)
- End-diastolic frame identification for optimal QFR analysis
- Special filtering for Siemens curved ECG display format

---

## API Reference

All API endpoints require the `X-Session-ID` header (except `/health` and `/auth/*`).

| Endpoint Group | Base Path | Description |
|---------------|-----------|-------------|
| **Health** | `GET /health` | Service health check |
| **Auth** | `/auth/*` | Registration, login, email verification, password reset |
| **DICOM** | `/dicom/*` | Upload DICOM files, retrieve frames (binary PNG) |
| **Segmentation** | `/segmentation/*` | Run AI segmentation, retrieve masks |
| **QFR** | `/qfr/*` | Full QFR analysis pipeline |
| **RWS** | `/rws/*` | Radial wall strain computation |
| **QCA** | `/qca/*` | Quantitative coronary analysis |
| **ECG** | `/ecg/*` | ECG extraction & cardiac phase detection |
| **Calibration** | `/calibration/*` | Pixel-to-mm spatial calibration |
| **Tracking** | `/tracking/*` | Vessel tracking across frames |
| **Mask Edit** | `/mask-edit/*` | Interactive segmentation mask editing |
| **Motion** | `/motion/*` | Motion analysis |
| **Export** | `/export/*` | Data export (JSON, CSV) |
| **Sessions** | `/sessions/*` | Session management |

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error description",
    "details": {}
  }
}
```

---

## Frontend Panels

| Panel | Description |
|-------|-------------|
| **Segmentation** | Run AI segmentation, visualize masks, toggle overlay visibility |
| **QFR Analysis** | Dual-view setup, centerline definition, 3D reconstruction, flow results |
| **RWS Analysis** | Cross-section placement, strain curve visualization, cardiac cycle analysis |
| **QCA** | Lumen measurements, stenosis grading, reference vessel selection |
| **ECG** | ECG waveform display, R-peak markers, phase selection |
| **Tracking** | Frame-to-frame vessel tracking, contour propagation |
| **Calibration** | Known-distance calibration, DICOM metadata–based calibration |
| **Export** | Download analysis results, screenshots, measurements |

### 3D Visualization

The **QFR Mesh 3D Viewer** provides an interactive Three.js-powered visualization of the reconstructed vessel geometry, rendered with React Three Fiber and Drei controls for orbit, zoom, and inspection.

---

## Testing

### Backend Tests

```bash
cd python-backend

# Run all tests with coverage
python -m pytest tests/ -v --cov=app

# Run only unit tests (core algorithms)
python -m pytest tests/unit/core/ -v

# Run integration tests (API endpoints)
python -m pytest tests/integration/api/ -v
```

### Frontend Tests

```bash
# Run all tests
npm run test

# Watch mode
npm run test:watch

# Coverage report
npm run test:coverage
```

### Testing Philosophy

- **Core algorithm tests** use synthetic data with known geometry → known result patterns
- **API tests** verify endpoint contracts, status codes, and response schemas
- **Frontend tests** use React Testing Library + Vitest for component behavior

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `JWT_SECRET` | Secret key for JWT token signing | ✅ |
| `TUNNEL_TOKEN` | Cloudflare Tunnel token for remote access | ❌ |
| `TUNNEL_HOSTNAME` | Domain name for the tunnel | ❌ |
| `RESEND_API_KEY` | Resend API key for email service | ❌ |
| `EMAIL_FROM` | Sender email address | ❌ |
| `APP_URL` | Application URL for email links | ❌ |
| `CRA_DEBUG` | Enable debug mode (`true`/`false`) | ❌ |
| `CRA_DEVICE` | PyTorch device (`cpu`/`cuda`) | ❌ |
| `CRA_LOG_LEVEL` | Log level (`DEBUG`/`INFO`/`WARNING`) | ❌ |
| `CRA_PORT` | Backend port (default: `8000`) | ❌ |
| `CRA_DB_PATH` | SQLite database path | ❌ |

Generate a strong JWT secret:
```bash
openssl rand -hex 32
```

---

## ML Models

The following pre-trained models should be placed in the `models/` directory:

| Model | Framework | Purpose |
|-------|-----------|---------|
| **nnU-Net** | nnU-Net v2 | Primary coronary vessel segmentation |
| **AngioPy** | segmentation-models-pytorch | Angiographic image analysis |
| **SeedModel** | Custom architecture (timm) | Seed-point based segmentation |

> [!NOTE]
> Model weights are not included in the repository due to their large file sizes. Contact the repository maintainers for access to pre-trained model weights.

---

## Development Notes

### Backend Conventions
- `core/` is **pure** — no I/O, no logging, no external dependencies
- All state is managed through `SessionStore` via dependency injection
- Frame responses are served as binary PNG for minimal latency
- `X-Session-ID` header required on all analysis endpoints

### Frontend Conventions
- Path alias `@/*` maps to `src/*`
- Cross-store communication uses `eventBus` exclusively (no `getState()` across stores)
- Theme via CSS custom properties + Tailwind `darkMode: 'class'`
- "For Research Use Only" disclaimer must appear in the status bar and all exports

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

<p align="center">
  <sub>Built with ❤️ for coronary artery research</sub>
</p>
