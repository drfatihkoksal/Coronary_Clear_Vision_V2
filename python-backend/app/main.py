import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Resolve relative nnUNet_results path to absolute (must happen before any import)
BACKEND_DIR = Path(__file__).resolve().parent.parent
_nnunet_results = os.environ.get("nnUNet_results")
if _nnunet_results and not os.path.isabs(_nnunet_results):
    abs_path = (BACKEND_DIR / _nnunet_results).resolve()
    if abs_path.exists():
        os.environ["nnUNet_results"] = str(abs_path)

from app.api.middleware.error_handler import add_error_handlers
from app.api.middleware.logging_middleware import LoggingMiddleware
from app.api.routes.calibration import router as calibration_router
from app.api.routes.dicom import router as dicom_router
from app.api.routes.ecg import router as ecg_router
from app.api.routes.export import router as export_router
from app.api.routes.health import router as health_router
from app.api.routes.mask_edit import router as mask_edit_router
from app.api.routes.motion import router as motion_router
from app.api.routes.qca import router as qca_router
from app.api.routes.qfr import router as qfr_router
from app.api.routes.rws import router as rws_router
from app.api.routes.segmentation import router as segmentation_router
from app.api.routes.sessions import router as sessions_router
from app.api.routes.tracking import router as tracking_router
from app.api.routes.auth import router as auth_router
from app.config.settings import settings
from app.infra.persistence.db import Database


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("app").info("Starting %s v%s", settings.app_name, settings.version)

    db = Database(settings.db_path)
    db.connect()
    app.state.db = db

    yield

    db.close()
    logging.getLogger("app").info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Coronary artery RWS and QFR analysis backend",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(LoggingMiddleware)

    add_error_handlers(app)

    app.include_router(health_router)
    app.include_router(dicom_router)
    app.include_router(segmentation_router)
    app.include_router(qca_router)
    app.include_router(calibration_router)
    app.include_router(ecg_router)
    app.include_router(motion_router)
    app.include_router(rws_router)
    app.include_router(qfr_router)
    app.include_router(mask_edit_router)
    app.include_router(export_router)
    app.include_router(sessions_router)
    app.include_router(tracking_router)
    app.include_router(auth_router)

    return app


app = create_app()
