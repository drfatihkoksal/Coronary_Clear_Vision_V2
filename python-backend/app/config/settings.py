from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Coronary RWS Analyser"
    version: str = "2.0.0"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:1420",
        "http://127.0.0.1:1420",
    ])
    device: str = "auto"  # auto, cpu, cuda, mps
    ml_models_path: Path = Path("./models")
    data_dir: Path = Path("./data")
    dicom_dir: Path = Path("./dicom")
    log_level: str = "INFO"
    max_upload_size_mb: int = 500
    session_timeout_minutes: int = 120
    max_sessions: int = 5
    db_path: Path = Path.home() / ".coronary-rws" / "sessions.db"

    model_config = {"env_prefix": "CRA_", "env_file": ".env"}


settings = Settings()
