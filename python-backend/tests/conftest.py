import pytest
from httpx import ASGITransport, AsyncClient

from app.infra.persistence.session_store import Session, SessionStore
from app.main import app


@pytest.fixture
def session_store() -> SessionStore:
    return SessionStore(max_sessions=3)


@pytest.fixture
def sample_session(session_store: SessionStore) -> Session:
    return session_store.create()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Trigger lifespan manually since ASGITransport doesn't do it
        import time

        app.state.start_time = time.time()
        yield ac
