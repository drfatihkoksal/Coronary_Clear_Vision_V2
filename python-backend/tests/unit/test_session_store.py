from app.infra.persistence.session_store import SessionStore


def test_create_session(session_store: SessionStore):
    session = session_store.create()
    assert session.id is not None
    assert session_store.active_count == 1


def test_get_session(session_store: SessionStore):
    session = session_store.create()
    retrieved = session_store.get(session.id)
    assert retrieved is not None
    assert retrieved.id == session.id


def test_get_nonexistent_returns_none(session_store: SessionStore):
    result = session_store.get("nonexistent-id")
    assert result is None


def test_delete_session(session_store: SessionStore):
    session = session_store.create()
    assert session_store.delete(session.id) is True
    assert session_store.active_count == 0
    assert session_store.get(session.id) is None


def test_lru_eviction(session_store: SessionStore):
    # max_sessions is 3 in fixture
    s1 = session_store.create()
    s2 = session_store.create()
    s3 = session_store.create()
    assert session_store.active_count == 3

    # Access s2 so s1 becomes LRU
    session_store.get(s2.id)
    session_store.get(s3.id)

    # Creating a 4th should evict s1 (least recently used)
    s4 = session_store.create()
    assert session_store.active_count == 3
    assert session_store.get(s1.id) is None
    assert session_store.get(s2.id) is not None
    assert session_store.get(s3.id) is not None
    assert session_store.get(s4.id) is not None
