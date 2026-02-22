from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_session_store
from app.infra.persistence.session_store import SessionStore

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.get("/")
async def list_sessions(store: SessionStore = Depends(get_session_store)):
    sessions = []
    for sid, s in store._sessions.items():
        sessions.append({
            "id": s.id,
            "has_study": s.study is not None,
            "num_segmentations": len(s.segmentations),
            "num_qca": len(s.qca_measurements),
            "num_rws": len(s.rws_results),
            "created_at": s.created_at.isoformat(),
            "last_accessed": s.last_accessed.isoformat(),
        })
    return {"sessions": sessions}


@router.delete("/{session_id}")
async def delete_session(session_id: str, store: SessionStore = Depends(get_session_store)):
    deleted = store.delete(session_id)
    if not deleted:
        raise HTTPException(404, detail={"code": "SESSION_NOT_FOUND"})
    return {"status": "deleted"}
