from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from app.api.dependencies import get_session
from app.infra.persistence.session_store import Session
from app.services.export_service import export_csv, export_json

router = APIRouter(prefix="/export", tags=["Export"])


@router.post("/csv")
async def export_as_csv(
    session: Session = Depends(get_session),
    anonymize: bool = Query(True),
):
    content = export_csv(session.id, session.qca_measurements, session.rws_results, anonymize)
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=rws_analysis_{session.id[:8]}.csv"},
    )


@router.post("/json")
async def export_as_json(
    session: Session = Depends(get_session),
    anonymize: bool = Query(True),
):
    content = export_json(session.id, session.qca_measurements, session.rws_results, anonymize)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=rws_analysis_{session.id[:8]}.json"},
    )
