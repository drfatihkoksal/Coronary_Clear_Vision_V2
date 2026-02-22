import pytest
from httpx import AsyncClient

from tests.unit.test_dicom_handler import create_test_dicom


async def _upload_study(client: AsyncClient) -> str:
    """Helper to upload a study and return session_id."""
    data = create_test_dicom(num_frames=5, width=64, height=64)
    resp = await client.post(
        "/dicom/upload", files={"file": ("test.dcm", data, "application/dicom")}
    )
    return resp.json()["session_id"]


@pytest.mark.asyncio
async def test_segment_frame(client: AsyncClient):
    session_id = await _upload_study(client)
    resp = await client.post(
        "/segmentation/segment",
        json={"frame_index": 0, "engine": "nnunet"},
        headers={"X-Session-ID": session_id},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["frame_index"] == 0
    assert "confidence" in body


@pytest.mark.asyncio
async def test_segment_and_extract(client: AsyncClient):
    session_id = await _upload_study(client)
    resp = await client.post(
        "/segmentation/segment-and-extract",
        json={"frame_index": 0, "engine": "nnunet"},
        headers={"X-Session-ID": session_id},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "centerline" in body
    assert isinstance(body["centerline"], list)


@pytest.mark.asyncio
async def test_get_engines(client: AsyncClient):
    session_id = await _upload_study(client)
    resp = await client.get(
        "/segmentation/engines",
        headers={"X-Session-ID": session_id},
    )
    assert resp.status_code == 200
    assert "engines" in resp.json()


@pytest.mark.asyncio
async def test_get_mask_after_segment(client: AsyncClient):
    session_id = await _upload_study(client)
    # Segment first
    await client.post(
        "/segmentation/segment",
        json={"frame_index": 0, "engine": "nnunet"},
        headers={"X-Session-ID": session_id},
    )
    # Get mask
    resp = await client.get(
        "/segmentation/mask/0",
        headers={"X-Session-ID": session_id},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_get_mask_not_segmented(client: AsyncClient):
    session_id = await _upload_study(client)
    resp = await client.get(
        "/segmentation/mask/0",
        headers={"X-Session-ID": session_id},
    )
    assert resp.status_code == 404
