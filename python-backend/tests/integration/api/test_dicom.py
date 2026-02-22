import pytest
from httpx import AsyncClient

from tests.unit.test_dicom_handler import create_test_dicom


@pytest.mark.asyncio
async def test_upload_dicom(client: AsyncClient):
    data = create_test_dicom(num_frames=5, width=32, height=32)
    response = await client.post(
        "/dicom/upload",
        files={"file": ("test.dcm", data, "application/dicom")},
        params={"anonymize": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "session_id" in body
    assert body["num_frames"] == 5
    assert body["image_width"] == 32
    assert body["image_height"] == 32


@pytest.mark.asyncio
async def test_get_frame(client: AsyncClient):
    # Upload first
    data = create_test_dicom(num_frames=5)
    upload = await client.post(
        "/dicom/upload",
        files={"file": ("test.dcm", data, "application/dicom")},
    )
    session_id = upload.json()["session_id"]

    # Get frame
    response = await client.get(
        "/dicom/frame/0",
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:4] == b"\x89PNG"


@pytest.mark.asyncio
async def test_get_frame_out_of_range(client: AsyncClient):
    data = create_test_dicom(num_frames=5)
    upload = await client.post(
        "/dicom/upload",
        files={"file": ("test.dcm", data, "application/dicom")},
    )
    session_id = upload.json()["session_id"]

    response = await client.get(
        "/dicom/frame/999",
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_metadata(client: AsyncClient):
    data = create_test_dicom()
    upload = await client.post(
        "/dicom/upload",
        files={"file": ("test.dcm", data, "application/dicom")},
    )
    session_id = upload.json()["session_id"]

    response = await client.get(
        "/dicom/metadata",
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 200
    assert response.json()["num_frames"] == 10


@pytest.mark.asyncio
async def test_clear_study(client: AsyncClient):
    data = create_test_dicom()
    upload = await client.post(
        "/dicom/upload",
        files={"file": ("test.dcm", data, "application/dicom")},
    )
    session_id = upload.json()["session_id"]

    response = await client.post(
        "/dicom/clear",
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_num_frames(client: AsyncClient):
    data = create_test_dicom(num_frames=7)
    upload = await client.post(
        "/dicom/upload",
        files={"file": ("test.dcm", data, "application/dicom")},
    )
    session_id = upload.json()["session_id"]

    response = await client.get(
        "/dicom/num-frames",
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 200
    assert response.json()["num_frames"] == 7


@pytest.mark.asyncio
async def test_upload_without_anonymize(client: AsyncClient):
    data = create_test_dicom(num_frames=3, width=16, height=16)
    response = await client.post(
        "/dicom/upload",
        files={"file": ("test.dcm", data, "application/dicom")},
        params={"anonymize": "false"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["patient"]["name"] == "TEST^PATIENT"
    assert body["patient"]["patient_id"] == "12345"
