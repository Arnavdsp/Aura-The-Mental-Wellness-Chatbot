"""End-to-end tests through the HTTP surface, on the echo engine."""

from __future__ import annotations

import base64
import json

import pytest


def test_health_reports_engine_and_capabilities(client) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["engine"] == "echo"
    assert payload["engine_ready"] is True
    assert "vision" in payload["capabilities"]


def test_index_serves_the_ui(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Aura" in response.text


def test_chat_round_trip(client) -> None:
    response = client.post("/api/chat", json={"message": "I'm exhausted and can't sleep"})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"].startswith("ses_")
    assert body["reply"]["text"]
    assert body["reply"]["role"] == "assistant"
    assert body["affect"]["label"] == "tired"
    assert len(body["suggestions"]) == 3


def test_chat_requires_content(client) -> None:
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422


def test_session_is_continuous_across_turns(client) -> None:
    first = client.post("/api/chat", json={"message": "Work is crushing me"}).json()
    session_id = first["session_id"]
    client.post("/api/chat", json={"session_id": session_id, "message": "And I can't sleep"})

    detail = client.get(f"/api/sessions/{session_id}").json()
    assert len(detail["turns"]) == 4  # two user turns, two replies
    assert "work" in detail["insights"]["graph"]["topics"]


def test_crisis_message_short_circuits_generation(client) -> None:
    body = client.post("/api/chat", json={"message": "I want to kill myself"}).json()
    assert body["safety"]["risk"] == "crisis"
    assert body["reply"]["meta"]["short_circuited"] is True
    assert "988" in body["reply"]["text"]
    assert body["safety"]["resources"]


def test_streaming_emits_meta_tokens_and_done(client) -> None:
    with client.stream(
        "POST", "/api/chat/stream", json={"message": "I feel anxious about work"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events, payloads = [], []
        current = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
                events.append(current)
            elif line.startswith("data:"):
                payloads.append((current, json.loads(line.split(":", 1)[1].strip())))

    assert events[0] == "meta"
    assert events[-1] == "done"
    assert events.count("token") > 1

    meta = next(data for name, data in payloads if name == "meta")
    assert meta["session_id"].startswith("ses_")
    assert meta["affect"]["label"] == "anxious"

    done = next(data for name, data in payloads if name == "done")
    assert done["reply"]["text"]
    assert done["suggestions"]


def test_streamed_tokens_reconstruct_the_final_reply(client) -> None:
    streamed, final = "", None
    with client.stream("POST", "/api/chat/stream", json={"message": "I'm lonely"}) as response:
        current = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if current == "token":
                    streamed += data["text"]
                elif current == "done":
                    final = data["reply"]["text"]
    assert final and streamed.strip() == final.strip()


def test_image_upload_is_normalised_and_attached(client, png_bytes: bytes) -> None:
    upload = client.post(
        "/api/uploads", files={"file": ("photo.png", png_bytes, "image/png")}
    ).json()
    assert upload["attachment"]["kind"] == "image"

    body = client.post(
        "/api/chat",
        json={"message": "This is my quiet place", "attachment_ids": [upload["attachment_id"]]},
    ).json()
    attachments = body["turn"]["attachments"]
    assert len(attachments) == 1
    assert attachments[0]["media_type"] == "image/png"  # re-encoded, EXIF stripped


def test_inline_image_needs_no_prior_upload(client, png_bytes: bytes) -> None:
    """The serverless path: one request carries both the text and the bytes."""
    body = client.post(
        "/api/chat",
        json={
            "message": "This is my quiet place",
            "attachments": [
                {
                    "kind": "image",
                    "media_type": "image/png",
                    "filename": "photo.png",
                    "data": base64.b64encode(png_bytes).decode(),
                }
            ],
        },
    ).json()
    attachments = body["turn"]["attachments"]
    assert len(attachments) == 1
    assert attachments[0]["media_type"] == "image/png"
    # Nothing was staged server-side, so there is no id to dereference later.
    assert attachments[0]["uri"] is None


def test_inline_attachment_accepts_a_data_url(client, png_bytes: bytes) -> None:
    encoded = base64.b64encode(png_bytes).decode()
    response = client.post(
        "/api/chat",
        json={
            "attachments": [
                {"kind": "image", "media_type": "image/png", "data": f"data:image/png;base64,{encoded}"}
            ]
        },
    )
    assert response.status_code == 200


def test_inline_attachment_alone_is_enough_content(client, png_bytes: bytes) -> None:
    response = client.post(
        "/api/chat",
        json={
            "attachments": [
                {"kind": "image", "media_type": "image/png", "data": base64.b64encode(png_bytes).decode()}
            ]
        },
    )
    assert response.status_code == 200


def test_inline_attachment_rejects_junk_base64(client) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "hi", "attachments": [{"kind": "image", "data": "not base64!"}]},
    )
    assert response.status_code == 422


def test_inline_attachment_enforces_the_size_limit(client, settings) -> None:
    oversized = base64.b64encode(b"\0" * (settings.max_upload_bytes + 1)).decode()
    response = client.post(
        "/api/chat",
        json={"attachments": [{"kind": "image", "media_type": "image/png", "data": oversized}]},
    )
    assert response.status_code == 413


def test_attachments_are_consumed_once(client, png_bytes: bytes) -> None:
    upload = client.post(
        "/api/uploads", files={"file": ("photo.png", png_bytes, "image/png")}
    ).json()
    attachment_id = upload["attachment_id"]
    first = client.post("/api/chat", json={"message": "a", "attachment_ids": [attachment_id]})
    second = client.post("/api/chat", json={"message": "b", "attachment_ids": [attachment_id]})
    assert len(first.json()["turn"]["attachments"]) == 1
    assert second.json()["turn"]["attachments"] == []


def test_upload_rejects_unsupported_and_empty_files(client) -> None:
    assert client.post(
        "/api/uploads", files={"file": ("notes.txt", b"hello", "text/plain")}
    ).status_code == 415
    assert client.post(
        "/api/uploads", files={"file": ("empty.png", b"", "image/png")}
    ).status_code == 422


def test_upload_enforces_the_size_limit(client, settings) -> None:
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * (settings.max_upload_bytes + 1)
    response = client.post(
        "/api/uploads", files={"file": ("big.png", oversized, "image/png")}
    )
    assert response.status_code == 413


def test_corrupt_image_is_reported_not_crashed(client) -> None:
    upload = client.post(
        "/api/uploads", files={"file": ("fake.png", b"not really a png", "image/png")}
    ).json()
    body = client.post(
        "/api/chat", json={"message": "look", "attachment_ids": [upload["attachment_id"]]}
    )
    assert body.status_code == 200
    assert body.json()["turn"]["attachments"] == []


def test_audio_upload_without_an_asr_backend_degrades_gracefully(client, wav_bytes: bytes) -> None:
    """The echo engine can't transcribe; the turn must still succeed."""
    upload = client.post(
        "/api/uploads", files={"file": ("voice.wav", wav_bytes, "audio/wav")}
    ).json()
    assert upload["attachment"]["kind"] == "audio"
    response = client.post(
        "/api/chat", json={"message": "", "attachment_ids": [upload["attachment_id"]]}
    )
    assert response.status_code == 200
    assert response.json()["reply"]["text"]


def test_session_deletion_removes_everything(client) -> None:
    session_id = client.post("/api/chat", json={"message": "hello there"}).json()["session_id"]
    assert client.delete(f"/api/sessions/{session_id}").json() == {"deleted": True}
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_missing_session_and_attachment_return_404(client) -> None:
    assert client.get("/api/sessions/ses_nope").status_code == 404
    assert client.get("/api/attachments/att_nope").status_code == 404
    assert client.get("/api/audio/aud_nope").status_code == 404


@pytest.mark.parametrize(("region", "expected"), [("US", "988"), ("IN", "14416")])
def test_resources_endpoint_is_region_aware(client, region: str, expected: str) -> None:
    resources = client.get(f"/api/resources?region={region}").json()
    assert any(expected in item["contact"] for item in resources)


def test_request_id_header_is_echoed(client) -> None:
    response = client.get("/api/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"
    assert "x-response-time-ms" in response.headers
