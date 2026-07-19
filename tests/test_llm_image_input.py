from __future__ import annotations

from types import SimpleNamespace

import pytest

from intent_trade.analysis import llm_client


class _StreamResponse:
    def __init__(self, data: bytes, content_length: int | None = None) -> None:
        self.data = data
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)

    def __enter__(self) -> "_StreamResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self):
        yield self.data


def test_image_source_downloads_and_encodes_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jpeg = b"\xff\xd8\xff\xe0image-bytes"
    captured = SimpleNamespace(url="")

    def fake_stream(method: str, url: str, **kwargs: object) -> _StreamResponse:
        captured.url = url
        assert method == "GET"
        assert kwargs["follow_redirects"] is False
        return _StreamResponse(jpeg, len(jpeg))

    monkeypatch.setattr(llm_client.httpx, "stream", fake_stream)
    source = llm_client.image_source_from_url(
        "https://pbs.twimg.com/media/example.jpg"
    )

    assert captured.url == "https://pbs.twimg.com/media/example.jpg"
    assert source == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "/9j/4GltYWdlLWJ5dGVz",
    }


def test_image_source_rejects_untrusted_host() -> None:
    with pytest.raises(ValueError, match="untrusted image URL host"):
        llm_client.image_source_from_url("https://example.com/image.jpg")


def test_image_source_enforces_download_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_client.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse(b"", content_length=11),
    )
    with pytest.raises(ValueError, match="exceeds 10 byte limit"):
        llm_client.image_source_from_url(
            "https://pbs.twimg.com/media/example.jpg", max_bytes=10
        )
