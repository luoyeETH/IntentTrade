from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from types import SimpleNamespace

from intent_trade.models.domain import SocialPost
from intent_trade.pipeline.runner import Pipeline
from intent_trade.storage.db import Storage


def _post(post_id: str, text: str, *, minutes: int = 0) -> SocialPost:
    return SocialPost(
        id=post_id,
        author_username="history_kol",
        author_display_name="History KOL",
        text=text,
        created_at=datetime(2026, 7, 19, 5, 0) + timedelta(minutes=minutes),
        url=f"https://x.com/history_kol/status/{post_id}",
        media_urls=[f"https://img.example/{post_id}.jpg"],
        media_alt_texts=[f"alt for {post_id}"],
        media_transcripts=[f"transcript for {post_id}"],
        raw={"id": post_id, "text": text, "version": "first-seen"},
        fetched_at=datetime(2026, 7, 19, 5, 10),
    )


def test_first_fetched_post_snapshot_is_never_overwritten(tmp_path) -> None:
    storage = Storage(tmp_path / "history.db")
    original = _post("100", "original text")
    later_response = SocialPost(
        id="100",
        author_username="history_kol",
        text="changed or incomplete text",
        created_at=datetime(2026, 7, 19, 5, 0),
        media_urls=[],
        raw={"id": "100", "version": "later-response"},
        fetched_at=datetime(2026, 7, 19, 6, 0),
    )

    assert storage.insert_post(original) is True
    assert storage.insert_post(later_response) is False

    archived = storage.get_post("100")
    assert archived is not None
    assert archived.text == "original text"
    assert archived.media_urls == original.media_urls
    assert archived.media_alt_texts == original.media_alt_texts
    assert archived.media_transcripts == original.media_transcripts
    assert archived.raw == original.raw
    assert archived.fetched_at == original.fetched_at
    assert storage.counts()["posts"] == 1


def test_existing_post_archive_is_migrated_without_rewriting_rows(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE social_posts (
            id VARCHAR(32) PRIMARY KEY,
            platform VARCHAR(32),
            author_username VARCHAR(128),
            author_display_name VARCHAR(256),
            text TEXT,
            created_at DATETIME,
            url VARCHAR(512),
            media_urls_json TEXT,
            media_transcripts_json TEXT,
            raw_json TEXT,
            fetched_at DATETIME
        )
        """
    )
    connection.execute(
        """
        INSERT INTO social_posts VALUES (
            'legacy-1', 'twitter', 'history_kol', 'History KOL',
            'legacy text', '2026-07-19 05:00:00',
            'https://x.com/history_kol/status/legacy-1',
            '["https://img.example/legacy.jpg"]', '[]',
            '{"id":"legacy-1"}', '2026-07-19 05:10:00'
        )
        """
    )
    connection.commit()
    connection.close()

    storage = Storage(db_path)

    archived = storage.get_post("legacy-1")
    assert archived is not None
    assert archived.text == "legacy text"
    assert archived.media_urls == ["https://img.example/legacy.jpg"]
    assert archived.media_alt_texts == []
    assert archived.raw == {"id": "legacy-1"}
    assert storage.counts()["posts"] == 1


def test_later_timeline_does_not_prune_deleted_or_replace_existing_posts(
    tmp_path,
) -> None:
    deleted_later = _post("100", "keep after source deletion")
    still_visible = _post("200", "first visible snapshot", minutes=1)
    changed_visible = _post("200", "later changed snapshot", minutes=1)

    class BatchFeed:
        def __init__(self) -> None:
            self.batches = [
                [deleted_later, still_visible],
                [changed_visible],
            ]

        def fetch_kols(self, usernames, limit_per_user):
            return self.batches.pop(0)

    pipe = Pipeline.__new__(Pipeline)
    pipe.settings = SimpleNamespace(
        kols=[SimpleNamespace(username="history_kol", enabled=True)],
        twitter=SimpleNamespace(max_posts_per_kol=20),
    )
    pipe.storage = Storage(tmp_path / "timeline.db")
    pipe.feed = BatchFeed()
    pipe.feed_error = ""

    assert [post.id for post in pipe.ingest()] == ["100", "200"]
    assert pipe.ingest() == []

    assert pipe.storage.get_post("100").text == "keep after source deletion"
    assert pipe.storage.get_post("200").text == "first visible snapshot"
    assert pipe.storage.counts()["posts"] == 2
