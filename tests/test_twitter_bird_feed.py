from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from intent_trade.config import Settings, TwitterConfig
from intent_trade.twitter.client import BirdSocialFeed, create_social_feed


def _feed(monkeypatch: pytest.MonkeyPatch) -> BirdSocialFeed:
    monkeypatch.setattr("intent_trade.twitter.client.shutil.which", lambda _: "/bin/bird")
    return BirdSocialFeed("auth-secret", "csrf-secret", command="bird")


def test_bird_feed_parses_json_and_keeps_secrets_out_of_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feed = _feed(monkeypatch)
    monkeypatch.setenv("NODE_CHANNEL_FD", "3")
    monkeypatch.setenv("NODE_CHANNEL_SERIALIZATION_MODE", "json")
    payload = {
        "tweets": [
            {
                "id": "200",
                "text": "new post",
                "createdAt": "Sat Jul 19 02:00:00 +0000 2026",
                "author": {"username": "xtony1314", "name": "XTony"},
                "media": [{"type": "photo", "url": "https://img/one.jpg"}],
            },
            {
                "id": "100",
                "text": "old post",
                "createdAt": "Sat Jul 19 01:00:00 +0000 2026",
                "author": {"username": "xtony1314", "name": "XTony"},
                "quotedTweet": {
                    "media": [
                        {"type": "photo", "url": "https://img/quoted.png"}
                    ]
                },
            },
        ],
        "nextCursor": "ignored",
    }
    captured: dict = {}

    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("intent_trade.twitter.client.subprocess.run", fake_run)
    posts = feed.fetch_user_posts("@xtony1314", limit=2)

    assert [post.id for post in posts] == ["200", "100"]
    assert posts[0].media_urls == ["https://img/one.jpg"]
    assert posts[1].media_urls == ["https://img/quoted.png"]
    assert posts[0].url == "https://x.com/xtony1314/status/200"
    assert captured["env"]["AUTH_TOKEN"] == "auth-secret"
    assert captured["env"]["CT0"] == "csrf-secret"
    assert "NODE_CHANNEL_FD" not in captured["env"]
    assert "NODE_CHANNEL_SERIALIZATION_MODE" not in captured["env"]
    command_text = " ".join(captured["args"])
    assert "auth-secret" not in command_text
    assert "csrf-secret" not in command_text
    assert captured["args"][-5:] == [
        "--max-pages",
        "1",
        "--json",
        "--plain",
        "--no-color",
    ]


def test_bird_feed_redacts_secrets_from_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feed = _feed(monkeypatch)
    monkeypatch.setattr(
        "intent_trade.twitter.client.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="request failed auth-secret csrf-secret",
        ),
    )

    with pytest.raises(RuntimeError, match=r"<redacted> <redacted>") as exc_info:
        feed.fetch_user_posts("xtony1314", limit=5)
    assert "auth-secret" not in str(exc_info.value)
    assert "csrf-secret" not in str(exc_info.value)


def test_bird_feed_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    feed = _feed(monkeypatch)

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="bird", timeout=60)

    monkeypatch.setattr("intent_trade.twitter.client.subprocess.run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        feed.fetch_user_posts("xtony1314", limit=5)


def test_create_bird_feed_requires_both_cookies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TWITTER_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TWITTER_CT0", raising=False)
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CT0", raising=False)
    settings = Settings(twitter=TwitterConfig(source="bird"))

    with pytest.raises(RuntimeError, match="TWITTER_AUTH_TOKEN/TWITTER_CT0"):
        create_social_feed(settings)


def test_create_bird_feed_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWITTER_AUTH_TOKEN", "auth-secret")
    monkeypatch.setenv("TWITTER_CT0", "csrf-secret")
    monkeypatch.setattr("intent_trade.twitter.client.shutil.which", lambda _: "/bin/bird")
    settings = Settings(
        twitter=TwitterConfig(source="bird", bird_timeout_seconds=33)
    )

    feed = create_social_feed(settings)

    assert isinstance(feed, BirdSocialFeed)
    assert feed.timeout_seconds == 33
