"""Social feed abstraction: mock (Phase 1 default) + optional X API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from intent_trade.config import Settings
from intent_trade.models.domain import SocialPost


class SocialFeed(ABC):
    @abstractmethod
    def fetch_user_posts(
        self, username: str, limit: int = 50
    ) -> list[SocialPost]:
        ...

    def fetch_kols(
        self, usernames: list[str], limit_per_user: int = 50
    ) -> list[SocialPost]:
        posts: list[SocialPost] = []
        for u in usernames:
            posts.extend(self.fetch_user_posts(u, limit=limit_per_user))
        posts.sort(key=lambda p: p.created_at)
        return posts


class UnavailableSocialFeed(SocialFeed):
    """Keeps the dashboard usable when a live feed is not configured."""

    def __init__(self, error: str) -> None:
        self.error = error

    def fetch_user_posts(
        self, username: str, limit: int = 50
    ) -> list[SocialPost]:
        raise RuntimeError(self.error)


class MockSocialFeed(SocialFeed):
    """Loads sample KOL posts from data/sample/kol_posts.json."""

    def __init__(self, sample_path: Path) -> None:
        self.sample_path = sample_path
        self._posts: list[SocialPost] | None = None

    def _load(self) -> list[SocialPost]:
        if self._posts is not None:
            return self._posts
        if not self.sample_path.exists():
            self._posts = []
            return self._posts
        with self.sample_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        posts: list[SocialPost] = []
        for item in raw:
            created = item.get("created_at")
            if isinstance(created, str):
                created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if created_at.tzinfo is not None:
                    created_at = created_at.replace(tzinfo=None)
            else:
                created_at = datetime.utcnow()
            posts.append(
                SocialPost(
                    id=item.get("id") or f"mock_{len(posts)}",
                    platform=item.get("platform", "twitter"),
                    author_username=item["author_username"],
                    author_display_name=item.get("author_display_name", ""),
                    text=item["text"],
                    created_at=created_at,
                    url=item.get("url"),
                    media_urls=item.get("media_urls") or [],
                    media_alt_texts=item.get("media_alt_texts") or [],
                    media_transcripts=item.get("media_transcripts") or [],
                    raw=item.get("raw") or item,
                )
            )
        self._posts = posts
        return posts

    def fetch_user_posts(
        self, username: str, limit: int = 50
    ) -> list[SocialPost]:
        all_posts = self._load()
        user = username.lstrip("@").lower()
        matched = [
            p for p in all_posts if p.author_username.lstrip("@").lower() == user
        ]
        matched.sort(key=lambda p: p.created_at, reverse=True)
        return matched[:limit]


class XApiSocialFeed(SocialFeed):
    """Optional official X API client via tweepy (Bearer token)."""

    def __init__(self, bearer_token: str) -> None:
        try:
            import tweepy
        except ImportError as e:
            raise RuntimeError("tweepy not installed") from e
        self._client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    def fetch_user_posts(
        self, username: str, limit: int = 50
    ) -> list[SocialPost]:
        user = self._client.get_user(username=username.lstrip("@"))
        if not user or not user.data:
            return []
        uid = user.data.id
        resp = self._client.get_users_tweets(
            id=uid,
            max_results=min(limit, 100),
            tweet_fields=["created_at", "entities", "attachments"],
            expansions=["attachments.media_keys"],
            media_fields=["url", "preview_image_url", "alt_text", "type"],
        )
        media_map: dict[str, dict] = {}
        if resp.includes and "media" in resp.includes:
            for m in resp.includes["media"]:
                media_map[m.media_key] = {
                    "url": getattr(m, "url", None)
                    or getattr(m, "preview_image_url", None),
                    "alt_text": getattr(m, "alt_text", None) or "",
                }
        posts: list[SocialPost] = []
        for t in resp.data or []:
            media_urls: list[str] = []
            media_alts: list[str] = []
            if t.attachments and "media_keys" in t.attachments:
                for mk in t.attachments["media_keys"]:
                    info = media_map.get(mk) or {}
                    if info.get("url"):
                        media_urls.append(info["url"])
                        media_alts.append(info.get("alt_text") or "")
            created = t.created_at
            if created and created.tzinfo is not None:
                created = created.replace(tzinfo=None)
            posts.append(
                SocialPost(
                    id=str(t.id),
                    platform="twitter",
                    author_username=username.lstrip("@"),
                    text=t.text or "",
                    created_at=created or datetime.utcnow(),
                    url=f"https://x.com/{username.lstrip('@')}/status/{t.id}",
                    media_urls=media_urls,
                    media_alt_texts=media_alts,
                    raw={"id": str(t.id)},
                )
            )
        return posts


def _parse_tweet_time(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if not value:
        return datetime.utcnow()
    s = str(value).strip()
    # twitterapi.io often returns "Mon Jul 14 08:12:00 +0000 2026"
    for fmt in (
        "%a %b %d %H:%M:%S %z %Y",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except ValueError:
        return datetime.utcnow()


def _media_from_tweet(t: dict) -> tuple[list[str], list[str]]:
    urls: list[str] = []
    alts: list[str] = []
    for key in ("extendedEntities", "extended_entities", "entities"):
        block = t.get(key) or {}
        media = block.get("media") or []
        for m in media:
            u = m.get("media_url_https") or m.get("media_url") or m.get("url")
            if u and u not in urls:
                urls.append(u)
                alts.append(m.get("ext_alt_text") or m.get("alt_text") or "")
    # flat list variants
    for m in t.get("media") or []:
        if isinstance(m, str):
            if m not in urls:
                urls.append(m)
                alts.append("")
        elif isinstance(m, dict):
            u = m.get("url") or m.get("media_url_https")
            if u and u not in urls:
                urls.append(u)
                alts.append(m.get("alt_text") or "")
    return urls, alts


class TwitterApiIoFeed(SocialFeed):
    """Third-party twitterapi.io — free signup credits, cheap user timeline reads.

    Docs: https://docs.twitterapi.io/
    Endpoint: GET /twitter/user/last_tweets?userName=...
    Auth: header X-API-Key
    """

    BASE = "https://api.twitterapi.io"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch_user_posts(
        self, username: str, limit: int = 50
    ) -> list[SocialPost]:
        import httpx

        user = username.lstrip("@")
        posts: list[SocialPost] = []
        cursor = ""
        headers = {"X-API-Key": self.api_key}
        with httpx.Client(timeout=30.0, headers=headers) as client:
            while len(posts) < limit:
                params: dict[str, object] = {
                    "userName": user,
                    "includeReplies": "false",
                }
                if cursor:
                    params["cursor"] = cursor
                r = client.get(
                    f"{self.BASE}/twitter/user/last_tweets",
                    params=params,
                )
                if r.status_code == 401 or r.status_code == 403:
                    raise RuntimeError(
                        f"twitterapi.io auth failed ({r.status_code}): {r.text[:300]}. "
                        "Get a free key at https://twitterapi.io"
                    )
                r.raise_for_status()
                data = r.json()
                tweets = data.get("tweets") or data.get("data") or []
                if not tweets:
                    break
                for t in tweets:
                    if len(posts) >= limit:
                        break
                    if not isinstance(t, dict):
                        continue
                    tid = str(t.get("id") or t.get("id_str") or "")
                    if not tid:
                        continue
                    text = t.get("text") or t.get("full_text") or ""
                    author = t.get("author") or {}
                    uname = (
                        author.get("userName")
                        or author.get("username")
                        or author.get("screen_name")
                        or user
                    )
                    dname = (
                        author.get("name")
                        or author.get("displayName")
                        or ""
                    )
                    media_urls, media_alts = _media_from_tweet(t)
                    created = _parse_tweet_time(
                        t.get("createdAt") or t.get("created_at")
                    )
                    posts.append(
                        SocialPost(
                            id=tid,
                            platform="twitter",
                            author_username=str(uname).lstrip("@"),
                            author_display_name=str(dname),
                            text=str(text),
                            created_at=created,
                            url=t.get("url")
                            or f"https://x.com/{str(uname).lstrip('@')}/status/{tid}",
                            media_urls=media_urls,
                            media_alt_texts=media_alts,
                            raw=t,
                        )
                    )
                if not data.get("has_next_page"):
                    break
                cursor = data.get("next_cursor") or ""
                if not cursor:
                    break
        return posts


class BirdSocialFeed(SocialFeed):
    """Read public X timelines through the cookie-authenticated bird CLI."""

    def __init__(
        self,
        auth_token: str,
        ct0: str,
        *,
        command: str = "bird",
        timeout_seconds: int = 60,
    ) -> None:
        if not shutil.which(command):
            raise RuntimeError(
                f"bird CLI not found: {command}. Install @jtsang/bird or set "
                "TWITTER_BIRD_BIN to its executable path."
            )
        self.auth_token = auth_token
        self.ct0 = ct0
        self.command = command
        self.timeout_seconds = max(10, int(timeout_seconds))

    def _safe_error(self, value: str) -> str:
        message = value.strip()
        for secret in (self.auth_token, self.ct0):
            if secret:
                message = message.replace(secret, "<redacted>")
        return message[-800:] or "unknown bird error"

    @staticmethod
    def _tweet_list(payload: object) -> list[dict]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            tweets = payload.get("tweets") or payload.get("data") or []
            if isinstance(tweets, list):
                return [item for item in tweets if isinstance(item, dict)]
        raise RuntimeError("bird returned an unexpected JSON response")

    @staticmethod
    def _post_from_tweet(tweet: dict, fallback_username: str) -> Optional[SocialPost]:
        tid = str(tweet.get("id") or "")
        text = str(tweet.get("text") or "")
        if not tid or not text:
            return None
        author = tweet.get("author") if isinstance(tweet.get("author"), dict) else {}
        username = str(author.get("username") or fallback_username).lstrip("@")
        display_name = str(author.get("name") or "")
        media_urls: list[str] = []
        media_alts: list[str] = []
        quoted = tweet.get("quotedTweet")
        media_sources = [tweet]
        if isinstance(quoted, dict):
            media_sources.append(quoted)
        for source in media_sources:
            for media in source.get("media") or []:
                if not isinstance(media, dict):
                    continue
                url = (
                    media.get("url")
                    or media.get("previewUrl")
                    or media.get("videoUrl")
                )
                if url and str(url) not in media_urls:
                    media_urls.append(str(url))
                    media_alts.append(str(media.get("altText") or ""))
        return SocialPost(
            id=tid,
            platform="twitter",
            author_username=username,
            author_display_name=display_name,
            text=text,
            created_at=_parse_tweet_time(tweet.get("createdAt")),
            url=f"https://x.com/{username}/status/{tid}",
            media_urls=media_urls,
            media_alt_texts=media_alts,
            raw=tweet,
        )

    def fetch_user_posts(
        self, username: str, limit: int = 50
    ) -> list[SocialPost]:
        user = username.lstrip("@")
        count = min(20, max(1, int(limit)))
        env = os.environ.copy()
        # PM2's Node IPC variables are inherited by the Python service. If they
        # reach bird, its Node runtime treats an unrelated fd as an IPC channel.
        env.pop("NODE_CHANNEL_FD", None)
        env.pop("NODE_CHANNEL_SERIALIZATION_MODE", None)
        env.update(
            {
                "AUTH_TOKEN": self.auth_token,
                "CT0": self.ct0,
                "NO_COLOR": "1",
            }
        )
        args = [
            self.command,
            "user-tweets",
            f"@{user}",
            "--count",
            str(count),
            "--max-pages",
            "1",
            "--json",
            "--plain",
            "--no-color",
        ]
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"bird timed out after {self.timeout_seconds}s while reading @{user}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"could not run bird CLI: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"bird failed for @{user}: {self._safe_error(result.stderr)}"
            )
        try:
            tweets = self._tweet_list(json.loads(result.stdout))
        except json.JSONDecodeError as exc:
            raise RuntimeError("bird returned invalid JSON") from exc
        posts = [self._post_from_tweet(tweet, user) for tweet in tweets]
        parsed = [post for post in posts if post is not None]
        parsed.sort(key=lambda post: post.created_at, reverse=True)
        return parsed[:count]


class RapidApiTwitter241Feed(SocialFeed):
    """RapidAPI Twttr API (twitter241) — user resolve + user-tweets timeline.

    Env:
      TWITTER_RAPIDAPI_KEY
      TWITTER_RAPIDAPI_HOST  (default twitter241.p.rapidapi.com)
    """

    def __init__(self, api_key: str, host: str | None = None) -> None:
        self.api_key = api_key
        self.host = host or os.getenv(
            "TWITTER_RAPIDAPI_HOST", "twitter241.p.rapidapi.com"
        )
        self.base = f"https://{self.host}"
        self._uid_cache: dict[str, str] = {
            # known live target
            "xtony1314": "1589272961235619840",
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-rapidapi-host": self.host,
            "x-rapidapi-key": self.api_key,
        }

    def resolve_user_id(self, username: str) -> Optional[str]:
        user = username.lstrip("@").lower()
        if user in self._uid_cache:
            return self._uid_cache[user]
        import httpx

        with httpx.Client(timeout=30.0, headers=self._headers()) as client:
            r = client.get(f"{self.base}/user", params={"username": user})
            if r.status_code >= 400:
                raise RuntimeError(
                    f"rapidapi /user failed ({r.status_code}): {r.text[:300]}"
                )
            data = r.json()
        uid = self._extract_user_id(data)
        if uid:
            self._uid_cache[user] = uid
        return uid

    @staticmethod
    def _extract_user_id(data: dict) -> Optional[str]:
        # common shapes
        try:
            u = data["result"]["data"]["user"]["result"]
            rid = u.get("rest_id") or u.get("id")
            if rid and str(rid).isdigit():
                return str(rid)
            # rest_id sometimes base64-ish GraphQL id; prefer legacy/id_str
            leg = u.get("legacy") or {}
            if leg.get("id_str"):
                return str(leg["id_str"])
        except Exception:
            pass

        def walk(o: object) -> Optional[str]:
            if isinstance(o, dict):
                if o.get("rest_id") and str(o["rest_id"]).isdigit():
                    # only accept if looks like user object
                    if "legacy" in o or o.get("__typename") == "User":
                        return str(o["rest_id"])
                if o.get("id_str") and str(o["id_str"]).isdigit() and "screen_name" in o:
                    return str(o["id_str"])
                for v in o.values():
                    found = walk(v)
                    if found:
                        return found
            elif isinstance(o, list):
                for v in o:
                    found = walk(v)
                    if found:
                        return found
            return None

        return walk(data)

    def fetch_user_posts(
        self, username: str, limit: int = 50
    ) -> list[SocialPost]:
        import httpx

        user = username.lstrip("@")
        uid = self.resolve_user_id(user)
        if not uid:
            raise RuntimeError(f"Could not resolve user id for @{user}")

        posts: list[SocialPost] = []
        cursor: Optional[str] = None
        with httpx.Client(timeout=45.0, headers=self._headers()) as client:
            while len(posts) < limit:
                params: dict[str, object] = {
                    "user": uid,
                    "count": str(min(20, max(5, limit - len(posts)))),
                }
                if cursor:
                    params["cursor"] = cursor
                r = client.get(f"{self.base}/user-tweets", params=params)
                if r.status_code >= 400:
                    raise RuntimeError(
                        f"rapidapi /user-tweets failed ({r.status_code}): {r.text[:300]}"
                    )
                data = r.json()
                batch = self._parse_timeline(data, fallback_username=user)
                if not batch:
                    break
                # de-dupe within fetch
                seen = {p.id for p in posts}
                new_items = [p for p in batch if p.id not in seen]
                if not new_items:
                    break
                posts.extend(new_items)
                cursor = None
                if isinstance(data.get("cursor"), dict):
                    cursor = data["cursor"].get("bottom")
                if not cursor:
                    break
        posts.sort(key=lambda p: p.created_at, reverse=True)
        return posts[:limit]

    def _parse_timeline(
        self, data: dict, fallback_username: str
    ) -> list[SocialPost]:
        raw_tweets: list[dict] = []

        def consider(node: dict) -> None:
            # Tweet / TweetWithVisibilityResults
            if node.get("__typename") in (
                "Tweet",
                "TweetWithVisibilityResults",
            ) or "legacy" in node:
                t = node
                if "tweet" in node and isinstance(node["tweet"], dict):
                    t = node["tweet"]
                if "result" in node and isinstance(node["result"], dict):
                    # sometimes nested
                    inner = node["result"]
                    if inner.get("__typename") == "Tweet" or "legacy" in inner:
                        t = inner
                    elif "tweet" in inner:
                        t = inner["tweet"]
                leg = t.get("legacy") if isinstance(t.get("legacy"), dict) else None
                if leg and (leg.get("full_text") or leg.get("text")):
                    raw_tweets.append(t)
                    return
            # entry content itemContent tweet_results
            item = node.get("itemContent") or node.get("content")
            if isinstance(item, dict):
                tr = item.get("tweet_results") or item.get("tweetResults")
                if isinstance(tr, dict) and isinstance(tr.get("result"), dict):
                    consider(tr["result"])

        def walk(o: object) -> None:
            if isinstance(o, dict):
                consider(o)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(data)

        posts: list[SocialPost] = []
        seen: set[str] = set()
        for t in raw_tweets:
            leg = t.get("legacy") or {}
            tid = str(
                t.get("rest_id")
                or leg.get("id_str")
                or leg.get("id")
                or ""
            )
            if not tid or tid in seen:
                continue
            seen.add(tid)
            text = leg.get("full_text") or leg.get("text") or ""
            created = _parse_tweet_time(leg.get("created_at") or t.get("created_at"))
            # author
            uname = fallback_username
            dname = ""
            try:
                core = (
                    t.get("core", {})
                    .get("user_results", {})
                    .get("result", {})
                )
                if isinstance(core, dict):
                    c = core.get("core") or {}
                    uname = c.get("screen_name") or uname
                    dname = c.get("name") or ""
                    leg_u = core.get("legacy") or {}
                    uname = leg_u.get("screen_name") or uname
                    dname = leg_u.get("name") or dname
            except Exception:
                pass
            media_urls, media_alts = _media_from_tweet(leg)
            # also entities on legacy
            posts.append(
                SocialPost(
                    id=tid,
                    platform="twitter",
                    author_username=str(uname).lstrip("@"),
                    author_display_name=str(dname),
                    text=str(text),
                    created_at=created,
                    url=f"https://x.com/{str(uname).lstrip('@')}/status/{tid}",
                    media_urls=media_urls,
                    media_alt_texts=media_alts,
                    raw=t,
                )
            )
        posts.sort(key=lambda p: p.created_at, reverse=True)
        return posts


def create_social_feed(settings: Settings) -> SocialFeed:
    source = (settings.twitter.source or "mock").lower()
    if source in ("bird", "bird_cli", "cookie"):
        auth_token = os.getenv("TWITTER_AUTH_TOKEN") or os.getenv("AUTH_TOKEN") or ""
        ct0 = os.getenv("TWITTER_CT0") or os.getenv("CT0") or ""
        if not auth_token or not ct0:
            raise RuntimeError(
                "twitter.source=bird but TWITTER_AUTH_TOKEN/TWITTER_CT0 are not set "
                "in .env; copy auth_token and ct0 from an active x.com browser session"
            )
        command = os.getenv("TWITTER_BIRD_BIN") or "bird"
        return BirdSocialFeed(
            auth_token,
            ct0,
            command=command,
            timeout_seconds=settings.twitter.bird_timeout_seconds,
        )
    if source in ("rapidapi", "twitter241", "twttr", "rapidapi_twitter241"):
        key = os.getenv("TWITTER_RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY") or ""
        if not key:
            raise RuntimeError(
                "twitter.source=rapidapi but TWITTER_RAPIDAPI_KEY is not set in .env"
            )
        host = os.getenv("TWITTER_RAPIDAPI_HOST") or None
        return RapidApiTwitter241Feed(key, host=host)
    if source in ("twitterapi_io", "twitterapi.io", "third_party"):
        key = os.getenv("TWITTERAPI_IO_KEY") or os.getenv("TWITTERAPI_IO_API_KEY") or ""
        if not key:
            raise RuntimeError(
                "twitter.source=twitterapi_io but TWITTERAPI_IO_KEY is not set. "
                "Sign up free (no card, ~$0.1 credits) at https://twitterapi.io "
                "and put the key in .env"
            )
        return TwitterApiIoFeed(key)
    if source == "x_api":
        token = os.getenv("X_BEARER_TOKEN", "")
        if not token:
            raise RuntimeError(
                "twitter.source=x_api but X_BEARER_TOKEN is not set; "
                "prefer rapidapi for current live reads, or set the token."
            )
        return XApiSocialFeed(token)
    sample = settings.data_dir / "sample" / "kol_posts.json"
    return MockSocialFeed(sample)
