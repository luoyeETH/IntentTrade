"""Multimodal helpers: attach image transcripts (OCR / vision) to posts.

Phase 1: uses media_transcripts already present on posts (sample data / pre-OCR).
When ANTHROPIC_API_KEY is set and media_urls are real images, optionally call
Claude vision to fill media_transcripts.
"""

from __future__ import annotations

import os
from typing import Optional

from intent_trade.analysis.llm_client import default_model
from intent_trade.models.domain import SocialPost


def enrich_post_with_vision(
    post: SocialPost,
    model: Optional[str] = None,
) -> SocialPost:
    """If post has media_urls but empty transcripts, try vision captioning."""
    if not post.media_urls:
        return post
    if post.media_transcripts and any(post.media_transcripts):
        return post
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("INTENT_TRADE_LLM_KEY")
    if not api_key:
        return post

    # Only attempt http(s) URLs that look reachable (skip example.com placeholders)
    urls = [
        u
        for u in post.media_urls
        if u.startswith("http") and "example.com" not in u
    ]
    if not urls:
        return post

    model = model or os.getenv("INTENT_TRADE_VISION_MODEL") or default_model()

    try:
        import anthropic
    except ImportError:
        return post

    timeout = float(os.getenv("INTENT_TRADE_LLM_TIMEOUT", "20"))
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    transcripts: list[str] = []
    for url in urls:
        try:
            content = [
                {
                    "type": "image",
                    "source": {"type": "url", "url": url},
                },
                {
                    "type": "text",
                    "text": (
                        "Extract any trading plan from this chart/image: "
                        "ticker, direction, entry, stop loss, take profit. "
                        "Reply in one short plain line."
                    ),
                },
            ]
            msg = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": content}],
            )
            text = ""
            for block in msg.content:
                if hasattr(block, "text"):
                    text += block.text
            if text.strip():
                transcripts.append(text.strip())
        except Exception:
            continue
    if transcripts:
        post.media_transcripts = transcripts
    return post
