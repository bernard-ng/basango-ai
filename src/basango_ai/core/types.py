from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl


class ArticleRecord(BaseModel):
    """Normalized news article representation used across the stack."""

    id: str
    title: str
    content: str
    source: str
    language: str
    published_at: Optional[datetime] = None
    url: Optional[HttpUrl] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["ArticleRecord"]
