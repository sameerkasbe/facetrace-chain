"""URL and content classification for discovered reverse search candidates.

Categorizes discovered web resources into:
- Profiles (social profiles, author/artist channels)
- Posts & Images (individual articles, feed posts, standalone image assets)
- Reels & Videos (short-form videos, YouTube videos, Instagram Reels, TikToks)
"""
import re
import urllib.parse
from enum import Enum
from typing import Dict, Any, Optional


class ContentType(str, Enum):
    PROFILE = "profile"
    POST = "post"
    IMAGE = "image"
    REEL = "reel"
    VIDEO = "video"
    WEBPAGE = "webpage"
    UNKNOWN = "unknown"


class ResultCategory(str, Enum):
    PROFILES = "Profiles"
    POSTS_IMAGES = "Posts & Images"
    REELS_VIDEOS = "Reels & Videos"


# Image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".svg"}


class ResultClassifier:
    """Classifies candidate URLs and metadata into distinct content types and categories."""

    @classmethod
    def classify(cls, url: str, metadata: Optional[Dict[str, Any]] = None) -> ContentType:
        """Determines the content type from URL path structure and optional metadata."""
        if not url:
            return ContentType.UNKNOWN

        meta = metadata or {}
        explicit_type = meta.get("content_type")
        if explicit_type and hasattr(ContentType, explicit_type.upper()):
            return ContentType(explicit_type.lower())

        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        query = parsed.query.lower()
        path_lower = path.lower()

        # Check for direct image files
        clean_path = path_lower.split("?")[0]
        for ext in IMAGE_EXTENSIONS:
            if clean_path.endswith(ext):
                return ContentType.IMAGE

        # 1. Instagram
        if "instagram.com" in netloc:
            if "/reel/" in path_lower or "/reels/" in path_lower or "/tv/" in path_lower:
                return ContentType.REEL
            if "/p/" in path_lower:
                return ContentType.POST
            # Path like /username/ or /username
            parts = [p for p in path.split("/") if p]
            if len(parts) == 1 and parts[0] not in {"explore", "accounts", "about", "developer"}:
                return ContentType.PROFILE
            return ContentType.POST

        # 2. YouTube
        if "youtube.com" in netloc or "youtu.be" in netloc:
            if "/shorts/" in path_lower:
                return ContentType.REEL
            if "/watch" in path_lower or "youtu.be" in netloc:
                return ContentType.VIDEO
            if any(path_lower.startswith(p) for p in ["/@", "/channel/", "/c/", "/user/"]):
                return ContentType.PROFILE
            return ContentType.VIDEO

        # 3. TikTok
        if "tiktok.com" in netloc:
            if "/video/" in path_lower:
                return ContentType.REEL
            parts = [p for p in path.split("/") if p]
            if len(parts) == 1 and parts[0].startswith("@"):
                return ContentType.PROFILE
            return ContentType.REEL

        # 4. Facebook
        if "facebook.com" in netloc or "fb.com" in netloc:
            if any(p in path_lower for p in ["/reel/", "/reels/", "/videos/", "/watch"]):
                return ContentType.REEL if "/reel" in path_lower else ContentType.VIDEO
            if any(p in path_lower for p in ["/posts/", "/photos/", "/photo.php", "/story.php"]):
                return ContentType.POST
            if any(p in path_lower for p in ["/pages/", "/people/", "/profile.php"]):
                return ContentType.PROFILE
            parts = [p for p in path.split("/") if p]
            if len(parts) == 1 and parts[0] not in {"watch", "groups", "events", "marketplace"}:
                return ContentType.PROFILE
            return ContentType.POST

        # 5. X / Twitter
        if "twitter.com" in netloc or "x.com" in netloc:
            if "/status/" in path_lower:
                return ContentType.POST
            parts = [p for p in path.split("/") if p]
            if len(parts) == 1 and parts[0] not in {"home", "explore", "notifications", "messages", "settings", "i"}:
                return ContentType.PROFILE
            return ContentType.POST

        # 6. LinkedIn
        if "linkedin.com" in netloc:
            if any(path_lower.startswith(p) for p in ["/in/", "/company/"]):
                return ContentType.PROFILE
            if any(path_lower.startswith(p) for p in ["/posts/", "/feed/update/", "/pulse/"]):
                return ContentType.POST
            return ContentType.WEBPAGE

        # 7. Reddit
        if "reddit.com" in netloc:
            if any(path_lower.startswith(p) for p in ["/user/", "/u/"]):
                return ContentType.PROFILE
            if "/comments/" in path_lower:
                return ContentType.POST
            return ContentType.WEBPAGE

        # 8. Pinterest
        if "pinterest.com" in netloc:
            if "/pin/" in path_lower:
                return ContentType.POST
            parts = [p for p in path.split("/") if p]
            if len(parts) == 1:
                return ContentType.PROFILE
            return ContentType.WEBPAGE

        # 9. Wikimedia / Wikipedia
        if "wikimedia.org" in netloc or "wikipedia.org" in netloc:
            if "wiki/file:" in path_lower or "wiki/special:filepath" in path_lower:
                return ContentType.IMAGE
            if "wiki/user:" in path_lower:
                return ContentType.PROFILE
            return ContentType.WEBPAGE

        # Video metadata or dedicated query parameter detection
        query_params = urllib.parse.parse_qs(query)
        has_video_query = (
            "v" in query_params
            or "video_id" in query_params
            or "clip_id" in query_params
            or any(k in {"watch", "video", "media_type"} and "video" in "".join(v).lower() for k, v in query_params.items())
        )
        if meta.get("is_video") or has_video_query or any(segment in path_lower for segment in ["/video/", "/videos/", "/watch/"]):
            return ContentType.VIDEO

        # Default fallback: webpage
        return ContentType.WEBPAGE

    @classmethod
    def get_category(cls, content_type: ContentType) -> ResultCategory:
        """Maps a detailed ContentType to one of the 3 primary presentation categories."""
        if content_type == ContentType.PROFILE:
            return ResultCategory.PROFILES
        elif content_type in {ContentType.REEL, ContentType.VIDEO}:
            return ResultCategory.REELS_VIDEOS
        else:
            return ResultCategory.POSTS_IMAGES

    @classmethod
    def get_category_for_url(cls, url: str, metadata: Optional[Dict[str, Any]] = None) -> ResultCategory:
        """Direct helper to classify a URL directly to its UI display category."""
        ctype = cls.classify(url, metadata)
        return cls.get_category(ctype)
