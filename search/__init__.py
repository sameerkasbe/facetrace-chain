"""Modular reverse image search and candidate collection module."""
from .search_provider import (
    BaseSearchProvider,
    TempImageUploader,
    LiveReverseImageSearchProvider,
    PublicWebSearchProvider,
    SearchProviderError,
    get_search_provider,
    classify_platform,
)
from .candidate_collector import CandidateCollector

__all__ = [
    "BaseSearchProvider",
    "TempImageUploader",
    "LiveReverseImageSearchProvider",
    "PublicWebSearchProvider",
    "SearchProviderError",
    "get_search_provider",
    "classify_platform",
    "CandidateCollector",
]
