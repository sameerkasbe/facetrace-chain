"""Unit and integration tests for search providers, image staging, platform & content classification."""
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

from search.search_provider import (
    TempImageUploader,
    LiveReverseImageSearchProvider,
    PublicWebSearchProvider,
    SearchProviderError,
    get_search_provider,
    classify_platform
)
from search.candidate_collector import CandidateCollector
from face.detector import FaceDetector
from face.encoder import get_face_recognizer
from face.matcher import FaceMatcher
from utils.classifier import ResultClassifier, ContentType, ResultCategory
from models.match_record import CandidateResult


def test_classify_platform():
    assert classify_platform("https://www.instagram.com/p/C_12345/") == "Instagram"
    assert classify_platform("https://facebook.com/user/photo") == "Facebook"
    assert classify_platform("https://x.com/user/status/12345") == "X/Twitter"
    assert classify_platform("https://twitter.com/user/status/12345") == "X/Twitter"
    assert classify_platform("https://www.linkedin.com/in/profile") == "LinkedIn"
    assert classify_platform("https://www.youtube.com/watch?v=abc") == "YouTube"
    assert classify_platform("https://commons.wikimedia.org/wiki/File:Test.jpg") == "Wikimedia"
    assert classify_platform("https://en.wikipedia.org/wiki/Albert_Einstein") == "Wikipedia"
    assert classify_platform("https://unknown-personal-blog.io/photo.jpg") == "Web"


def test_temp_image_uploader_url_passthrough():
    uploader = TempImageUploader()
    url = "https://example.org/sample_portrait.jpg"
    assert uploader.upload_image(url) == url
    assert uploader.stage_image(url) == url


def test_temp_image_uploader_image_prep(tmp_path):
    uploader = TempImageUploader()
    img = np.zeros((1600, 1600, 3), dtype=np.uint8)
    test_file = tmp_path / "large_face.jpg"
    cv2.imwrite(str(test_file), img)

    raw_bytes, name = uploader._prepare_image_bytes(test_file)
    assert len(raw_bytes) > 0
    assert name == "search_face.jpg"


@patch("requests.post")
def test_temp_image_uploader_uguu_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "files": [{"url": "https://h.uguu.se/test_portrait.jpg"}]
    }
    mock_post.return_value = mock_resp

    uploader = TempImageUploader()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy_img)

    result_url = uploader.upload_image(buf.tobytes())
    assert result_url == "https://h.uguu.se/test_portrait.jpg"


@patch("requests.post")
def test_temp_image_uploader_freeimage_success(mock_post, tmp_path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status_code": 200,
        "image": {"url": "https://iili.io/test_image.jpg"}
    }
    mock_post.return_value = mock_resp

    uploader = TempImageUploader()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy_img)

    result_url = uploader.upload_image(buf.tobytes())
    assert result_url == "https://iili.io/test_image.jpg"


@patch("requests.post")
def test_temp_image_uploader_fallback_tmpfiles(mock_post):
    def mock_post_side_effect(url, **kwargs):
        resp = MagicMock()
        if "freeimage.host" in url:
            resp.status_code = 500
            return resp
        elif "tmpfiles.org" in url:
            resp.status_code = 200
            resp.json.return_value = {
                "status": "success",
                "data": {"url": "https://tmpfiles.org/12345/test.jpg"}
            }
            return resp
        return resp

    mock_post.side_effect = mock_post_side_effect

    uploader = TempImageUploader()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy_img)

    result_url = uploader.upload_image(buf.tobytes())
    assert result_url == "https://tmpfiles.org/dl/12345/test.jpg"


def test_live_reverse_search_missing_key():
    provider = LiveReverseImageSearchProvider(api_key="")
    with pytest.raises(SearchProviderError) as exc_info:
        provider.search_by_image("sample_data/sample_portrait_a.jpg")
    assert "Missing SERPAPI_KEY" in str(exc_info.value)


@patch("requests.get")
@patch.object(TempImageUploader, "stage_image", return_value="https://iili.io/staged.jpg")
def test_live_reverse_search_with_mock_results(mock_stage, mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "visual_matches": [
            {
                "position": 1,
                "title": "Albert Einstein — Nobel Prize Archives",
                "link": "https://www.nobelprize.org/prizes/physics/1921/einstein/biographical/",
                "source": "nobelprize.org",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR...",
                "snippet": "Biographical profile of Albert Einstein",
                "date": "1921"
            },
            {
                "position": 2,
                "title": "Einstein Instagram Fan Page",
                "link": "https://instagram.com/p/einstein_portrait",
                "source": "Instagram",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT..."
            }
        ]
    }
    mock_get.return_value = mock_resp

    provider = LiveReverseImageSearchProvider(api_key="mock_serpapi_key_123")
    assert "Google Lens" in provider.mode_name

    candidates = provider.search_by_image("sample_data/sample_portrait_a.jpg", max_results=5)
    assert len(candidates) == 2
    assert candidates[0].title == "Albert Einstein — Nobel Prize Archives"
    assert candidates[0].source_domain == "www.nobelprize.org"
    assert candidates[0].platform == "Web"
    assert candidates[0].search_provider == "Google Lens via SerpAPI"
    assert candidates[0].snippet == "Biographical profile of Albert Einstein"
    assert candidates[0].publication_date == "1921"
    assert candidates[0].category == "Posts & Images"

    assert candidates[1].platform == "Instagram"
    assert candidates[1].content_type == "post"
    assert candidates[1].category == "Posts & Images"


def test_search_provider_factory():
    p_live = get_search_provider(mode="live")
    assert isinstance(p_live, LiveReverseImageSearchProvider)

    p_serp = get_search_provider(mode="serpapi")
    assert isinstance(p_serp, LiveReverseImageSearchProvider)

    # Offline mode is removed; defaults to live reverse search
    p_fallback = get_search_provider(mode="offline")
    assert isinstance(p_fallback, LiveReverseImageSearchProvider)

    p_web = get_search_provider(mode="web")
    assert isinstance(p_web, PublicWebSearchProvider)


def test_result_classifier_profiles():
    assert ResultClassifier.classify("https://www.instagram.com/johndoe/") == ContentType.PROFILE
    assert ResultClassifier.classify("https://www.youtube.com/@techlead") == ContentType.PROFILE
    assert ResultClassifier.classify("https://www.youtube.com/channel/UC123456789") == ContentType.PROFILE
    assert ResultClassifier.classify("https://x.com/elonmusk") == ContentType.PROFILE
    assert ResultClassifier.classify("https://twitter.com/billgates") == ContentType.PROFILE
    assert ResultClassifier.classify("https://www.linkedin.com/in/satyanadella") == ContentType.PROFILE
    assert ResultClassifier.classify("https://www.tiktok.com/@tiktokcreator") == ContentType.PROFILE
    assert ResultClassifier.classify("https://www.facebook.com/zuck") == ContentType.PROFILE


def test_result_classifier_posts_and_images():
    assert ResultClassifier.classify("https://www.instagram.com/p/DB123456/") == ContentType.POST
    assert ResultClassifier.classify("https://x.com/user/status/123456789") == ContentType.POST
    assert ResultClassifier.classify("https://twitter.com/user/status/987654321") == ContentType.POST
    assert ResultClassifier.classify("https://www.facebook.com/posts/101589123") == ContentType.POST
    assert ResultClassifier.classify("https://example.org/photos/portrait.jpg") == ContentType.IMAGE
    assert ResultClassifier.classify("https://commons.wikimedia.org/wiki/File:Einstein.jpg") == ContentType.IMAGE


def test_result_classifier_reels_and_videos():
    assert ResultClassifier.classify("https://www.instagram.com/reel/C89abcdef/") == ContentType.REEL
    assert ResultClassifier.classify("https://www.instagram.com/reels/C89abcdef/") == ContentType.REEL
    assert ResultClassifier.classify("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == ContentType.VIDEO
    assert ResultClassifier.classify("https://youtu.be/dQw4w9WgXcQ") == ContentType.VIDEO
    assert ResultClassifier.classify("https://www.youtube.com/shorts/abcdefgh") == ContentType.REEL
    assert ResultClassifier.classify("https://www.tiktok.com/@user/video/7123456789") == ContentType.REEL
    assert ResultClassifier.classify("https://www.facebook.com/watch/?v=123456") == ContentType.VIDEO
    assert ResultClassifier.classify("https://www.facebook.com/reel/123456") == ContentType.REEL


def test_result_categories_mapping():
    assert ResultClassifier.get_category(ContentType.PROFILE) == ResultCategory.PROFILES
    assert ResultClassifier.get_category(ContentType.POST) == ResultCategory.POSTS_IMAGES
    assert ResultClassifier.get_category(ContentType.IMAGE) == ResultCategory.POSTS_IMAGES
    assert ResultClassifier.get_category(ContentType.WEBPAGE) == ResultCategory.POSTS_IMAGES
    assert ResultClassifier.get_category(ContentType.REEL) == ResultCategory.REELS_VIDEOS
    assert ResultClassifier.get_category(ContentType.VIDEO) == ResultCategory.REELS_VIDEOS


def test_video_query_parameter_precision():
    # Marketing query parameter should NOT turn a blog/webpage into a video
    assert ResultClassifier.classify("https://example.org/article?utm_source=video_campaign") == ContentType.WEBPAGE
    # Actual video indicator query parameter or metadata
    assert ResultClassifier.classify("https://example.org/player?v=abc12345") == ContentType.VIDEO
    assert ResultClassifier.classify("https://example.org/content", metadata={"is_video": True}) == ContentType.VIDEO


def test_empty_search_candidates_handling():
    """Verifies that empty candidates returned by search are handled safely without crashing verification."""
    collector = CandidateCollector(
        search_provider=PublicWebSearchProvider(),
        detector=FaceDetector(),
        encoder=get_face_recognizer("sface"),
        matcher=FaceMatcher(),
        top_k=5
    )
    dummy_emb = np.zeros(128, dtype=np.float32)
    verified, match_record = collector.verify_candidates(
        input_embedding=dummy_emb,
        candidates=[],
        input_image_bytes=b"fake",
        verbose=False
    )
    assert verified == []
    assert match_record is None


def test_live_search_missing_key_exception():
    """Verifies that LiveReverseImageSearchProvider cleanly raises SearchProviderError when key is empty."""
    provider = LiveReverseImageSearchProvider(api_key="")
    with pytest.raises(SearchProviderError) as exc_info:
        provider.search_by_image(b"dummy_bytes")
    assert "Missing SERPAPI_KEY" in str(exc_info.value)


@patch("requests.get")
@patch("search.search_provider.TempImageUploader.stage_image")
def test_live_search_api_json_error(mock_stage, mock_get):
    """Verifies that SerpAPI error responses are cleanly wrapped in SearchProviderError."""
    mock_stage.return_value = "https://example.com/staged.jpg"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"error": "Invalid API key provided."}
    mock_get.return_value = mock_resp

    provider = LiveReverseImageSearchProvider(api_key="invalid_key")
    with pytest.raises(SearchProviderError) as exc_info:
        provider.search_by_image(b"dummy_bytes")
    assert "Invalid API key" in str(exc_info.value)

