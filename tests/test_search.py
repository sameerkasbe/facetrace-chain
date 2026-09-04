"""Unit and integration tests for search providers and image staging."""
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

from search.search_provider import (
    TempImageUploader,
    LiveReverseImageSearchProvider,
    OfflineDemoCorpusProvider,
    SearchProviderError,
    get_search_provider,
)
from models.match_record import CandidateResult


def test_temp_image_uploader_url_passthrough():
    uploader = TempImageUploader()
    url = "https://example.org/sample_portrait.jpg"
    assert uploader.upload_image(url) == url


def test_temp_image_uploader_image_prep(tmp_path):
    uploader = TempImageUploader()
    # Create test image
    img = np.zeros((1600, 1600, 3), dtype=np.uint8)
    test_file = tmp_path / "large_face.jpg"
    cv2.imwrite(str(test_file), img)

    raw_bytes, name = uploader._prepare_image_bytes(test_file)
    assert len(raw_bytes) > 0
    assert name == "search_face.jpg"


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
    # Freeimage fails, tmpfiles succeeds
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


def test_offline_demo_corpus_provider(tmp_path):
    # Create mock corpus with 2 portraits
    img1 = tmp_path / "einstein.jpg"
    img2 = tmp_path / "lincoln.png"
    cv2.imwrite(str(img1), np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(img2), np.zeros((100, 100, 3), dtype=np.uint8))

    provider = OfflineDemoCorpusProvider(corpus_dir=tmp_path)
    assert provider.mode_name == "Offline Demo Mode (Local Corpus Simulation)"

    candidates = provider.search_by_image("dummy_input.jpg", max_results=5)
    assert len(candidates) == 2
    for c in candidates:
        assert c.is_live is False
        assert c.source_domain == "local.public_corpus"
        assert c.metadata["provenance"] == "offline_simulation_corpus"


def test_live_reverse_search_missing_key():
    provider = LiveReverseImageSearchProvider(api_key="")
    with pytest.raises(SearchProviderError) as exc_info:
        provider.search_by_image("sample_data/sample_portrait_a.jpg")
    assert "SerpAPI key not found" in str(exc_info.value)
    assert "--mode offline" in str(exc_info.value)


@patch("requests.get")
@patch.object(TempImageUploader, "upload_image", return_value="https://iili.io/staged.jpg")
def test_live_reverse_search_with_mock_results(mock_upload, mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "visual_matches": [
            {
                "position": 1,
                "title": "Albert Einstein — Nobel Prize Archives",
                "link": "https://www.nobelprize.org/prizes/physics/1921/einstein/biographical/",
                "source": "nobelprize.org",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR..."
            },
            {
                "position": 2,
                "title": "Einstein Archives Online",
                "link": "https://albert-einstein.org/portrait.jpg",
                "source": "albert-einstein.org",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT..."
            }
        ]
    }
    mock_get.return_value = mock_resp

    provider = LiveReverseImageSearchProvider(api_key="mock_serpapi_key_123")
    assert provider.mode_name == "Live Reverse Image Search (Google Lens API)"

    candidates = provider.search_by_image("sample_data/sample_portrait_a.jpg", max_results=5)
    assert len(candidates) == 2
    assert candidates[0].title == "Albert Einstein — Nobel Prize Archives"
    assert candidates[0].source_domain == "www.nobelprize.org"
    assert candidates[0].is_live is True
    assert candidates[0].content_identifier == "google_lens_1"


def test_search_provider_factory():
    p_live = get_search_provider(mode="live")
    assert isinstance(p_live, LiveReverseImageSearchProvider)

    p_serp = get_search_provider(mode="serpapi")
    assert isinstance(p_serp, LiveReverseImageSearchProvider)

    p_offline = get_search_provider(mode="offline")
    assert isinstance(p_offline, OfflineDemoCorpusProvider)

    p_local = get_search_provider(mode="local")
    assert isinstance(p_local, OfflineDemoCorpusProvider)
