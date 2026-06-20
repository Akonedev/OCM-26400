"""Tests apprentissage YouTube (OCM-26400) — audit H9."""
import pytest
from ocm26400.youtube_learner import (
    available, _is_youtube_url, _clean_vtt, fetch_transcript, learn_from_youtube,
)


def test_available_flag():
    assert isinstance(available(), bool)


def test_is_youtube_url():
    assert _is_youtube_url("https://www.youtube.com/watch?v=abc") is True
    assert _is_youtube_url("https://youtu.be/abc") is True
    assert _is_youtube_url("https://example.com") is False
    assert _is_youtube_url("https://evil.com/youtube") is False


def test_clean_vtt_strips_timestamps():
    vtt = "WEBVTT\n\n00:01\n00:00:01.000 --> 00:00:03.000\nHello world\nHello world\n<c>Bonjour</c>"
    cleaned = _clean_vtt(vtt)
    assert "Hello world" in cleaned
    assert "00:00:01" not in cleaned        # timestamps retirés
    assert "WEBVTT" not in cleaned
    assert "Bonjour" in cleaned             # balises <c> retirées, texte gardé


def test_clean_vtt_dedup():
    vtt = "00:00:01.000 --> 00:00:02.000\nmot\n00:00:02.000 --> 00:00:03.000\nmot\nautre"
    cleaned = _clean_vtt(vtt)
    # "mot" apparaît 2× en VTT mais 1× après dédup
    assert cleaned.count("mot") == 1
    assert "autre" in cleaned


def test_fetch_transcript_rejects_non_youtube():
    """SÉCURITÉ : fetch_transcript refuse les URLs non-YouTube."""
    assert fetch_transcript("https://example.com") is None
    assert fetch_transcript("file:///etc/passwd") is None


@pytest.mark.skipif(not available(), reason="yt-dlp absent")
def test_fetch_transcript_runs_on_real_video():
    """Best-effort : essaie une vidéo réelle (peut retourner None si pas de sous-titres)."""
    tr = fetch_transcript("https://www.youtube.com/watch?v=jNQXAC9IVRw")  # "Me at the zoo"
    # accepte None (sous-titres absents) ou un texte non vide
    assert tr is None or len(tr) > 0


def test_learn_from_youtube_safe_on_bad_url():
    from ocm26400.document_learner import DocumentLearner
    dl = DocumentLearner()
    r = learn_from_youtube("https://example.com", dl)
    assert r["ok"] is False     # URL non-YouTube → échec propre (pas d'exception)
