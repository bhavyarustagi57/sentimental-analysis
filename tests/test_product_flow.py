"""Offline tests for the optional product-report workflow (no Groq key / network needed)."""

from __future__ import annotations

import csv
import io

import pytest
from fastapi.testclient import TestClient

import main
from json_synthesizer import _select_for_prompt, friendly_groq_error


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "_groq_key", lambda: "")
    with TestClient(main.app) as test_client:
        yield test_client


def _upload(client, name: str, data: bytes):
    return client.post("/upload-csv", files={"file": (name, data, "text/csv")})


def test_sample_csv_is_a_valid_wide_file(client):
    response = client.get("/sample-csv")
    assert response.status_code == 200
    rows = list(csv.reader(io.StringIO(response.text)))
    assert len(rows[0]) == 3 and all(len(row) == 3 for row in rows)


def test_upload_rejects_bad_extension_and_empty_file(client):
    assert _upload(client, "reviews.txt", b"a,b\n1,2").status_code == 400
    assert _upload(client, "reviews.csv", b"").status_code == 422


def test_upload_requires_target_reviews_and_headers(client):
    assert _upload(client, "r.csv", b"Phone,Other\nonly one review,x\n").status_code == 422  # < 3 target reviews
    assert _upload(client, "r.csv", b",Other\nabc review text,def\nfoo bar baz,x\nlong text here,y\n").status_code == 422


def test_valid_upload_without_key_gets_clear_503(client):
    response = _upload(client, "r.csv", main._sample_csv_text().encode("utf-8"))
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_latin1_csv_is_accepted_by_the_parser():
    table = main._read_uploaded_table("Phone\ncafé is great\n".encode("cp1252"), ".csv")
    assert list(table.columns) == ["Phone"]


def test_build_uploaded_reviews_drops_empty_competitors_and_caps_rows():
    import pandas as pd
    frame = pd.DataFrame({"Target": [f"review number {i}" for i in range(10)], "Empty": [None] * 10})
    built = main._build_uploaded_reviews(["Target", "Empty"], frame, max_per_column=4)
    assert len(built["target"]["posts"]) == 4
    assert built["competitors"] == {}


def test_history_item_404_and_health_flag(client):
    assert client.get("/history/does-not-exist").status_code == 404
    assert client.get("/health").json()["groq_configured"] is False


def test_prompt_sample_is_bounded_per_product():
    data = [{"product": "A", "is_competitor": False, "text": "x" * 50, "sentiment": "POSITIVE"} for _ in range(100)]
    data += [{"product": "B", "is_competitor": True, "text": "y" * 50, "sentiment": "NEGATIVE"} for _ in range(100)]
    chosen = _select_for_prompt(data, {"synthesis_posts_target": 30, "synthesis_posts_per_competitor": 12})
    assert sum(1 for item in chosen if item["product"] == "A") == 30
    assert sum(1 for item in chosen if item["product"] == "B") == 12


def test_friendly_groq_errors():
    import groq
    import httpx
    request = httpx.Request("POST", "https://api.groq.com/x")
    unauthorized = groq.AuthenticationError("bad", response=httpx.Response(401, request=request), body=None)
    limited = groq.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)
    assert "API key" in friendly_groq_error(unauthorized)
    assert "rate limit" in friendly_groq_error(limited).lower()
    assert "internet" in friendly_groq_error(groq.APIConnectionError(request=request)).lower()


def test_collect_posts_terminates_when_reddit_keeps_returning_the_same_posts(monkeypatch):
    try:
        import reddit_scraper
    except Exception as exc:  # NLTK data unavailable offline
        pytest.skip(f"reddit_scraper needs NLTK stopwords: {exc}")
    calls = {"count": 0}

    def fake_fetch(**_kwargs):
        calls["count"] += 1
        return [{"text": "same post", "utc": 0.0}, {"text": "another post", "utc": 0.0}]

    monkeypatch.setattr(reddit_scraper, "_fetch_reddit_posts", fake_fetch)
    monkeypatch.setattr(reddit_scraper.time, "sleep", lambda _seconds: None)
    posts = reddit_scraper._collect_posts("anything", 70, "agent")
    assert len(posts) == 2
    assert calls["count"] <= 3  # previously this looped forever
