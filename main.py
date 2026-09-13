"""FastAPI entry point for the frozen comparative sentiment-analysis project.

It serves the read-only academic ML dashboard and saved-model prediction API,
plus an optional independent VADER/Groq product-review workflow.
"""

from __future__ import annotations

import io
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from data_manager import DataStore
from ml.config import DATASETS, MODEL_NAMES
from ml.dataset_loader import describe_all_datasets
from ml.predictor import predict_sentiment
from ml.results_analysis import analyze_results
from ml.trainers import describe_models
from ml.utils import ResultStore


BASE_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = BASE_DIR / "settings.json"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
load_dotenv(BASE_DIR / ".env")

data_store = DataStore()
_status_lock = threading.RLock()
_pipeline_status: dict[str, Any] = {
    "status": "idle", "product": None, "source": None, "stage": "",
    "message": "No analysis has been started.", "started_at": None,
    "completed_at": None, "session_id": None,
}
_ml_status: dict[str, Any] = {
    "status": "frozen", "scope": None,
    "message": "Canonical ML results are read-only in the web application.",
    "started_at": None, "completed_at": None, "completed": 24, "failures": [],
}
ml_results = ResultStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    data_store.initialize()
    yield


app = FastAPI(
    title="Comparative Sentiment Analysis API",
    description="Frozen academic ML results and saved-model sentiment prediction.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    product: str = Field(min_length=2, max_length=120)


class MLPredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    dataset: str
    model: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_settings() -> dict[str, Any]:
    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise RuntimeError("settings.json is missing.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("settings.json contains invalid JSON.") from exc


def _require_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured. Copy .env.example to .env and add a valid key before starting an analysis.",
        )
    return key


def _status_snapshot() -> dict[str, Any]:
    with _status_lock:
        return dict(_pipeline_status)


def _set_status(status: Literal["idle", "running", "done", "error"], stage: str, message: str, *, product: str | None = None, source: str | None = None, session_id: str | None = None, completed: bool = False) -> None:
    with _status_lock:
        _pipeline_status["status"] = status
        _pipeline_status["stage"] = stage
        _pipeline_status["message"] = message
        if product is not None:
            _pipeline_status["product"] = product
        if source is not None:
            _pipeline_status["source"] = source
        if session_id is not None:
            _pipeline_status["session_id"] = session_id
        if completed:
            _pipeline_status["completed_at"] = _utc_now()


def _begin_analysis(product: str, source: str) -> str:
    global _pipeline_status
    with _status_lock:
        if _pipeline_status["status"] == "running":
            raise HTTPException(status_code=409, detail=f"An analysis for '{_pipeline_status['product']}' is already running.")
        session_id = uuid.uuid4().hex
        _pipeline_status = {
            "status": "running", "product": product, "source": source,
            "stage": "queued", "message": "Analysis is queued.",
            "started_at": _utc_now(), "completed_at": None, "session_id": session_id,
        }
        return session_id


def _build_uploaded_reviews(columns: list[str], dataframe: Any) -> dict[str, Any]:
    """Convert the supported wide spreadsheet format into the scraper data shape."""
    target_posts: list[dict[str, Any]] = []
    competitors: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for index, column in enumerate(columns):
        reviews = dataframe.iloc[:, index].dropna().tolist()
        posts = [{"text": str(review).strip(), "utc": 0.0} for review in reviews if str(review).strip()]
        if index == 0:
            target_posts = posts
        else:
            competitors[column] = {"posts": posts}
    if not target_posts:
        raise HTTPException(status_code=422, detail="The first column (target product) contains no review text.")
    return {"target": {"name": columns[0], "posts": target_posts}, "competitors": competitors}


def _run_pipeline(product: str, source: Literal["reddit", "upload"], session_id: str, raw_reviews: dict[str, Any] | None = None) -> None:
    """Run the full analysis in a background task; expose failures through /status."""
    try:
        # Deferred imports let /health start without attempting an NLTK download.
        from groq import Groq
        from aso_router import route
        from json_synthesizer import synthesize
        from preprocessor import preprocess
        from reddit_scraper import scrape

        settings = _load_settings()
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        if source == "reddit":
            _set_status("running", "collecting", "Discovering competitors and collecting relevant Reddit posts…")
            raw_reviews = scrape(product, settings, client)
        else:
            _set_status("running", "preprocessing", "Cleaning uploaded reviews…")

        if raw_reviews is None:
            raise RuntimeError("No reviews were available for analysis.")
        _set_status("running", "preprocessing", "Cleaning and deduplicating reviews…")
        cleaned_reviews = preprocess(raw_reviews)
        if not cleaned_reviews.get("target", {}).get("posts", []):
            raise RuntimeError("No usable target-product reviews remained after quality filtering.")

        _set_status("running", "sentiment", "Computing sentiment statistics…")
        tagged_reviews, statistics = route(cleaned_reviews, settings, client)
        if not tagged_reviews:
            raise RuntimeError("No usable reviews were available for sentiment analysis.")

        _set_status("running", "synthesis", "Generating the final analysis report with Groq…")
        payload = synthesize(tagged_reviews, product, settings, client).model_dump()
        payload.update(statistics)
        data_store.append_session({
            "id": session_id, "product": product, "source": source, "created_at": _utc_now(),
            "review_count": len(tagged_reviews), "payload": payload,
        })
        _set_status("done", "complete", f"Analysis complete: {len(tagged_reviews)} usable reviews processed.", completed=True)
    except Exception as exc:
        _set_status("error", "failed", f"Analysis failed: {str(exc) or type(exc).__name__}", completed=True)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok", "service": "Comparative Sentiment Analysis API", "version": "1.0.0",
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "stored_sessions": data_store.count(), "pipeline": _status_snapshot(),
        "ml_status": _ml_status["status"],
    }


@app.get("/status")
async def status() -> dict[str, Any]:
    return _status_snapshot()


@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    product = request.product.strip()
    if len(product) < 2:
        raise HTTPException(status_code=422, detail="Product name must contain at least two characters.")
    _require_groq_key()
    session_id = _begin_analysis(product, "reddit")
    background_tasks.add_task(_run_pipeline, product, "reddit", session_id)
    return {"status": "started", "session_id": session_id, "product": product, "message": "Analysis started. Poll /status for progress."}


@app.post("/upload-csv")
async def upload_reviews(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a .csv, .xlsx, or .xls file.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds the 5 MB prototype limit.")
    try:
        import pandas as pd
        buffer = io.BytesIO(contents)
        dataframe = pd.read_csv(buffer, dtype=str) if extension == ".csv" else pd.read_excel(buffer, dtype=str)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse the uploaded file: {exc}") from exc
    if dataframe.empty or len(dataframe.columns) == 0:
        raise HTTPException(status_code=422, detail="The file must include product-name headers and at least one review.")
    columns = [str(column).strip() for column in dataframe.columns]
    if not columns[0]:
        raise HTTPException(status_code=422, detail="The first column header (target product name) is empty.")
    if any(not column for column in columns):
        raise HTTPException(status_code=422, detail="All populated review columns need a product-name header.")
    if len(set(columns)) != len(columns):
        raise HTTPException(status_code=422, detail="Product column headers must be unique.")
    raw_reviews = _build_uploaded_reviews(columns, dataframe)
    _require_groq_key()
    session_id = _begin_analysis(columns[0], "upload")
    background_tasks.add_task(_run_pipeline, columns[0], "upload", session_id, raw_reviews)
    review_count = sum(len(item["posts"]) for item in [raw_reviews["target"], *raw_reviews["competitors"].values()])
    return {
        "status": "started", "session_id": session_id, "product": columns[0],
        "competitors": columns[1:], "review_count": review_count,
        "message": "Upload accepted and analysis started. Poll /status for progress.",
    }


@app.get("/latest")
async def latest() -> dict[str, Any]:
    session = data_store.get_latest()
    if session is None:
        raise HTTPException(status_code=404, detail="No completed analyses are stored yet.")
    return session


@app.get("/history")
async def history() -> dict[str, list[dict[str, Any]]]:
    sessions = data_store.get_all_sessions()
    return {"sessions": [{
        "id": session.get("id"), "product": session.get("product"), "source": session.get("source"),
        "created_at": session.get("created_at"), "review_count": session.get("review_count", 0),
        "score": session.get("payload", {}).get("overallScore"),
    } for session in reversed(sessions)]}


# Academic ML endpoints are offline and independent of Groq/VADER product analysis.
@app.get("/ml/datasets")
async def ml_datasets() -> dict[str, list[dict[str, Any]]]:
    return {"datasets": describe_all_datasets()}


@app.get("/ml/models")
async def ml_models() -> dict[str, list[dict[str, str]]]:
    return {"models": describe_models()}


@app.get("/ml/status")
async def ml_status() -> dict[str, Any]:
    return dict(_ml_status)


@app.get("/ml/results")
async def ml_result_list() -> dict[str, list[dict[str, Any]]]:
    try:
        return {"results": ml_results.read()}
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/ml/summary")
async def ml_summary() -> dict[str, Any]:
    """Read-only presentation summary derived from frozen experiment results."""
    try:
        return analyze_results(ml_results.read())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Canonical ML results are unavailable or invalid: {exc}") from exc


@app.post("/ml/train", status_code=403)
@app.post("/ml/train-all", status_code=403)
async def ml_training_disabled() -> None:
    """Keep the published academic baseline immutable through the web API."""
    raise HTTPException(
        status_code=403,
        detail="Training is disabled in the frozen showcase application. Use the documented CLI only when intentionally rebuilding the academic baseline.",
    )


@app.post("/ml/predict")
async def ml_predict(request: MLPredictRequest) -> dict[str, Any]:
    if request.dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{request.dataset}'.")
    if request.model not in MODEL_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown model '{request.model}'.")
    try:
        return predict_sentiment(request.text, request.dataset, request.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
