import json
import time

import groq
from groq import Groq
from pydantic import BaseModel, ValidationError, field_validator, Field
from typing import Optional

from rate_limiter import groq_rate_limiter, model_extras
from templates import PROMPTS


def _sleep(seconds: int) -> None:
    time.sleep(seconds)

class KeywordItem(BaseModel):
    word: str
    size: str        # e.g. "2.25rem"
    color: str       # hex
    weight: int      # font-weight integer e.g. 700


class DemographicItem(BaseModel):
    """User demographic segment."""
    label: str       # e.g. "Gen Z (18-24)"
    pct: int         # integer percentage 0-100
    color: str       # hex


class ClusterItem(BaseModel):
    """Community bubble cluster."""
    label: str       # e.g. "Tech"
    users: str       # e.g. "12.4k"
    size: int        # pixel size of bubble
    color: str       # rgba background
    border: str      # rgba border
    textColor: str   # hex text color
    top: str         # CSS top offset e.g. "10%"
    left: str        # CSS left offset e.g. "20%"


class SentimentBrand(BaseModel):
    """Comparative brand sentiment entry for Market Pulse."""
    name: str
    score: float     # 0.0 - 10.0
    color: str       # hex or named color
    pct: int         # integer percentage 0-100


class TrendingTopic(BaseModel):
    """Trending market topic card."""
    badge: str        # e.g. "High Velocity"
    badgeColor: str   # hex
    title: str
    desc: str
    positive: str     # e.g. "84% Positive"
    icon: str         # Google Material Symbols icon name
    iconColor: str    # hex
    discussions: str  # e.g. "2.4k discussions"
    bg: str           # CSS gradient string


# ─── Statistics Models (Computed in Backend) ──────────────────────────────────

class TimelineItem(BaseModel):
    name: str
    positive: int
    neutral: int
    negative: int

class DonutItem(BaseModel):
    name: str
    value: int
    color: str

class BarChartItem(BaseModel):
    name: str
    value: int
    color: str

class MarketShareItem(BaseModel):
    name: str
    size: int


class PerformanceMetrics(BaseModel):
    """Aggregate analysis performance statistics."""
    totalMentions: int
    avgResponseTime: str    # e.g. "2.4 hours"
    analysisAccuracy: int   # integer 0-100


class LuminaPayload(BaseModel):
    """
    Root model for the complete UI-ready analysis payload.
    Matches the data contract expected by the React frontend.
    """
    brandName: str
    overallScore: float           # 0.0 - 10.0

    keywords: list[KeywordItem]
    demographics: list[DemographicItem]
    clusters: list[ClusterItem]
    platformPosts: list[str]

    sentimentBrands: list[SentimentBrand] = Field(default_factory=list)
    trendingTopics: list[TrendingTopic]
    performanceMetrics: PerformanceMetrics

    # Fields computed purely by stats backend (LLM may ignore)
    donut: list[DonutItem] = Field(default_factory=list)
    timeline: list[TimelineItem] = Field(default_factory=list)
    barChart: list[BarChartItem] = Field(default_factory=list)
    marketShare: list[MarketShareItem] = Field(default_factory=list)

    @field_validator("overallScore")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Ensure score stays within 0.0 - 10.0 regardless of LLM output."""
        return max(0.0, min(10.0, v))

    @field_validator("demographics")
    @classmethod
    def clamp_demographics_pct(cls, items: list[DemographicItem]) -> list[DemographicItem]:
        """Clamp individual demographic percentages to 0-100."""
        for item in items:
            item.pct = max(0, min(100, item.pct))
        return items


# ─── Prompt Assembly ──────────────────────────────────────────────────────────

def _select_for_prompt(sentiment_data: list[dict], settings: dict, shrink: float = 1.0) -> list[dict]:
    """
    Pick a bounded, balanced sample of tagged posts for the LLM prompt.

    Statistics (VADER) are computed on ALL posts; the LLM only needs a sample.
    Sending everything easily exceeds Groq's free-tier tokens-per-minute limit.
    """
    target_cap = max(4, int(settings.get("synthesis_posts_target", 30) * shrink))
    comp_cap = max(3, int(settings.get("synthesis_posts_per_competitor", 12) * shrink))
    taken: dict[str, int] = {}
    selected: list[dict] = []
    for item in sentiment_data:
        cap = comp_cap if item["is_competitor"] else target_cap
        if taken.get(item["product"], 0) >= cap:
            continue
        taken[item["product"]] = taken.get(item["product"], 0) + 1
        selected.append(item)
    return selected


def _build_synthesis_prompt(
    sentiment_data: list[dict],
    product_name: str,
    settings: dict,
    shrink: float = 1.0,
) -> str:
    """
    Assemble the full JSON synthesis prompt by injecting runtime values
    into the PROMPTS['json_payload_construction'] template.
    """
    colors = settings["design_colors"]
    cluster_labels = ", ".join(settings["community_clusters"])
    max_chars = max(80, int(settings.get("synthesis_chars_per_post", 180) * shrink))

    # Compact summary line: [PRODUCT | TARGET/COMPETITOR | SENTIMENT]: text
    summary_lines: list[str] = []
    for item in _select_for_prompt(sentiment_data, settings, shrink):
        role = "COMPETITOR" if item["is_competitor"] else "TARGET"
        summary_lines.append(
            f"[{item['product']} | {role} | {item['sentiment']}]: {item['text'][:max_chars]}"
        )

    return PROMPTS["json_payload_construction"]["template"].format(
        product_name=product_name,
        color_primary=colors["primary"],
        color_positive=colors["positive"],
        color_negative=colors["negative"],
        color_neutral=colors["neutral"],
        cluster_labels=cluster_labels,
        sentiment_data="\n".join(summary_lines),
    )


class SynthesisError(RuntimeError):
    """Raised with a user-readable message when the Groq report cannot be produced."""


def friendly_groq_error(exc: Exception) -> str:
    """Translate Groq/network exceptions into one clear sentence for the UI."""
    if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
        return ("Groq rejected the API key. Check GROQ_API_KEY in .env (it should start with 'gsk_') and try again.")
    if isinstance(exc, groq.NotFoundError):
        return ("The configured Groq model was not found. Check model_heavy / model_light in settings.json.")
    if isinstance(exc, groq.RateLimitError):
        return ("Groq's free-tier rate limit was reached. Wait about a minute and try again.")
    if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
        return "Could not reach Groq. Check your internet connection and try again."
    if isinstance(exc, groq.APIStatusError):
        return f"Groq returned an error (HTTP {exc.status_code}). Try again in a moment."
    return str(exc) or type(exc).__name__


def _call_groq_heavy(
    prompt: str,
    groq_client: Groq,
    model_heavy: str,
    token_limit: int,
) -> str:

    groq_rate_limiter.wait()  # ← enforce RPM cap before heavy model call
    response = groq_client.chat.completions.create(
        model=model_heavy,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Ethereal Analyst. "
                    "Output ONLY valid JSON. No markdown. No explanation text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=token_limit,
        temperature=0.4,
        **model_extras(model_heavy),
    )
    return response.choices[0].message.content.strip()


def _strip_markdown_fences(raw: str) -> str:
    """Remove accidental markdown code fences the LLM may wrap JSON in."""
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] contains the content between first pair of fences
        inner = parts[1] if len(parts) > 1 else raw
        if inner.startswith("json"):
            inner = inner[4:].strip()
        return inner.strip()
    return raw



def synthesize(
    sentiment_data: list[dict],
    product_name: str,
    settings: dict,
    groq_client: Groq,
) -> LuminaPayload:

    model_heavy: str = settings["model_heavy"]
    token_limit: int = settings["token_limit_heavy"]
    sleep_seconds: int = settings["rate_limit_sleep_seconds"]
    max_attempts = 3

    shrink = 1.0
    extra_instructions = ""
    last_problem = "unknown error"

    for attempt in range(1, max_attempts + 1):
        prompt = _build_synthesis_prompt(sentiment_data, product_name, settings, shrink) + extra_instructions
        print(f"[json_synthesizer] Synthesis attempt {attempt}/{max_attempts} using '{model_heavy}'...")
        try:
            raw = _call_groq_heavy(prompt, groq_client, model_heavy, token_limit)
            _sleep(sleep_seconds)
            parsed = json.loads(_strip_markdown_fences(raw))
            payload = LuminaPayload(**parsed)
            print(f"[json_synthesizer] Payload validated on attempt {attempt}. Score: {payload.overallScore}/10")
            return payload

        except (groq.AuthenticationError, groq.PermissionDeniedError, groq.NotFoundError) as e:
            # Retrying cannot fix a bad key or a missing model.
            raise SynthesisError(friendly_groq_error(e)) from e

        except groq.RateLimitError as e:
            last_problem = friendly_groq_error(e)
            print(f"[json_synthesizer] Rate limited (attempt {attempt}); waiting then sending a smaller prompt.")
            shrink *= 0.5
            if attempt < max_attempts:
                _sleep(25)

        except groq.APIStatusError as e:
            last_problem = friendly_groq_error(e)
            if e.status_code == 413:  # request too large for the model's token limit
                shrink *= 0.5
            print(f"[json_synthesizer] Groq error {e.status_code} (attempt {attempt}).")
            if attempt < max_attempts:
                _sleep(sleep_seconds)

        except (groq.APIConnectionError, groq.APITimeoutError) as e:
            last_problem = friendly_groq_error(e)
            if attempt < max_attempts:
                _sleep(sleep_seconds)

        except json.JSONDecodeError as e:
            last_problem = "Groq returned text that was not valid JSON."
            print(f"[json_synthesizer] Attempt {attempt}: JSON parse error: {e}")
            extra_instructions = (
                f"\n\nPREVIOUS ATTEMPT FAILED - JSON PARSE ERROR:\n{str(e)[:300]}\n"
                "Fix the JSON syntax and output ONLY valid, parseable JSON."
            )
            shrink *= 0.8  # a shorter prompt leaves more room for the JSON answer
            if attempt < max_attempts:
                _sleep(sleep_seconds)

        except ValidationError as e:
            last_problem = "Groq's report was missing required fields."
            print(f"[json_synthesizer] Attempt {attempt}: schema validation error:\n{e}")
            extra_instructions = (
                f"\n\nPREVIOUS ATTEMPT FAILED - SCHEMA VALIDATION ERROR:\n{str(e)[:300]}\n"
                "Ensure ALL required fields are present with the correct types."
            )
            if attempt < max_attempts:
                _sleep(sleep_seconds)

    raise SynthesisError(f"Could not generate the report after {max_attempts} attempts: {last_problem}")
