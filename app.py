import logging
import os
import traceback
import uuid
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from analyzer import BookOfBusinessAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("book_of_business")

# Set DEBUG=1 in the environment to include server tracebacks in API error responses.
# Leave unset in production so internal stack traces aren't exposed to clients.
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

app = FastAPI(title="Book of Business Intelligent Analyzer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SESSION_COOKIE_NAME = "bob_session_id"
MAX_SESSIONS = 100  # simple cap so a long-running server doesn't grow unbounded

# Two independent per-session slots: the primary uploaded file, and an optional
# comparison snapshot (a second file) used for period-over-period comparisons.
PRIMARY_CACHE: dict[str, BookOfBusinessAnalyzer] = {}
COMPARE_CACHE: dict[str, BookOfBusinessAnalyzer] = {}


# --------------------------------------------------------------------------- #
# Request schemas
# --------------------------------------------------------------------------- #

class ProfitCenterRequest(BaseModel):
    profit_center_column: Optional[str] = None


class AgencyCodeRequest(BaseModel):
    agency_code_column: Optional[str] = None


class ColumnValuesRequest(BaseModel):
    column: Optional[str] = None


class DateRangeRequest(BaseModel):
    timeline_column: Optional[str] = None


class GoalConfig(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = "Goal"
    period: str = "annual"
    scope_type: str = "overall"
    scope_value: Optional[str] = None
    target_value: float = 0.0


class AnalyzeRequest(BaseModel):
    mapping: dict
    profit_center: str = "ALL"
    projection_target: str = "premium"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_future_dates: bool = False
    selected_agency_codes: list = Field(default_factory=list)
    goal_value: float = 0.0
    goals: list[GoalConfig] = Field(default_factory=list)
    business_view: str = "all"
    target: str = "primary"  # "primary" or "compare" — which uploaded file to analyze


class SuggestGoalsRequest(BaseModel):
    mapping: dict
    projection_target: str = "premium"
    period: str = "annual"
    business_view: str = "all"
    top_n: int = 3


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #

def get_or_create_session_id(request: Request, response: Response) -> str:
    """
    Each browser gets its own session id (a cookie), so concurrent users don't
    clobber each other's uploaded file. Sessions expire after 24h of inactivity.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)

    if not session_id:
        session_id = uuid.uuid4().hex

    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24
    )

    return session_id


def _store_in_cache(cache: dict, session_id: str, analyzer: BookOfBusinessAnalyzer) -> None:
    """Insert into a session cache, evicting the oldest entry once the cap is hit."""
    if session_id not in cache and len(cache) >= MAX_SESSIONS:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)

    cache[session_id] = analyzer


def get_active_analyzer(session_id: str, target: str = "primary") -> BookOfBusinessAnalyzer:
    """Fetch the analyzer for this session/slot, or raise a clear 400 error."""
    cache = COMPARE_CACHE if target == "compare" else PRIMARY_CACHE
    analyzer = cache.get(session_id)

    if not analyzer:
        label = "comparison" if target == "compare" else "primary"
        raise HTTPException(
            status_code=400,
            detail=f"No active {label} data file found for this session. Upload a file first."
        )

    return analyzer


def error_response(exc: Exception, status_code: int = 500) -> JSONResponse:
    """Log the full traceback server-side and return a clean, bounded error to the client."""
    logger.error("Request failed: %s", exc, exc_info=True)

    message = str(exc) or exc.__class__.__name__

    if DEBUG_MODE:
        message = f"{message}\n\nTraceback:\n{traceback.format_exc()}"

    return JSONResponse(status_code=status_code, content={"error": message})


def _model_to_dict(model: BaseModel) -> dict:
    """Support both pydantic v2 (model_dump) and v1 (dict) without pinning a version."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/api/health")
async def health_check(request: Request, response: Response):
    session_id = get_or_create_session_id(request, response)

    return JSONResponse(content={
        "status": "ok",
        "service": "Book of Business Intelligent Analyzer",
        "primary_active": session_id in PRIMARY_CACHE,
        "compare_active": session_id in COMPARE_CACHE
    })


@app.post("/api/upload")
async def upload_file(request: Request, response: Response, file: UploadFile = File(...)):
    try:
        session_id = get_or_create_session_id(request, response)
        contents = await file.read()

        if not contents:
            raise ValueError("The uploaded file is empty.")

        analyzer = BookOfBusinessAnalyzer(contents, file.filename)
        schema = analyzer.infer_schema()

        _store_in_cache(PRIMARY_CACHE, session_id, analyzer)

        # A fresh primary upload invalidates any prior comparison snapshot for this session,
        # since it was likely set up to compare against the previous primary file.
        COMPARE_CACHE.pop(session_id, None)

        return JSONResponse(content=schema)

    except Exception as e:
        return error_response(e, status_code=400)


@app.post("/api/compare-upload")
async def upload_compare_file(request: Request, response: Response, file: UploadFile = File(...)):
    """Upload a second file (e.g. last month's export) to compare against the primary one."""
    try:
        session_id = get_or_create_session_id(request, response)
        contents = await file.read()

        if not contents:
            raise ValueError("The uploaded comparison file is empty.")

        analyzer = BookOfBusinessAnalyzer(contents, file.filename)
        schema = analyzer.infer_schema()

        _store_in_cache(COMPARE_CACHE, session_id, analyzer)

        return JSONResponse(content=schema)

    except Exception as e:
        return error_response(e, status_code=400)


@app.post("/api/profit-centers")
async def refresh_profit_centers(body: ProfitCenterRequest, request: Request, response: Response):
    try:
        session_id = get_or_create_session_id(request, response)
        analyzer = get_active_analyzer(session_id)

        return JSONResponse(content={
            "profit_centers": analyzer.get_profit_centers(body.profit_center_column)
        })

    except HTTPException:
        raise
    except Exception as e:
        return error_response(e)


@app.post("/api/agency-codes")
async def refresh_agency_codes(body: AgencyCodeRequest, request: Request, response: Response):
    try:
        session_id = get_or_create_session_id(request, response)
        analyzer = get_active_analyzer(session_id)

        return JSONResponse(content={
            "agency_codes": analyzer.get_agency_codes(body.agency_code_column)
        })

    except HTTPException:
        raise
    except Exception as e:
        return error_response(e)


@app.post("/api/column-values")
async def column_values(body: ColumnValuesRequest, request: Request, response: Response):
    try:
        session_id = get_or_create_session_id(request, response)
        analyzer = get_active_analyzer(session_id)

        return JSONResponse(content={
            "values": analyzer.get_unique_column_values(body.column)
        })

    except HTTPException:
        raise
    except Exception as e:
        return error_response(e)


@app.post("/api/date-range")
async def refresh_date_range(body: DateRangeRequest, request: Request, response: Response):
    try:
        session_id = get_or_create_session_id(request, response)
        analyzer = get_active_analyzer(session_id)

        return JSONResponse(content=analyzer.get_date_range(body.timeline_column))

    except HTTPException:
        raise
    except Exception as e:
        return error_response(e)


@app.post("/api/analyze")
async def analyze_data(body: AnalyzeRequest, request: Request, response: Response):
    try:
        session_id = get_or_create_session_id(request, response)
        target = "compare" if body.target == "compare" else "primary"
        analyzer = get_active_analyzer(session_id, target=target)

        results = analyzer.run_analysis(
            mapping=body.mapping,
            selected_profit_center=body.profit_center,
            projection_target=body.projection_target,
            start_date=body.start_date,
            end_date=body.end_date,
            include_future_dates=body.include_future_dates,
            selected_agency_codes=body.selected_agency_codes,
            goal_value=body.goal_value,
            goals=[_model_to_dict(g) for g in body.goals],
            business_view=body.business_view
        )

        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        return error_response(e)


@app.post("/api/suggest-goals")
async def suggest_goals(body: SuggestGoalsRequest, request: Request, response: Response):
    try:
        session_id = get_or_create_session_id(request, response)
        analyzer = get_active_analyzer(session_id)

        suggestions = analyzer.suggest_goal_candidates(
            mapping=body.mapping,
            projection_target=body.projection_target,
            period=body.period,
            business_view=body.business_view,
            top_n=body.top_n
        )

        return JSONResponse(content={"suggestions": suggestions})

    except HTTPException:
        raise
    except Exception as e:
        return error_response(e)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Template index.html missing from repository layout structure."
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
