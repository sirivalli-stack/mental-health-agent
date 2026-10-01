"""FastAPI application entry point.

Run from the `backend/` directory:

    uvicorn main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import chat, health
from app.config.logging_config import configure_logging, get_logger
from app.config.settings import get_settings
from app.models.schemas import ErrorResponse

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001, ANN201
    configure_logging()
    logger.info(
        "Starting %s v%s | environment=%s | system_profile=%s",
        settings.app_name,
        settings.app_version,
        settings.environment,
        settings.system_profile,
    )
    logger.warning("Research prototype - %s", settings.safety_disclaimer)
    yield
    # Phase 15: release database handles ourselves (SQLite memory backend +
    # turn audit) instead of leaving them to the OS at process exit.
    from app.pipelines.turn_log import reset_turn_log
    from app.state_engine import close_state_engine

    close_state_engine()
    reset_turn_log()
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-driven, state-aware conversational support research prototype. "
        "Not a medical diagnostic system and not a replacement for a "
        "mental-health professional."
    ),
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(chat.router)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "system_profile": settings.system_profile,
        "docs": "/docs",
        "disclaimer": settings.safety_disclaimer,
    }


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:  # noqa: ANN001
    # pydantic puts raw exceptions into err["ctx"] (e.g. ValueError from a
    # field validator), which json.dumps cannot encode - keep only safe keys.
    detail = [
        {
            "type": err.get("type"),
            "loc": list(err.get("loc", ())),
            "msg": str(err.get("msg", "")),
        }
        for err in exc.errors()
    ]
    logger.info("Validation error on %s: %s", request.url.path, detail)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="validation_error", detail=detail).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: ANN001
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error", detail="Unexpected server error."
        ).model_dump(),
    )
