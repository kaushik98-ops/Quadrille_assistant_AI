"""Chat endpoints. Replies stream as server-sent events (SSE)."""
import json
from collections.abc import AsyncIterator
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StrictBool, field_validator

from assistant.api.deps import Principal, get_principal, scoped_conversation
from assistant.api.orchestrator import ConfirmEvent, Event, RouteEvent, SourcesEvent, TextEvent
from assistant.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
MAX_MESSAGE_CHARS = 4000


class ChatRequest(BaseModel):
    conversation_id: str = Field(pattern=ID_PATTERN)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class ConfirmRequest(BaseModel):
    conversation_id: str = Field(pattern=ID_PATTERN)
    run_id: str = Field(pattern=ID_PATTERN)
    approved: StrictBool  # strict: "false" or 0 must never be read as an answer


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _to_sse(event: Event) -> str:
    if isinstance(event, RouteEvent):
        return _sse("route", asdict(event))
    if isinstance(event, SourcesEvent):
        return _sse("sources", {"sources": [asdict(s) for s in event.sources]})
    if isinstance(event, TextEvent):
        return _sse("text", {"text": event.text})
    if isinstance(event, ConfirmEvent):
        return _sse("confirm", asdict(event))
    raise TypeError(f"Unknown event type: {type(event).__name__}")


async def _stream(events: AsyncIterator[Event]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield _to_sse(event)
        yield _sse("done", {})
    except Exception:
        # The details go to the log; the client only gets a generic message.
        log.exception("stream_error")
        yield _sse("error", {"message": "Something went wrong. Please try again."})


def _response(events: AsyncIterator[Event]) -> StreamingResponse:
    return StreamingResponse(
        _stream(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat")
async def chat(body: ChatRequest, request: Request, principal: Principal = Depends(get_principal)):
    orchestrator = request.app.state.services.orchestrator
    conversation_id = scoped_conversation(principal, body.conversation_id)
    return _response(orchestrator.handle(conversation_id, body.message))


@router.post("/chat/confirm")
async def confirm(body: ConfirmRequest, request: Request, principal: Principal = Depends(get_principal)):
    orchestrator = request.app.state.services.orchestrator
    conversation_id = scoped_conversation(principal, body.conversation_id)
    if not orchestrator.has_pending(conversation_id, body.run_id):
        raise HTTPException(status_code=404, detail="No pending action for this conversation and run.")
    return _response(orchestrator.resume_agent(conversation_id, body.run_id, body.approved))
