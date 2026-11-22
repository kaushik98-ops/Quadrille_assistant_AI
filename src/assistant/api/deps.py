"""Request dependencies. Authentication itself happens in the gate middleware."""
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class Principal:
    key_id: str


def get_principal(request: Request) -> Principal:
    key_id = getattr(request.state, "key_id", None)  # set by GateMiddleware
    if not key_id:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return Principal(key_id)


def scoped_conversation(principal: Principal, conversation_id: str) -> str:
    """Namespace conversation ids by API key, so keys can never see each other's history."""
    return f"{principal.key_id}:{conversation_id}"
