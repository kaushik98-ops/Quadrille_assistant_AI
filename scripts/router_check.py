"""Live check of the router on a few labeled messages. Needs only your LLM key.

  python scripts/router_check.py
"""
import asyncio

from assistant.core.config import settings
from assistant.core.llm.base import Message
from assistant.core.llm.factory import get_llm_client
from assistant.core.logging import setup_logging
from assistant.router.router import Router

CASES = [
    ("Explain what an embedding is", "chat"),
    ("Write a short thank-you note to my landlord", "chat"),
    ("Thanks, that was helpful!", "chat"),
    ("What does my lease say about early termination?", "rag"),
    ("According to my documents, what is the notice period?", "rag"),
    ("Summarize the PDF I uploaded about my insurance", "rag"),
    ("Read goals.txt and tell me what's in it", "agent"),
    ("Save a note titled Ideas with the text 'try MCP'", "agent"),
    ("Search the web for the latest Python release", "agent"),
    ("Find my notes about Project X and summarize them", "agent"),
]

FOLLOW_UP_HISTORY = [
    Message("user", "What does my lease say about early termination?"),
    Message("assistant", "It requires sixty days written notice [1]."),
]


async def main() -> None:
    setup_logging("WARNING")
    router = Router(
        get_llm_client(),
        min_confidence=settings.router_min_confidence,
        history_messages=settings.router_history_messages,
    )
    correct = 0
    for text, expected in CASES:
        d = await router.decide(text, [])
        ok = d.route.value == expected
        correct += ok
        print(
            f"{'PASS' if ok else 'FAIL'}  expected={expected:5} got={d.route.value:5} "
            f"conf={d.confidence:.2f} ({d.reason})  {text}"
        )
    print(f"\n{correct}/{len(CASES)} correct")

    d = await router.decide("What about the security deposit?", FOLLOW_UP_HISTORY)
    print(f"\nFollow-up -> route={d.route.value} conf={d.confidence:.2f}\n  rewritten: {d.query}")


asyncio.run(main())
