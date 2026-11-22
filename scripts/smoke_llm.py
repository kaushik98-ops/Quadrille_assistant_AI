"""Live check: sends one prompt to your configured provider, streaming the reply."""
import asyncio

from assistant.core.config import settings
from assistant.core.llm.base import Message
from assistant.core.llm.factory import get_llm_client
from assistant.core.logging import get_logger, new_request_id, setup_logging


async def main() -> None:
    setup_logging(settings.log_level)
    log = get_logger("smoke")
    new_request_id()
    log.info("starting", extra={"fields": {"provider": settings.llm_provider}})

    llm = get_llm_client()
    messages = [Message("user", "Say hello in exactly five words.")]

    print("generate:", await llm.generate(messages, system="Be brief."))
    print("stream: ", end="", flush=True)
    async for chunk in llm.stream(messages, system="Be brief."):
        print(chunk, end="", flush=True)
    print()


asyncio.run(main())
