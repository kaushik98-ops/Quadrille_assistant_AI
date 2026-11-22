"""Standalone terminal chat. Proves the chat service works without the API."""
import asyncio
import sys

from assistant.chat.service import ChatService
from assistant.core.config import settings
from assistant.core.llm.factory import get_llm_client
from assistant.core.logging import new_request_id, setup_logging
from assistant.core.memory import ConversationMemory


async def main(conversation_id: str) -> None:
    setup_logging("WARNING")  # keep the terminal readable
    service = ChatService(get_llm_client(), ConversationMemory(settings.db_path))
    print(f"Chatting in conversation '{conversation_id}'. Ctrl+C to quit.\n")

    while True:
        try:
            text = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not text:
            continue
        new_request_id()
        print("bot> ", end="", flush=True)
        async for chunk in service.stream_reply(conversation_id, text):
            print(chunk, end="", flush=True)
        print("\n")


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "cli"))
