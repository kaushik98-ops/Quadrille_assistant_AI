"""Chat service: history, system prompt, streaming. Depends only on core."""
from collections.abc import AsyncIterator

from assistant.chat.prompts import build_system
from assistant.core.llm.base import LLMClient, Message
from assistant.core.logging import get_logger
from assistant.core.memory import ConversationMemory

log = get_logger(__name__)


class ChatService:
    def __init__(
        self,
        llm: LLMClient,
        memory: ConversationMemory,
        max_history_messages: int = 20,
        max_history_chars: int = 8000,
    ):
        self._llm = llm
        self._memory = memory
        self._max_messages = max_history_messages
        self._max_chars = max_history_chars

    def _window(self, history: list[Message]) -> list[Message]:
        """Keep recent messages within the budget; always start on a user message."""
        msgs = history[-self._max_messages:]
        total = sum(len(m.content) for m in msgs)
        while msgs and (total > self._max_chars or msgs[0].role != "user"):
            total -= len(msgs[0].content)
            msgs = msgs[1:]
        return msgs

    async def stream_reply(
        self,
        conversation_id: str,
        user_text: str,
        system_extra: str | None = None,
    ) -> AsyncIterator[str]:
        history = self._window(self._memory.history(conversation_id, limit=self._max_messages))
        messages = history + [Message("user", user_text)]
        system = build_system(system_extra)

        log.info(
            "chat_start",
            extra={"fields": {"conversation_id": conversation_id, "history_len": len(history)}},
        )

        chunks: list[str] = []
        async for chunk in self._llm.stream(messages, system=system):
            chunks.append(chunk)
            yield chunk

        # Reached only if streaming completed without error.
        reply = "".join(chunks)
        self._memory.add(conversation_id, "user", user_text)
        self._memory.add(conversation_id, "assistant", reply)
        log.info("chat_done", extra={"fields": {"reply_chars": len(reply)}})

    async def reply(
        self, conversation_id: str, user_text: str, system_extra: str | None = None
    ) -> str:
        """Non-streaming convenience: collects the stream into one string."""
        return "".join(
            [c async for c in self.stream_reply(conversation_id, user_text, system_extra)]
        )
