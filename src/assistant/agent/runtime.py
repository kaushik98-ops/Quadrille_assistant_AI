"""The agent's public interface: start a task, then resume it if it needs approval."""
from langgraph.types import Command

from assistant.agent.graph import build_graph
from assistant.agent.types import AgentEvent, ToolProvider
from assistant.core.errors import AssistantError
from assistant.core.llm.base import LLMClient
from assistant.core.logging import get_logger

log = get_logger(__name__)


class AgentRuntime:
    def __init__(self, graph, max_steps: int):
        self._graph = graph
        self._max_steps = max_steps

    @classmethod
    async def create(
        cls,
        llm: LLMClient,
        tools: ToolProvider,
        max_steps: int = 6,
        max_observation_chars: int = 4000,
    ) -> "AgentRuntime":
        infos = await tools.list_tools()  # read once; restart to pick up new tools
        if not infos:
            raise AssistantError("The MCP server exposed no tools.")
        return cls(build_graph(llm, tools, infos, max_steps, max_observation_chars), max_steps)

    def _config(self, run_id: str) -> dict:
        # Each loop is 3 graph steps, so raise LangGraph's default limit of 25 as needed.
        return {"configurable": {"thread_id": run_id}, "recursion_limit": self._max_steps * 3 + 5}

    async def start(self, run_id: str, task: str) -> AgentEvent:
        """Begin a task. Use a fresh run_id for every task."""
        log.info("agent_run_start", extra={"fields": {"run_id": run_id, "task_chars": len(task)}})
        state = {"task": task, "scratch": [], "pending": None, "approved": False,
                 "tainted": False, "answer": None, "steps": 0}
        await self._graph.ainvoke(state, self._config(run_id))
        return await self._event(run_id)

    async def resume(self, run_id: str, approved: bool) -> AgentEvent:
        """Answer a pending confirmation. Approval applies to the exact call that was shown."""
        snapshot = await self._graph.aget_state(self._config(run_id))
        if not snapshot.next:
            raise ValueError(f"Run '{run_id}' is not waiting for confirmation.")
        await self._graph.ainvoke(Command(resume=approved), self._config(run_id))
        return await self._event(run_id)

    async def _event(self, run_id: str) -> AgentEvent:
        snapshot = await self._graph.aget_state(self._config(run_id))
        if snapshot.next:  # paused at the gate
            payload = snapshot.tasks[0].interrupts[0].value
            return AgentEvent("confirm", run_id, tool=payload["tool"], args=payload["args"])
        return AgentEvent("final", run_id, answer=snapshot.values.get("answer") or "")
