"""The agent loop: plan -> gate -> act -> plan ... until a final answer or the step limit."""
import json
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from assistant.agent.decision import parse_decision
from assistant.agent.prompts import build_system, render_task
from assistant.agent.types import ToolInfo, ToolProvider
from assistant.core.guardrails import needs_confirmation, preview_args, truncate, wrap_observation
from assistant.core.llm.base import LLMClient, Message
from assistant.core.logging import get_logger

log = get_logger(__name__)


class AgentState(TypedDict, total=False):
    task: str
    scratch: list[str]      # what happened so far: the agent's calls and the observations
    pending: dict | None    # {"tool": ..., "args": ...} waiting to be run
    approved: bool
    tainted: bool           # True once any tool has returned content to this run
    answer: str | None
    steps: int


def build_graph(
    llm: LLMClient,
    tools: ToolProvider,
    infos: list[ToolInfo],
    max_steps: int,
    max_observation_chars: int,
):
    by_name = {t.name: t for t in infos}
    system = build_system(infos)

    async def plan(state: AgentState) -> dict:
        steps = state["steps"]
        if steps >= max_steps:
            return {
                "answer": (
                    f"I stopped after {max_steps} steps without finishing the task. "
                    "You can ask me to continue or narrow the request."
                ),
                "pending": None,
            }

        raw = await llm.generate([Message("user", render_task(state["task"], state["scratch"]))], system=system)
        decision = parse_decision(raw)
        scratch = list(state["scratch"])

        if decision is None:
            log.warning("agent_bad_decision", extra={"fields": {"reply_chars": len(raw)}})
            scratch.append(
                "System: your last reply was not a valid JSON object in the required format. "
                "Reply with exactly one JSON object."
            )
            return {"scratch": scratch, "steps": steps + 1, "pending": None}

        if decision.action == "final":
            return {"answer": decision.answer, "pending": None}

        if decision.tool not in by_name:
            scratch.append(
                f'System: there is no tool named "{decision.tool[:60]}". '
                f"Available tools: {', '.join(by_name)}."
            )
            return {"scratch": scratch, "steps": steps + 1, "pending": None}

        scratch.append(f"You called {decision.tool} with arguments {json.dumps(decision.args)}")
        return {
            "scratch": scratch,
            "steps": steps + 1,
            "pending": {"tool": decision.tool, "args": decision.args},
        }

    def route_after_plan(state: AgentState) -> str:
        if state.get("answer") is not None:
            return END
        if state.get("pending"):
            return "gate"
        return "plan"  # invalid reply or unknown tool: let the model try again

    def gate(state: AgentState) -> dict:
        call = state["pending"]
        info = by_name[call["tool"]]
        if not needs_confirmation(info.read_only_hint, info.open_world_hint, state.get("tainted", False)):
            return {"approved": True}
        # Pauses the run here. When resumed, this node runs again from the top and
        # interrupt() returns the resume value, so nothing before it may have side effects.
        decision = interrupt({"tool": call["tool"], "args": call["args"]})
        return {"approved": bool(decision)}

    async def act(state: AgentState) -> dict:
        call = state["pending"]
        approved = bool(state.get("approved"))
        if approved:
            try:
                obs = await tools.call_tool(call["tool"], call["args"])
            except Exception as exc:  # tool failures are information for the model, not crashes
                obs = f"ERROR: {exc}"
            obs = truncate(obs, max_observation_chars)
        else:
            obs = "DENIED: the user did not approve this action. It was not executed. Do not retry it."

        log.info(
            "tool_call",
            extra={"fields": {
                "tool": call["tool"], "args_preview": preview_args(call["args"]),
                "approved": approved, "result_chars": len(obs),
            }},
        )
        return {
            "scratch": state["scratch"] + [wrap_observation(call["tool"], obs)],
            "pending": None,
            "approved": False,
            # From here on, this run has seen tool output, which may be untrusted.
            "tainted": bool(state.get("tainted")) or approved,
        }

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("gate", gate)
    graph.add_node("act", act)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan, {"gate": "gate", "plan": "plan", END: END})
    graph.add_edge("gate", "act")
    graph.add_edge("act", "plan")
    # interrupt() needs a checkpointer. MemorySaver keeps paused runs in memory only.
    return graph.compile(checkpointer=MemorySaver())
