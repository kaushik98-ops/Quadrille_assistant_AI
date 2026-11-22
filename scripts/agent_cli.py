"""Run the agent from the terminal. Risky tool calls ask for your approval.

  python scripts/agent_cli.py "your task here"

Tool calls are logged to stderr as JSON lines. Set LOG_LEVEL=WARNING to hide them.
"""
import asyncio
import json
import sys
import uuid

from assistant.agent.mcp_client import MCPToolClient
from assistant.agent.runtime import AgentRuntime
from assistant.core.config import settings
from assistant.core.llm.factory import get_llm_client
from assistant.core.logging import new_request_id, setup_logging


async def main(task: str) -> None:
    setup_logging(settings.log_level)
    new_request_id()
    llm = get_llm_client()
    async with MCPToolClient.from_settings() as tools:
        runtime = await AgentRuntime.create(
            llm, tools, settings.agent_max_steps, settings.agent_max_observation_chars
        )
        run_id = uuid.uuid4().hex[:12]
        event = await runtime.start(run_id, task)
        while event.kind == "confirm":
            print(f"\nThe agent wants to run: {event.tool}")
            print(json.dumps(event.args, indent=2, ensure_ascii=False))
            ok = input("Approve? [y/N] ").strip().lower() in ("y", "yes")
            event = await runtime.resume(run_id, ok)
        print(f"\n{event.answer}")


if len(sys.argv) < 2:
    print(__doc__)
else:
    asyncio.run(main(" ".join(sys.argv[1:])))
