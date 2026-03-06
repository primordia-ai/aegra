"""Minimal sleep agent for graceful-drain e2e testing.

No LLM calls — just asyncio.sleep so the run stays 'running' long enough
for the test to send SIGTERM.
"""

import asyncio

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    messages: list


async def sleep_node(state: State) -> State:
    await asyncio.sleep(60)
    return state


builder = StateGraph(State)
builder.add_node("sleep", sleep_node)
builder.set_entry_point("sleep")
builder.add_edge("sleep", END)

graph = builder.compile(name="sleep-agent")
