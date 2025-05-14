from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import StateGraph, END, add_messages
from langchain.tools import tool, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    AIMessageChunk,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from uuid import uuid4
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import json
import datetime

load_dotenv()

# LLM with streaming
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", temperature=0.7, model_kwargs={"streaming": True}
)


# --- Tool Definitions ---
class SearchToolSchema(BaseModel):
    query: str


def search_web(query: str):
    return TavilySearchResults(max_results=2).invoke({"query": query})


search_tool = Tool.from_function(
    func=search_web,
    name="search",
    description=(
        "Search the web for current events, launch dates, or real-time data. "
        "Use this when you need **any updated information**, like SpaceX launches, live news, or today's weather."
    ),
    args_schema=SearchToolSchema,
)


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Use this tool to get the current system time and date. Especially useful for answering questions that require knowing the current date or time."""
    return datetime.datetime.now().strftime(format)


@tool
def calculate_days_between(input: str) -> str:
    """Use this tool to get Calculated number of days between 'YYYY-MM-DD to YYYY-MM-DD'."""
    try:
        parts = input.replace('"', "").replace("'", "").split(" to ")
        if len(parts) != 2:
            return "Format: 'YYYY-MM-DD to YYYY-MM-DD'"
        start = datetime.datetime.strptime(parts[0].strip(), "%Y-%m-%d")
        end = datetime.datetime.strptime(parts[1].strip(), "%Y-%m-%d")
        return f"{(end - start).days} days"
    except Exception as e:
        return f"Error: {e}"


tools = [search_tool, get_system_time, calculate_days_between]
llm_with_tools = llm.bind_tools(tools)

# Memory
memory = MemorySaver()


# LangGraph state
class State(TypedDict):
    messages: Annotated[List, add_messages]


# Model node (streaming chunk-by-chunk)
async def model(state: State):
    async for chunk in llm_with_tools.astream(state["messages"]):
        yield {"messages": state["messages"] + [chunk]}


# Tool router
async def tools_router(state: State):
    last = state["messages"][-1]
    return "tool_node" if getattr(last, "tool_calls", []) else "end"


# Tool node
async def tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for call in tool_calls:
        for tool in tools:
            if tool.name == call["name"]:
                try:
                    output = tool.invoke(call["args"])
                    results.append(
                        ToolMessage(content=str(output), tool_call_id=call["id"])
                    )
                except Exception as e:
                    results.append(
                        ToolMessage(content=f"Error: {str(e)}", tool_call_id=call["id"])
                    )
    return {"messages": state["messages"] + results}


# Build graph
builder = StateGraph(State)
builder.add_node("model", model)
builder.add_node("tool_node", tool_node)
builder.set_entry_point("model")
builder.add_edge("tool_node", "model")
builder.add_conditional_edges(
    "model", tools_router, {"tool_node": "tool_node", "end": END}
)
graph = builder.compile(checkpointer=memory)

# FastAPI setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# Utility
def safe(text: str):
    return text.replace('"', '\\"').replace("\n", "\\n")


system_prompt = SystemMessage(
    content="Use available tools for any current or factual information. If a question asks about the date, time, or current events, use the tools instead of guessing."
)


# Stream handler
@limiter.limit("10/minute")
@app.get("/chat_stream/{message}")
async def chat_stream(
    request: Request, message: str, checkpoint_id: Optional[str] = Query(None)
):
    async def event_stream():
        is_new = checkpoint_id is None

        if is_new:
            thread_id = str(uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            events = graph.astream_events(
                {"messages": [HumanMessage(content=message)]},
                version="v2",
                config=config,
            )
            yield f'data: {{"type": "checkpoint", "checkpoint_id": "{thread_id}"}}\n\n'
        else:
            config = {"configurable": {"thread_id": checkpoint_id}}
            events = graph.astream_events(
                {"messages": [HumanMessage(content=message)]},
                version="v2",
                config=config,
            )

        async for event in events:
            etype = event["event"]

            if etype == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                for word in content.split(" "):
                    yield f'data: {{"type": "content", "content": "{safe(word)} "}}\n\n'

            elif etype == "on_chat_model_end":
                tool_calls = getattr(event["data"]["output"], "tool_calls", [])
                for call in tool_calls:
                    if call["name"] == "search":
                        query = call["args"].get("query", "")
                        safe_query = (
                            query.replace('"', '\\"')
                            .replace("'", "\\'")
                            .replace("\n", "\\n")
                        )
                        yield f'data: {{"type": "search_start", "query": "{safe_query}"}}\n\n'

            elif etype == "on_tool_end" and event["name"] == "search":
                output = event["data"]["output"]
                if isinstance(output, list):
                    urls = [
                        item["url"]
                        for item in output
                        if isinstance(item, dict) and "url" in item
                    ]
                    urls_json = json.dumps(urls)
                    yield f'data: {{"type": "search_results", "urls": {urls_json}}}\n\n'

        yield 'data: {"type": "end"}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream")
