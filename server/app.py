from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain.tools import Tool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from uuid import uuid4
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import datetime
import json
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", temperature=0.7, model_kwargs={"streaming": True}
)


# --- Tool Definitions ---
class SearchToolSchema(BaseModel):
    query: str


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def search_web(query: str):
    return TavilySearchResults(max_results=2).invoke({"query": query})


search_tool = Tool.from_function(
    func=search_web,
    name="realtime_web_search",
    description="Use this to fetch live information (like current events, weather, news, launch dates) from the web.",
    args_schema=SearchToolSchema,
)


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current system time in the specified format."""
    return datetime.datetime.now().strftime(format)


@tool
def calculate_days_between(input: str) -> str:
    """Calculate number of days between 'YYYY-MM-DD to YYYY-MM-DD'."""
    try:
        parts = input.strip().replace('"', "").replace("'", "").split(" to ")
        if len(parts) != 2:
            return "Invalid format. Use: 'YYYY-MM-DD to YYYY-MM-DD'"
        start = datetime.datetime.strptime(parts[0].strip(), "%Y-%m-%d")
        end = datetime.datetime.strptime(parts[1].strip(), "%Y-%m-%d")
        return str((end - start).days) + " days"
    except Exception as e:
        return f"Error: {e}"


tools = [search_tool, get_system_time, calculate_days_between]
llm_with_tools = llm.bind_tools(tools=tools)


# LangGraph setup
class State(TypedDict):
    messages: Annotated[list, add_messages]


memory = MemorySaver()


async def model(state: State):
    messages = state["messages"][-12:]  # Keep only last 12 for context
    async for chunk in llm_with_tools.astream(messages):
        if isinstance(chunk, AIMessageChunk):
            yield {"messages": messages + [chunk]}


async def tools_router(state: State):
    last_msg = state["messages"][-1]
    return "tool_node" if getattr(last_msg, "tool_calls", []) else "end"


async def tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for call in tool_calls:
        tool_name, tool_args, tool_id = call["name"], call["args"], call["id"]
        for tool in tools:
            if tool.name == tool_name:
                try:
                    output = tool.invoke(tool_args)
                    tool_messages.append(
                        ToolMessage(content=str(output), tool_call_id=tool_id)
                    )
                except Exception as e:
                    tool_messages.append(
                        ToolMessage(
                            content=f"Tool error: {str(e)}", tool_call_id=tool_id
                        )
                    )
    return {"messages": state["messages"] + tool_messages}


graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_edge("tool_node", "model")
graph_builder.add_conditional_edges(
    "model", tools_router, {"tool_node": "tool_node", "end": END}
)
graph = graph_builder.compile(checkpointer=memory)

# --- FastAPI App with Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize_chunk(chunk):
    return chunk.content if isinstance(chunk, AIMessageChunk) else str(chunk)


@limiter.limit("5/minute")
@app.get("/chat_stream/{message}")
async def chat_stream(request: Request, message: str, checkpoint_id: Optional[str] = Query(None)):
    async def event_stream():
        is_new = checkpoint_id is None
        thread_id = str(uuid4()) if is_new else checkpoint_id

        if is_new:
            yield f'data: {{"type": "checkpoint", "checkpoint_id": "{thread_id}"}}\n\n'

        config = {"configurable": {"thread_id": thread_id}}
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]}, version="v2", config=config
        )

        async for event in events:
            etype = event["event"]

            if etype == "on_chat_model_stream":
                chunk = (
                    event["data"]["chunk"]
                    .content.replace("\n", "\\n")
                    .replace('"', '\\"')
                )
                for word in chunk.split(" "):
                    yield f'data: {{"type": "content", "content": "{word} "}}\n\n'

            elif etype == "on_chat_model_end":
                tool_calls = getattr(event["data"]["output"], "tool_calls", [])
                for call in tool_calls:
                    if call["name"] == "search":
                        query = call["args"].get("query", "")
                        safe = (
                            query.replace('"', '\\"')
                            .replace("'", "\\'")
                            .replace("\n", "\\n")
                        )
                        yield f'data: {{"type": "search_start", "query": "{safe}"}}\n\n'

            elif etype == "on_tool_end" and event["name"] == "search":
                output = event["data"]["output"]
                if isinstance(output, list):
                    urls = [
                        i["url"] for i in output if isinstance(i, dict) and "url" in i
                    ]
                    yield f'data: {{"type": "search_results", "urls": {json.dumps(urls)} }}\n\n'

        yield f'data: {{"type": "end"}}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream")