from typing import TypedDict, Annotated, Optional # Using for defining the state of our agent graph
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, tool, AgentType
# from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from uuid import uuid4
from dotenv import load_dotenv
import datetime
import json

load_dotenv()

# LangChain-compatible Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

from pydantic import BaseModel

def search_web(query: str):
    search_results = TavilySearchResults(max_results=2).invoke({"query": query})

    if not search_results:
        return "No relevant search results found."

    return search_results

class SearchToolSchema(BaseModel):
    query: str

# search_tool = Tool.from_function(
#     func=search_web,
#     name="tavily_Search",
#     description="Retrieve real-time information like weather, news, or current events using web search.",
#     args_schema= SearchToolSchema
# ) # Find only given number of result
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specific format"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

from langchain.tools import tool

@tool
def calculate_days_between(input: str) -> int:
    """Calculate the number of days between two dates in 'YYYY-MM-DD to YYYY-MM-DD' format."""
    from datetime import datetime
    try:
        # Clean and split
        parts = input.strip().replace('"', '').replace("'", "").split(" to ")
        if len(parts) != 2:
            return "Input must be in format: 'YYYY-MM-DD to YYYY-MM-DD'"
        
        start_date = parts[0].strip()
        end_date = parts[1].strip()

        fmt = "%Y-%m-%d"
        delta = datetime.strptime(end_date, fmt) - datetime.strptime(start_date, fmt)
        return delta.days

    except Exception as e:
        return f"Error parsing input: {e}"

search_tool = TavilySearchResults(search_depth='basic') # Find only given number of result

tools = [search_tool, get_system_time, calculate_days_between] # storing search tool in general tool array

memory = MemorySaver()


# llm_with_tools = llm.bind_tools(tools=tools)
# llm_with_tools = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)
llm_with_tools = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)
# llm_with_tools = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
# response = llm_with_tools.invoke("What's the todays weather of Gwalior India")
# response


class State(TypedDict):
    messages: Annotated[list, add_messages]

async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return{
        "messages": state["messages"] + [AIMessage(content=result["output"])]
    }

async def tools_router(state: State):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return "end"
    
async def tool_node(state):
    """Custom tool node that handles tool calls from the LLM."""
    # Get the tool calls from the last message
    tool_calls = state["messages"][-1].tool_calls

    # Initialize list to store tool messages
    tool_messages = []

    # Process each tool call
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Handle the search tool
        if tool_name == search_tool.name:
            # Execute the search tool with the provided arguments
            search_results = await search_tool.ainvoke(tool_args)

            # Create a ToolMessage for this result
            tool_message = ToolMessage(
                content = str(search_results),
                tool_call_id=tool_id,
                name = tool_name
            )

            tool_messages.append(tool_message)

    # Add the tool messages to the state
    return {
        "messages": tool_messages
    }

graph_builder = StateGraph(State)

graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")

graph_builder.add_edge("tool_node", "model")
graph_builder.add_conditional_edges("model", tools_router, {"tool_node": "tool_node", "end": END})

graph = graph_builder.compile(checkpointer=memory)


# Initialize FastAPI
app = FastAPI()

# Add Cors Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=["Content-Type"]
)

def serialize_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )

async def generate_chat_responses(message: str, checkpoint_id: Optional[str]=None):
    is_new_conversation = checkpoint_id is None

    if is_new_conversation:
        # Generate new checkpoint Id for first message in conversation
        new_checkpoint_id = str(uuid4())

        config = {
            "configurable":{
                "thread_id": new_checkpoint_id
            }
        }

        # Initialize with first message
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

        # First send the checkpoint ID
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        new_checkpoint_id = checkpoint_id
        config = {
            "configurable":{
                "thread_id": new_checkpoint_id
            }
        }

        # Continue existing conversation
        events = graph.astream_events(
            {"messages":[HumanMessage(content=message)]},
            version="v2",
            config=config
        )
    
    async for event in events:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk_content = serialize_ai_message_chunk(event["data"]["chunk"])
            # Escape single quotes and newlines for safe JSON parsing
            safe_content = chunk_content.replace("'", "\\").replace("\n", "\\n")

            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"
        
        elif event_type == "on_chat_model_end":
            # Check if there are tool calls for each
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "tavily_search_results_json"]

            if search_calls:
                # Signal that a search is starting
                search_query = search_calls[0]["args"].get("query", "")
                # Escape quotes and special characters
                safe_query = search_query.replace('"', '\\"').replace("'", "\\'").replace("\n","\\n")
                
                yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"
        
        elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
            # Search completed send result or error
            output = event["data"]["output"]

            # Check if output is a list
            if isinstance(output, list):
                urls = []
                for item in output:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])
                
                # Convert urls to JSON and yield them
                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": \{urls_json}\}}\n\n"
    
    # Send an end event
    yield f"data: {{\"type\": \"end\"}}\n\n"
        

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id),
        media_type="text/event-stream"
    )