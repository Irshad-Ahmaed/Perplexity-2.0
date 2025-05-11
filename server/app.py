from typing import TypedDict, Annotated, Optional # Using for defining the state of our agent graph
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from dotenv import load_dotenv
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

search_tool = Tool.from_function(
    func=search_web,
    name="tavily_Search",
    description="Retrieve real-time information like weather, news, or current events using web search.",
    args_schema= SearchToolSchema
) # Find only given number of result

tools = [search_tool] # storing search tool in general tool array

memory = MemorySaver()

llm_with_tools = llm.bind_tools(tools=tools)


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return{
        "messages": [result],
    }

async def tools_router(state: State):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and last_message.tool_calls):
        return "tool_node"
    else:
        return "model_end"
    
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
        if tool_name == "tavily_Search":
            search_query = tool_args.get("query", None) or state["messages"][-1].content
            # Execute the search tool with the provided arguments
            search_results = await search_tool.ainvoke(search_query)

            # Create a ToolMessage for this result
            tool_message = ToolMessage(
                content = str(search_results),
                tool_call_id=tool_id,
                name = tool_name
            )

            tool_messages.append(tool_message)

    # Add the tool messages to the state
    return {"messages": state["messages"] + tool_messages}

graph_builder = StateGraph(State)

graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_node("model_end", lambda state: {"messages": state["messages"]})

graph_builder.add_conditional_edges("model", tools_router, {"tool_node": "tool_node", "model_end": END})
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)
graph_instance = graph