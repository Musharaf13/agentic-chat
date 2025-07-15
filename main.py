import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Load Groq API key (set in Render environment variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load LLaMA 3 Model
llm = ChatGroq(model="llama3-8b-8192")

# Define Tools
search_tool = TavilySearch(max_results=2)

def multiply(a: int, b: int) -> int:
    return a * b

tools = [search_tool, multiply]
llm_with_tools = llm.bind_tools(tools)

# Define State for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Node function (LLM + Tools)
def agent_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build Agentic Graph
graph_builder = StateGraph(State)
graph_builder.add_node("agent_node", agent_node)
graph_builder.add_edge(START, "agent_node")
graph_builder.add_edge("agent_node", END)
graph = graph_builder.compile()

# Input Schema
class AgentRequest(BaseModel):
    message: str

# API Endpoint
@app.post("/agent/run")
async def run_agent(request: AgentRequest):
    input_state = {"messages": [request.message]}
    result = graph.invoke(input_state)
    response_message = result["messages"][-1].content
    return {"response": response_message}

# Health Check
@app.get("/")
def health():
    return {"status": "Agentic AI backend is running"}
